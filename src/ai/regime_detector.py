"""
Market Regime Detector
======================
Unsupervised clustering on market features to classify regimes.

Three regimes:
1. TRENDING — High ADX, consistent slope → use trend-following params
2. RANGING  — Low ADX, low slope → tighter entries or skip trading
3. VOLATILE — High ATR, unstable direction → tighten stops, reduce size

Uses KMeans with 3 clusters. Features are normalized before clustering
so the model works across different instruments (USDJPY vs XAUJPY).

The detector labels clusters post-fit by their characteristics,
not by cluster number (which is arbitrary).
"""

import logging
import statistics
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Local imports
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from backtest.indicators import (
    calculate_adx,
    calculate_atr,
    calculate_slope,
)

logger = logging.getLogger(__name__)

# Regime labels
TRENDING = "TRENDING"
RANGING = "RANGING"
VOLATILE = "VOLATILE"


@dataclass
class RegimeResult:
    """Result of regime classification for a single window."""
    regime: str  # TRENDING, RANGING, or VOLATILE
    confidence: float  # 0.0 to 1.0 (distance-based)
    features: Dict[str, float]  # The computed features

    def to_dict(self) -> Dict:
        return {
            "regime": self.regime,
            "confidence": round(self.confidence, 3),
            "features": {k: round(v, 6) for k, v in self.features.items()},
        }


class RegimeDetector:
    """
    Detects market regime using KMeans clustering on market features.

    Usage:
        detector = RegimeDetector()
        detector.fit(historical_prices, window_size=50)
        regime = detector.detect(recent_prices)
        print(f"Current regime: {regime.regime}")
    """

    def __init__(self, n_clusters: int = 3, window_size: int = 50):
        """
        Initialize regime detector.

        Args:
            n_clusters: Number of regimes (default 3)
            window_size: Number of candles per feature window
        """
        self.n_clusters = n_clusters
        self.window_size = window_size
        self.model: Optional[KMeans] = None
        self.scaler: Optional[StandardScaler] = None
        self.cluster_labels: Dict[int, str] = {}
        self.is_fitted = False

    def _compute_features(self, window: List[Dict]) -> Dict[str, float]:
        """
        Compute feature vector from a price window.

        Features:
        1. ADX — trend strength (0-100)
        2. ATR percentile — volatility relative to price level
        3. Slope magnitude — trend direction strength
        4. Slope consistency — how stable the trend is
        """
        adx = calculate_adx(window)
        atr = calculate_atr(window)
        slope = calculate_slope(window)

        # ATR as percentage of price (normalizes across instruments)
        avg_price = statistics.mean(c["close"] for c in window)
        atr_pct = (atr / avg_price * 100) if avg_price > 0 else 0

        # Slope consistency: compare slope of first half vs second half
        half = len(window) // 2
        slope_first = calculate_slope(window[:half]) if half >= 2 else 0
        slope_second = calculate_slope(window[half:]) if half >= 2 else 0

        # Consistency = 1 when both halves agree, 0 when they disagree
        if slope_first != 0 or slope_second != 0:
            if (slope_first > 0 and slope_second > 0) or \
               (slope_first < 0 and slope_second < 0):
                consistency = 1.0
            else:
                consistency = 0.0
        else:
            consistency = 0.5

        return {
            "adx": adx,
            "atr_pct": atr_pct,
            "slope_abs": abs(slope),
            "slope_consistency": consistency,
        }

    def _extract_feature_matrix(self, prices: List[Dict]) -> Tuple[np.ndarray, List[Dict]]:
        """
        Extract feature matrix from price history using sliding windows.

        Args:
            prices: Full price history

        Returns:
            Tuple of (feature_matrix, list_of_feature_dicts)
        """
        features_list = []
        step = max(1, self.window_size // 4)  # Overlap windows for more data

        for i in range(self.window_size, len(prices), step):
            window = prices[i - self.window_size:i]
            features = self._compute_features(window)
            features_list.append(features)

        if not features_list:
            return np.array([]), features_list

        # Convert to numpy matrix
        feature_names = ["adx", "atr_pct", "slope_abs", "slope_consistency"]
        matrix = np.array([
            [f[name] for name in feature_names]
            for f in features_list
        ])

        return matrix, features_list

    def _label_clusters(self, matrix: np.ndarray, labels: np.ndarray) -> Dict[int, str]:
        """
        Label clusters by their characteristics.

        TRENDING: highest ADX + highest slope_abs
        VOLATILE: highest ATR percentile
        RANGING: everything else (lowest ADX)
        """
        cluster_means = {}
        for cluster_id in range(self.n_clusters):
            mask = labels == cluster_id
            if mask.sum() == 0:
                continue
            cluster_means[cluster_id] = {
                "adx": matrix[mask, 0].mean(),
                "atr_pct": matrix[mask, 1].mean(),
                "slope_abs": matrix[mask, 2].mean(),
                "slope_consistency": matrix[mask, 3].mean(),
            }

        if not cluster_means:
            return {i: RANGING for i in range(self.n_clusters)}

        # Sort by characteristics to assign labels
        ids = list(cluster_means.keys())

        # Highest ATR = VOLATILE
        volatile_id = max(ids, key=lambda x: cluster_means[x]["atr_pct"])

        # Among remaining, highest ADX + slope = TRENDING
        remaining = [x for x in ids if x != volatile_id]
        if remaining:
            trending_id = max(remaining, key=lambda x: (
                cluster_means[x]["adx"] + cluster_means[x]["slope_abs"] * 100
            ))
            ranging_ids = [x for x in remaining if x != trending_id]
        else:
            trending_id = volatile_id
            ranging_ids = []

        result = {}
        result[volatile_id] = VOLATILE
        result[trending_id] = TRENDING
        for rid in ranging_ids:
            result[rid] = RANGING

        return result

    def fit(self, prices: List[Dict]) -> bool:
        """
        Fit the regime detector on historical price data.

        Args:
            prices: Historical price data (OHLC dicts)

        Returns:
            True if fitting succeeded
        """
        if len(prices) < self.window_size * 2:
            logger.warning(
                f"Not enough data to fit regime detector "
                f"(need {self.window_size * 2}, got {len(prices)})"
            )
            return False

        matrix, features_list = self._extract_feature_matrix(prices)

        if len(matrix) < self.n_clusters:
            logger.warning(f"Not enough windows ({len(matrix)}) for {self.n_clusters} clusters")
            return False

        # Normalize features
        self.scaler = StandardScaler()
        scaled = self.scaler.fit_transform(matrix)

        # Fit KMeans
        self.model = KMeans(
            n_clusters=self.n_clusters,
            n_init=10,
            random_state=42,
        )
        labels = self.model.fit_predict(scaled)

        # Label clusters by characteristics
        self.cluster_labels = self._label_clusters(matrix, labels)
        self.is_fitted = True

        # Log cluster info
        for cid, label in self.cluster_labels.items():
            count = (labels == cid).sum()
            logger.info(f"Regime {label}: {count} windows ({count/len(labels)*100:.1f}%)")

        return True

    def detect(self, prices: List[Dict]) -> RegimeResult:
        """
        Detect the current market regime.

        Args:
            prices: Recent price data (at least window_size candles)

        Returns:
            RegimeResult with regime label and confidence
        """
        if not self.is_fitted or self.model is None or self.scaler is None:
            # Default to RANGING when not fitted
            return RegimeResult(
                regime=RANGING,
                confidence=0.0,
                features={"adx": 0, "atr_pct": 0, "slope_abs": 0, "slope_consistency": 0},
            )

        if len(prices) < self.window_size:
            return RegimeResult(
                regime=RANGING,
                confidence=0.0,
                features={"adx": 0, "atr_pct": 0, "slope_abs": 0, "slope_consistency": 0},
            )

        window = prices[-self.window_size:]
        features = self._compute_features(window)

        feature_names = ["adx", "atr_pct", "slope_abs", "slope_consistency"]
        feature_vector = np.array([[features[name] for name in feature_names]])
        scaled = self.scaler.transform(feature_vector)

        cluster_id = self.model.predict(scaled)[0]
        regime = self.cluster_labels.get(cluster_id, RANGING)

        # Confidence based on distance to cluster center
        distances = self.model.transform(scaled)[0]
        min_dist = distances[cluster_id]
        max_dist = distances.max()
        confidence = 1.0 - (min_dist / max_dist) if max_dist > 0 else 0.5

        return RegimeResult(
            regime=regime,
            confidence=float(confidence),
            features=features,
        )

    def get_regime_params(
        self,
        regime: str,
        base_params: Dict,
    ) -> Dict:
        """
        Adjust trading parameters based on detected regime.

        Args:
            regime: Current regime (TRENDING, RANGING, VOLATILE)
            base_params: Base trading parameters

        Returns:
            Adjusted parameters for the regime
        """
        params = base_params.copy()

        if regime == TRENDING:
            # Trending: widen TP, keep normal SL, relax position filter
            params["tp_mult"] = min(base_params.get("tp_mult", 2.0) * 1.3, 4.0)
            params["buy_position"] = min(base_params.get("buy_position", 0.5) + 0.05, 0.7)
            params["sell_position"] = max(base_params.get("sell_position", 0.5) - 0.05, 0.3)

        elif regime == RANGING:
            # Ranging: tighter TP, tighter entries, higher ADX threshold
            params["tp_mult"] = max(base_params.get("tp_mult", 2.0) * 0.8, 1.0)
            params["adx_threshold"] = min(base_params.get("adx_threshold", 5) + 5, 30)

        elif regime == VOLATILE:
            # Volatile: tighten SL, reduce TP, stricter RSI
            params["sl_mult"] = max(base_params.get("sl_mult", 1.5) * 0.8, 0.8)
            params["tp_mult"] = max(base_params.get("tp_mult", 2.0) * 0.7, 1.0)
            params["rsi_buy_max"] = max(base_params.get("rsi_buy_max", 75) - 5, 60)
            params["rsi_sell_min"] = min(base_params.get("rsi_sell_min", 25) + 5, 40)

        return params
