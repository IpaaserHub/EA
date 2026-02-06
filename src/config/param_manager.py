"""
Parameter Manager
=================
Handles loading, saving, and validating trading parameters.

This is a core component that:
1. Loads parameters from JSON config files
2. Validates parameters are within safe limits
3. Saves updated parameters after optimization
4. Tracks parameter history for rollback
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional

# Parameter limits - AI cannot suggest values outside these ranges
PARAM_LIMITS = {
    "adx_threshold": {"min": 3, "max": 30, "type": "int"},
    "slope_threshold": {"min": 0.000001, "max": 0.0001, "type": "float"},
    "buy_position": {"min": 0.3, "max": 0.7, "type": "float"},
    "sell_position": {"min": 0.3, "max": 0.7, "type": "float"},
    "rsi_buy_max": {"min": 60, "max": 85, "type": "int"},
    "rsi_sell_min": {"min": 15, "max": 40, "type": "int"},
    "tp_mult": {"min": 1.0, "max": 4.0, "type": "float"},
    "sl_mult": {"min": 0.8, "max": 3.0, "type": "float"},
    # Multi-timeframe parameters
    "h4_adx_min": {"min": 5, "max": 25, "type": "int"},
    # Adaptive exit parameters
    "trail_activation_atr": {"min": 0.5, "max": 2.0, "type": "float"},
    "trail_distance_mult": {"min": 0.5, "max": 3.0, "type": "float"},
    "max_bars_in_trade": {"min": 10, "max": 200, "type": "int"},
}

# Default parameters (same as current system)
DEFAULT_PARAMS = {
    "adx_threshold": 5,
    "slope_threshold": 0.00001,
    "buy_position": 0.50,
    "sell_position": 0.50,
    "rsi_buy_max": 75,
    "rsi_sell_min": 25,
    "tp_mult": 2.0,
    "sl_mult": 1.5,
}


class ParamManager:
    """Manages trading parameters for a specific symbol."""

    def __init__(self, config_dir: str = "config/params"):
        """
        Initialize the parameter manager.

        Args:
            config_dir: Directory where parameter JSON files are stored
        """
        self.config_dir = config_dir
        os.makedirs(config_dir, exist_ok=True)

    def _get_config_path(self, symbol: str) -> str:
        """Get the config file path for a symbol."""
        return os.path.join(self.config_dir, f"{symbol}.json")

    def _get_history_path(self) -> str:
        """Get the optimization history file path."""
        return os.path.join(self.config_dir, "optimization_history.json")

    def load(self, symbol: str) -> Dict[str, Any]:
        """
        Load parameters for a symbol.

        Args:
            symbol: Trading symbol (e.g., "XAUJPY")

        Returns:
            Dictionary of parameters
        """
        config_path = self._get_config_path(symbol)

        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                params = json.load(f)
                # Merge with defaults for any missing keys
                return {**DEFAULT_PARAMS, **params}

        # Return defaults if no config exists
        return DEFAULT_PARAMS.copy()

    def save(self, symbol: str, params: Dict[str, Any], reason: str = "") -> bool:
        """
        Save parameters for a symbol.

        Args:
            symbol: Trading symbol
            params: Dictionary of parameters to save
            reason: Reason for the update (for logging)

        Returns:
            True if saved successfully
        """
        # Validate before saving
        validated_params = self.validate(params)

        config_path = self._get_config_path(symbol)
        with open(config_path, "w") as f:
            json.dump(validated_params, f, indent=2)

        # Log to history
        self._log_history(symbol, validated_params, reason)

        return True

    def validate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and clamp parameters to safe limits.

        Args:
            params: Parameters to validate

        Returns:
            Validated parameters (clamped to limits)
        """
        validated = {}

        for key, value in params.items():
            if key in PARAM_LIMITS:
                limits = PARAM_LIMITS[key]
                # Clamp to min/max
                clamped = max(limits["min"], min(limits["max"], value))
                # Cast to correct type
                if limits["type"] == "int":
                    clamped = int(round(clamped))
                validated[key] = clamped
            else:
                # Keep unknown params as-is
                validated[key] = value

        return validated

    def is_valid(self, params: Dict[str, Any]) -> bool:
        """
        Check if parameters are within valid limits.

        Args:
            params: Parameters to check

        Returns:
            True if all parameters are valid
        """
        for key, value in params.items():
            if key in PARAM_LIMITS:
                limits = PARAM_LIMITS[key]
                if value < limits["min"] or value > limits["max"]:
                    return False
        return True

    def _log_history(self, symbol: str, params: Dict[str, Any], reason: str):
        """Log parameter changes to history file."""
        history_path = self._get_history_path()

        history = []
        if os.path.exists(history_path):
            with open(history_path, "r") as f:
                history = json.load(f)

        entry = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "params": params,
            "reason": reason,
        }
        history.append(entry)

        # Keep last 100 entries
        history = history[-100:]

        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

    def get_history(self, symbol: Optional[str] = None, limit: int = 10) -> list:
        """
        Get parameter change history.

        Args:
            symbol: Filter by symbol (None for all)
            limit: Maximum entries to return

        Returns:
            List of history entries
        """
        history_path = self._get_history_path()

        if not os.path.exists(history_path):
            return []

        with open(history_path, "r") as f:
            history = json.load(f)

        if symbol:
            history = [h for h in history if h.get("symbol") == symbol]

        return history[-limit:]

    def rollback(self, symbol: str, steps: int = 1) -> Optional[Dict[str, Any]]:
        """
        Rollback to previous parameters.

        Args:
            symbol: Trading symbol
            steps: How many versions to go back

        Returns:
            The restored parameters, or None if not possible
        """
        history = self.get_history(symbol, limit=steps + 1)

        if len(history) <= steps:
            return None

        # Get the params from 'steps' ago
        old_params = history[-(steps + 1)]["params"]
        self.save(symbol, old_params, reason=f"Rollback {steps} step(s)")

        return old_params
