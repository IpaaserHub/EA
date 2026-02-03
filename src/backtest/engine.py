"""
Backtest Engine
===============
Simulates trading with given parameters on historical data.

This is the core component that:
1. Checks entry conditions (ADX, slope, RSI, position)
2. Calculates SL/TP based on ATR
3. Simulates trade outcomes
4. Returns performance metrics
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from .indicators import (
    calculate_rsi,
    calculate_adx,
    calculate_atr,
    calculate_slope,
    calculate_position_in_range,
)


@dataclass
class Trade:
    """Represents a single trade."""
    entry_index: int
    entry_price: float
    direction: str  # "BUY" or "SELL"
    sl: float
    tp: float
    exit_index: Optional[int] = None
    exit_price: Optional[float] = None
    profit: float = 0.0
    outcome: str = "OPEN"  # "WIN", "LOSS", "OPEN"


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    profit_factor: float
    total_profit: float
    gross_profit: float
    gross_loss: float
    max_drawdown: float
    avg_win: float
    avg_loss: float
    trades: List[Trade]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_trades": self.total_trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": round(self.win_rate, 2),
            "profit_factor": round(self.profit_factor, 2),
            "total_profit": round(self.total_profit, 2),
            "gross_profit": round(self.gross_profit, 2),
            "gross_loss": round(self.gross_loss, 2),
            "max_drawdown": round(self.max_drawdown, 2),
            "avg_win": round(self.avg_win, 2),
            "avg_loss": round(self.avg_loss, 2),
        }


class BacktestEngine:
    """
    Runs backtests on historical price data.

    Usage:
        engine = BacktestEngine(prices)
        result = engine.run(params)
        print(f"Win rate: {result.win_rate}%")
    """

    def __init__(self, prices: List[Dict], lookback: int = 50):
        """
        Initialize backtest engine.

        Args:
            prices: List of OHLC dicts [{"open": x, "high": x, "low": x, "close": x}, ...]
            lookback: Number of candles to use for indicator calculation
        """
        self.prices = prices
        self.lookback = lookback

    def run(self, params: Dict[str, Any], max_trades: int = 1000) -> BacktestResult:
        """
        Run backtest with given parameters.

        Args:
            params: Trading parameters dict
            max_trades: Maximum trades to simulate

        Returns:
            BacktestResult with performance metrics
        """
        if len(self.prices) < self.lookback + 20:
            return self._empty_result()

        trades = []
        in_trade = False
        current_trade = None

        # Extract parameters with defaults
        adx_threshold = params.get("adx_threshold", 5)
        slope_threshold = params.get("slope_threshold", 0.00001)
        buy_position = params.get("buy_position", 0.5)
        sell_position = params.get("sell_position", 0.5)
        rsi_buy_max = params.get("rsi_buy_max", 75)
        rsi_sell_min = params.get("rsi_sell_min", 25)
        tp_mult = params.get("tp_mult", 2.0)
        sl_mult = params.get("sl_mult", 1.5)

        # Simulate trading
        for i in range(self.lookback, len(self.prices) - 1):
            current_price = self.prices[i]["close"]
            next_candle = self.prices[i + 1]

            # If in a trade, check for exit
            if in_trade and current_trade:
                exit_result = self._check_exit(current_trade, next_candle)
                if exit_result:
                    current_trade.exit_index = i + 1
                    current_trade.exit_price = exit_result["price"]
                    current_trade.profit = exit_result["profit"]
                    current_trade.outcome = exit_result["outcome"]
                    trades.append(current_trade)
                    in_trade = False
                    current_trade = None

                    if len(trades) >= max_trades:
                        break
                continue

            # Check for entry signal
            window = self.prices[i - self.lookback:i]
            signal = self._check_entry(
                window,
                current_price,
                adx_threshold,
                slope_threshold,
                buy_position,
                sell_position,
                rsi_buy_max,
                rsi_sell_min,
            )

            if signal:
                atr = calculate_atr(window)

                if signal == "BUY":
                    sl = current_price - (atr * sl_mult)
                    tp = current_price + (atr * tp_mult)
                else:  # SELL
                    sl = current_price + (atr * sl_mult)
                    tp = current_price - (atr * tp_mult)

                current_trade = Trade(
                    entry_index=i,
                    entry_price=current_price,
                    direction=signal,
                    sl=sl,
                    tp=tp,
                )
                in_trade = True

        return self._calculate_results(trades)

    def _check_entry(
        self,
        window: List[Dict],
        current_price: float,
        adx_threshold: float,
        slope_threshold: float,
        buy_position: float,
        sell_position: float,
        rsi_buy_max: float,
        rsi_sell_min: float,
    ) -> Optional[str]:
        """Check if entry conditions are met."""
        adx = calculate_adx(window)
        rsi = calculate_rsi(window)
        slope = calculate_slope(window)
        position = calculate_position_in_range(window)

        # BUY conditions
        if (
            adx > adx_threshold
            and slope > slope_threshold
            and position < buy_position
            and rsi < rsi_buy_max
        ):
            return "BUY"

        # SELL conditions
        if (
            adx > adx_threshold
            and slope < -slope_threshold
            and position > sell_position
            and rsi > rsi_sell_min
        ):
            return "SELL"

        return None

    def _check_exit(self, trade: Trade, candle: Dict) -> Optional[Dict]:
        """Check if trade hits SL or TP."""
        high = candle["high"]
        low = candle["low"]

        if trade.direction == "BUY":
            # Check TP first (more favorable)
            if high >= trade.tp:
                return {
                    "price": trade.tp,
                    "profit": trade.tp - trade.entry_price,
                    "outcome": "WIN",
                }
            # Check SL
            if low <= trade.sl:
                return {
                    "price": trade.sl,
                    "profit": trade.sl - trade.entry_price,
                    "outcome": "LOSS",
                }
        else:  # SELL
            # Check TP first
            if low <= trade.tp:
                return {
                    "price": trade.tp,
                    "profit": trade.entry_price - trade.tp,
                    "outcome": "WIN",
                }
            # Check SL
            if high >= trade.sl:
                return {
                    "price": trade.sl,
                    "profit": trade.entry_price - trade.sl,
                    "outcome": "LOSS",
                }

        return None

    def _calculate_results(self, trades: List[Trade]) -> BacktestResult:
        """Calculate performance metrics from trades."""
        if not trades:
            return self._empty_result()

        wins = [t for t in trades if t.outcome == "WIN"]
        losses = [t for t in trades if t.outcome == "LOSS"]

        gross_profit = sum(t.profit for t in wins)
        gross_loss = abs(sum(t.profit for t in losses))
        total_profit = gross_profit - gross_loss

        # Calculate drawdown
        cumulative = 0
        peak = 0
        max_drawdown = 0
        for trade in trades:
            cumulative += trade.profit
            if cumulative > peak:
                peak = cumulative
            drawdown = peak - cumulative
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        return BacktestResult(
            total_trades=len(trades),
            wins=len(wins),
            losses=len(losses),
            win_rate=float(len(wins) / len(trades) * 100) if trades else 0.0,
            profit_factor=float(gross_profit / gross_loss) if gross_loss > 0 else 0.0,
            total_profit=total_profit,
            gross_profit=gross_profit,
            gross_loss=gross_loss,
            max_drawdown=max_drawdown,
            avg_win=(gross_profit / len(wins)) if wins else 0,
            avg_loss=(gross_loss / len(losses)) if losses else 0,
            trades=trades,
        )

    def _empty_result(self) -> BacktestResult:
        """Return empty result when no trades possible."""
        return BacktestResult(
            total_trades=0,
            wins=0,
            losses=0,
            win_rate=0,
            profit_factor=0,
            total_profit=0,
            gross_profit=0,
            gross_loss=0,
            max_drawdown=0,
            avg_win=0,
            avg_loss=0,
            trades=[],
        )


def run_backtest(
    prices: List[Dict],
    params: Dict[str, Any],
    lookback: int = 50,
    max_trades: int = 1000,
) -> BacktestResult:
    """
    Convenience function to run a backtest.

    Args:
        prices: List of OHLC dicts
        params: Trading parameters
        lookback: Indicator calculation window
        max_trades: Maximum trades to simulate

    Returns:
        BacktestResult with performance metrics
    """
    engine = BacktestEngine(prices, lookback)
    return engine.run(params, max_trades)
