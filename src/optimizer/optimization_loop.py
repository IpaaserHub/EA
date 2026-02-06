"""
Optimization Loop
=================
Main orchestration module that ties together all components:
- Parameter manager for loading/saving configs
- Backtest engine for running simulations
- Optuna optimizer for finding better parameters
- AI analyzer for intelligent refinements

This module:
1. Loads current parameters for a symbol
2. Runs optimization using backtest as objective
3. Validates improvements meet safety thresholds
4. Saves new parameters with history tracking
"""

import os
import sys
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config.param_manager import ParamManager, PARAM_LIMITS
from backtest.data_loader import load_prices, find_data_file
from backtest.engine import BacktestEngine, BacktestResult
from optimizer.optuna_optimizer import OptunaOptimizer, HybridOptimizer, OptimizationResult
from optimizer.ai_analyzer import AIAnalyzer, AnalysisResult

logger = logging.getLogger(__name__)


@dataclass
class OptimizationRun:
    """Complete result of an optimization run."""
    symbol: str
    old_params: Dict[str, Any]
    new_params: Dict[str, Any]
    old_result: BacktestResult
    new_result: BacktestResult
    improvement_pct: float
    optimization_result: OptimizationResult
    ai_analysis: Optional[AnalysisResult]
    applied: bool
    reason: str
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "symbol": self.symbol,
            "old_params": self.old_params,
            "new_params": self.new_params,
            "old_result": self.old_result.to_dict(),
            "new_result": self.new_result.to_dict(),
            "improvement_pct": round(self.improvement_pct, 2),
            "applied": self.applied,
            "reason": self.reason,
            "timestamp": self.timestamp,
        }


class OptimizationLoop:
    """
    Main optimization orchestrator.

    Usage:
        loop = OptimizationLoop(data_dir="data", config_dir="config/params")
        run = loop.optimize_symbol("XAUJPY", n_trials=50)
        print(f"Improvement: {run.improvement_pct}%")
    """

    # Safety thresholds
    MIN_IMPROVEMENT_PCT = 5.0  # Minimum improvement to apply changes
    MIN_TRADES = 20  # Minimum trades to consider results valid
    MAX_DRAWDOWN_INCREASE_PCT = 20.0  # Max allowed increase in drawdown

    def __init__(
        self,
        data_dir: str = "data",
        config_dir: str = "config/params",
        openai_api_key: Optional[str] = None,
        use_ai: bool = True,
    ):
        """
        Initialize optimization loop.

        Args:
            data_dir: Directory containing price data CSVs
            config_dir: Directory containing parameter JSON files
            openai_api_key: OpenAI API key for AI analysis
            use_ai: Whether to use AI analysis (default True)
        """
        self.data_dir = data_dir
        self.config_dir = config_dir
        self.use_ai = use_ai

        # Initialize parameter manager
        self.param_manager = ParamManager(config_dir)

        # Initialize AI analyzer if requested
        self.ai_analyzer = None
        if use_ai:
            self.ai_analyzer = AIAnalyzer(api_key=openai_api_key)

    def optimize_symbol(
        self,
        symbol: str,
        n_trials: int = 50,
        ai_refinement_trials: int = 10,
        auto_apply: bool = False,
        min_improvement: Optional[float] = None,
    ) -> OptimizationRun:
        """
        Run optimization for a single symbol.

        Args:
            symbol: Trading symbol (e.g., "XAUJPY")
            n_trials: Number of Optuna trials
            ai_refinement_trials: Number of AI refinement trials
            auto_apply: Automatically apply improvements (default False)
            min_improvement: Minimum improvement % to apply (overrides class default)

        Returns:
            OptimizationRun with complete results
        """
        timestamp = datetime.now().isoformat()
        min_imp = min_improvement or self.MIN_IMPROVEMENT_PCT

        logger.info(f"Starting optimization for {symbol}")

        # Load current parameters
        old_params = self.param_manager.load(symbol)
        if not old_params or old_params == {}:
            raise ValueError(f"No parameters found for {symbol}")

        # Load price data
        data_file = find_data_file(symbol, self.data_dir)
        if not data_file:
            raise ValueError(f"No data file found for {symbol}")

        prices = load_prices(data_file)
        if len(prices) < 100:
            raise ValueError(f"Insufficient price data for {symbol}: {len(prices)} candles")

        logger.info(f"Loaded {len(prices)} candles for {symbol}")

        # Run backtest with current params (baseline)
        engine = BacktestEngine(prices)
        old_result = engine.run(old_params)

        logger.info(f"Baseline: PF={old_result.profit_factor:.2f}, WR={old_result.win_rate:.1f}%, Trades={old_result.total_trades}")

        # Define objective function
        def objective(params: Dict[str, Any]) -> float:
            result = engine.run(params)
            # Primary objective: profit factor
            # Secondary: penalize low trade count
            if result.total_trades < self.MIN_TRADES:
                return result.profit_factor * 0.5  # Penalize
            return result.profit_factor

        # Define backtest function for AI (returns full result dict)
        def backtest_fn(params: Dict[str, Any]) -> Dict[str, Any]:
            result = engine.run(params)
            return result.to_dict()

        # Run optimization
        if self.ai_analyzer:
            optimizer = HybridOptimizer(
                optuna_optimizer=OptunaOptimizer(study_name=f"{symbol}_optimization"),
                ai_analyzer=self.ai_analyzer,
            )
            opt_result = optimizer.optimize(
                objective_fn=objective,
                backtest_fn=backtest_fn,
                initial_params=old_params,
                optuna_trials=n_trials,
                ai_refinement_trials=ai_refinement_trials,
            )
        else:
            optuna_opt = OptunaOptimizer(study_name=f"{symbol}_optimization")
            opt_result = optuna_opt.optimize(
                objective,
                n_trials=n_trials,
                initial_params=old_params,
            )

        # Merge best params with non-optimized params
        new_params = old_params.copy()
        for key, value in opt_result.best_params.items():
            if key in PARAM_LIMITS:
                new_params[key] = value

        # Run backtest with new params
        new_result = engine.run(new_params)

        # Calculate improvement
        old_pf = old_result.profit_factor if old_result.profit_factor > 0 else 0.001
        new_pf = new_result.profit_factor if new_result.profit_factor > 0 else 0.001
        improvement_pct = ((new_pf - old_pf) / old_pf) * 100

        logger.info(f"Optimized: PF={new_result.profit_factor:.2f}, WR={new_result.win_rate:.1f}%, Improvement={improvement_pct:.1f}%")

        # Get AI analysis if available
        ai_analysis = None
        if self.ai_analyzer:
            ai_analysis = self.ai_analyzer.analyze(
                new_result.to_dict(),
                new_params,
                symbol=symbol,
            )

        # Decide whether to apply
        applied = False
        reason = ""

        if new_result.total_trades < self.MIN_TRADES:
            reason = f"Insufficient trades ({new_result.total_trades} < {self.MIN_TRADES})"
        elif improvement_pct < min_imp:
            reason = f"Improvement too small ({improvement_pct:.1f}% < {min_imp}%)"
        elif new_result.max_drawdown > old_result.max_drawdown * (1 + self.MAX_DRAWDOWN_INCREASE_PCT / 100):
            reason = f"Drawdown increased too much"
        elif new_result.profit_factor < 1.0:
            reason = f"New profit factor below 1.0"
        else:
            reason = "Meets all criteria"
            if auto_apply:
                self.param_manager.save(symbol, new_params, reason="optimizer auto-apply")
                applied = True
                reason = "Applied automatically"
                logger.info(f"Applied new parameters for {symbol}")

        run = OptimizationRun(
            symbol=symbol,
            old_params=old_params,
            new_params=new_params,
            old_result=old_result,
            new_result=new_result,
            improvement_pct=improvement_pct,
            optimization_result=opt_result,
            ai_analysis=ai_analysis,
            applied=applied,
            reason=reason,
            timestamp=timestamp,
        )

        return run

    def optimize_all_symbols(
        self,
        symbols: Optional[List[str]] = None,
        n_trials: int = 50,
        auto_apply: bool = False,
    ) -> List[OptimizationRun]:
        """
        Run optimization for multiple symbols.

        Args:
            symbols: List of symbols to optimize (None = auto-detect)
            n_trials: Number of trials per symbol
            auto_apply: Automatically apply improvements

        Returns:
            List of OptimizationRun results
        """
        if symbols is None:
            # Auto-detect from config files
            symbols = []
            if os.path.exists(self.config_dir):
                for f in os.listdir(self.config_dir):
                    if f.endswith('.json'):
                        symbols.append(f.replace('.json', ''))

        results = []
        for symbol in symbols:
            try:
                run = self.optimize_symbol(
                    symbol,
                    n_trials=n_trials,
                    auto_apply=auto_apply,
                )
                results.append(run)
            except Exception as e:
                logger.error(f"Failed to optimize {symbol}: {e}")

        return results

    def apply_run(self, run: OptimizationRun) -> bool:
        """
        Apply a previous optimization run.

        Args:
            run: OptimizationRun to apply

        Returns:
            True if applied successfully
        """
        if run.applied:
            logger.warning(f"Run for {run.symbol} already applied")
            return False

        self.param_manager.save(run.symbol, run.new_params, reason="optimizer manual apply")
        run.applied = True
        run.reason = "Applied manually"
        logger.info(f"Applied optimization for {run.symbol}")
        return True

    def rollback(self, symbol: str, versions_back: int = 1) -> Optional[Dict[str, Any]]:
        """
        Rollback parameters to a previous version.

        Args:
            symbol: Trading symbol
            versions_back: How many versions to rollback

        Returns:
            Restored parameters, or None if failed
        """
        return self.param_manager.rollback(symbol, versions_back)


def run_single_optimization(
    symbol: str,
    data_dir: str = "data",
    config_dir: str = "config/params",
    n_trials: int = 50,
    auto_apply: bool = False,
) -> OptimizationRun:
    """
    Convenience function to run optimization for a single symbol.

    Args:
        symbol: Trading symbol
        data_dir: Directory containing price data
        config_dir: Directory containing parameters
        n_trials: Number of optimization trials
        auto_apply: Whether to automatically apply improvements

    Returns:
        OptimizationRun with complete results
    """
    loop = OptimizationLoop(data_dir=data_dir, config_dir=config_dir)
    return loop.optimize_symbol(symbol, n_trials=n_trials, auto_apply=auto_apply)


def print_run_summary(run: OptimizationRun) -> None:
    """Print a human-readable summary of an optimization run."""
    print(f"\n{'='*60}")
    print(f"Optimization Summary for {run.symbol}")
    print(f"{'='*60}")
    print(f"Timestamp: {run.timestamp}")
    print(f"\nBaseline Performance:")
    print(f"  Profit Factor: {run.old_result.profit_factor:.2f}")
    print(f"  Win Rate: {run.old_result.win_rate:.1f}%")
    print(f"  Total Trades: {run.old_result.total_trades}")
    print(f"  Max Drawdown: {run.old_result.max_drawdown:.2f}")
    print(f"\nOptimized Performance:")
    print(f"  Profit Factor: {run.new_result.profit_factor:.2f}")
    print(f"  Win Rate: {run.new_result.win_rate:.1f}%")
    print(f"  Total Trades: {run.new_result.total_trades}")
    print(f"  Max Drawdown: {run.new_result.max_drawdown:.2f}")
    print(f"\nImprovement: {run.improvement_pct:+.1f}%")
    print(f"Status: {'APPLIED' if run.applied else 'NOT APPLIED'}")
    print(f"Reason: {run.reason}")

    if run.old_params != run.new_params:
        print(f"\nParameter Changes:")
        for key in run.new_params:
            if key in run.old_params and run.old_params[key] != run.new_params[key]:
                print(f"  {key}: {run.old_params[key]} -> {run.new_params[key]}")

    print(f"{'='*60}\n")
