"""
Optuna Optimizer
================
Uses Optuna for Bayesian hyperparameter optimization of trading parameters.

This module:
1. Defines the parameter search space from PARAM_LIMITS
2. Runs Optuna trials with backtest as objective function
3. Tracks optimization history
4. Supports warm-starting from previous best params
"""

import os
import sys
import logging
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config.param_manager import PARAM_LIMITS

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result from an optimization run."""
    best_params: Dict[str, Any]
    best_value: float  # Objective value (profit factor)
    n_trials: int
    study_name: str
    optimization_history: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "best_params": self.best_params,
            "best_value": round(self.best_value, 4),
            "n_trials": self.n_trials,
            "study_name": self.study_name,
            "timestamp": datetime.now().isoformat(),
        }


class OptunaOptimizer:
    """
    Optimizes trading parameters using Optuna.

    Usage:
        def objective(params):
            result = run_backtest(prices, params)
            return result.profit_factor

        optimizer = OptunaOptimizer()
        result = optimizer.optimize(objective, n_trials=50)
        print(f"Best profit factor: {result.best_value}")
    """

    def __init__(
        self,
        study_name: str = "trading_optimization",
        storage: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize optimizer.

        Args:
            study_name: Name for the Optuna study
            storage: SQLite URL for persistence (e.g., "sqlite:///optuna.db")
            seed: Random seed for reproducibility
        """
        self.study_name = study_name
        self.storage = storage
        self.seed = seed
        self._study = None

    def optimize(
        self,
        objective_fn: Callable[[Dict[str, Any]], float],
        n_trials: int = 50,
        initial_params: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
        show_progress: bool = True,
    ) -> OptimizationResult:
        """
        Run optimization to find best parameters.

        Args:
            objective_fn: Function that takes params dict and returns score (higher is better)
            n_trials: Number of trials to run
            initial_params: Starting parameters (for warm-start)
            timeout: Max seconds to run (None = no limit)
            show_progress: Show progress bar

        Returns:
            OptimizationResult with best parameters
        """
        try:
            import optuna
            from optuna.samplers import TPESampler
        except ImportError:
            logger.error("Optuna not installed. Install with: pip install optuna")
            raise ImportError("Optuna required. Install with: pip install optuna")

        # Suppress Optuna info logs unless debugging
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Create sampler with seed
        sampler = TPESampler(seed=self.seed)

        # Create or load study
        self._study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            load_if_exists=True,
            direction="maximize",  # We want to maximize profit factor
            sampler=sampler,
        )

        # Enqueue initial params if provided
        if initial_params:
            valid_params = self._filter_valid_params(initial_params)
            if valid_params:
                self._study.enqueue_trial(valid_params)

        # Define objective wrapper
        def objective(trial):
            params = self._suggest_params(trial)
            try:
                score = objective_fn(params)
                return score if score is not None else 0.0
            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return 0.0

        # Run optimization
        self._study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=show_progress,
        )

        # Build result
        best_params = self._study.best_params

        # Ensure types are correct
        for param_name, value in best_params.items():
            if param_name in PARAM_LIMITS:
                if PARAM_LIMITS[param_name]["type"] == "int":
                    best_params[param_name] = int(round(value))

        # Build optimization history
        history = []
        for trial in self._study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                history.append({
                    "number": trial.number,
                    "value": trial.value,
                    "params": trial.params,
                })

        return OptimizationResult(
            best_params=best_params,
            best_value=self._study.best_value,
            n_trials=len(self._study.trials),
            study_name=self.study_name,
            optimization_history=history,
        )

    def _suggest_params(self, trial) -> Dict[str, Any]:
        """
        Suggest parameters for a trial using Optuna.

        Maps PARAM_LIMITS to Optuna's suggest_* methods.
        """
        params = {}

        for param_name, limits in PARAM_LIMITS.items():
            param_type = limits["type"]
            min_val = limits["min"]
            max_val = limits["max"]

            if param_type == "int":
                params[param_name] = trial.suggest_int(param_name, min_val, max_val)
            elif param_type == "float":
                # Use log scale for very small ranges like slope_threshold
                if max_val / min_val > 100:
                    params[param_name] = trial.suggest_float(
                        param_name, min_val, max_val, log=True
                    )
                else:
                    params[param_name] = trial.suggest_float(param_name, min_val, max_val)
            else:
                # Default to float
                params[param_name] = trial.suggest_float(param_name, min_val, max_val)

        return params

    def _filter_valid_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Filter params to only include those in PARAM_LIMITS."""
        return {
            k: v for k, v in params.items()
            if k in PARAM_LIMITS
        }

    def get_study(self):
        """Get the underlying Optuna study."""
        return self._study

    def get_best_trials(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get the top N trials by objective value."""
        if not self._study:
            return []

        import optuna
        completed = [
            t for t in self._study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
        sorted_trials = sorted(completed, key=lambda t: t.value, reverse=True)

        return [
            {
                "number": t.number,
                "value": t.value,
                "params": t.params,
            }
            for t in sorted_trials[:n]
        ]


class HybridOptimizer:
    """
    Combines Optuna optimization with AI analysis.

    Strategy:
    1. Run Optuna for broad exploration
    2. Use AI to analyze best results and suggest refinements
    3. Test AI suggestions with more Optuna trials
    """

    def __init__(
        self,
        optuna_optimizer: Optional[OptunaOptimizer] = None,
        ai_analyzer=None,
    ):
        """
        Initialize hybrid optimizer.

        Args:
            optuna_optimizer: OptunaOptimizer instance
            ai_analyzer: AIAnalyzer instance (optional)
        """
        self.optuna = optuna_optimizer or OptunaOptimizer()
        self.ai_analyzer = ai_analyzer

    def optimize(
        self,
        objective_fn: Callable[[Dict[str, Any]], float],
        backtest_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
        initial_params: Dict[str, Any],
        optuna_trials: int = 30,
        ai_refinement_trials: int = 10,
    ) -> OptimizationResult:
        """
        Run hybrid optimization.

        Args:
            objective_fn: Function returning score from params
            backtest_fn: Function returning full backtest result dict
            initial_params: Starting parameters
            optuna_trials: Number of Optuna exploration trials
            ai_refinement_trials: Number of trials for AI refinements

        Returns:
            OptimizationResult with best parameters
        """
        # Phase 1: Optuna exploration
        logger.info(f"Phase 1: Running {optuna_trials} Optuna trials")
        result = self.optuna.optimize(
            objective_fn,
            n_trials=optuna_trials,
            initial_params=initial_params,
            show_progress=True,
        )

        # If no AI analyzer, return Optuna results
        if not self.ai_analyzer:
            return result

        # Phase 2: AI refinement
        logger.info("Phase 2: AI analysis of best result")
        best_params = result.best_params

        # Get full backtest result for AI analysis
        backtest_result = backtest_fn(best_params)

        # Get AI suggestions
        analysis = self.ai_analyzer.analyze(
            backtest_result,
            best_params,
            additional_context=f"This was the best result from {optuna_trials} optimization trials",
        )

        if not analysis.suggestions:
            logger.info("AI has no additional suggestions")
            return result

        # Apply AI suggestions and test
        ai_params = analysis.get_new_params(best_params)
        logger.info(f"Phase 3: Testing AI suggestions with {ai_refinement_trials} trials")

        # Run more Optuna trials starting from AI suggestions
        refined_result = self.optuna.optimize(
            objective_fn,
            n_trials=ai_refinement_trials,
            initial_params=ai_params,
            show_progress=True,
        )

        # Return best overall result
        if refined_result.best_value > result.best_value:
            logger.info(f"AI refinement improved result: {result.best_value:.4f} -> {refined_result.best_value:.4f}")
            return refined_result
        else:
            logger.info(f"Original Optuna result was better: {result.best_value:.4f}")
            return result


def run_optimization(
    objective_fn: Callable[[Dict[str, Any]], float],
    n_trials: int = 50,
    initial_params: Optional[Dict[str, Any]] = None,
    study_name: str = "trading_optimization",
) -> OptimizationResult:
    """
    Convenience function to run optimization.

    Args:
        objective_fn: Function that takes params and returns score
        n_trials: Number of trials
        initial_params: Starting parameters
        study_name: Name for the study

    Returns:
        OptimizationResult with best parameters
    """
    optimizer = OptunaOptimizer(study_name=study_name)
    return optimizer.optimize(
        objective_fn,
        n_trials=n_trials,
        initial_params=initial_params,
    )
