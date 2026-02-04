"""
Walk-Forward Optimization
=========================
Validates trading strategy robustness by testing on out-of-sample data.

This module:
1. Splits data into multiple time-based folds
2. Optimizes parameters on in-sample (training) data
3. Validates on out-of-sample (test) data
4. Calculates robustness metrics
"""

from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class WalkForwardResult:
    """Result from walk-forward analysis."""
    in_sample_results: List[Dict[str, Any]]   # Results from optimization periods
    out_sample_results: List[Dict[str, Any]]  # Results from validation periods
    avg_in_sample_pf: float                   # Average in-sample profit factor
    avg_out_sample_pf: float                  # Average out-of-sample profit factor
    robustness_ratio: float                   # out/in ratio (closer to 1.0 = better)
    is_robust: bool                           # True if strategy passes robustness test

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "in_sample_results": self.in_sample_results,
            "out_sample_results": self.out_sample_results,
            "avg_in_sample_pf": round(self.avg_in_sample_pf, 4),
            "avg_out_sample_pf": round(self.avg_out_sample_pf, 4),
            "robustness_ratio": round(self.robustness_ratio, 4),
            "is_robust": self.is_robust,
        }
