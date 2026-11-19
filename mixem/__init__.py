"""mixem: Gaussian mixture models from first principles in NumPy."""

from .bayesian import VariationalGaussianMixture
from .gmm import (
    COVARIANCE_TYPES,
    ConvergenceWarning,
    GaussianMixture,
    estimate_log_gaussian_prob,
    n_parameters,
)
from .kmeans import kmeans, kmeans_plusplus
from .selection import SelectionResult, select_model

__all__ = [
    "COVARIANCE_TYPES",
    "ConvergenceWarning",
    "GaussianMixture",
    "VariationalGaussianMixture",
    "SelectionResult",
    "estimate_log_gaussian_prob",
    "kmeans",
    "kmeans_plusplus",
    "n_parameters",
    "select_model",
]

__version__ = "1.0.0"
