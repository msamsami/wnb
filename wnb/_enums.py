from enum import Enum


__all__ = [
    "Distribution"
]


class Distribution(str, Enum):
    """
    Names of probability distributions.
    """
    NORMAL = "Normal"
    LOGNORMAL = "Lognormal"
    EXPONENTIAL = "Exponential"
    UNIFORM = "Uniform"
    CATEGORICAL = "Categorical"
    MULTINOMIAL = "Multinomial"
    POISSON = "Poisson"
