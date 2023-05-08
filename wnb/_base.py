from abc import ABCMeta


class ContinuousDistMixin(metaclass=ABCMeta):
    """
    Mixin class for all continuous probability distributions in wnb.
    """

    name = None

    def __init__(self, **kwargs):
        """Initializes an instance of the probability distribution with given parameters.

        """
        pass

    @classmethod
    def from_data(cls, data):
        """Creates an instance of the class from given data. Distribution parameters will be estimated from the data.

        Returns:
            self: An instance of the class.
        """
        pass

    def pdf(self, x: float) -> float:
        """Returns the value of probability density function at x.

        Args:
            x (float): Input value.

        Returns:
            float: Probability density.
        """
        pass

    def __call__(self, x: float) -> float:
        return self.pdf(x)

    def __repr__(self) -> str:
        pass


class DiscreteDistMixin(metaclass=ABCMeta):
    """
    Mixin class for all discrete probability distributions in wnb.
    """

    name = None

    def __init__(self, **kwargs):
        """Initializes an instance of the probability distribution with given parameters.

        """
        pass

    @classmethod
    def from_data(cls, data):
        """Creates an instance of the class from given data. Distribution parameters will be estimated from the data.

        Returns:
            self: An instance of the class.
        """
        pass

    def pmf(self, x: float) -> float:
        """Returns the value of probability density function at x.

        Args:
            x (float): Input value.

        Returns:
            float: Probability density.
        """
        pass

    def __call__(self, x: float) -> float:
        return self.pmf(x)

    def __repr__(self) -> str:
        pass
