from abc import ABC, abstractmethod
import torch


class Domain:
   Naturals = 1
   Reals = 2


class Space(ABC):
    @abstractmethod
    def __init__(self, domain: Domain, dim: int, is_finite: bool, is_countable: bool):
        """
        Abstract base class for a space, e.g. R^2
        :param domain: The domain of the space, e.g., real or natural numbers
        :param dim: The dimensionality of the space, e.g., 2
        :param is_finite: boolean for finite spaces
        :param is_countable: boolean for countable spaces
        """
        # Save properties
        self.domain = domain
        self.dim = dim
        self.is_finite = is_finite
        self.is_countable = is_countable


class ContinuousSpace(Space):
    def __init__(self, dim: int):
        domain = Domain.Reals
        super().__init__(domain=domain, dim=dim, is_finite=False, is_countable=False)


class RealSpace(ContinuousSpace):
    def __init__(self, dim: int):
        super().__init__(dim=dim)


class BoundedRealSpace(ContinuousSpace):
    def __init__(self, lower, upper):
        self.bound_lower = torch.as_tensor(lower)
        self.bound_upper = torch.as_tensor(upper)
        if torch.is_tensor(lower):
            ldim = lower.numel()
            if not torch.is_tensor(upper):
                raise AttributeError('If lower bound is tenor, the upper needs to be as well.')
            udim = upper.numel()
            if ldim != udim:
                raise AttributeError('Upper and lower bound need to be of the same dimensions.')
        else:
            ldim = 1

        super().__init__(dim=ldim)

    def sample(self, n):
        bound_len = self.bound_upper - self.bound_lower
        return torch.empty((n, self.dim)).uniform_(0, 1) * bound_len[None] + self.bound_lower[None]


class CountableSpace(Space, ABC):
    def __init__(self, dim: int, is_finite: bool):
        domain = Domain.Naturals
        super().__init__(domain=domain, dim=dim, is_finite=is_finite, is_countable=True)


class FiniteIntSpace(CountableSpace):
    def __init__(self, n: int):
        super().__init__(dim=1, is_finite=True)
        self.n = n


class InfiniteIntSpace(CountableSpace):
    def __init__(self, dim=1):
        super().__init__(dim=dim, is_finite=False)
