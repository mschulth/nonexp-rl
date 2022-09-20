from abc import ABC, abstractmethod


class RewardModel(ABC):

    @abstractmethod
    def r(self, x, u):
        pass


class SeparableRewardModel(RewardModel):
    def __init__(self, r_x, r_u):
        self.r_x = r_x
        self.r_u = r_u

    def r(self, x, u):
        return self.r_x(x) + self.r_u(u)
