from abc import ABC


class DiscountingModel(ABC):
    pass


class HyperbolicDiscountingModel(DiscountingModel):
    def __init__(self, alpha_0, beta_0):
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0


class ExponentialDiscountingModel(DiscountingModel):
    def __init__(self, hazard):
        self.hazard = hazard
