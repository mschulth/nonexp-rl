from torch import nn
from problems.system import System
from type.discounting_models import DiscountingModel, HyperbolicDiscountingModel, ExponentialDiscountingModel


class TimeIndependentValueNetwork(nn.Sequential):

    def __init__(self, xdim=1, layer_size=64):
        input_dim = xdim  # input dim is state wo time
        layers = (nn.Linear(input_dim, layer_size),
            nn.Sigmoid(),
            nn.Linear(layer_size, layer_size),
            nn.Sigmoid(),
            nn.Linear(layer_size, 1))
        super().__init__(*layers)


class TimeDependentValueNetwork(nn.Sequential):

    def __init__(self, xdim=1, layer_size=64):
        input_dim = xdim + 1  # input dim is state + time
        layers = (nn.Linear(input_dim, layer_size),
            nn.Sigmoid(),
            nn.Linear(layer_size, layer_size),
            nn.Sigmoid(),
            nn.Linear(layer_size, 1))
        super().__init__(*layers)


class SensitivityNetwork(nn.Sequential):

    def __init__(self, xdim=1, p_dim=2, layer_size=64):
        input_dim = xdim + 1  # input dim is state + time
        layers = (nn.Linear(input_dim, layer_size),
            nn.Sigmoid(),
            nn.Linear(layer_size, layer_size),
            nn.Sigmoid(),
            nn.Linear(layer_size, p_dim))
        super().__init__(*layers)


class ValueNetworkFactory:
    @classmethod
    def get_value_network(cls, system: System, discounting_model: DiscountingModel):
        if isinstance(discounting_model, HyperbolicDiscountingModel):
            return TimeDependentValueNetwork(system.s_space.dim)
        elif isinstance(discounting_model, ExponentialDiscountingModel):
            return TimeIndependentValueNetwork(system.s_space.dim)
        else:
            raise AttributeError('Discounting model is not supported.')
