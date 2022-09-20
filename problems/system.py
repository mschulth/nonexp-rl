from type.spaces import Space
from type.transition_models import DynamicsModel
from type.reward_models import RewardModel


class System:

    def __init__(self,
                 s_space: Space,
                 a_space: Space,
                 t_model: DynamicsModel,
                 r_model: RewardModel):
        self.s_space = s_space
        self.a_space = a_space
        self.t_model = t_model
        self.r_model = r_model
