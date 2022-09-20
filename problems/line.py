import torch
from problems.system import System
from type.spaces import BoundedRealSpace, FiniteIntSpace
from type.transition_models import DynamicsModel
from type.reward_models import SeparableRewardModel


class LineProblem(System):

    def __init__(self, diffusion=0.05):
        # state space
        s_space = BoundedRealSpace(-1., 1.)

        # action space
        a_space = FiniteIntSpace(3)

        # dynamics model
        f = lambda x, u: (u - 1.)
        G = lambda x, u: torch.abs(u - 1.)[:, None] * diffusion ** 2
        t_model = DynamicsModel(f, G)

        # reward function
        q_u = 0.1  # action cose
        ll_reward = 3.  # larger later reward
        ll_start = -0.95  # first rewarding state for larger later reward
        a = - ll_reward / (ll_start + 1)
        b = ll_reward * ll_start / (ll_start + 1)
        r_x = lambda x: torch.where(x > 0, torch.clamp(x, max=0.5), torch.clamp(a * x + b, min=0., max=ll_reward))[..., 0]
        r_u = lambda u: -q_u * (u - 1.) ** 2
        r_model = SeparableRewardModel(r_x, r_u)

        super().__init__(s_space, a_space, t_model, r_model)
