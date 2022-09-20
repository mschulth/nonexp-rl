import torch
from problems.system import System
from type.spaces import BoundedRealSpace, FiniteIntSpace
from type.transition_models import DynamicsModel
from type.reward_models import SeparableRewardModel


class InvestmentProblem(System):

    def __init__(self, ru_mult=0.1, interest_diffusion=0.01):
        # state space
        s_bound_l = torch.tensor([0., 0.])
        s_bound_u = torch.tensor([1., 1.])

        # action space
        s_space = BoundedRealSpace(s_bound_l, s_bound_u)
        a_space = FiniteIntSpace(2)

        # dynamics model
        def f(x, u):
            xdot = torch.zeros_like(x)
            xdot[:, 0] = torch.where(x[:, :1] <= 1, u * 0.1, u * 0.)[:, 0]
            return xdot

        def G(x, u):
            s_dim = self.s_space.dim
            G = torch.zeros(x.shape[0], s_dim, s_dim)
            G[:, 1, 1] = interest_diffusion ** 2
            return G

        t_model = DynamicsModel(f, G)

        # reward function
        r_x = lambda x: torch.clamp(x[:, 0], max=1.) * torch.clamp(x[:, 1], min=0., max=1.)
        r_u = lambda u: (1. - u) * ru_mult
        r_model = SeparableRewardModel(r_x, r_u)

        super().__init__(s_space, a_space, t_model, r_model)
