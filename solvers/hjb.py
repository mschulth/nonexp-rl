import torch
from abc import ABC, abstractmethod


class HJBActionMaximizer(ABC):
    """Maximizes the right hand side of the HJB equation"""
    @classmethod
    @abstractmethod
    def max_rhs(cls, mdp, xs, vx, vt, vxx=None):
        pass


class HJBDiscreteActionMaximizer(HJBActionMaximizer):
    @classmethod
    def max_rhs(cls, mdp, xs, vx, vt, vxx=None):
        n_actions = mdp.a_space.n
        x_dim = mdp.s_space.dim
        u = torch.arange(n_actions)[None]

        # flatten states and actions to push them through the dynamics model
        xs_flt = torch.repeat_interleave(xs[:, None, :], n_actions, 1).reshape((-1, x_dim))
        us_flt = torch.repeat_interleave(u, xs.shape[0], 0).flatten()[:, None]

        # compute right hand side
        fx = mdp.t_model.f(xs_flt, us_flt).reshape(xs.shape[0], -1, xs.shape[1])
        vxf = torch.sum(vx[:, None, :] * fx, axis=2)
        rhs_all = mdp.r_model.r_x(xs)[:, None] + mdp.r_model.r_u(u) + vxf + vt[:, None]
        if vxx is not None:
            gx = mdp.t_model.G(xs_flt, us_flt)
            vxx_rep = torch.repeat_interleave(vxx, n_actions, 0)
            vxxgg = 0.5 * torch.sum(vxx_rep * gx, axis=(1, 2)).reshape(-1, n_actions)
            rhs_all += vxxgg

        # maximize right hand side
        rhs_max, u_max = torch.max(rhs_all, dim=1)

        return u_max, rhs_max, rhs_all
