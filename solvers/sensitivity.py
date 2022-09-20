import torch
from torch import nn
from solvers.collocation import CollocationSolver
from solvers.hjb import HJBDiscreteActionMaximizer
from util.features import TimeFeatures
from util.value import TimeDependentValueFunction


class HyperbolicCollocationSensitivitySolver(CollocationSolver):
    """Collocation method solver for the IRL method"""

    def __init__(self, mdp, value_model, value_fun_model, t_exp_lamb=0.2, alpha0=3., beta0=1.):
        super().__init__(mdp, value_model, name='irl')
        self.value_fun_model = value_fun_model
        self.t_exp_lambd = t_exp_lamb  # lambda of exponential distribution to sample t from
        self.alpha_0 = alpha0
        self.beta_0 = beta0

    def collocation_loss(self, xs, episode):
        ys = torch.rand(self.n_samples)
        vp_fun = TimeDependentValueFunction(self.network_model, self.t_exp_lambd)
        v_fun = TimeDependentValueFunction(self.value_fun_model, self.t_exp_lambd)

        compute_hessian = self.system.t_model.hasG
        if compute_hessian:
            vp, vpx, vpt, vpxx = vp_fun.dv(xs, y=ys, compute_hessian=compute_hessian)
            v, vx, vt, vxx = v_fun.dv(xs, y=ys, compute_hessian=compute_hessian)
            pass
        else:
            vp, vpx, vpt = vp_fun.dv(xs, y=ys)
            v, vx, vt = v_fun.dv(xs, y=ys)
            vpxx = None
            vxx = None

        # get action that maximizes RHS
        u_max, _, _ = HJBDiscreteActionMaximizer.max_rhs(self.system, xs, vx, vt, vxx)

        # compute function H that should be minimized
        ts = TimeFeatures.y_to_t(1 - ys, self.t_exp_lambd)
        Fp_0 = 1 / (self.beta_0 + ts)
        Fp_1 = -self.alpha_0 / (self.beta_0 + ts) ** 2
        lhs = torch.vstack((Fp_0, Fp_1)).T * v
        lhs += vp * (self.alpha_0 / (self.beta_0 + ts[:, None]))
        rhs = torch.sum(self.system.t_model.f(xs, u_max[:, None])[..., None] * vpx, axis=-1)
        rhs += vpt
        if compute_hessian:
            G = self.system.t_model.G(xs, u_max[:, None])
            gvpxx = 0.5 * torch.sum(vpxx * G[:, None], axis=(2, 3))
            rhs += gvpxx

        # compute loss to minimize H
        loss = nn.MSELoss()
        output = loss(rhs, lhs)
        return loss, output
