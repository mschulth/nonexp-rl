import torch
import numpy as np
import scipy.linalg

from solvers.hjb import HJBDiscreteActionMaximizer
from type.spaces import FiniteIntSpace, BoundedRealSpace
from util.features import TimeFeatures
from util.value import TimeDependentValueFunction


class Simulator:
    """Simulates a system based on the Euler–Maruyama method."""

    def __init__(self, system, value_net, t_exp_lambd):
        self.system = system
        self.t_exp_lambd = t_exp_lambd
        self.value_fun = TimeDependentValueFunction(value_net, t_exp_lambd)

    def f(self, x, t, ret_rhs=False, ret_G=False):
        # convert time points to time features
        y_in = TimeFeatures.t_to_y(t, self.t_exp_lambd)

        # adapt dimensions of input
        if x.ndim == 1:
            x_in = x[None]
        else:
            x_in = x
        if y_in.ndim == 0:
            y_in = y_in[None]

        # compute value function and obtain optimal policy
        if ret_G:
            v, vx, vt, vxx = self.value_fun.dv(x_in, y=y_in, compute_hessian=ret_G)
        else:
            v, vx, vt = self.value_fun.dv(x_in, y=y_in, compute_hessian=ret_G)
        u_max, rhs_max, rhs_all = HJBDiscreteActionMaximizer.max_rhs(self.system, x_in, vx, vt, vxx)
        u_flt = u_max

        # follow system dynamics
        f_val = self.system.t_model.f(x, u_flt)
        if ret_G:
            if self.system.t_model.hasG is not None:
                g_val = self.system.t_model.G(x, u_flt)
            else:
                g_val = None
            if ret_rhs:
                return f_val, g_val, u_flt, rhs_max, rhs_all, v, vx, vt, vxx
            else:
                return f_val, g_val, u_flt
        if ret_rhs:
            return f_val, u_flt, rhs_max, rhs_all, v, vx, vt
        else:
            return f_val, u_flt

    def ode_int_euler(self, x0, t):
        """Solves the ODE using the Euler–Maruyama method."""
        xdim = self.system.s_space.dim
        udim = self.system.a_space.dim
        assert isinstance(self.system.a_space, FiniteIntSpace)
        nu = self.system.a_space.n
        assert(xdim == torch.numel(x0))
        n_t = torch.numel(t)

        # initialize arrays
        x_traj = torch.zeros((n_t, xdim))
        u_traj = torch.zeros((n_t - 1, udim))
        rhs_traj = torch.zeros((n_t - 1, nu))
        x_traj[0] = x0

        for i in range(n_t - 1):
            xv = x_traj[i]
            tv = t[i]
            dt = t[i + 1] - t[i]

            # compute dynamics
            fv, g_val, u, rhs_max, rhs_all, v, vx, vt, vxx = self.f(xv[None], tv[None], ret_rhs=True, ret_G=True)
            fv = fv[0]
            g_val = g_val[0]
            u = u[0]
            xvn = xv + fv * dt
            if g_val is not None:
                x_dim = self.system.s_space.dim
                lu, d, perm = scipy.linalg.ldl(g_val.detach())
                g_sqrt = torch.as_tensor(lu @ np.sqrt(d) @ lu.T)
                noise = torch.randn(x_dim)
                xvn = xvn + g_sqrt @ noise

            # respect bounds
            if isinstance(self.system.s_space, BoundedRealSpace):
                lower = self.system.s_space.bound_lower
                upper = self.system.s_space.bound_upper
                if torch.sum(xvn - lower) < 0:
                    xvn = lower
                elif torch.sum(xvn - upper) > 0:
                    xvn = upper

            # set values in array
            x_traj[i + 1] = xvn
            u_traj[i] = u
            rhs_traj[i] = rhs_all[0]

        return x_traj, u_traj, rhs_traj
