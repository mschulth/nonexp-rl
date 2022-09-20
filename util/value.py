import torch
from util.features import TimeFeatures


class ValueFunction:

    def __init__(self, value_net):
        self.value_net = value_net

    def dv(self, x, compute_hessian=False):
        v_in = x
        v_in.requires_grad_()

        # push input through network
        v = self.value_net(v_in)

        # compute gradient
        dv = torch.autograd.grad(v, v_in, grad_outputs=torch.ones_like(v), create_graph=True, retain_graph=True)[0]
        vx = dv

        if compute_hessian:
            vxx = torch.stack(
                ([torch.autograd.grad(dv[:, i], v_in, grad_outputs=torch.ones_like(dv[:, i]), create_graph=True,
                                      retain_graph=True)[0] for i in range(v_in.shape[1])]), dim=-1)

        if compute_hessian:
            return v, vx, vxx
        else:
            return v, vx


class TimeDependentValueFunction:

    def __init__(self, value_net, t_exp_lambd):
        self.value_net = value_net
        self.t_exp_lambd = t_exp_lambd

    def dv(self, x, y=None, t=None, compute_hessian=False):
        """
        Computes the value function and its partial derivatives wrt x and t
        """
        if y is None and t is not None:
            y = TimeFeatures.t_to_y(t, self.t_exp_lambd)
        elif y is not None and t is None:
            t = TimeFeatures.y_to_t(y, self.t_exp_lambd)

        # create network input
        v_in = torch.cat((x, y[..., None]), axis=1)
        v_in.requires_grad_()

        # push input through network
        v = self.value_net(v_in)

        # compute gradient
        v_out_dim = v.shape[-1]
        v_in_dim = v_in.shape[-1]
        x_dim = v_in_dim - 1
        if v_out_dim == 1:
            dv = torch.autograd.grad(v, v_in, grad_outputs=torch.ones_like(v), create_graph=True, retain_graph=True)[0]

            vx = dv[..., :-1]
            vy = dv[..., -1]
            dy_dt = TimeFeatures.dy_dt(t, self.t_exp_lambd)
            vt = vy * dy_dt

            if compute_hessian:
                vxx = torch.stack(
                    ([torch.autograd.grad(dv[:, i], v_in, grad_outputs=torch.ones_like(dv[:, i]), create_graph=True,
                                          retain_graph=True)[0] for i in range(v_in.shape[1])]), dim=-1)[:, :x_dim, :x_dim]
        else:
            dvs = tuple(torch.autograd.grad(v[:, i], v_in, grad_outputs=torch.ones_like(v[:, i]), create_graph=True)[0]
                        for i in range(v_out_dim))
            dv = torch.stack(dvs, axis=1)
            vx = dv[..., :-1]
            vy = dv[..., -1]
            dy_dt = TimeFeatures.dy_dt(t, self.t_exp_lambd)
            vt = vy * dy_dt[..., None]
            if compute_hessian:
                vxx = torch.stack([torch.stack([
                    torch.autograd.grad(dv[:, i, j], v_in, grad_outputs=torch.ones_like(dv[:, i, j]),
                                        create_graph=True, retain_graph=True)[0]
                    for j in range(dv.shape[2])], dim=-1)
                    for i in range(dv.shape[1])], dim=1)[:, :, :x_dim, :x_dim]

        if compute_hessian:
            return v, vx, vt, vxx
        else:
            return v, vx, vt
