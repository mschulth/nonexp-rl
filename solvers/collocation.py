import torch
from abc import ABC, abstractmethod
from torch import nn, optim

from problems.system import System
from solvers.hjb import HJBDiscreteActionMaximizer
from solvers.networks import ValueNetworkFactory
from type.discounting_models import DiscountingModel, HyperbolicDiscountingModel, ExponentialDiscountingModel
from util.features import TimeFeatures
from util.value import TimeDependentValueFunction, ValueFunction
from datetime import datetime


class CollocationSolver(ABC):

    def __init__(self, system, value_model, discount_model):
        # number of samples per iteration
        self.n_samples = 10000

        # numer of total episodes
        self.n_episodes = 100000

        # learning rate for the optimizer
        self.opt_lr = 0.003

        # the name used for checkpoint dump
        self.name = 'collocation'

        # models
        self.system = system
        self.network_model = value_model
        self.discount_model = discount_model

    @abstractmethod
    def collocation_loss(self, xs, episode):
        pass

    def solve(self):
        optimizer = optim.Adam(self.network_model.parameters(), lr=self.opt_lr)

        for e in range(self.n_episodes):
            optimizer.zero_grad()

            # sample states
            xs = self.system.s_space.sample(self.n_samples)

            # compute collocation loss
            loss, output = self.collocation_loss(xs, e)

            # optimization steps
            output.backward()
            optimizer.step()

            if e % 5000 == 0 or e == self.n_episodes - 1:
                now = datetime.now()
                print("Episode {}, loss {}, at {}".format(e, output.item(), now))
                torch.save({
                    'epoch': e,
                    'model_state_dict': self.network_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, 'checkpoints/model_{}_{}.pt'.format(self.name, str(e + 1)))


class CollocationHJBSolver(CollocationSolver, ABC):

    def __init__(self, system, value_model, discount_model):
        super().__init__(system, value_model, discount_model)
        # alpha offset in the beginning to stabilize learning, set to zero to disable
        self.alpha_offset_max = 50

        # time until which alpha is decreased
        self.alpha_decay_time = 50000


class ExponentialCollocationHJBSolver(CollocationHJBSolver):

    def collocation_loss(self, xs, episode):
        value_fun = ValueFunction(self.network_model)

        if self.system.t_model.G_ is None:
            v, vx = value_fun.dv(xs)
            vxx = None
        else:
            v, vx, vxx = value_fun.dv(xs, compute_hessian=True)

        # vt is zero
        vt = torch.zeros(1)

        # left hand side
        hazard = self.discount_model.hazard
        tau = 1. / hazard
        lhs = v[:, 0] / tau

        # right hand side
        _, rhs, _ = HJBDiscreteActionMaximizer.max_rhs(self.system, xs, vx, vt, vxx)

        loss = nn.MSELoss()
        output = loss(rhs, lhs)
        return loss, output


class HyperbolicCollocationHJBSolver(CollocationHJBSolver):

    def __init__(self, system, value_model, discount_model, t_exp_lamb=0.2):
        super().__init__(system, value_model, discount_model)
        self.t_exp_lambd = t_exp_lamb  # lambda of exponential distribution to sample t from

    def collocation_loss(self, xs, episode):
        # sample random time features
        ys = torch.rand(self.n_samples)

        # evaluate value function
        value_fun = TimeDependentValueFunction(self.network_model, self.t_exp_lambd)
        if self.system.t_model.hasG is None:
            v, vx, vt = value_fun.dv(xs, y=ys)
            vxx = None
        else:
            v, vx, vt, vxx = value_fun.dv(xs, y=ys, compute_hessian=True)

        # max right hand side of hjb equation
        _, rhs, _ = HJBDiscreteActionMaximizer.max_rhs(self.system, xs, vx, vt, vxx)

        alpha = self.discount_model.alpha_0 + self.alpha_offset(episode)
        beta = self.discount_model.beta_0
        ts = TimeFeatures.y_to_t(ys, self.t_exp_lambd)
        lhs = alpha * v[..., 0] / (beta + ts)

        loss = nn.MSELoss()
        output = loss(rhs, lhs)
        return loss, output

    def alpha_offset(self, episode):
        if episode < self.alpha_decay_time:
            alpha_offset = self.alpha_offset_max * (1 - episode/self.alpha_decay_time)
        else:
            alpha_offset = 0

        return alpha_offset


class CollocationSolverFactory:

    @classmethod
    def get_collocation_solver(cls, system: System, discounting_model: DiscountingModel,
                               value_model=None):
        if value_model is None:
            value_model = ValueNetworkFactory.get_value_network(system, discounting_model)
        if isinstance(discounting_model, HyperbolicDiscountingModel):
            return HyperbolicCollocationHJBSolver(system, value_model, discounting_model)
        elif isinstance(discounting_model, ExponentialDiscountingModel):
            return ExponentialCollocationHJBSolver(system, value_model, discounting_model)
        else:
            raise AttributeError('Discounting model is not supported.')
