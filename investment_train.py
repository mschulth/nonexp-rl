import torch
from problems.investment import InvestmentProblem
from solvers.collocation import CollocationSolverFactory
from type.discounting_models import HyperbolicDiscountingModel


seed = 0
torch.manual_seed(seed)

# setup problem
p = InvestmentProblem()
discount_model = HyperbolicDiscountingModel(alpha_0=3., beta_0=1.)

# setup HJB solver
solver = CollocationSolverFactory.get_collocation_solver(p, discount_model)
value_net = solver.network_model
solver.name = 'investment'
solver.n_episodes = 125000

# start solver
solver.solve()
