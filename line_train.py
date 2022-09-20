import torch
from problems.line import LineProblem
from solvers.collocation import CollocationSolverFactory
from type.discounting_models import HyperbolicDiscountingModel


seed = 0
torch.manual_seed(seed)

# setup problem
p = LineProblem()
discount_model = HyperbolicDiscountingModel(alpha_0=5., beta_0=1.)

# setup HJB solver
solver = CollocationSolverFactory.get_collocation_solver(p, discount_model)
value_net = solver.network_model
solver.name = 'line'

# start solver
solver.solve()
