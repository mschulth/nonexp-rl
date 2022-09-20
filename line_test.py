import torch
import matplotlib.pyplot as plt
from problems.line import LineProblem
from solvers.hjb import HJBDiscreteActionMaximizer
from solvers.networks import ValueNetworkFactory
from type.discounting_models import HyperbolicDiscountingModel
from util.features import TimeFeatures
from util.simulation import Simulator
from util.value import TimeDependentValueFunction


seed = 0
torch.manual_seed(seed)

# load checkpoint
cpdict = torch.load('checkpoints/model_line_100000.pt')

# setup problem
p = LineProblem()
discount_model = HyperbolicDiscountingModel(alpha_0=5., beta_0=1.)
t_exp_lambd = 0.2

# setup HJB solver
value_net = ValueNetworkFactory.get_value_network(p, discount_model)
value_net.load_state_dict(cpdict['model_state_dict'])

# generate combinations of states and time points
N = 200
xl = torch.linspace(-1., 1., N)
yl = torch.linspace(0, 0.6, N)
X, Y = torch.meshgrid(xl, yl, indexing='ij')
x = X.flatten()[:, None]
y = Y.flatten()

# compute value function
vf = TimeDependentValueFunction(value_net, t_exp_lambd)
v, vx, vt, vxx = vf.dv(x, y=y, compute_hessian=True)

# compute rhs of HJB equation
u_max, rhs, rhs_all = HJBDiscreteActionMaximizer.max_rhs(p, x, vx, vt, vxx)

v_plt = v.detach()[:, 0].reshape_as(X)
u_plt = u_max.detach().reshape_as(X)
ts_plt = TimeFeatures.y_to_t(y, t_exp_lambd).reshape_as(X)

# just value function
cf = plt.contourf(ts_plt.T, X.T, v_plt.T, levels=15)
plt.colorbar(cf)
plt.title('Value function')
plt.xlabel('t')
plt.ylabel('x')
plt.show()

# optimal actions
cf = plt.contourf(ts_plt.T, X.T, u_plt.T, levels=1)
plt.colorbar(cf)
plt.title('Optimal policy')
plt.xlabel('t')
plt.ylabel('x')
plt.show()

# advantage values
x_id = N // 2
rhs_np = rhs_all.reshape_as(X[..., None].repeat(1, 1, 3)).detach()
rhsplt_t = ts_plt[x_id]
plt.figure()
plt.plot(rhsplt_t, rhs_np[x_id, :, 0], label='left')
plt.plot(rhsplt_t, rhs_np[x_id, :, 1], label='stay')
plt.plot(rhsplt_t, rhs_np[x_id, :, 2], label='right')
plt.title('Advantage values for s={}'.format(xl[x_id]))
plt.xlabel('t')
plt.ylabel('Q')
plt.legend()
plt.show()

# simulation
sim = Simulator(p, value_net, t_exp_lambd)
x0 = torch.tensor(0.)
t = torch.linspace(0, 5, 200)
x_traj, u_traj, rhs_traj = sim.ode_int_euler(x0, t)

plt.figure()
fig, axs = plt.subplots(2)
plt.title('Simulation')
axs[0].plot(t, x_traj[:, 0])
axs[1].step(t[:-1], u_traj[:, 0])
axs[0].set_ylabel('x')
axs[1].set_xlabel('t')
axs[1].set_ylabel('u')
plt.show()
