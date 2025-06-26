import numpy as np
import casadi
from NMPC_get_max_terminal import NMPC_get_max_terminal
from draw_ellip import draw_ellip
import time
import matplotlib.pyplot as plt

# Nonlinear MPC with guaranteed stability
# NMPC controller by Guido Sanchez

#  System description (From Chen et al. IJACSP 2003 https://doi.org/10.1002/acs.731, and 
#  Chen et al. Automatica 1998 https://doi.org/10.1016/S0005-1098(98)00073-9)

#  x1' = x2 + u * (mu + (1 - mu) * x1)
#  x2' = x1 + u * (mu - 4 * (1 - mu) * x2)
#  mu = 0.9

# The designed local state-feeback control law is u = K * x.

#  Initialization
dt = 0.1  # The sampling period
N = 15   # The prediction horizon N * dt
T = 100   # The simulation time T * dt

Nx = 2
Nu = 1
def ode(x,u):
    dxdt = [
        x[1] + u[0] * (mu + (1 - mu) * x[0]),
        x[0] + u[0] * (mu - 4 * (1 - mu) * x[1])
    ]
    return np.array(dxdt)

Q = np.diag([0.5, 0.5])
R = np.diag([1])
mu = 0.9

# Initial settings
x0 = np.array([2, -3])  # x0
# x0 = [-.683, -.864]  # x0
uM = 2  # The constraint of u for set M of the LDI
xM = 2  # The constraint of x for set M of the LDI
alphaM = 10
# Define the variables storing the actual states and controls
xc = np.zeros((Nx, T+1))
uc = np.zeros((Nu, T))
xc[:, 0] = x0

# Get the enlarged terminal region and terminal penalty
[P, K, alpha] = NMPC_get_max_terminal (mu, Q, R, uM, xM, alphaM)  # alphaM is the given upper bound of alpha. -1 means no given bound.

# Define CasADi symbolic variables.
x_sym = casadi.SX.sym("x", Nx)
u_sym = casadi.SX.sym("u", Nu)

# Make integrator object.
ode_integrator = dict(x=x_sym, p=u_sym, ode=ode(x_sym, u_sym))
intoptions = {
    "abstol": 1e-8,
    "reltol": 1e-8,
}
vdp = casadi.integrator("int_ode", "cvodes", ode_integrator, 0, dt, intoptions)

# Then get nonlinear casadi function and Euler discretization
# ode_casadi = casadi.Function(
#     "ode",[x_sym,u_sym],[ode(x_sym,u_sym)])
xnext_euler_casadi = casadi.Function("xnext_euler_casadi", [x_sym, u_sym], [x_sym + dt * ode(x_sym, u_sym)])

# Define stage cost and terminal cost.
lfunc = (casadi.mtimes([x_sym.T, Q, x_sym]) + casadi.mtimes([u_sym.T, R, u_sym]))
l = casadi.Function("l", [x_sym, u_sym], [lfunc])

Pffunc = casadi.mtimes([x_sym.T, P, x_sym])
Pf = casadi.Function("Pf", [x_sym], [Pffunc])

# Bounds on x.
xlb = [-4, -4]
xub = [4, 4]
# Bounds on u.
ulb = [-2]
uub = [2]

# Create Opti instance
opti = casadi.Opti()

# Decision variables
x_opti = opti.variable(Nx, N+1)   # States
u_opti = opti.variable(Nu, N)     # Controls
x0_par = opti.parameter(Nx, 1)

# Objective function
obj = 0
for k in range(N):
    obj += l(x_opti[:, k], u_opti[:, k])
obj += Pf(x_opti[:, N])
opti.minimize(obj)

# Constraints
for k in range(N):
    # Dynamics constraint
    opti.subject_to(x_opti[:, k+1] == xnext_euler_casadi(x_opti[:, k], u_opti[:, k]))
    # Control bounds
    opti.subject_to(opti.bounded(ulb, u_opti[:, k], uub))
    # State bounds
    opti.subject_to(opti.bounded(xlb, u_opti[:, k], xub))
# State bounds for last state
opti.subject_to(opti.bounded(xlb, x_opti[:, N], xub))
# Initial state constraint
opti.subject_to(x_opti[:, 0] == x0_par)
# Terminal set constraint
opti.subject_to(opti.bounded(0, casadi.mtimes([x_opti[:, N].T, P, x_opti[:, N]]), alpha))
    
# Solver options
opts = {
    "ipopt.print_level": 0,
    "ipopt.max_cpu_time": 60,
    "ipopt.max_iter": 100,
    "print_time": False
}

# Solve NLP
opti.solver('ipopt', opts)

times = dt*T*np.linspace(0, 1, T+1)
# x = np.zeros((T+1, Nx))
# x[0, :] = x0
# u = np.zeros((T, Nu))
iter_time = np.zeros((T, 1))

for t in range(T):
    t0 = time.time()
    
    opti.set_value(x0_par, xc[:, t])
    sol = opti.solve()
       
    t1 = time.time()
    
    if t == 0:
        xc_com = sol.value(x_opti)  # Store the optimized states at the first time instant
        uc_com = sol.value(u_opti)  # Store the optimized states at the first time instant
    
    # Print stats
    print(f"{t}: Solved in {t1 - t0:.4f} seconds")
    
    # Store solution
    uc[:, t] = sol.value(u_opti[:, 0])
    iter_time[t] = t1 - t0
    
    # Simulate
    vdpargs = dict(x0=xc[:, t], p=uc[:, t])
    out = vdp(**vdpargs)
    xc[:, t+1] = np.array(out["xf"]).flatten()

 # u plots. Need to repeat last element
# for stairstep plot.
uc = np.concatenate((uc, uc[:,-1:]), axis=1)

plt.figure(1)
plt.plot(times, xc[0, :], label="x0")
plt.plot(times, xc[1, :], label="x1")
plt.grid()
plt.legend()

plt.figure(2)
plt.step(times, uc[0, :], where="post")
plt.step(np.linspace(0, 1, N), uc_com, where='post')
plt.grid()

plt.figure(3)
plt.plot(xc[0, :], xc[1, :], '*')
plt.plot(xc_com[0, :], xc_com[1, :], 'ro')
draw_ellip(P, alpha, 'k')