import numpy as np
from NMPC_get_max_terminal import NMPC_get_max_terminal
from draw_ellip import draw_ellip

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
N = 300   # The prediction horizon N * dt
T = 100   # The simulation time T * dt

Q = np.diag([0.5, 0.5])
R = np.diag([1])
mu = 0.9

# Initial settings
x0 = [2, -3]  # x0
# x0 = [-.683, -.864]  # x0
uM = 2  # The constraint of u for set M of the LDI
xM = 2  # The constraint of x for set M of the LDI
alphaM = 10
# Define the variables storing the actual states and controls
xc = np.zeros((2, T))
uc = np.zeros((1, T))
xc[:, 1] = x0
# Get the enlarged terminal region and terminal penalty
[P, K, alpha] = NMPC_get_max_terminal (mu, Q, R, uM, xM, alphaM)  # alphaM is the given upper bound of alpha. -1 means no given bound.


draw_ellip(P, alpha, 'k')