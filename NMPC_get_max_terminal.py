import numpy as np
import cvxpy

def NMPC_get_max_terminal(mu, Q, R, ucon, xcon, alpha):
    # On the terminal region of model predictive control for non-linear systems with input/state constraints (2003)
    # Based on the MATLAB code by get the maximal terminal region By Zehua Jia, jiazehua@sjtu.edu.cn

    # Simulation reproduction of Chen et. al 2003.

    # Input
    # ucon: |u| <= ucon; 
    # xcon: |x| <= xcon
    # Here x(i) has the same upper and lower bounds, i = 1, 2.
    # alpha: x' * P * x <= alpha
    # Syst NMPC_get_max_terminalem description
    # x1' = x2 + u * (mu + (1 - mu) * x1)
    # x2' = x1 + u * (mu - 4 * (1 - mu) * x2)

    # The designed local state-feeback control law is u = K * x.

    # Some Tips
    # A. The constraints should be Linear, which means nonlinear terms are not
    # permitted (such as x*y and x^2, where x,y are both sdpvars).

    # B. The inverse of a sdpvar is not desired in constraints or objective.
    # logdet(X^(-1)) = -logdet(X).

    # C. logdet is concave, and then -logdet is convex.

    # D. One should always try to reformulate the problem to obtain a convex 
    # problem. 

    # E. -logdet(a * X) (sdpvar a, X = sdpvar(n,n)) can be reformulated as 
    # -logdet(Y) (sdpvar a, Y = sdpvar(n,n)). Then multiply LMI constraints by 
    # a in both sides, and replace Y = a * X in constraints, 

    # Initiallization
    umax = ucon
    umin = -ucon
    xmax = xcon
    xmin = -xcon
      
    # LDI approximation within the selected set x <= 2
    N = 8  # The number of LDI
    A = np.array([[0, 1], [1, 0]])
    # B: (x1, x2) in (min,min),(min,max),(max,min)(max,max) with minimal u
    B = [] 
    B += [np.array([[mu + (1 - mu) * xmin], [mu - 4 * (1 - mu) * xmin]])]
    B += [np.array([[mu + (1 - mu) * xmin], [mu - 4 * (1 - mu) * xmax]])]
    B += [np.array([[mu + (1 - mu) * xmax], [mu - 4 * (1 - mu) * xmin]])]
    B += [np.array([[mu + (1 - mu) * xmax], [mu - 4 * (1 - mu) * xmax]])] 
    B += [np.array([[mu + (1 - mu) * xmin], [mu - 4 * (1 - mu) * xmin]])]
    B += [np.array([[mu + (1 - mu) * xmin], [mu - 4 * (1 - mu) * xmax]])] 
    B += [np.array([[mu + (1 - mu) * xmax], [mu - 4 * (1 - mu) * xmin]])]
    B += [np.array([[mu + (1 - mu) * xmax], [mu - 4 * (1 - mu) * xmax]])]  # (x1, x2) in (min,min),(min,max),(max,min)(max,max) with maximal u

    F = []
    for i in range(N):
        F += [np.hstack((A, B[i]))]
        
    # Solve the optimization problem using 
    # Express the state/Input constraints in standard form
    c = []
    c += [np.array([1/xmax, 0])]  # for x1 < 2
    c += [np.array([0, 1/xmax])]  # for x2 < 2
    c += [np.array([-1/xmax, 0])]  # for x1 > -2
    c += [np.array([0, -1/xmax])]  # for x2 > -2
    c += [np.zeros((1, 2))]
    c += [np.zeros((1, 2))]
    
    d = []
    d += [np.array([[0]])]
    d += [np.array([[0]])]
    d += [np.array([[0]])]
    d += [np.array([[0]])]
    d += [np.array([[1/umax]])]
    d += [np.array([[-1/umax]])]
    
    Nc = len(d)  # Number of constraints

    # Define decision matrix variables
    n = Q.shape[0]
    m = R.shape[1]

    alpha0 = cvxpy.Variable((1))
    W1 = cvxpy.Variable((n,n))
    W2 = cvxpy.Variable((m,n))
    W = cvxpy.bmat([[W1, W2.T]])

    P = None
    K = None
    alpha1 = None
    
    MAT1 = []  # casadi.DM.zeros(2*n+m, 2*n+m, N)
    MAT2 = []  # casadi.DM.zeros(1+n, 1+n, Nc)

    # Define LMI constraints
    for i in range(N):
        M11 = cvxpy.bmat([[cvxpy.matmul(-F[i], W.T) + cvxpy.matmul(-W, F[i].T), cvxpy.matmul(W1,Q**(0.5)), W2.T]])
        M22 = cvxpy.bmat([[cvxpy.matmul( W1, Q**(0.5)), W2.T]]).T
        M23 =  alpha0 *cvxpy.bmat([[np.eye(Nx), np.zeros((Nx,Nu))], [np.zeros((Nu,Nx)), np.linalg.inv(R)]])
        MAT1 += [cvxpy.bmat([[M11], [M22, M23]])]

    for i in range(Nc):
        MAT2 += [cvxpy.bmat([
            [cvxpy.bmat([[np.array([[1.0]]), cvxpy.matmul(c[i], W1) + cvxpy.matmul(d[i], W2)]])],
            [cvxpy.bmat([[(cvxpy.matmul(c[i], W1) + cvxpy.matmul(d[i], W2)).T, W1]])]])]

    LMI = []
    LMI += [alpha0>=0]
    LMI += [W1 >> 0]
    LMI += [W1 == W1.T]

    for i in range(N):
        LMI += [MAT1[i]>>0]
        # LMI += [MAT1[i] == MAT1[i].T]

    for i in range(Nc):
        LMI += [MAT2[i]>>0]
        # LMI += [MAT2[i] == MAT2[i].T]

    obj = -cvxpy.log_det(W1)

    LMI1 = LMI + [alpha0 <= alpha]  # The LMI method with limited alpha

    # Solving
    if alpha == -1:
        prob = cvxpy.Problem(cvxpy.Minimize(obj), LMI)
        prob.solve(solver=cvxpy.SCS, verbose=True, max_iters=100000, eps=1e-3)
    elif alpha <= 0:
        raise('The alphaM must be larger than 0')
    else:

        prob = cvxpy.Problem(cvxpy.Minimize(obj), LMI1)
        prob.solve(solver=cvxpy.SCS, verbose=True, max_iters=100000, eps=1e-3)

    print(cvxpy.installed_solvers())

    print("optimal value with:", prob.value)
    print("status:", prob.status)
    print("obj=", obj.value)

    W1 = W1.value / alpha0.value
    W2 = W2.value / alpha0.value
    P = np.linalg.inv(W1)
    K = np.matmul(W2, np.linalg.inv(W1))
    alpha1 = alpha0.value
    
    return (P, K, alpha1)