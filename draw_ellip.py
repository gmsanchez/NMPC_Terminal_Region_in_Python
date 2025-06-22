import numpy as np
import matplotlib.pyplot as plt
import math

def draw_ellip(P, alpha, color):
    # Draw the ellipse centered at origin
    # The ellipsoid a * x^2 + c * y^2 + b * x * y = f
    # P = [a, b/2; b/2, c]
    a = P[0,0]
    b = 2 * P[0,1]
    c = P[1,1]
    d = 0
    e = 0
    f = alpha
    delta = b**2 - 4*a*c
    if delta >= 0:
        Warning("This is not an ellipse.")
        return
    x0 = (b*e-2*c*d)/delta
    y0 = (b*d-2*a*e)/delta
    r = a*x0**2 + b*x0*y0 +c*y0**2 + f
    if r <= 0:
        Warning('This is not an ellipse')
        return

    aa = math.sqrt(r/a); 
    bb = math.sqrt(-4*a*r/delta);
    t = np.linspace(0, 2*np.pi, 60);
    # xy = np.array([[1, -b/(2*a)],[0, 1]]) * np.array([aa*cos(t);bb*sin(t)];
    # plt.plot(xy(1,:)-x0,xy(2,:)-y0, color, 'linewidth', 2);
    xy = np.array([[1, -b/(2*a)],[0, 1]]) @ np.vstack([aa * np.cos(t), bb * np.sin(t)])

    plt.plot(xy[0,:] - x0, xy[1,:] - y0)
    plt.grid()
    plt.show()