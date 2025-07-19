import casadi as ca
import numpy as np

x = ca.MX.sym('x', 4)   # state vector
u = ca.MX.sym('u', 1)   # control input
Q = ca.diag([10, 10, 1, 1])  # state cost matrix
xf = ca.DM([3, 2, 1, 5])  # final state
objective = ca.sumsqr(u)+(x-xf).T @ Q @ (x-xf)  # cost function

g1 = x[0]+x[1]-5
g2 = x[3]-3
g3 = 5-x[3]
g4 = u-3
g5 = -u-1

g = ca.vertcat(g1, g2, g3, g4, g5)  # constraints
lbg = ca.DM([0, 0, 0, 0, 0])  # lower bounds
ubg = ca.DM([ca.inf, ca.inf, ca.inf, ca.inf, ca.inf])  # upper bounds

nlp = {'x': ca.vertcat(x, u), 'f': objective, 'g': g}
opts = {'print_time': False}
solver = ca.nlpsol('solver', 'fatrop', nlp, opts)

sol = solver(x0=ca.DM([0, 0, 0, 0, 0]), lbg=lbg, ubg=ubg)
# Print solution
print('Optimal x:', sol['x'])
print('Optimal objective:', sol['f'])