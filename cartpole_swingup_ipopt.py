import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# Parameters
M = 2.0 # cart mass [kg]
m = 0.5 # pole mass [kg]
l = 1.0 # pole length [m]
g = 9.81 # gravity [m/s²]
horizon = 3.0 # Total time horizon
dt = 0.01 # Time step
N  = int(horizon/dt) # discretization segments

# Define symbolic state and control
state = ca.MX.sym('x',4)
u = ca.MX.sym('u') 

# Set up cartpole dynamics with RK4 integration
def cartpole_dynamics(X, U):
    x, xdot, theta, thdot = X[0], X[1], X[2], X[3]
    F = u[0]
    den = -(M + m) + m * ca.cos(theta)**2
    ddx  = (-F-m*l*thdot**2*ca.sin(theta)- m*g * ca.sin(theta)*ca.cos(theta))/den
    ddth = (F*ca.cos(theta)+M*l*thdot**2*ca.cos(theta)*ca.sin(theta)+(M + m)*g*ca.sin(theta))/(l*den)
    
    return ca.vertcat(xdot, ddx, thdot, ddth)

f = ca.Function('f', [state, u], [cartpole_dynamics(state, u)])

def rk4(f, x, u, dt):
    k1 = f(x, u)
    k2 = f(x + dt/2 * k1, u)
    k3 = f(x + dt/2 * k2, u)
    k4 = f(x + dt * k3, u)
    return dt/6 * (k1 + 2*k2 + 2*k3 + k4)


# Opti problem setup
opti = ca.Opti()
X = opti.variable(4, N+1) 
U = opti.variable(1, N)   

# Dynamics constraints

J = 0 # Cost function
for k in range(N):
    xk = X[:, k]
    uk = U[:, k]
    opti.subject_to(X[:, k+1] - X[:, k] == rk4(f, xk, uk, dt))
    J += uk**2 * dt

# Boundary constraint
opti.subject_to(X[:, 0] == [0, 0, 0, 0])
opti.subject_to(X[:, -1] == [0, 0, np.pi, 0])

# Path constraints
opti.subject_to(opti.bounded(-5, X[0, :], 5))
opti.subject_to(opti.bounded(-2*np.pi, X[2, :], 2*np.pi))
opti.subject_to(opti.bounded(-30, U, 30))

# Initial guess
opti.set_initial(X[0, :], np.linspace(0,0,N+1))
opti.set_initial(X[1, :], np.linspace(0,np.pi,N+1))
opti.set_initial(X[2, :], 0)
opti.set_initial(X[3, :], 0)
opti.set_initial(U, 0)

# Solve
opti.minimize(J)
opti.solver('ipopt') # ('fatrop')
sol = opti.solve()

t = np.linspace(0, horizon, N+1)
x, x_dot, theta, omega = sol.value(X)
u = sol.value(U)
        
# Plotting results
plt.figure(figsize=(8,6))
plt.suptitle('Cart-Pole Swingup Optimization')
plt.subplot(3,2,1)
plt.plot(t, x, label='Cart position (m)')
plt.xlabel('Time [s]')
plt.ylabel('x')
plt.subplot(3,2,2)
plt.plot(t, theta, label='Pole angle (rad)')
plt.xlabel('Time [s]')
plt.ylabel(r'$\dot{x}$')
plt.subplot(3,2,3)
plt.plot(t, x_dot, label=r'$\dot{x}$')
plt.xlabel('Time [s]')
plt.ylabel(r'$\theta$')
plt.subplot(3,2,4)
plt.plot(t, omega, label=r'$\dot{\theta}$')
plt.xlabel('Time [s]')
plt.ylabel(r'$\dot{\theta}$')
plt.subplot(3,2,5)
plt.step(t[:-1], u)
plt.xlabel('Time [s]')
plt.ylabel(r'$u_{opt}$')
plt.tight_layout()
plt.savefig('cartpole_swingup.png')
plt.show()

