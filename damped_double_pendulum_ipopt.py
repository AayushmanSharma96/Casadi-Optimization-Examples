import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# Parameters
m1, m2   = 1.0, 1.0      # link masses          [kg]
l1, l2   = 1.0, 1.0      # link lengths         [m]
r1, r2   = 0.5, 0.5      # com distances        [m]
I1, I2   = 0.02, 0.02    # link inertias        [kg·m²]
b1, b2   = 0.05, 0.05    # viscous damping      [N·m·s/rad]
g        = 9.81          # gravity              [m/s²]

horizon = 3.0 # Total time horizon
dt = 0.01 # Time step
N  = int(horizon/dt) # discretization segments

# Define symbolic state and control
state = ca.MX.sym('x',4) # theta1, theta2, omega1, omega2
u = ca.MX.sym('u', 2) # control inputs (torques) 

# Set up cartpole dynamics with RK4 integration
def double_pendulum_dynamics(X, U):
    theta1, theta2, omega1, omega2 = X[0], X[1], X[2], X[3]
    F1 = U[0]
    F2 = U[1]

    # Inertia matrix
    c2 = ca.cos(theta2)
    M11 = I1 + I2 + m1*r1**2 + m2*(l1**2 + r2**2 + 2*l1*r2*c2)
    M12 = I2 + m2*(r2**2 + l1*r2*c2)
    M22 = I2 + m2*r2**2
    M = ca.vertcat(ca.hcat([M11, M12]), ca.hcat([M12, M22]))

    # Coriolis matrix
    s2 = ca.sin(theta2)
    C11 = -m2*l1*r2*s2*omega2
    C12 = -m2*l1*r2*s2*(omega1 + omega2)
    C21 =  m2*l1*r2*s2*omega1
    C  = ca.vertcat(ca.hcat([C11, C12]), ca.hcat([C21, 0]))
    Comega = C @ ca.vertcat(omega1, omega2)

    # Gravitational forces
    G1 = (m1*r1 + m2*l1)*g*ca.sin(theta1) + m2*r2*g*ca.sin(theta1+theta2)
    G2 = m2*r2*g*ca.sin(theta1+theta2)
    G  = ca.vertcat(G1, G2)

    # Viscous damping forces
    B = ca.diag(ca.vertcat(b1, b2))
    Bomega = B @ ca.vertcat(omega1, omega2)

    # M·domega = F-C*omega-G-B*omega 
    domega = ca.solve(M, U - Comega - G - Bomega)
    
    return ca.vertcat(omega1, omega2, domega[0], domega[1])

f = ca.Function('f', [state, u], [double_pendulum_dynamics(state, u)])

def rk4(f, x, u, dt):
    k1 = f(x, u)
    k2 = f(x + dt/2 * k1, u)
    k3 = f(x + dt/2 * k2, u)
    k4 = f(x + dt * k3, u)
    return dt/6 * (k1 + 2*k2 + 2*k3 + k4)


# Opti problem setup
opti = ca.Opti()
X = opti.variable(4, N+1) 
U = opti.variable(2, N)   

# Dynamics constraints

J = 0 # Cost function
for k in range(N):
    xk = X[:, k]
    uk = U[:, k]
    opti.subject_to(X[:, k+1] - X[:, k] == rk4(f, xk, uk, dt))
    J += ca.sumsqr(uk) * dt
    J += 1e-2 * ca.sumsqr(U[:,k] - U[:,k-1]) # Smoothing term

# Boundary constraint
opti.subject_to(X[:, 0] == [0, 0, 0, 0])
opti.subject_to(X[:, -1] == [np.pi, 0, 0, 0])

# Path constraints
opti.subject_to(opti.bounded(-2*np.pi, X[0, :], 2*np.pi))
opti.subject_to(opti.bounded(-2*np.pi, X[1, :], 2*np.pi))
opti.subject_to(opti.bounded(-10, U[0], 10))
opti.subject_to(opti.bounded(-10, U[1], 10))

# Initial guess
opti.set_initial(X[0, :], np.linspace(0,np.pi,N+1))
opti.set_initial(X[1, :], np.linspace(0,0,N+1))
opti.set_initial(X[2, :], 0)
opti.set_initial(X[3, :], 0)
opti.set_initial(U, 0)

# Solve
opti.minimize(J)
opti.solver('ipopt', {"ipopt.max_iter": 1000, "ipopt.tol": 1e-6})  # Set max iterations
t = np.linspace(0, horizon, N+1)

try:
    sol = opti.solve()
    
except RuntimeError as e:
    print("Solver failed to converge, returning best found solution.")
    sol = opti.debug
    
theta1, theta2, omega1, omega2 = sol.value(X)
u = sol.value(U)
        
# Plotting results
plt.figure(figsize=(8,6))
plt.suptitle('Double-Pendulum Swingup Optimization')
plt.subplot(3,2,1)
plt.plot(t, theta1, label='Angle 1 (rad)')
plt.xlabel('Time [s]')
plt.ylabel(r'$\theta_1$')
plt.subplot(3,2,2)
plt.plot(t, theta2, label='Angle 2 (rad)')
plt.xlabel('Time [s]')
plt.ylabel(r'$\theta_2$')
plt.subplot(3,2,3)
plt.plot(t, omega1, label=r'$\dot{\theta}_1$')
plt.xlabel('Time [s]')
plt.ylabel(r'$\dot{\theta}_1$')
plt.subplot(3,2,4)
plt.plot(t, omega2, label=r'$\dot{\theta}_2$')
plt.xlabel('Time [s]')
plt.ylabel(r'$\dot{\theta}_2$')
plt.subplot(3,2,5)
plt.step(t[:-1], u[0], label='Torque 1 (N·m)')
plt.step(t[:-1], u[1], label='Torque 2 (N·m)')
plt.xlabel('Time [s]')
plt.ylabel(r'$u_{opt}$')
plt.tight_layout()
plt.savefig('Double_pendulum_swingup.png')
plt.show()

# Animation of the double pendulum trajectory
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

x1 =  l1 * np.sin(theta1)
y1 = -l1 * np.cos(theta1)
x2 = x1 + l2 * np.sin(theta1 + theta2)
y2 = y1 - l2 * np.cos(theta1 + theta2)

fig, ax = plt.subplots(figsize=(5,5))
ax.set_xlim(-(l1+l2)*1.1, (l1+l2)*1.1)
ax.set_ylim(-2.2, 2.2)
ax.set_aspect('equal')
line, = ax.plot([], [], 'o-', lw=2)

def init():
    line.set_data([], [])
    return (line,)

def update(i):
    thisx = [0, x1[i], x2[i]]
    thisy = [0, y1[i], y2[i]]
    line.set_data(thisx, thisy)
    return (line,)

speedup = 3               
fps     = 30              
interval = 1000/fps       
ani = animation.FuncAnimation(
    fig, update,
    frames=range(0, len(theta1), speedup),  
    init_func=init,
    blit=True,
    interval=interval    
)

# Save as GIF
ani.save('double_pendulum_trajectory.gif', writer=PillowWriter(fps=fps))