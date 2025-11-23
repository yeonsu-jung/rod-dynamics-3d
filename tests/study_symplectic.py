import numpy as np
import matplotlib.pyplot as plt

m = 1.0
h = 0.01
T = 50.0
N = int(T / h)

def V(x):
    # double-well potential
    return 0.25*x**4 - 0.5*x**2

def F(x):
    # force = -dV/dx
    return - (x**3 - x)  # = -x^3 + x

t = np.linspace(0, T, N+1)
x = np.zeros(N+1)
v = np.zeros(N+1)
E = np.zeros(N+1)

# initial condition: start on one side of the well
x[0] = 1.0
v[0] = 0.5   # nonzero velocity

# initial energy
E[0] = 0.5*m*v[0]**2 + V(x[0])

# Velocity-Verlet
F_curr = F(x[0])
for n in range(N):
    v_half = v[n] + 0.5*h*F_curr/m
    x[n+1] = x[n] + h*v_half
    F_next = F(x[n+1])
    v[n+1] = v_half + 0.5*h*F_next/m
    F_curr = F_next

    E[n+1] = 0.5*m*v[n+1]**2 + V(x[n+1])

plt.figure()
plt.subplot(2,1,1)
plt.plot(t, x)
plt.ylabel("x(t)")

plt.subplot(2,1,2)
plt.plot(t, E)
plt.ylabel("Energy")
plt.xlabel("t")
plt.tight_layout()
plt.show()
