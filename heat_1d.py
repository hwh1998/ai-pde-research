"""
1D Heat Equation Solver using FEniCS
AI-generated code, reviewed by human.
Equation: u_t = u_xx, x in [0,1], t in [0,T]
Boundary: u(0,t)=u(1,t)=0
Initial:  u(x,0) = sin(pi*x)
"""
from fenics import *
import matplotlib.pyplot as plt

# Mesh and function space
nx = 50
mesh = UnitIntervalMesh(nx)
V = FunctionSpace(mesh, 'P', 1)

# Boundary condition
u_D = Constant(0.0)
bc = DirichletBC(V, u_D, 'on_boundary')

# Initial condition
u_0 = Expression('sin(pi*x[0])', degree=2)
u_n = interpolate(u_0, V)

# Time stepping
T = 1.0
dt = 0.001
num_steps = int(T/dt)

u = TrialFunction(V)
v = TestFunction(V)

F = u*v*dx + dt*dot(grad(u), grad(v))*dx - u_n*v*dx
a, L = lhs(F), rhs(F)

u = Function(V)
t = 0

for n in range(num_steps):
    t += dt
    solve(a == L, u, bc)
    u_n.assign(u)

# Plot final solution
plot(u)
plt.title(f"Solution at t={T}")
plt.savefig("heat_1d_result.png")
print("Simulation complete. Figure saved as heat_1d_result.png")
