import utils
from pathlib import Path

from basix.ufl import element
from dolfinx.fem import functionspace, Function


# define mesh and timestep
mesher = utils.Mesh()
mesh = mesher.build_simple_squared(nx=20, ny=20)
t = 0
T = 10
num_steps = 500
dt = T / num_steps

# define fields
v_cg2 = element("Lagrange", mesh.topology.cell_name(), 2, shape=(mesh.geometry.dim, ))
s_cg1 = element("Lagrange", mesh.topology.cell_name(), 1)
V = functionspace(mesh, v_cg2)
Q = functionspace(mesh, s_cg1)
var = utils.Variables(V,Q)

# set system parameters
sys = utils.System(mesh)
sys.set_parameters(timestep=dt, viscosity=1., density=1.)

# set boundary conditions
bc = utils.Boundary(sys)
bc.set_top_bottom(V, velocity=0.)
bc.set_left_pressure(Q, pressure=8.)
bc.set_right_pressure(Q, pressure=0.)

# set solver engine
solver = utils.Solver(variables=var, system=sys, boundary=bc)
solver.setup_variational_problem()
solver.setup_solver()

# set folder for storing results
folder = Path("results")
folder.mkdir(exist_ok=True, parents=True)

# exact solution
exact = utils.Exact(solver, V, utils.u_exact)
# and actual values
values = utils.Values(V, Q, var)

for i in range(num_steps):
    
    # Update current time step
    t += dt
    solver.step()
    
    # write current u_n and p_n
    u, p = values.get_actual()

    # Compute error at current time-step
    error_L2, error_max = exact.get_errors()
    # Print error only every 20th step and at the last step
    if (i % 20 == 0) or (i == num_steps - 1):
        print(f"Time {t:.2f}, L2-error {error_L2:.2e}, Max error {error_max:.2e}")

solver.clean()

# plot stationary solution
plot = utils.Plot(folder)
u, p = values.get_actual()
plot.plot_velocity(u, values.u_geometry)
plot.plot_pressure(p, values.p_geometry)