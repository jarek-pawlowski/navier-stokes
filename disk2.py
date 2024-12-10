# example adapted from: https://jsdokken.com/dolfinx-tutorial/chapter2/ns_code2.html

import utils
from pathlib import Path

from basix.ufl import element
from dolfinx.fem import functionspace, Function

flow_velocity = 20.0
viscosity_mu = 0.01
density_rho = 1.
disk_radius = 0.07
box_size = 1.0
print("Re = ", density_rho*flow_velocity*disk_radius/viscosity_mu)

# define mesh and timestep
mesher = utils.Mesh()
mesh, ft = mesher.build_disk_in_a_square(L=box_size*3., H=box_size*2, d_x=box_size/2., d_y=box_size*1.1, r=disk_radius, resolution=5)
t = 0
T = 0.3
dt = 1./100000
num_steps = int(T / dt)

# define fields
v_cg2 = element("Lagrange", mesh.topology.cell_name(), 2, shape=(mesh.geometry.dim, ))
s_cg1 = element("Lagrange", mesh.topology.cell_name(), 1)
V = functionspace(mesh, v_cg2)
Q = functionspace(mesh, s_cg1)
var = utils.Variables(V,Q)

# set system parameters
sys = utils.System(mesh)
sys.set_parameters(timestep=dt, viscosity=viscosity_mu, density=density_rho)

# set boundary conditions
bc = utils.Boundary(sys)
# left_marker, right_marker, top_bottom_marker, disk_marker = 2, 3, 4, 5
bc.set_velocity_marker(V, ft, 2, velocity = (flow_velocity,0.))  # velocity on the left side
bc.set_velocity_marker(V, ft, 3, velocity = (flow_velocity,0.))    # pressure on the right side 
bc.set_velocity_marker(V, ft, 4, velocity = (flow_velocity,0.))  # velocity on the top/bottom sides
bc.set_velocity_marker(V, ft, 5, velocity = (0.,0.))  # velocity at disk (no slip)

# set solver engine
solver = utils.Solver(variables=var, system=sys, boundary=bc)
solver.setup_variational_problem()
solver.setup_solver()

# TBA, implement alternative variational form:
# https://jsdokken.com/dolfinx-tutorial/chapter2/ns_code2.html#variational-form

# set folder for storing results
folder = Path("results")
folder.mkdir(exist_ok=True, parents=True)

# to get and actual values
values = utils.Values(V, Q, var)

for i in range(num_steps):
    
    # Update current time step
    t += dt
    solver.step()
    
    # write current u_n and p_n
    u, p = values.get_actual()

    if (i % 20 == 0) or (i == num_steps - 1):
        print(f"Time {t:.2f}")

solver.clean()

# plot stationary solution
plot = utils.Plot(folder)
u, p = values.get_actual()
plot.plot_velocity(u, values.u_geometry, scaling=1000.)
plot.plot_pressure(p, values.p_geometry, pointsize=10.)