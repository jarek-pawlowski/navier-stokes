# example adapted from: https://jsdokken.com/dolfinx-tutorial/chapter2/ns_code2.html
import numpy as np
import utils
from pathlib import Path

from basix.ufl import element
from dolfinx.fem import functionspace, Function

flow_velocity = 5.0
viscosity_mu = 0.01
viscosity_dzeta = 0.02
viscosity_eta = -0.015
density_rho = 1.
disk_radius = 0.1
box_size = 1.0
print("Re = ", density_rho*flow_velocity*disk_radius/viscosity_mu)

# define mesh and timestep
mesher = utils.Mesh()
mesh, ft = mesher.build_disk_in_a_square(L=box_size*3., H=box_size*2, d_x=box_size/2., d_y=box_size*1.1, r=disk_radius, resolution=5)
t = 0
T = 0.05
dt = 1./10000
num_steps = int(T / dt)

# define fields
v_cg2 = element("Lagrange", mesh.topology.cell_name(), 2, shape=(mesh.geometry.dim, ))
s_cg1 = element("Lagrange", mesh.topology.cell_name(), 1)
V = functionspace(mesh, v_cg2)
Q = functionspace(mesh, s_cg1)
R = functionspace(mesh, s_cg1)
var = utils.Variables(V,Q,R)

# set system parameters
sys = utils.System(mesh)
sys.set_parameters(timestep=dt, viscosity=viscosity_mu, density=density_rho, bulk_viscosity=viscosity_dzeta, odd_viscosity=viscosity_eta)

# set boundary conditions
bc = utils.Boundary(sys)
# left_marker, right_marker, top_bottom_marker, disk_marker = 2, 3, 4, 5
bc.set_velocity_marker(V, ft, 2, velocity = (flow_velocity,0.))  # velocity on the left side
bc.set_velocity_marker(V, ft, 3, velocity = (flow_velocity,0.))  # pressure on the right side 
bc.set_velocity_marker(V, ft, 4, velocity = (flow_velocity,0.))  # velocity on the top/bottom sides
bc.set_velocity_marker(V, ft, 5, velocity = (0.,0.))  # velocity at disk (no slip)
bc.set_pressure_marker(Q, ft, 2, pressure = (100.))  # inflow pressure
# density
bc.set_density_marker(R, ft, 2, density = sys.rho)
bc.set_density_marker(R, ft, 3, density = sys.rho)
bc.set_density_marker(R, ft, 4, density = sys.rho)
bc.set_density_marker(R, ft, 5, density = sys.rho)

# set solver engine
solver = utils.Solver(variables=var, system=sys, boundary=bc)
solver.setup_variational_problem_c_odd()
solver.setup_solver()

# TBA, implement alternative variational form:
# https://jsdokken.com/dolfinx-tutorial/chapter2/ns_code2.html#variational-form

# set folder for storing results
folder = Path("results")
folder.mkdir(exist_ok=True, parents=True)

# to get and actual values
values = utils.Values(V, Q, var, R)
measure = utils.Measurements(mesh, ft, var)
measure.setup_disk_forces(5, mu=viscosity_mu, rho=density_rho)
forces = []
times = []


for i in range(num_steps):
    
    # Update current time step
    t += dt
    times.append(t)

    solver.step()
    
    # write current u_n and p_n
    u, p, r = values.get_actual()

    if (i % 20 == 0) or (i == num_steps - 1):
        print(f"Time {t:.2f}")
    
    forces.append(measure.get_disk_forces())
forces = np.array(forces)
solver.clean()

# plot stationary solution
plot = utils.Plot(folder)
u, p, r = values.get_actual()
plot.plot_velocity(u, values.u_geometry, scaling=100.)
plot.plot_pressure(p, values.p_geometry, pointsize=1.)
plot.plot_density(r, values.r_geometry, pointsize=1.)
plot.plot_in_time(times, forces[:,0]/density_rho/flow_velocity/flow_velocity/disk_radius, ylabel='$C_d$', filename="drag_coeff.png")
plot.plot_in_time(times, forces[:,1]/density_rho/flow_velocity/flow_velocity/disk_radius, ylabel='$C_l$', filename="lift_coeff.png", ranges=[-5.,0.])
print("C_d = ", forces[-1,0]/density_rho/flow_velocity/flow_velocity/disk_radius)
print("C_l = ", forces[-1,1]/density_rho/flow_velocity/flow_velocity/disk_radius)