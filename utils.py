# implementation adapted from:
# https://jsdokken.com/dolfinx-tutorial/chapter2/navierstokes.html
import numpy as np
import gmsh
from petsc4py import PETSc
from mpi4py import MPI
import matplotlib.pyplot as plt
from pathlib import Path

from dolfinx.fem import Constant, Function, assemble_scalar, dirichletbc, form, locate_dofs_geometrical, locate_dofs_topological
from dolfinx.mesh import create_unit_square
from dolfinx.io import (gmshio)
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, create_vector, set_bc
from dolfinx.plot import vtk_mesh
from ufl import (FacetNormal, Identity, TestFunction, TrialFunction,
                 div, dot, ds, dx, inner, lhs, nabla_grad, rhs, sym)


# geometry
def walls(x):
    return np.logical_or(np.isclose(x[1], 0), np.isclose(x[1], 1))

def inflow(x):
    return np.isclose(x[0], 0)

def outflow(x):
    return np.isclose(x[0], 1)


# auxiliary functions
def epsilon(u):
    # Define strain-rate tensor
    return sym(nabla_grad(u))

def sigma(u, p, mu):
    # Define stress tensor
    return 2 * mu * epsilon(u) - p * Identity(len(u))

def u_exact(x):
    values = np.zeros((2, x.shape[1]), dtype=PETSc.ScalarType)
    values[0] = 4 * x[1] * (1.0 - x[1])
    return values


class Variables:
    def __init__(self, V, Q):   
        self.u = TrialFunction(V)
        self.v = TestFunction(V)
        self.p = TrialFunction(Q)
        self.q = TestFunction(Q)
        #
        self.u_n = Function(V)
        self.u_n.name = "u_n"
        self.U = 0.5 * (self.u_n + self.u)
        self.u_ = Function(V)
        self.p_n = Function(Q)
        self.p_n.name = "p_n"
        self.p_ = Function(Q)


class System:
    def __init__(self, mesh):
        self.mesh = mesh
    def set_parameters(self, timestep, viscosity=1., density=1.):
        self.n = FacetNormal(self.mesh)
        self.f = Constant(self.mesh, PETSc.ScalarType((0, 0)))  # force
        self.k = Constant(self.mesh, PETSc.ScalarType(timestep))  # timestep
        self.mu = Constant(self.mesh, PETSc.ScalarType(viscosity))  # viscosity
        self.rho = Constant(self.mesh, PETSc.ScalarType(density))  # density


class Mesh:
    
    def build_simple_squared(self, nx=10, ny=10):
        mesh = create_unit_square(MPI.COMM_WORLD, nx, ny)
        return mesh
    
    def build_disk_in_a_square(self, L=1.0, H=1.0, d_x=0.5, d_y=0.5, r=0.05, resolution=3):
        gmsh.initialize()
        gdim = 2
        #
        mesh_comm = MPI.COMM_WORLD
        model_rank = 0
        if mesh_comm.rank == model_rank:
            rectangle = gmsh.model.occ.addRectangle(0, 0, 0, L, H, tag=1)
            obstacle = gmsh.model.occ.addDisk(d_x, d_y, 0, r, r)
        # subtract the obstacle from the channel, such that we do not mesh the interior of the circle.
        if mesh_comm.rank == model_rank:
            fluid = gmsh.model.occ.cut([(gdim, rectangle)], [(gdim, obstacle)])
            gmsh.model.occ.synchronize()
        # add marker to fluid    
        fluid_marker = 1
        if mesh_comm.rank == model_rank:
            volumes = gmsh.model.getEntities(dim=gdim)
            assert (len(volumes) == 1)
            gmsh.model.addPhysicalGroup(volumes[0][0], [volumes[0][1]], fluid_marker)
            gmsh.model.setPhysicalName(volumes[0][0], fluid_marker, "Fluid")
        # To tag the different surfaces of the mesh, we tag 
        # - the inflow (left hand side) with marker 2, 
        # - the outflow (right hand side) with marker 3 
        # - and the fluid walls (top, bottom) with 4 
        # - and obstacle with 5
        inlet_marker, outlet_marker, wall_marker, obstacle_marker = 2, 3, 4, 5
        inflow, outflow, walls, obstacle = [], [], [], []
        if mesh_comm.rank == model_rank:
            boundaries = gmsh.model.getBoundary(volumes, oriented=False)
            for boundary in boundaries:
                center_of_mass = gmsh.model.occ.getCenterOfMass(boundary[0], boundary[1])
                if np.allclose(center_of_mass, [0, H / 2, 0]):
                    inflow.append(boundary[1])
                elif np.allclose(center_of_mass, [L, H / 2, 0]):
                    outflow.append(boundary[1])
                elif np.allclose(center_of_mass, [L / 2, H, 0]) or np.allclose(center_of_mass, [L / 2, 0, 0]):
                    walls.append(boundary[1])
                else:
                    obstacle.append(boundary[1])
            gmsh.model.addPhysicalGroup(1, walls, wall_marker)
            gmsh.model.setPhysicalName(1, wall_marker, "Walls")
            gmsh.model.addPhysicalGroup(1, inflow, inlet_marker)
            gmsh.model.setPhysicalName(1, inlet_marker, "Inlet")
            gmsh.model.addPhysicalGroup(1, outflow, outlet_marker)
            gmsh.model.setPhysicalName(1, outlet_marker, "Outlet")
            gmsh.model.addPhysicalGroup(1, obstacle, obstacle_marker)
            gmsh.model.setPhysicalName(1, obstacle_marker, "Obstacle")  
        # set meshing parameters
        res_min = r / resolution
        if mesh_comm.rank == model_rank:
            distance_field = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(distance_field, "EdgesList", obstacle)
            threshold_field = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(threshold_field, "IField", distance_field)
            gmsh.model.mesh.field.setNumber(threshold_field, "LcMin", res_min)
            gmsh.model.mesh.field.setNumber(threshold_field, "LcMax", 0.25 * H)
            gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", r)
            gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", 2 * H)
            min_field = gmsh.model.mesh.field.add("Min")
            gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", [threshold_field])
            gmsh.model.mesh.field.setAsBackgroundMesh(min_field)
        # generate the mesh    
        if mesh_comm.rank == model_rank:
            gmsh.option.setNumber("Mesh.Algorithm", 8)
            gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
            gmsh.option.setNumber("Mesh.RecombineAll", 1)
            gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
            gmsh.model.mesh.generate(gdim)
            gmsh.model.mesh.setOrder(2)
            gmsh.model.mesh.optimize("Netgen")
        mesh, _, ft = gmshio.model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=gdim)
        ft.name = "Facet markers"
        return mesh, ft


class Boundary:
    def __init__(self, system):
        self.system = system
        self.bc_u = []
        self.bc_p = []
    def set_top_bottom(self, V, velocity = 0.):
        # velocity on top and bottom edges
        wall_dofs = locate_dofs_geometrical(V, walls)
        u_walls = np.array((velocity,) * self.system.mesh.geometry.dim, dtype=PETSc.ScalarType)
        self.bc_u.append(dirichletbc(u_walls, wall_dofs, V))
    def set_left_pressure(self, Q, pressure = 0.):
        # left edge pressure
        inflow_dofs = locate_dofs_geometrical(Q, inflow)
        self.bc_p.append(dirichletbc(PETSc.ScalarType(pressure), inflow_dofs, Q))
    def set_right_pressure(self, Q, pressure = 0.):
        # right edge pressure
        outflow_dofs = locate_dofs_geometrical(Q, outflow)
        self.bc_p.append(dirichletbc(PETSc.ScalarType(pressure), outflow_dofs, Q))
    # using markers
    def set_velocity_marker(self, V, ft, marker, velocity = (0.,0.)):
        u_to_set = np.array(velocity, dtype=PETSc.ScalarType)
        self.bc_u.append(dirichletbc(u_to_set, locate_dofs_topological(V, self.system.mesh.topology.dim - 1, ft.find(marker)), V))
    def set_pressure_marker(self, Q, ft, marker, pressure = 0.):
        self.bc_p.append(dirichletbc(PETSc.ScalarType(pressure), locate_dofs_topological(Q, self.system.mesh.topology.dim - 1, ft.find(marker)), Q))


class Solver:
    def __init__(self, variables, system, boundary):
        self.variables = variables
        self.system = system
        self.boundary = boundary
    def setup_variational_problem(self):
        r = self.variables
        s = self.system
        # Define the variational problem for the 1st step
        F1 = s.rho * dot((r.u - r.u_n) / s.k, r.v) * dx
        F1 += s.rho * dot(dot(r.u_n, nabla_grad(r.u_n)), r.v) * dx
        F1 += inner(sigma(r.U, r.p_n, s.mu), epsilon(r.v)) * dx
        F1 += dot(r.p_n * s.n, r.v) * ds - dot(s.mu * nabla_grad(r.U) * s.n, r.v) * ds
        F1 -= dot(s.f, r.v) * dx
        self.a1 = form(lhs(F1))
        self.L1 = form(rhs(F1))
        self.A1 = assemble_matrix(self.a1, bcs=self.boundary.bc_u)
        self.A1.assemble()
        self.b1 = create_vector(self.L1)
        # Define variational problem for step 2
        self.a2 = form(dot(nabla_grad(r.p), nabla_grad(r.q)) * dx)
        self.L2 = form(dot(nabla_grad(r.p_n), nabla_grad(r.q)) * dx - (s.rho / s.k) * div(r.u_) * r.q * dx)
        self.A2 = assemble_matrix(self.a2, bcs=self.boundary.bc_p)
        self.A2.assemble()
        self.b2 = create_vector(self.L2)
        # Define variational problem for step 3
        self.a3 = form(s.rho * dot(r.u, r.v) * dx)
        self.L3 = form(s.rho * dot(r.u_, r.v) * dx - s.k * dot(nabla_grad(r.p_ - r.p_n), r.v) * dx)
        self.A3 = assemble_matrix(self.a3)
        self.A3.assemble()
        self.b3 = create_vector(self.L3)
    def setup_solver(self):
        # Solver for step 1
        self.solver1 = PETSc.KSP().create(self.system.mesh.comm)
        self.solver1.setOperators(self.A1)
        self.solver1.setType(PETSc.KSP.Type.BCGS)
        pc1 = self.solver1.getPC()
        pc1.setType(PETSc.PC.Type.HYPRE)
        pc1.setHYPREType("boomeramg")
        # Solver for step 2
        self.solver2 = PETSc.KSP().create(self.system.mesh.comm)
        self.solver2.setOperators(self.A2)
        self.solver2.setType(PETSc.KSP.Type.BCGS)
        pc2 = self.solver2.getPC()
        pc2.setType(PETSc.PC.Type.HYPRE)
        pc2.setHYPREType("boomeramg")
        # Solver for step 3
        self.solver3 = PETSc.KSP().create(self.system.mesh.comm)
        self.solver3.setOperators(self.A3)
        self.solver3.setType(PETSc.KSP.Type.CG)
        pc3 = self.solver3.getPC()
        pc3.setType(PETSc.PC.Type.SOR)
    def step(self):
        r = self.variables
        # Step 1: Tentative veolcity step
        with self.b1.localForm() as loc_1:
            loc_1.set(0)
        assemble_vector(self.b1, self.L1)
        apply_lifting(self.b1, [self.a1], [self.boundary.bc_u])
        self.b1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(self.b1, self.boundary.bc_u)
        self.solver1.solve(self.b1, r.u_.x.petsc_vec)
        r.u_.x.scatter_forward()
        # Step 2: Pressure corrrection step
        with self.b2.localForm() as loc_2:
            loc_2.set(0)
        assemble_vector(self.b2, self.L2)
        apply_lifting(self.b2, [self.a2], [self.boundary.bc_p])
        self.b2.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(self.b2, self.boundary.bc_p)
        self.solver2.solve(self.b2, r.p_.x.petsc_vec)
        r.p_.x.scatter_forward()
        # Step 3: Velocity correction step
        with self.b3.localForm() as loc_3:
            loc_3.set(0)
        assemble_vector(self.b3, self.L3)
        self.b3.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        self.solver3.solve(self.b3, r.u_.x.petsc_vec)
        r.u_.x.scatter_forward()
        # Update variable with solution form this time step
        r.u_n.x.array[:] = r.u_.x.array[:]
        r.p_n.x.array[:] = r.p_.x.array[:]
    def clean(self):
        self.b1.destroy()
        self.b2.destroy()
        self.b3.destroy()
        self.solver1.destroy()
        self.solver2.destroy()
        self.solver3.destroy()
        
class Exact:
    def __init__(self, solver, V, solution):
        self.mesh = solver.system.mesh
        self.variables = solver.variables
        self.u_ex = Function(V)
        self.u_ex.interpolate(solution)
        self.L2_error = form(dot(self.variables.u_ - self.u_ex, self.variables.u_ - self.u_ex) * dx)  
    def get_errors(self):
        error_L2 = np.sqrt(self.mesh.comm.allreduce(assemble_scalar(self.L2_error), op=MPI.SUM))
        error_max = self.mesh.comm.allreduce(np.max(self.variables.u_.x.petsc_vec.array - self.u_ex.x.petsc_vec.array), op=MPI.MAX)
        return error_L2, error_max
    
class Values:
    def __init__(self, V, Q, variables):
        topology, cell_types, self.u_geometry = vtk_mesh(V)
        topology, cell_types, self.p_geometry = vtk_mesh(Q)
        self.variables = variables
    def get_actual(self): 
        self.u = np.zeros((self.u_geometry.shape[0], 2), dtype=np.float64)
        self.u[:, :len(self.variables.u_n)] = self.variables.u_n.x.array.real.reshape((self.u_geometry.shape[0], len(self.variables.u_n)))
        self.p = np.zeros(self.p_geometry.shape[0], dtype=np.float64)
        self.p[:] = self.variables.p_n.x.array.real
        return self.u, self.p
    
class Plot:
    def __init__(self, path=Path('./')):   
        self.path = path
    def plot_velocity(self, u, geometry, filename='u.png', scaling=None):
        fig, ax = plt.subplots()
        x = geometry[:,0]
        y = geometry[:,1]
        #c = ax.pcolormesh(x, y, u[:,0])
        if scaling is not None:
            c = ax.quiver(x, y, u[:,0], u[:,1], np.sqrt(u[:,0]**2+u[:,1]**2), scale=scaling)
        else:
            c = ax.quiver(x, y, u[:,0], u[:,1], np.sqrt(u[:,0]**2+u[:,1]**2))
        ax.set_title('velocity field')
        # set the limits of the plot to the limits of the data
        ax.axis([x.min(), x.max(), y.min(), y.max()])
        ax.set_aspect('equal')
        fig.colorbar(c, ax=ax)
        plt.savefig(str(self.path / filename), bbox_inches='tight', dpi=300)
    def plot_pressure(self, p, geometry, filename='p.png', pointsize=None):
        fig, ax = plt.subplots()
        x = geometry[:,0]
        y = geometry[:,1]
        if pointsize is not None:
            c = ax.scatter(x, y, c=p, s=pointsize)
        else:
            c = ax.scatter(x, y, c=p)            
        ax.set_title('pressure field')
        # set the limits of the plot to the limits of the data
        ax.axis([x.min(), x.max(), y.min(), y.max()])
        ax.set_aspect('equal')
        fig.colorbar(c, ax=ax)
        plt.savefig(str(self.path / filename), bbox_inches='tight', dpi=300)