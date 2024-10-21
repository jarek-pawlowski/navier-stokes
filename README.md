# (not so) simple Navier-Stokes boundary 2D problems 
solved via Fenics FEM-solver
### Prerequisites
`conda install -c conda-forge fenics-dolfinx mpich pyvista`\
`conda install -c conda-forge gmsh python-gmsh`
### Problems
1. Laminar flow trough pipe, just run `python channel.py`
2. Disc obstracle added, `python disk.py`

Pipe flow | Disk flow
:--:|:--:
![](/results/channel_u.png)  |  ![](/results/disk_u.png)
