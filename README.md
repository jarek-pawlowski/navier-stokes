# (not so) simple Navier-Stokes 2D boundary problems 
solved via Fenics FEM engine
### Prerequisites
`conda install -c conda-forge fenics-dolfinx mpich pyvista`\
`conda install -c conda-forge gmsh python-gmsh`
### Problems
1. Laminar flow trough a pipe, just run `python channel.py`
2. Disc obstracle added, `python disk.py`

Pipe flow | Disk flow
:--:|:--:
![](/results/channel_u.png)  |  ![](/results/disk_u.png)
