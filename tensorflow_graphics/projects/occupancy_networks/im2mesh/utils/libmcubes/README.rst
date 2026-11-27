========
PyMCubes
========

PyMCubes is an implementation of the marching cubes algorithm to extract
isosurfaces from volumetric data. The volumetric data can be given as a
three-dimensional NumPy array or as a Python function ``f(x, y, z)``. The first
option is much faster, but it requires more memory and becomes unfeasible for
very large volumes.

PyMCubes also provides a function to export the results of the marching cubes as
COLLADA ``(.dae)`` files. This requires the
`PyCollada <https://github.com/pycollada/pycollada>`_ library.

Installation
============

Just as any standard Python package, clone or download the project
and run::

  $ cd path/to/PyMCubes
  $ python setup.py build
  $ python setup.py install

If you do not have write permission on the directory of Python packages,
install with the ``--user`` option::

  $ python setup.py install --user

Example
=======

The following example creates a data volume with spherical isosurfaces and
extracts one of them (i.e., a sphere) with PyMCubes. The result is exported as
``sphere.dae``::

  >>> import numpy as np
  >>> import mcubes
  
  # Create a data volume (30 x 30 x 30)
  >>> X, Y, Z = np.mgrid[:30, :30, :30]
  >>> u = (X-15)**2 + (Y-15)**2 + (Z-15)**2 - 8**2
  
  # Extract the 0-isosurface
  >>> vertices, triangles = mcubes.marching_cubes(u, 0)
  
  # Export the result to sphere.dae
  >>> mcubes.export_mesh(vertices, triangles, "sphere.dae", "MySphere")

The second example is very similar to the first one, but it uses a function
to represent the volume instead of a NumPy array::

  >>> import numpy as np
  >>> import mcubes
  
  # Create the volume
  >>> f = lambda x, y, z: x**2 + y**2 + z**2
  
  # Extract the 16-isosurface
  >>> vertices, triangles = mcubes.marching_cubes_func((-10,-10,-10), (10,10,10),
  ... 100, 100, 100, f, 16)
  
  # Export the result to sphere2.dae
  >>> mcubes.export_mesh(vertices, triangles, "sphere2.dae", "MySphere")
