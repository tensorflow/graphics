Dependency
-------------------------------------------------------------------
`Differentiable MPM` depends on `Python 3` with packages:
 - `tensorflow`
 - `cv2`
 - `numpy`

We will add CUDA support later. For now the dependency is simple.

Examples
------------------------------
Please see `jump.py`


Building the documentation
-------------------------------------

.. code-block:: bash

    sudo pip3 install Sphinx sphinx_rtd_theme
    sphinx-build . build
