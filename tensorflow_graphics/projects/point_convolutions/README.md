# TensorFlow Graphics Point Cloud Convolutions

This module contains a python TensorFlow module `pylib` and a custom ops package in `tfg_custom_ops`.
While it is possible to run without the custom ops package, it is strongly advised to install it for performance and memory efficiency.

## Content

This code contains all necessary operations to perform point cloud convolutions

1. Datastructure
    - Point cloud class for batches of arbitrary sized point clouds.
    - Memory efficient regular grid data structure
2. Point cloud operations
    - Neighborhood computation
    - Point density estimation
    - Spatial sampling
      - Poisson Disk sampling
      - Cell average sampling
3. Convolution kernels 
    - Kernel Point Convolutions
      - linear interpolation
      - gaussian interpolation
      - deformable points with regularization loss as in [KPConv](https://arxiv.org/abs/1904.08889)
    - MLP
      - multiple MLPs as in [MCConv](https://arxiv.org/abs/1806.01759)
      - single MLP as in [PointConv](https://arxiv.org/abs/1811.07246)
4. Feature aggregation inside receptive fields
    - Monte-Carlo integration with pdf
    - Constant summation
5. Easy to use classes for building models
    - `PointCloud` class
    - `PointHierarchy` for sequential downsampling of point clouds
    - layer classes
      - `MCConv`
      - `PointConv`
      - `KPConv`
    - ResNet building blocks
      - `ResNet`
      - `ResNetBottleNeck`
      - `ResNetSpatialBottleNeck` 

## Installation

Precompiled versions of the custom ops package are provided in `custom_ops/pkg_builds/tf_*` for the latest TensorFlow versions.
For compilation instructions see the [README](custom_ops/README.md) in the `custom_ops` folder.

To install it run the following command (replace `VERSION` with your installed TensorFlow version, e.g. `2.3.0`)
```bash
  pip install custom_ops/tf_VERSION/*.whl
```

## Tutorials

Check out the Colab notebooks for an introduction to the code

- [Introduction](pylib/notebooks/Introduction.ipynb)
- [Classification on ModelNet40](pylib/notebooks/ModelNet40.ipynb) 

## Unit tests

Unit tests can be evaluated using

```bash
  pip install -r pytest_requirements.txt
  pytest pylib/
```

These include tests of the custom ops if they are installed.

## Additional Information

You may use this software under the
[Apache 2.0 License](https://github.com/schellmi42/tensorflow_graphics_point_clouds/blob/master/LICENSE).