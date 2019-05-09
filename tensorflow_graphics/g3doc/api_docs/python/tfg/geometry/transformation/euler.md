<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.geometry.transformation.euler" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tfg.geometry.transformation.euler

This modules implements Euler angles functionalities.



Defined in [`geometry/transformation/euler.py`](https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/geometry/transformation/euler.py).

<!-- Placeholder for "Used in" -->

The Euler angles are defined using a vector $$[\theta, \gamma, \beta]^T \in
\mathbb{R}^3$$, where $$\theta$$ is the angle about $$x$$, $$\gamma$$ the angle
about $$y$$, and $$\beta$$ is the angle about $$z$$

More details about Euler angles can be found on [this page.]
(https://en.wikipedia.org/wiki/Euler_angles)

Note: The angles are defined in radians.

## Functions

[`from_axis_angle(...)`](../../../tfg/geometry/transformation/euler/from_axis_angle.md): Converts axis-angle to Euler angles.

[`from_quaternion(...)`](../../../tfg/geometry/transformation/euler/from_quaternion.md): Converts quaternions to Euler angles.

[`from_rotation_matrix(...)`](../../../tfg/geometry/transformation/euler/from_rotation_matrix.md): Converts rotation matrices to Euler angles.

[`inverse(...)`](../../../tfg/geometry/transformation/euler/inverse.md): Computes the angles that would inverse a transformation by euler_angle.

