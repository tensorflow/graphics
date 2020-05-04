<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.geometry.transformation.axis_angle" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tfg.geometry.transformation.axis_angle

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/geometry/transformation/axis_angle.py">View source</a>



This module implements axis-angle functionalities.


The axis-angle representation is defined as $$\theta\mathbf{a}$$, where
$$\mathbf{a}$$ is a unit vector indicating the direction of rotation and
$$\theta$$ is a scalar controlling the angle of rotation. It is important to
note that the axis-angle does not perform rotation by itself, but that it can be
used to rotate any given vector $$\mathbf{v} \in {\mathbb{R}^3}$$ into
a vector $$\mathbf{v}'$$ using the Rodrigues' rotation formula:

$$\mathbf{v}'=\mathbf{v}\cos(\theta)+(\mathbf{a}\times\mathbf{v})\sin(\theta)
+\mathbf{a}(\mathbf{a}\cdot\mathbf{v})(1-\cos(\theta)).$$

More details about the axis-angle formalism can be found on [this page.]
(https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation)

Note: Some of the functions defined in the module expect
a normalized axis $$\mathbf{a} = [x, y, z]^T$$ as inputs where
$$x^2 + y^2 + z^2 = 1$$.

## Functions

[`from_euler(...)`](../../../tfg/geometry/transformation/axis_angle/from_euler.md): Converts Euler angles to an axis-angle representation.

[`from_euler_with_small_angles_approximation(...)`](../../../tfg/geometry/transformation/axis_angle/from_euler_with_small_angles_approximation.md): Converts small Euler angles to an axis-angle representation.

[`from_quaternion(...)`](../../../tfg/geometry/transformation/axis_angle/from_quaternion.md): Converts a quaternion to an axis-angle representation.

[`from_rotation_matrix(...)`](../../../tfg/geometry/transformation/axis_angle/from_rotation_matrix.md): Converts a rotation matrix to an axis-angle representation.

[`inverse(...)`](../../../tfg/geometry/transformation/axis_angle/inverse.md): Computes the axis-angle that is the inverse of the input axis-angle.

[`is_normalized(...)`](../../../tfg/geometry/transformation/axis_angle/is_normalized.md): Determines if the axis-angle is normalized or not.

[`rotate(...)`](../../../tfg/geometry/transformation/axis_angle/rotate.md): Rotates a 3d point using an axis-angle by applying the Rodrigues' formula.

