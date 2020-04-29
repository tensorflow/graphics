<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.geometry.transformation.quaternion" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tfg.geometry.transformation.quaternion

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/geometry/transformation/quaternion.py">View source</a>



This module implements TensorFlow quaternion utility functions.


A quaternion is written as $$q =  xi + yj + zk + w$$, where $$i,j,k$$ forms the
three bases of the imaginary part. The functions implemented in this file
use the Hamilton convention where $$i^2 = j^2 = k^2 = ijk = -1$$. A quaternion
is stored in a 4-D vector $$[x, y, z, w]^T$$.

More details about Hamiltonian quaternions can be found on [this page.]
(https://en.wikipedia.org/wiki/Quaternion)

Note: Some of the functions expect normalized quaternions as inputs where
$$x^2 + y^2 + z^2 + w^2 = 1$$.

## Functions

[`between_two_vectors_3d(...)`](../../../tfg/geometry/transformation/quaternion/between_two_vectors_3d.md): Computes quaternion over the shortest arc between two vectors.

[`conjugate(...)`](../../../tfg/geometry/transformation/quaternion/conjugate.md): Computes the conjugate of a quaternion.

[`from_axis_angle(...)`](../../../tfg/geometry/transformation/quaternion/from_axis_angle.md): Converts an axis-angle representation to a quaternion.

[`from_euler(...)`](../../../tfg/geometry/transformation/quaternion/from_euler.md): Converts an Euler angle representation to a quaternion.

[`from_euler_with_small_angles_approximation(...)`](../../../tfg/geometry/transformation/quaternion/from_euler_with_small_angles_approximation.md): Converts small Euler angles to quaternions.

[`from_rotation_matrix(...)`](../../../tfg/geometry/transformation/quaternion/from_rotation_matrix.md): Converts a rotation matrix representation to a quaternion.

[`inverse(...)`](../../../tfg/geometry/transformation/quaternion/inverse.md): Computes the inverse of a quaternion.

[`is_normalized(...)`](../../../tfg/geometry/transformation/quaternion/is_normalized.md): Determines if quaternion is normalized quaternion or not.

[`multiply(...)`](../../../tfg/geometry/transformation/quaternion/multiply.md): Multiplies two quaternions.

[`normalize(...)`](../../../tfg/geometry/transformation/quaternion/normalize.md): Normalizes a quaternion.

[`normalized_random_uniform(...)`](../../../tfg/geometry/transformation/quaternion/normalized_random_uniform.md): Random normalized quaternion following a uniform distribution law on SO(3).

[`normalized_random_uniform_initializer(...)`](../../../tfg/geometry/transformation/quaternion/normalized_random_uniform_initializer.md): Random unit quaternion initializer.

[`relative_angle(...)`](../../../tfg/geometry/transformation/quaternion/relative_angle.md): Computes the unsigned relative rotation angle between 2 unit quaternions.

[`rotate(...)`](../../../tfg/geometry/transformation/quaternion/rotate.md): Rotates a point using a quaternion.

