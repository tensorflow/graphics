<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.math.interpolation.slerp" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tfg.math.interpolation.slerp

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/math/interpolation/slerp.py">View source</a>



Tensorflow.graphics slerp interpolation module.


Spherical linear interpolation (slerp) is defined for both quaternions and for
regular M-D vectors, and act slightly differently because of inherent
ambiguity of quaternions. This module has two functions returning the
interpolation weights for quaternions (quaternion_weights) and for vectors
(vector_weights), which can then be used in a weighted sum to calculate the
final interpolated quaternions and vectors. A helper interpolate function is
also provided.

The main differences between two methods are:
vector_weights:
  can get any M-D tensor as input,
  does not expect normalized vectors as input,
  returns unnormalized outputs (in general) for unnormalized inputs.

quaternion_weights:
  expects M-D tensors with a last dimension of 4,
  assumes normalized input,
  checks for ambiguity by looking at the angle between quaternions,
  returns normalized quaternions naturally.

## Classes

[`class InterpolationType`](../../../tfg/math/interpolation/slerp/InterpolationType.md): Defines interpolation methods for slerp module.

## Functions

[`interpolate(...)`](../../../tfg/math/interpolation/slerp/interpolate.md): Applies slerp to vectors or quaternions.

[`interpolate_with_weights(...)`](../../../tfg/math/interpolation/slerp/interpolate_with_weights.md): Interpolates vectors by taking their weighted sum.

[`quaternion_weights(...)`](../../../tfg/math/interpolation/slerp/quaternion_weights.md): Calculates slerp weights for two normalized quaternions.

[`vector_weights(...)`](../../../tfg/math/interpolation/slerp/vector_weights.md): Spherical linear interpolation (slerp) between two unnormalized vectors.

