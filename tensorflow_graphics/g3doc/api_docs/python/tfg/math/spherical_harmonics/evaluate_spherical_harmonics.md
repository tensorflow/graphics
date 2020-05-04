<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.math.spherical_harmonics.evaluate_spherical_harmonics" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.math.spherical_harmonics.evaluate_spherical_harmonics

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/math/spherical_harmonics.py">View source</a>



Evaluates a point sample of a Spherical Harmonic basis function.

```python
tfg.math.spherical_harmonics.evaluate_spherical_harmonics(
    degree_l, order_m, theta, phi, name=None
)
```



<!-- Placeholder for "Used in" -->


#### Note:

This function is implementating the algorithm and variable names described
p. 12 of 'Spherical Harmonic Lighting: The Gritty Details.



#### Note:

In the following, A1 to An are optional batch dimensions.



#### Args:


* <b>`degree_l`</b>: An integer tensor of shape `[A1, ..., An, C]`, where the last
  dimension represents the band of the spherical harmonics. Note that
  `degree_l` must be non-negative.
* <b>`order_m`</b>: An integer tensor of shape `[A1, ..., An, C]`, where the last
  dimension represents the index of the spherical harmonics in the band
  `degree_l`. Note that `order_m` must satisfy `0 <= order_m <= l`.
* <b>`theta`</b>: A tensor of shape `[A1, ..., An, 1]`. This variable stores the polar
  angle of the sameple. Values of theta must be in [0, pi].
* <b>`phi`</b>: A tensor of shape `[A1, ..., An, 1]`. This variable stores the
  azimuthal angle of the sameple. Values of phi must be in [0, 2pi].
* <b>`name`</b>: A name for this op. Defaults to
  'spherical_harmonics_evaluate_spherical_harmonics'.


#### Returns:

A tensor of shape `[A1, ..., An, C]` containing the evaluation of each basis
of the spherical harmonics.



#### Raises:


* <b>`ValueError`</b>: if the shape of `theta` or `phi` is not supported.
* <b>`InvalidArgumentError`</b>: if at least an element of `l`, `m`, `theta` or `phi`
is outside the expected range.