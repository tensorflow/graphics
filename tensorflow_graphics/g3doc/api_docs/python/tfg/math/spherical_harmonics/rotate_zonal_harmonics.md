<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.math.spherical_harmonics.rotate_zonal_harmonics" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.math.spherical_harmonics.rotate_zonal_harmonics

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/math/spherical_harmonics.py">View source</a>



Rotates zonal harmonics.

```python
tfg.math.spherical_harmonics.rotate_zonal_harmonics(
    zonal_coeffs, theta, phi, name=None
)
```



<!-- Placeholder for "Used in" -->


#### Note:

In the following, A1 to An are optional batch dimensions.



#### Args:


* <b>`zonal_coeffs`</b>: A tensor of shape `[C]` storing zonal harmonics coefficients.
* <b>`theta`</b>: A tensor of shape `[A1, ..., An, 1]` storing polar angles.
* <b>`phi`</b>: A tensor of shape `[A1, ..., An, 1]` storing azimuthal angles.
* <b>`name`</b>: A name for this op. Defaults to
  'spherical_harmonics_rotate_zonal_harmonics'.


#### Returns:

A tensor of shape `[A1, ..., An, C*C]` storing coefficients of the rotated
harmonics.



#### Raises:


* <b>`ValueError`</b>: If the shape of `zonal_coeffs`, `theta` or `phi` is not
  supported.