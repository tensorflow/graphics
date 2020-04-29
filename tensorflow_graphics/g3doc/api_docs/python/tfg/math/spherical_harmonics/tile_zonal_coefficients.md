<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.math.spherical_harmonics.tile_zonal_coefficients" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.math.spherical_harmonics.tile_zonal_coefficients

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/math/spherical_harmonics.py">View source</a>



Tiles zonal coefficients.

```python
tfg.math.spherical_harmonics.tile_zonal_coefficients(
    coefficients, name=None
)
```



<!-- Placeholder for "Used in" -->

Zonal Harmonics only contains the harmonics where m=0. This function returns
these coefficients for -l <= m <= l, where l is the rank of `coefficients`.

#### Args:


* <b>`coefficients`</b>: A tensor of shape `[C]` storing zonal harmonics coefficients.
* <b>`name`</b>: A name for this op. Defaults to
  'spherical_harmonics_tile_zonal_coefficients'.
Return: A tensor of shape `[C*C]` containing zonal coefficients tiled as
  'regular' spherical harmonics coefficients.

#### Raises:


* <b>`ValueError`</b>: if the shape of `coefficients` is not supported.