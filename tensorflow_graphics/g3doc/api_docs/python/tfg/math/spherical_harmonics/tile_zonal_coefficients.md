<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.math.spherical_harmonics.tile_zonal_coefficients" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.math.spherical_harmonics.tile_zonal_coefficients

Tiles zonal coefficients.

``` python
tfg.math.spherical_harmonics.tile_zonal_coefficients(
    coefficients,
    name=None
)
```



Defined in [`math/spherical_harmonics.py`](https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/math/spherical_harmonics.py).

<!-- Placeholder for "Used in" -->

Zonal Harmonics only contains the harmonics where m=0. This function returns
these coefficients for -l <= m <= l, where l is the rank of `coefficients`.

#### Args:

coefficients: A tensor of shape `[C]` storing zonal harmonics coefficients.
name: A name for this op. Defaults to
  'spherical_harmonics_tile_zonal_coefficients'.
* <b>`Return`</b>: A tensor of shape `[C*C]` containing zonal coefficients tiled as
  'regular' spherical harmonics coefficients.


#### Raises:

* <b>`ValueError`</b>: if the shape of `coefficients` is not supported.