<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.math.spherical_harmonics.generate_l_m_zonal" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.math.spherical_harmonics.generate_l_m_zonal

Generates l and m coefficients for zonal harmonics.

``` python
tfg.math.spherical_harmonics.generate_l_m_zonal(
    max_band,
    name=None
)
```



Defined in [`math/spherical_harmonics.py`](https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/math/spherical_harmonics.py).

<!-- Placeholder for "Used in" -->

#### Args:

* <b>`max_band`</b>: An integer scalar storing the highest band.
* <b>`name`</b>: A name for this op. Defaults to
  'spherical_harmonics_generate_l_m_zonal'.


#### Returns:

Two tensors of shape `[max_band+1]`, one for degree l and one for order m.