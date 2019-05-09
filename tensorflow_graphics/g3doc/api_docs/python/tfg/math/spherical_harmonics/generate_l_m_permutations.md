<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.math.spherical_harmonics.generate_l_m_permutations" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.math.spherical_harmonics.generate_l_m_permutations

Generates permutations of degree l and order m for spherical harmonics.

``` python
tfg.math.spherical_harmonics.generate_l_m_permutations(
    max_band,
    name=None
)
```



Defined in [`math/spherical_harmonics.py`](https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/math/spherical_harmonics.py).

<!-- Placeholder for "Used in" -->

#### Args:

* <b>`max_band`</b>: An integer scalar storing the highest band.
* <b>`name`</b>: A name for this op. Defaults to
  'spherical_harmonics_generate_l_m_permutations'.


#### Returns:

Two tensors of shape `[max_band*max_band]`.