<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.math.spherical_harmonics.generate_l_m_permutations" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.math.spherical_harmonics.generate_l_m_permutations

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/math/spherical_harmonics.py">View source</a>



Generates permutations of degree l and order m for spherical harmonics.

```python
tfg.math.spherical_harmonics.generate_l_m_permutations(
    max_band, name=None
)
```



<!-- Placeholder for "Used in" -->


#### Args:


* <b>`max_band`</b>: An integer scalar storing the highest band.
* <b>`name`</b>: A name for this op. Defaults to
  'spherical_harmonics_generate_l_m_permutations'.


#### Returns:

Two tensors of shape `[max_band*max_band]`.
