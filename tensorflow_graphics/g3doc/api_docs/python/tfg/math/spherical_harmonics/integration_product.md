<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.math.spherical_harmonics.integration_product" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.math.spherical_harmonics.integration_product

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/math/spherical_harmonics.py">View source</a>



Computes the integral of harmonics1.harmonics2 over the sphere.

```python
tfg.math.spherical_harmonics.integration_product(
    harmonics1, harmonics2, keepdims=True, name=None
)
```



<!-- Placeholder for "Used in" -->


#### Note:

In the following, A1 to An are optional batch dimensions.



#### Args:


* <b>`harmonics1`</b>: A tensor of shape `[A1, ..., An, C]`, where the last dimension
  represents spherical harmonics coefficients.
* <b>`harmonics2`</b>: A tensor of shape `[A1, ..., An, C]`, where the last dimension
  represents spherical harmonics coefficients.
* <b>`keepdims`</b>: If True, retains reduced dimensions with length 1.
* <b>`name`</b>: A name for this op. Defaults to "spherical_harmonics_convolution".


#### Returns:

A tensor of shape `[A1, ..., An]` containing scalar values resulting from
integrating the product of the spherical harmonics `harmonics1` and
`harmonics2`.



#### Raises:


* <b>`ValueError`</b>: if the last dimension of `harmonics1` is different from the last
dimension of `harmonics2`.