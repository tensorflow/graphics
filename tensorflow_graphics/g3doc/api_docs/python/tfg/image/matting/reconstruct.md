<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.image.matting.reconstruct" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.image.matting.reconstruct

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/image/matting.py">View source</a>



Reconstruct the matte from the image using the linear coefficients.

```python
tfg.image.matting.reconstruct(
    image, coeff_mul, coeff_add, name=None
)
```



<!-- Placeholder for "Used in" -->

Reconstruct the matte from the image using the linear coefficients (a, b)
returned by the linear_coefficients function.

#### Args:


* <b>`image`</b>: A tensor of shape `[B, H, W, C]` .
* <b>`coeff_mul`</b>: A tensor of shape `[B, H, W, C]` representing the multiplicative
  part of the linear coefficients.
* <b>`coeff_add`</b>: A tensor of shape `[B, H, W, 1]` representing the additive part
  of the linear coefficients.
* <b>`name`</b>: A name for this op. Defaults to "matting_reconstruct".


#### Returns:

A tensor of shape `[B, H, W, 1]` containing the mattes.



#### Raises:


* <b>`ValueError`</b>: If `image`, `coeff_mul`, or `coeff_add` are not of rank 4. If
the last dimension of `coeff_add` is not 1. If the batch dimensions of
`image`, `coeff_mul`, and `coeff_add` do not match.