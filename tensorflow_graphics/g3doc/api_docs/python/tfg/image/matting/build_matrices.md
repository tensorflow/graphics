<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.image.matting.build_matrices" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.image.matting.build_matrices

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/image/matting.py">View source</a>



Generates the closed form matting Laplacian.

```python
tfg.image.matting.build_matrices(
    image, size=3, eps=1e-05, name=None
)
```



<!-- Placeholder for "Used in" -->

Generates the closed form matting Laplacian as proposed by Levin et
al. in "A Closed Form Solution to Natural Image Matting". This function also
return the pseudo-inverse matrix allowing to retrieve the matting linear
coefficient.

#### Args:


* <b>`image`</b>: A tensor of shape `[B, H, W, C]`.
* <b>`size`</b>: An `int` representing the size of the patches used to enforce
  smoothness.
* <b>`eps`</b>: A small number of type `float` to regularize the problem.
* <b>`name`</b>: A name for this op. Defaults to "matting_build_matrices".


#### Returns:

A tensor of shape `[B, H - pad, W - pad, size^2, size^2]` containing
the matting Laplacian matrices. A tensor of shape
`[B, H - pad, W - pad, C + 1, size^2]` containing the pseudo-inverse
matrices which can be used to retrieve the matting linear coefficients.
The padding `pad` is equal to `size - 1`.



#### Raises:


* <b>`ValueError`</b>: If `image` is not of rank 4.