<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.image.matting.laplacian_weights" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.image.matting.laplacian_weights

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/image/matting.py">View
source</a>

Generates the closed form matting Laplacian weights.

```python
tfg.image.matting.laplacian_weights(
    image,
    size=3,
    eps=1e-05,
    name=None
)
```

<!-- Placeholder for "Used in" -->

Generates the closed form matting Laplacian weights as proposed by Levin et al.
in "A Closed Form Solution to Natural Image Matting".

#### Args:

*   <b>`image`</b>: A tensor of shape `[B, H, W, C]`.
*   <b>`size`</b>: An `int` representing the size of the patches used to enforce
    smoothness.
*   <b>`eps`</b>: A small number of type `float` to regularize the problem.
*   <b>`name`</b>: A name for this op. Defaults to "matting_laplacian_weights".

#### Returns:

A tensor of shape `[B, H, W, size^2, size^2]` containing the matting Laplacian
weights .

#### Raises:

*   <b>`ValueError`</b>: If `image` is not of rank 4.
