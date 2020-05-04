<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.geometry.representation.ray.triangulate" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.geometry.representation.ray.triangulate

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/geometry/representation/ray.py">View source</a>



Triangulates 3d points by miminizing the sum of squared distances to rays.

```python
tfg.geometry.representation.ray.triangulate(
    startpoints, endpoints, weights, name=None
)
```



<!-- Placeholder for "Used in" -->

The rays are defined by their start points and endpoints. At least two rays
are required to triangulate any given point. Contrary to the standard
reprojection-error metric, the sum of squared distances to rays can be
minimized in a closed form.

#### Note:

In the following, A1 to An are optional batch dimensions.



#### Args:


* <b>`startpoints`</b>: A tensor of ray start points with shape `[A1, ..., An, V, 3]`,
  the number of rays V around which the solution points live should be
  greater or equal to 2, otherwise triangulation is impossible.
* <b>`endpoints`</b>: A tensor of ray endpoints with shape `[A1, ..., An, V, 3]`, the
  number of rays V around which the solution points live should be greater
  or equal to 2, otherwise triangulation is impossible. The `endpoints`
  tensor should have the same shape as the `startpoints` tensor.
* <b>`weights`</b>: A tensor of ray weights (certainties) with shape `[A1, ..., An,
  V]`. Weights should have all positive entries. Weight should have at least
  two non-zero entries for each point (at least two rays should have
  certainties > 0).
* <b>`name`</b>: A name for this op. The default value of None means "ray_triangulate".


#### Returns:

A tensor of triangulated points with shape `[A1, ..., An, 3]`.



#### Raises:


* <b>`ValueError`</b>: If the shape of the arguments is not supported.