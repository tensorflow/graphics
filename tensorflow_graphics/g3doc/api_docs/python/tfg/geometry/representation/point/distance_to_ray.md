<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.geometry.representation.point.distance_to_ray" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.geometry.representation.point.distance_to_ray

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/geometry/representation/point.py">View source</a>



Computes the distance from a M-d point to a M-d ray.

```python
tfg.geometry.representation.point.distance_to_ray(
    point, origin, direction, keepdims=True, name=None
)
```



<!-- Placeholder for "Used in" -->


#### Note:

In the following, A1 to An are optional batch dimensions, which must be
broadcast compatible.



#### Args:


* <b>`point`</b>: A tensor of shape `[A1, ..., An, M]`.
* <b>`origin`</b>: A tensor of shape `[A1, ..., An, M]`.
* <b>`direction`</b>: A tensor of shape `[A1, ..., An, M]`. The last dimension must be
  normalized.
* <b>`keepdims`</b>: A `bool`, whether to keep the last dimension with length 1 or to
  remove it.
* <b>`name`</b>: A name for this op. Defaults to "point_distance_to_ray".


#### Returns:

A tensor of shape `[A1, ..., An, 1]` containing the distance from each point
to the corresponding ray.



#### Raises:


* <b>`ValueError`</b>: If the shape of `point`, `origin`, or 'direction' is not
supported.