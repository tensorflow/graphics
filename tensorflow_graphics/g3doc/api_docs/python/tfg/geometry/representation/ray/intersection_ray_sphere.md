<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.geometry.representation.ray.intersection_ray_sphere" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.geometry.representation.ray.intersection_ray_sphere

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/geometry/representation/ray.py">View source</a>



Finds positions and surface normals where the sphere and the ray intersect.

```python
tfg.geometry.representation.ray.intersection_ray_sphere(
    sphere_center, sphere_radius, ray, point_on_ray, name=None
)
```



<!-- Placeholder for "Used in" -->


#### Note:

In the following, A1 to An are optional batch dimensions.



#### Args:


* <b>`sphere_center`</b>: A tensor of shape `[3]` representing the 3d sphere center.
* <b>`sphere_radius`</b>: A tensor of shape `[1]` containing a strictly positive value
  defining the radius of the sphere.
* <b>`ray`</b>: A tensor of shape `[A1, ..., An, 3]` containing normalized 3D vectors.
* <b>`point_on_ray`</b>: A tensor of shape `[A1, ..., An, 3]`.
* <b>`name`</b>: A name for this op. The default value of None means
  "ray_intersection_ray_sphere".


#### Returns:

A tensor of shape `[2, A1, ..., An, 3]` containing the position of the
intersections, and a tensor of shape `[2, A1, ..., An, 3]` the associated
surface normals at that point. Both tensors contain NaNs when there is no
intersections. The first dimension of the returned tensor provides access to
the first and second intersections of the ray with the sphere.



#### Raises:


* <b>`ValueError`</b>: if the shape of `sphere_center`, `sphere_radius`, `ray` or
  `point_on_ray` is not supported.
* <b>`tf.errors.InvalidArgumentError`</b>: If `ray` is not normalized.