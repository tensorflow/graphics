<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.rendering.reflectance.lambertian.brdf" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.rendering.reflectance.lambertian.brdf

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/rendering/reflectance/lambertian.py">View source</a>



Evaluates the brdf of a Lambertian surface.

```python
tfg.rendering.reflectance.lambertian.brdf(
    direction_incoming_light, direction_outgoing_light, surface_normal, albedo,
    name=None
)
```



<!-- Placeholder for "Used in" -->


#### Note:

In the following, A1 to An are optional batch dimensions, which must be
broadcast compatible.



#### Note:

The gradient of this function is not smooth when the dot product of the
normal with any light is 0.0.



#### Args:


* <b>`direction_incoming_light`</b>: A tensor of shape `[A1, ..., An, 3]`, where the
  last dimension represents a normalized incoming light vector.
* <b>`direction_outgoing_light`</b>: A tensor of shape `[A1, ..., An, 3]`, where the
  last dimension represents a normalized outgoing light vector.
* <b>`surface_normal`</b>: A tensor of shape `[A1, ..., An, 3]`, where the last
  dimension represents a normalized surface normal.
* <b>`albedo`</b>: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
  represents albedo with values in [0,1].
* <b>`name`</b>: A name for this op. Defaults to "lambertian_brdf".


#### Returns:

A tensor of shape `[A1, ..., An, 3]`, where the last dimension represents
  the amount of reflected light in any outgoing direction.



#### Raises:


* <b>`ValueError`</b>: if the shape of `direction_incoming_light`,
`direction_outgoing_light`, `surface_normal`, `shininess` or `albedo` is not
supported.
* <b>`InvalidArgumentError`</b>: if at least one element of `albedo` is outside of
[0,1].