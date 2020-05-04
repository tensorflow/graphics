<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.rendering.reflectance.blinn_phong.brdf" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.rendering.reflectance.blinn_phong.brdf

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/rendering/reflectance/blinn_phong.py">View source</a>



Evaluates the specular brdf of the Blinn-Phong model.

```python
tfg.rendering.reflectance.blinn_phong.brdf(
    direction_incoming_light, direction_outgoing_light, surface_normal, shininess,
    albedo, brdf_normalization=True, name=None
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
* <b>`shininess`</b>: A tensor of shape `[A1, ..., An, 1]`, where the last dimension
  represents a non-negative shininess coefficient.
* <b>`albedo`</b>: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
  represents albedo with values in [0,1].
* <b>`brdf_normalization`</b>: A `bool` indicating whether normalization should be
  applied to enforce the energy conservation property of BRDFs. Note that
  `brdf_normalization` must be set to False in order to use the original
  Blinn-Phong specular model.
* <b>`name`</b>: A name for this op. Defaults to "blinn_phong_brdf".


#### Returns:

A tensor of shape `[A1, ..., An, 3]`, where the last dimension represents
  the amount of light reflected in the outgoing light direction.



#### Raises:


* <b>`ValueError`</b>: if the shape of `direction_incoming_light`,
`direction_outgoing_light`, `surface_normal`, `shininess` or `albedo` is not
supported.
* <b>`InvalidArgumentError`</b>: if not all of shininess values are non-negative, or if
at least one element of `albedo` is outside of [0,1].