<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.rendering.opengl.math.perspective_correct_interpolation" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.rendering.opengl.math.perspective_correct_interpolation

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/rendering/opengl/math.py">View source</a>



Returns perspective corrected interpolation of attributes over triangles.

```python
tfg.rendering.opengl.math.perspective_correct_interpolation(
    triangle_vertices_model_space, attribute, pixel_position, camera_position,
    look_at, up_vector, vertical_field_of_view, screen_dimensions, near, far,
    lower_left_corner, name=None
)
```



<!-- Placeholder for "Used in" -->


#### Note:

In the following, A1 to An are optional batch dimensions.



#### Args:


* <b>`triangle_vertices_model_space`</b>: A tensor of shape `[A1, ..., An, 3, 3]`,
  where the last dimension represents the vertices of a triangle in model
  space.
* <b>`attribute`</b>: A tensor of shape `[A1, ..., An, 3, B]`, where the last dimension
  stores a per-vertex `B`-dimensional attribute.
* <b>`pixel_position`</b>: A tensor of shape `[A1, ..., An, 2]`, where the last
  dimension stores the position (in pixels) where the interpolation is
  requested.
* <b>`camera_position`</b>: A tensor of shape `[A1, ..., An, 3]`, where the last
  dimension represents the 3D position of the camera.
* <b>`look_at`</b>: A tensor of shape `[A1, ..., An, 3, 3]`, with the last dimension
  storing the position where the camera is looking at.
* <b>`up_vector`</b>: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
  defines the up vector of the camera.
* <b>`vertical_field_of_view`</b>: A tensor of shape `[A1, ..., An, 1]`, where the last
  dimension represents the vertical field of view of the frustum. Note that
  values for `vertical_field_of_view` must be in the range ]0,pi[.
* <b>`screen_dimensions`</b>: A tensor of shape `[A1, ..., An, 2]`, where the last
  dimension is expressed in pixels and captures the width and the height (in
  pixels) of the screen.
* <b>`near`</b>:  A tensor of shape `[A1, ..., An, 1]`, where the last dimension
  captures the distance between the viewer and the near clipping plane. Note
  that values for `near` must be non-negative.
* <b>`far`</b>:  A tensor of shape `[A1, ..., An, 1]`, where the last dimension
  captures the distance between the viewer and the far clipping plane. Note
  that values for `far` must be greater than those of `near`.
* <b>`lower_left_corner`</b>: A tensor of shape `[A1, ..., An, 2]`, where the last
  dimension captures the position (in pixels) of the lower left corner of
  the screen.
* <b>`name`</b>: A name for this op. Defaults to 'perspective_correct_interpolation'.


#### Raises:


* <b>`InvalidArgumentError`</b>: if any input contains data not in the specified range
  of valid values.
* <b>`ValueError`</b>: If any input is of an unsupported shape.


#### Returns:

A tensor of shape `[A1, ..., An, B]`, containing interpolated attributes.
