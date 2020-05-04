<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.rendering.opengl.math.eye_to_clip" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.rendering.opengl.math.eye_to_clip

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/rendering/opengl/math.py">View source</a>



Transforms points from eye to clip space.

```python
tfg.rendering.opengl.math.eye_to_clip(
    point_eye_space, vertical_field_of_view, aspect_ratio, near, far, name=None
)
```



<!-- Placeholder for "Used in" -->


#### Note:

In the following, A1 to An are optional batch dimensions which must be
broadcast compatible.



#### Args:


* <b>`point_eye_space`</b>: A tensor of shape `[A1, ..., An, 3]`, where the last
  dimension represents the 3D points in eye coordinates.
* <b>`vertical_field_of_view`</b>: A tensor of shape `[A1, ..., An, 1]`, where the last
  dimension represents the vertical field of view of the frustum. Note that
  values for `vertical_field_of_view` must be in the range ]0,pi[.
* <b>`aspect_ratio`</b>: A tensor of shape `[A1, ..., An, 1]`, where the last dimension
  stores the width over height ratio of the frustum. Note that values for
  `aspect_ratio` must be non-negative.
* <b>`near`</b>: A tensor of shape `[A1, ..., An, 1]`, where the last dimension
  captures the distance between the viewer and the near clipping plane. Note
  that values for `near` must be non-negative.
* <b>`far`</b>: A tensor of shape `[A1, ..., An, 1]`, where the last dimension captures
  the distance between the viewer and the far clipping plane. Note that
  values for `far` must be non-negative.
* <b>`name`</b>: A name for this op. Defaults to 'eye_to_clip'.


#### Raises:


* <b>`ValueError`</b>: If any input is of an unsupported shape.


#### Returns:

A tensor of shape `[A1, ..., An, 4]`, containing `point_eye_space` in
homogeneous clip coordinates.
