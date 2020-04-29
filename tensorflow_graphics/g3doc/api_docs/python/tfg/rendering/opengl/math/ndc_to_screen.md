<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.rendering.opengl.math.ndc_to_screen" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.rendering.opengl.math.ndc_to_screen

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/rendering/opengl/math.py">View source</a>



Transforms points from normalized device coordinates to screen coordinates.

```python
tfg.rendering.opengl.math.ndc_to_screen(
    point_ndc_space, lower_left_corner, screen_dimensions, near, far, name=None
)
```



<!-- Placeholder for "Used in" -->


#### Note:

In the following, A1 to An are optional batch dimensions which must be
broadcast compatible between `point_ndc_space` and the other variables.



#### Args:


* <b>`point_ndc_space`</b>: A tensor of shape `[A1, ..., An, 3]`, where the last
  dimension represents points in normalized device coordinates.
* <b>`lower_left_corner`</b>: A tensor of shape `[A1, ..., An, 2]`, where the last
  dimension captures the position (in pixels) of the lower left corner of
  the screen.
* <b>`screen_dimensions`</b>: A tensor of shape `[A1, ..., An, 2]`, where the last
  dimension is expressed in pixels and captures the width and the height (in
  pixels) of the screen.
* <b>`near`</b>:  A tensor of shape `[A1, ..., An, 1]`, where the last dimension
  captures the distance between the viewer and the near clipping plane. Note
  that values for `near` must be non-negative.
* <b>`far`</b>:  A tensor of shape `[A1, ..., An, 1]`, where the last dimension
  captures the distance between the viewer and the far clipping plane. Note
  that values for `far` must be greater than those of `near`.
* <b>`name`</b>: A name for this op. Defaults to 'ndc_to_screen'.


#### Raises:


* <b>`InvalidArgumentError`</b>: if any input contains data not in the specified range
  of valid values.
* <b>`ValueError`</b>: If any input is of an unsupported shape.


#### Returns:

A tensor of shape `[A1, ..., An, 3]`, containing `point_ndc_space` in
screen coordinates.
