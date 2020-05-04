<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.rendering.opengl.math.perspective_right_handed" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.rendering.opengl.math.perspective_right_handed

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/rendering/opengl/math.py">View source</a>



Generates the matrix for a right handed perspective projection.

```python
tfg.rendering.opengl.math.perspective_right_handed(
    vertical_field_of_view, aspect_ratio, near, far, name=None
)
```



<!-- Placeholder for "Used in" -->


#### Note:

In the following, A1 to An are optional batch dimensions.



#### Args:


* <b>`vertical_field_of_view`</b>: A tensor of shape `[A1, ..., An, 1]`, where the last
  dimension represents the vertical field of view of the frustum expressed
  in radians. Note that values for `vertical_field_of_view` must be in the
  range (0,pi).
* <b>`aspect_ratio`</b>: A tensor of shape `[A1, ..., An, 1]`, where the last dimension
  stores the width over height ratio of the frustum. Note that values for
  `aspect_ratio` must be non-negative.
* <b>`near`</b>:  A tensor of shape `[A1, ..., An, 1]`, where the last dimension
  captures the distance between the viewer and the near clipping plane. Note
  that values for `near` must be non-negative.
* <b>`far`</b>:  A tensor of shape `[A1, ..., An, 1]`, where the last dimension
  captures the distance between the viewer and the far clipping plane. Note
  that values for `far` must be greater than those of `near`.
* <b>`name`</b>: A name for this op. Defaults to 'perspective_rh'.


#### Raises:


* <b>`InvalidArgumentError`</b>: if any input contains data not in the specified range
  of valid values.
* <b>`ValueError`</b>: if the all the inputs are not of the same shape.


#### Returns:

A tensor of shape `[A1, ..., An, 4, 4]`, containing matrices of right
handed perspective-view frustum.
