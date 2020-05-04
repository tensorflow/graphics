<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.rendering.opengl.math.model_to_eye" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.rendering.opengl.math.model_to_eye

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/rendering/opengl/math.py">View source</a>



Transforms points from model to eye coordinates.

```python
tfg.rendering.opengl.math.model_to_eye(
    point_model_space, camera_position, look_at, up_vector, name=None
)
```



<!-- Placeholder for "Used in" -->


#### Note:

In the following, A1 to An are optional batch dimensions which must be
broadcast compatible.



#### Args:


* <b>`point_model_space`</b>: A tensor of shape `[A1, ..., An, 3]`, where the last
  dimension represents the 3D points in model space.
* <b>`camera_position`</b>: A tensor of shape `[A1, ..., An, 3]`, where the last
  dimension represents the 3D position of the camera.
* <b>`look_at`</b>: A tensor of shape `[A1, ..., An, 3]`, with the last dimension
  storing the position where the camera is looking at.
* <b>`up_vector`</b>: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
  defines the up vector of the camera.
* <b>`name`</b>: A name for this op. Defaults to 'model_to_eye'.


#### Raises:


* <b>`ValueError`</b>: if the all the inputs are not of the same shape, or if any input
of of an unsupported shape.


#### Returns:

A tensor of shape `[A1, ..., An, 3]`, containing `point_model_space` in eye
coordinates.
