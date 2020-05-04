<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.rendering.opengl.math.look_at_right_handed" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.rendering.opengl.math.look_at_right_handed

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/rendering/opengl/math.py">View source</a>



Builds a right handed look at view matrix.

```python
tfg.rendering.opengl.math.look_at_right_handed(
    camera_position, look_at, up_vector, name=None
)
```



<!-- Placeholder for "Used in" -->


#### Note:

In the following, A1 to An are optional batch dimensions.



#### Args:


* <b>`camera_position`</b>: A tensor of shape `[A1, ..., An, 3]`, where the last
  dimension represents the 3D position of the camera.
* <b>`look_at`</b>: A tensor of shape `[A1, ..., An, 3]`, with the last dimension
  storing the position where the camera is looking at.
* <b>`up_vector`</b>: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
  defines the up vector of the camera.
* <b>`name`</b>: A name for this op. Defaults to 'look_at_right_handed'.


#### Raises:


* <b>`ValueError`</b>: if the all the inputs are not of the same shape, or if any input
of of an unsupported shape.


#### Returns:

A tensor of shape `[A1, ..., An, 4, 4]`, containing right handed look at
matrices.
