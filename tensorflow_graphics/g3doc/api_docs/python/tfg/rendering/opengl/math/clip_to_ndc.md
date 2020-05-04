<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.rendering.opengl.math.clip_to_ndc" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.rendering.opengl.math.clip_to_ndc

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/rendering/opengl/math.py">View source</a>



Transforms points from clip to normalized device coordinates (ndc).

```python
tfg.rendering.opengl.math.clip_to_ndc(
    point_clip_space, name=None
)
```



<!-- Placeholder for "Used in" -->


#### Note:

In the following, A1 to An are optional batch dimensions.



#### Args:


* <b>`point_clip_space`</b>: A tensor of shape `[A1, ..., An, 4]`, where the last
  dimension represents points in clip space.
* <b>`name`</b>: A name for this op. Defaults to 'clip_to_ndc'.


#### Raises:


* <b>`ValueError`</b>: If `point_clip_space` is not of size 4 in its last dimension.


#### Returns:

A tensor of shape `[A1, ..., An, 3]`, containing `point_clip_space` in
normalized device coordinates.
