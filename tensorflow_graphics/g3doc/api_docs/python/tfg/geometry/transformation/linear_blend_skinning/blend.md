<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.geometry.transformation.linear_blend_skinning.blend" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.geometry.transformation.linear_blend_skinning.blend

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/geometry/transformation/linear_blend_skinning.py">View source</a>



Transforms the points using Linear Blend Skinning.

```python
tfg.geometry.transformation.linear_blend_skinning.blend(
    points, skinning_weights, bone_rotations, bone_translations, name=None
)
```



<!-- Placeholder for "Used in" -->


#### Note:

In the following, A1 to An are optional batch dimensions, which must be
broadcast compatible and allow transforming full 3D shapes at once.
In the following, B1 to Bm are optional batch dimensions, which allow
transforming multiple poses at once.



#### Args:


* <b>`points`</b>: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
  represents a 3d point.
* <b>`skinning_weights`</b>: A tensor of shape `[A1, ..., An, W]`, where the last
  dimension represents the skinning weights of each bone.
* <b>`bone_rotations`</b>: A tensor of shape `[B1, ..., Bm, W, 3, 3]`, which represents
  the 3d rotations applied to each bone.
* <b>`bone_translations`</b>: A tensor of shape `[B1, ..., Bm, W, 3]`, which represents
  the 3d translation vectors applied to each bone.
* <b>`name`</b>: A name for this op that defaults to "linear_blend_skinning_blend".


#### Returns:

A tensor of shape `[B1, ..., Bm, A1, ..., An, 3]`, where the last dimension
represents a 3d point.



#### Raises:


* <b>`ValueError`</b>: If the shape of the input tensors are not supported.