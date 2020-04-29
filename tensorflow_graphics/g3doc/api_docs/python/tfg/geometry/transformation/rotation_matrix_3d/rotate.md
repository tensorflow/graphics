<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.geometry.transformation.rotation_matrix_3d.rotate" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.geometry.transformation.rotation_matrix_3d.rotate

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/geometry/transformation/rotation_matrix_3d.py">View source</a>



Rotate a point using a rotation matrix 3d.

```python
tfg.geometry.transformation.rotation_matrix_3d.rotate(
    point, matrix, name=None
)
```



<!-- Placeholder for "Used in" -->


#### Note:

In the following, A1 to An are optional batch dimensions, which must be
broadcast compatible.



#### Args:


* <b>`point`</b>: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
  represents a 3d point.
* <b>`matrix`</b>: A tensor of shape `[A1, ..., An, 3,3]`, where the last dimension
  represents a 3d rotation matrix.
* <b>`name`</b>: A name for this op that defaults to "rotation_matrix_3d_rotate".


#### Returns:

A tensor of shape `[A1, ..., An, 3]`, where the last dimension represents
a 3d point.



#### Raises:


* <b>`ValueError`</b>: If the shape of `point` or `rotation_matrix_3d` is not
supported.