<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.geometry.transformation.rotation_matrix_3d.inverse" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.geometry.transformation.rotation_matrix_3d.inverse

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/geometry/transformation/rotation_matrix_3d.py">View source</a>



Computes the inverse of a 3D rotation matrix.

```python
tfg.geometry.transformation.rotation_matrix_3d.inverse(
    matrix, name=None
)
```



<!-- Placeholder for "Used in" -->


#### Note:

In the following, A1 to An are optional batch dimensions.



#### Args:


* <b>`matrix`</b>: A tensor of shape `[A1, ..., An, 3, 3]`, where the last two
  dimensions represent a 3d rotation matrix.
* <b>`name`</b>: A name for this op that defaults to "rotation_matrix_3d_inverse".


#### Returns:

A tensor of shape `[A1, ..., An, 3, 3]`, where the last two dimensions
represent a 3d rotation matrix.



#### Raises:


* <b>`ValueError`</b>: If the shape of `matrix` is not supported.