<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.geometry.transformation.rotation_matrix_3d.assert_rotation_matrix_normalized" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.geometry.transformation.rotation_matrix_3d.assert_rotation_matrix_normalized

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/geometry/transformation/rotation_matrix_3d.py">View source</a>



Checks whether a matrix is a rotation matrix.

```python
tfg.geometry.transformation.rotation_matrix_3d.assert_rotation_matrix_normalized(
    matrix, eps=0.001, name=None
)
```



<!-- Placeholder for "Used in" -->


#### Note:

In the following, A1 to An are optional batch dimensions.



#### Args:


* <b>`matrix`</b>: A tensor of shape `[A1, ..., An, 3, 3]`, where the last two
  dimensions represent a 3d rotation matrix.
* <b>`eps`</b>: The absolute tolerance parameter.
* <b>`name`</b>: A name for this op that defaults to
  'assert_rotation_matrix_normalized'.


#### Returns:

The input matrix, with dependence on the assertion operator in the graph.



#### Raises:


* <b>`tf.errors.InvalidArgumentError`</b>: If rotation_matrix_3d is not normalized.