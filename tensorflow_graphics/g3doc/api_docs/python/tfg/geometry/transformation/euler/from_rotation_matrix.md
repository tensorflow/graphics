<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.geometry.transformation.euler.from_rotation_matrix" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.geometry.transformation.euler.from_rotation_matrix

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/geometry/transformation/euler.py">View source</a>



Converts rotation matrices to Euler angles.

```python
tfg.geometry.transformation.euler.from_rotation_matrix(
    rotation_matrix, name=None
)
```



<!-- Placeholder for "Used in" -->

The rotation matrices are assumed to have been constructed by rotation around
the $$x$$, then $$y$$, and finally the $$z$$ axis.

#### Note:

There is an infinite number of solutions to this problem. There are

Gimbal locks when abs(rotation_matrix(2,0)) == 1, which are not handled.

#### Note:

In the following, A1 to An are optional batch dimensions.



#### Args:


* <b>`rotation_matrix`</b>: A tensor of shape `[A1, ..., An, 3, 3]`, where the last two
  dimensions represent a rotation matrix.
* <b>`name`</b>: A name for this op that defaults to "euler_from_rotation_matrix".


#### Returns:

A tensor of shape `[A1, ..., An, 3]`, where the last dimension represents
the three Euler angles.



#### Raises:


* <b>`ValueError`</b>: If the shape of `rotation_matrix` is not supported.