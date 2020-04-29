<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.geometry.transformation.quaternion.rotate" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.geometry.transformation.quaternion.rotate

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/geometry/transformation/quaternion.py">View source</a>



Rotates a point using a quaternion.

```python
tfg.geometry.transformation.quaternion.rotate(
    point, quaternion, name=None
)
```



<!-- Placeholder for "Used in" -->


#### Note:

In the following, A1 to An are optional batch dimensions.



#### Args:


* <b>`point`</b>: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
  represents a 3d point.
* <b>`quaternion`</b>: A tensor of shape `[A1, ..., An, 4]`, where the last dimension
  represents a normalized quaternion.
* <b>`name`</b>: A name for this op that defaults to "quaternion_rotate".


#### Returns:

A tensor of shape `[A1, ..., An, 3]`, where the last dimension represents a
3d point.



#### Raises:


* <b>`ValueError`</b>: If the shape of `point` or `quaternion` is not supported.