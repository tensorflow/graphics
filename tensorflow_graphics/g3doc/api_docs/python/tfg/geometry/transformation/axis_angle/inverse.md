<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.geometry.transformation.axis_angle.inverse" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.geometry.transformation.axis_angle.inverse

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/geometry/transformation/axis_angle.py">View source</a>



Computes the axis-angle that is the inverse of the input axis-angle.

```python
tfg.geometry.transformation.axis_angle.inverse(
    axis, angle, name=None
)
```



<!-- Placeholder for "Used in" -->


#### Note:

In the following, A1 to An are optional batch dimensions.



#### Args:


* <b>`axis`</b>: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
  represents a normalized axis.
* <b>`angle`</b>: A tensor of shape `[A1, ..., An, 1]` where the last dimension
  represents an angle.
* <b>`name`</b>: A name for this op that defaults to "axis_angle_inverse".


#### Returns:

A tuple of two tensors, respectively of shape `[A1, ..., An, 3]` and
`[A1, ..., An, 1]`, where the first tensor represents the axis, and the
second represents the angle. The resulting axis is a normalized vector.



#### Raises:


* <b>`ValueError`</b>: If the shape of `axis` or `angle` is not supported.