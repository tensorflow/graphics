<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.geometry.transformation.quaternion.from_axis_angle" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.geometry.transformation.quaternion.from_axis_angle

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/geometry/transformation/quaternion.py">View source</a>



Converts an axis-angle representation to a quaternion.

```python
tfg.geometry.transformation.quaternion.from_axis_angle(
    axis, angle, name=None
)
```



<!-- Placeholder for "Used in" -->


#### Note:

In the following, A1 to An are optional batch dimensions.



#### Args:


* <b>`axis`</b>: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
  represents a normalized axis.
* <b>`angle`</b>: A tensor of shape `[A1, ..., An, 1]`, where the last dimension
  represents an angle.
* <b>`name`</b>: A name for this op that defaults to "quaternion_from_axis_angle".


#### Returns:

A tensor of shape `[A1, ..., An, 4]`, where the last dimension represents
a normalized quaternion.



#### Raises:


* <b>`ValueError`</b>: If the shape of `axis` or `angle` is not supported.