<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.geometry.transformation.axis_angle.is_normalized" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.geometry.transformation.axis_angle.is_normalized

Determines if the axis-angle is normalized or not.

``` python
tfg.geometry.transformation.axis_angle.is_normalized(
    axis,
    angle,
    atol=0.001,
    name=None
)
```



Defined in [`geometry/transformation/axis_angle.py`](https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/geometry/transformation/axis_angle.py).

<!-- Placeholder for "Used in" -->

#### Note:

In the following, A1 to An are optional batch dimensions.


#### Args:

* <b>`axis`</b>: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
  represents a normalized axis.
* <b>`angle`</b>: A tensor of shape `[A1, ..., An, 1]` where the last dimension
  represents an angle.
* <b>`atol`</b>: The absolute tolerance parameter.
* <b>`name`</b>: A name for this op that defaults to "axis_angle_is_normalized".


#### Returns:

A tensor of shape `[A1, ..., An, 1]`, where False indicates that the axis is
not normalized.