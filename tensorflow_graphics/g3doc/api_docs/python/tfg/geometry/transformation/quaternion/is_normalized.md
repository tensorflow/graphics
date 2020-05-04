<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.geometry.transformation.quaternion.is_normalized" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.geometry.transformation.quaternion.is_normalized

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/geometry/transformation/quaternion.py">View source</a>



Determines if quaternion is normalized quaternion or not.

```python
tfg.geometry.transformation.quaternion.is_normalized(
    quaternion, atol=0.001, name=None
)
```



<!-- Placeholder for "Used in" -->


#### Note:

In the following, A1 to An are optional batch dimensions.



#### Args:


* <b>`quaternion`</b>:  A tensor of shape `[A1, ..., An, 4]`, where the last dimension
  represents a quaternion.
* <b>`atol`</b>: The absolute tolerance parameter.
* <b>`name`</b>: A name for this op that defaults to "quaternion_is_normalized".


#### Returns:

A tensor of type `bool` and shape `[A1, ..., An, 1]`, where False indicates
that the quaternion is not normalized.



#### Raises:


* <b>`ValueError`</b>: If the shape of `quaternion` is not supported.