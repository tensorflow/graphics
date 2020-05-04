<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.geometry.transformation.quaternion.multiply" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.geometry.transformation.quaternion.multiply

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/geometry/transformation/quaternion.py">View source</a>



Multiplies two quaternions.

```python
tfg.geometry.transformation.quaternion.multiply(
    quaternion1, quaternion2, name=None
)
```



<!-- Placeholder for "Used in" -->


#### Note:

In the following, A1 to An are optional batch dimensions.



#### Args:


* <b>`quaternion1`</b>:  A tensor of shape `[A1, ..., An, 4]`, where the last dimension
  represents a quaternion.
* <b>`quaternion2`</b>:  A tensor of shape `[A1, ..., An, 4]`, where the last dimension
  represents a quaternion.
* <b>`name`</b>: A name for this op that defaults to "quaternion_multiply".


#### Returns:

A tensor of shape `[A1, ..., An, 4]` representing quaternions.



#### Raises:


* <b>`ValueError`</b>: If the shape of `quaternion1` or `quaternion2` is not supported.