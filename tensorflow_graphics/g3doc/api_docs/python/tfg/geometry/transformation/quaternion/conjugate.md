<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.geometry.transformation.quaternion.conjugate" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.geometry.transformation.quaternion.conjugate

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/geometry/transformation/quaternion.py">View source</a>



Computes the conjugate of a quaternion.

```python
tfg.geometry.transformation.quaternion.conjugate(
    quaternion, name=None
)
```



<!-- Placeholder for "Used in" -->


#### Note:

In the following, A1 to An are optional batch dimensions.



#### Args:


* <b>`quaternion`</b>: A tensor of shape `[A1, ..., An, 4]`, where the last dimension
  represents a normalized quaternion.
* <b>`name`</b>: A name for this op that defaults to "quaternion_conjugate".


#### Returns:

A tensor of shape `[A1, ..., An, 4]`, where the last dimension represents
a normalized quaternion.



#### Raises:


* <b>`ValueError`</b>: If the shape of `quaternion` is not supported.