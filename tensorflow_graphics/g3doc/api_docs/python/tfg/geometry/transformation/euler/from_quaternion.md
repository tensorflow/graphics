<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.geometry.transformation.euler.from_quaternion" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.geometry.transformation.euler.from_quaternion

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/geometry/transformation/euler.py">View source</a>



Converts quaternions to Euler angles.

```python
tfg.geometry.transformation.euler.from_quaternion(
    quaternions, name=None
)
```



<!-- Placeholder for "Used in" -->


#### Args:


* <b>`quaternions`</b>: A tensor of shape `[A1, ..., An, 4]`, where the last dimension
  represents a normalized quaternion.
* <b>`name`</b>: A name for this op that defaults to "euler_from_quaternion".


#### Returns:

A tensor of shape `[A1, ..., An, 3]`, where the last dimension represents
the three Euler angles.
