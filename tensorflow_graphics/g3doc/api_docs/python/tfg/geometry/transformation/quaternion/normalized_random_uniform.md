<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.geometry.transformation.quaternion.normalized_random_uniform" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.geometry.transformation.quaternion.normalized_random_uniform

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/geometry/transformation/quaternion.py">View source</a>



Random normalized quaternion following a uniform distribution law on SO(3).

```python
tfg.geometry.transformation.quaternion.normalized_random_uniform(
    quaternion_shape, name=None
)
```



<!-- Placeholder for "Used in" -->


#### Args:


* <b>`quaternion_shape`</b>: A list representing the shape of the output tensor.
* <b>`name`</b>: A name for this op that defaults to
  "quaternion_normalized_random_uniform".


#### Returns:

A tensor of shape `[quaternion_shape[0],...,quaternion_shape[-1], 4]`
representing random normalized quaternions.
