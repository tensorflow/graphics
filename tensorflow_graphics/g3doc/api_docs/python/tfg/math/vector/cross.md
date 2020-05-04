<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.math.vector.cross" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.math.vector.cross

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/math/vector.py">View source</a>



Computes the cross product between two tensors along an axis.

```python
tfg.math.vector.cross(
    vector1, vector2, axis=-1, name=None
)
```



<!-- Placeholder for "Used in" -->


#### Note:

In the following, A1 to An are optional batch dimensions, which should be
broadcast compatible.



#### Args:


* <b>`vector1`</b>: A tensor of shape `[A1, ..., Ai = 3, ..., An]`, where the dimension
  i = axis represents a 3d vector.
* <b>`vector2`</b>: A tensor of shape `[A1, ..., Ai = 3, ..., An]`, where the dimension
  i = axis represents a 3d vector.
* <b>`axis`</b>: The dimension along which to compute the cross product.
* <b>`name`</b>: A name for this op which defaults to "vector_cross".


#### Returns:

A tensor of shape `[A1, ..., Ai = 3, ..., An]`, where the dimension i = axis
represents the result of the cross product.
