<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.math.vector.dot" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.math.vector.dot

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/math/vector.py">View source</a>



Computes the dot product between two tensors along an axis.

```python
tfg.math.vector.dot(
    vector1, vector2, axis=-1, keepdims=True, name=None
)
```



<!-- Placeholder for "Used in" -->


#### Note:

In the following, A1 to An are optional batch dimensions, which should be
broadcast compatible.



#### Args:


* <b>`vector1`</b>: Tensor of rank R and shape `[A1, ..., Ai, ..., An]`, where the
  dimension i = axis represents a vector.
* <b>`vector2`</b>: Tensor of rank R and shape `[A1, ..., Ai, ..., An]`, where the
  dimension i = axis represents a vector.
* <b>`axis`</b>: The dimension along which to compute the dot product.
* <b>`keepdims`</b>: If True, retains reduced dimensions with length 1.
* <b>`name`</b>: A name for this op which defaults to "vector_dot".


#### Returns:

A tensor of shape `[A1, ..., Ai = 1, ..., An]`, where the dimension i = axis
represents the result of the dot product.
