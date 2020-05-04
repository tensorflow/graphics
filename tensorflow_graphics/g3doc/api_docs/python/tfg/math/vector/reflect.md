<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.math.vector.reflect" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.math.vector.reflect

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/math/vector.py">View source</a>



Computes the reflection direction for an incident vector.

```python
tfg.math.vector.reflect(
    vector, normal, axis=-1, name=None
)
```



<!-- Placeholder for "Used in" -->

For an incident vector \\(\mathbf{v}\\) and normal $$\mathbf{n}$$ this
function computes the reflected vector as
\\(\mathbf{r} = \mathbf{v} - 2(\mathbf{n}^T\mathbf{v})\mathbf{n}\\).

#### Note:

In the following, A1 to An are optional batch dimensions, which should be
broadcast compatible.



#### Args:


* <b>`vector`</b>: A tensor of shape `[A1, ..., Ai, ..., An]`, where the dimension i =
  axis represents a vector.
* <b>`normal`</b>: A tensor of shape `[A1, ..., Ai, ..., An]`, where the dimension i =
  axis represents a normal around which the vector needs to be reflected.
  The normal vector needs to be normalized.
* <b>`axis`</b>: The dimension along which to compute the reflection.
* <b>`name`</b>: A name for this op which defaults to "vector_reflect".


#### Returns:

A tensor of shape `[A1, ..., Ai, ..., An]`, where the dimension i = axis
represents a reflected vector.
