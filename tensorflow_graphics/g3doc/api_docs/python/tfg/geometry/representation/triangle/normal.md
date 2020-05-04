<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.geometry.representation.triangle.normal" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.geometry.representation.triangle.normal

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/geometry/representation/triangle.py">View source</a>



Computes face normals (triangles).

```python
tfg.geometry.representation.triangle.normal(
    v0, v1, v2, clockwise=False, normalize=True, name=None
)
```



<!-- Placeholder for "Used in" -->


#### Note:

In the following, A1 to An are optional batch dimensions, which must be
broadcast compatible.



#### Args:


* <b>`v0`</b>: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
  represents the first vertex of a triangle.
* <b>`v1`</b>: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
  represents the second vertex of a triangle.
* <b>`v2`</b>: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
  represents the third vertex of a triangle.
* <b>`clockwise`</b>: Winding order to determine front-facing triangles.
* <b>`normalize`</b>: A `bool` indicating whether output normals should be normalized
  by the function.
* <b>`name`</b>: A name for this op. Defaults to "triangle_normal".


#### Returns:

A tensor of shape `[A1, ..., An, 3]`, where the last dimension represents
  a normalized vector.



#### Raises:


* <b>`ValueError`</b>: If the shape of `v0`, `v1`, or `v2` is not supported.