<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.geometry.representation.triangle.area" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.geometry.representation.triangle.area

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/geometry/representation/triangle.py">View source</a>



Computes triangle areas.

```python
tfg.geometry.representation.triangle.area(
    v0, v1, v2, name=None
)
```



<!-- Placeholder for "Used in" -->

  Note: Computed triangle area = 0.5 * | e1 x e2 | where e1 and e2 are edges
    of triangle. A degenerate triangle will return 0 area, whereas the normal
    for a degenerate triangle is not defined.


  In the following, A1 to An are optional batch dimensions, which must be
  broadcast compatible.

#### Args:


* <b>`v0`</b>: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
  represents the first vertex of a triangle.
* <b>`v1`</b>: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
  represents the second vertex of a triangle.
* <b>`v2`</b>: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
  represents the third vertex of a triangle.
* <b>`name`</b>: A name for this op. Defaults to "triangle_area".


#### Returns:

A tensor of shape `[A1, ..., An, 1]`, where the last dimension represents
  a normalized vector.
