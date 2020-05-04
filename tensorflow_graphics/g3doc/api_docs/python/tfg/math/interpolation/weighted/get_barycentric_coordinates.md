<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.math.interpolation.weighted.get_barycentric_coordinates" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.math.interpolation.weighted.get_barycentric_coordinates

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/math/interpolation/weighted.py">View source</a>



Computes the barycentric coordinates of pixels for 2D triangles.

```python
tfg.math.interpolation.weighted.get_barycentric_coordinates(
    triangle_vertices, pixels, name=None
)
```



<!-- Placeholder for "Used in" -->

Barycentric coordinates of a point `p` are represented as coefficients
$(w_1, w_2, w_3)$ corresponding to the masses placed at the vertices of a
reference triangle if `p` is the center of mass. Barycentric coordinates are
normalized so that $w_1 + w_2 + w_3 = 1$. These coordinates play an essential
role in computing the pixel attributes (e.g. depth, color, normals, and
texture coordinates) of a point lying on the surface of a triangle. The point
`p` is inside the triangle if all of its barycentric coordinates are positive.

#### Note:

In the following, A1 to An are optional batch dimensions.



#### Args:


* <b>`triangle_vertices`</b>: A tensor of shape `[A1, ..., An, 3, 2]`, where the last
  two dimensions represents the `x` and `y` coordinates for each vertex of a
  2D triangle.
* <b>`pixels`</b>: A tensor of shape `[A1, ..., An, N, 2]`, where `N` represents the
  number of pixels, and the last dimension represents the `x` and `y`
  coordinates of each pixel.
* <b>`name`</b>: A name for this op that defaults to
  "rasterizer_get_barycentric_coordinates".


#### Returns:


* <b>`barycentric_coordinates`</b>: A float tensor of shape `[A1, ..., An, N, 3]`,
  representing the barycentric coordinates.
* <b>`valid`</b>: A boolean tensor of shape `[A1, ..., An, N], which is `True` where
  pixels are inside the triangle, and `False` otherwise.