<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.geometry.representation.mesh.normals.face_normals" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.geometry.representation.mesh.normals.face_normals

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/geometry/representation/mesh/normals.py">View source</a>



Computes face normals for meshes.

```python
tfg.geometry.representation.mesh.normals.face_normals(
    faces, clockwise=True, normalize=True, name=None
)
```



<!-- Placeholder for "Used in" -->

This function supports planar convex polygon faces. Note that for
non-triangular faces, this function uses the first 3 vertices of each
face to calculate the face normal.

#### Note:

In the following, A1 to An are optional batch dimensions.



#### Args:


* <b>`faces`</b>: A tensor of shape `[A1, ..., An, M, 3]`, which stores vertices
  positions of each face, where M is the number of vertices of each face.
  The rank of this tensor should be at least 2.
* <b>`clockwise`</b>: Winding order to determine front-facing faces. The order of
  vertices should be either clockwise or counterclockwise.
* <b>`normalize`</b>: A `bool` defining whether output normals are normalized.
* <b>`name`</b>: A name for this op. Defaults to "normals_face_normals".


#### Returns:

A tensor of shape `[A1, ..., An, 3]` containing the face normals.



#### Raises:


* <b>`ValueError`</b>: If the shape of `vertices`, `faces` is not supported.