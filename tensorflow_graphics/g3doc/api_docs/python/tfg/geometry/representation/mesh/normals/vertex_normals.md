<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.geometry.representation.mesh.normals.vertex_normals" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.geometry.representation.mesh.normals.vertex_normals

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/geometry/representation/mesh/normals.py">View source</a>



Computes vertex normals from a mesh.

```python
tfg.geometry.representation.mesh.normals.vertex_normals(
    vertices, indices, clockwise=True, name=None
)
```



<!-- Placeholder for "Used in" -->

This function computes vertex normals as the weighted sum of the adjacent
face normals, where the weights correspond to the area of each face. This
function supports planar convex polygon faces. For non-triangular meshes,
this function converts them into triangular meshes to calculate vertex
normals.

#### Note:

In the following, A1 to An are optional batch dimensions.



#### Args:


* <b>`vertices`</b>: A tensor of shape `[A1, ..., An, V, 3]`, where V is the number of
  vertices.
* <b>`indices`</b>: A tensor of shape `[A1, ..., An, F, M]`, where F is the number of
  faces and M is the number of vertices per face.
* <b>`clockwise`</b>: Winding order to determine front-facing faces. The order of
  vertices should be either clockwise or counterclockwise.
* <b>`name`</b>: A name for this op. Defaults to "normals_vertex_normals".


#### Returns:

A tensor of shape `[A1, ..., An, V, 3]` containing vertex normals. If
vertices and indices have different batch dimensions, this function
broadcasts them into the same batch dimensions and the output batch
dimensions are the broadcasted.



#### Raises:


* <b>`ValueError`</b>: If the shape of `vertices`, `indices` is not supported.