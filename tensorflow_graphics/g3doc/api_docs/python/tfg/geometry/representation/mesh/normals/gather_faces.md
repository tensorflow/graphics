<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.geometry.representation.mesh.normals.gather_faces" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.geometry.representation.mesh.normals.gather_faces

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/geometry/representation/mesh/normals.py">View source</a>



Gather corresponding vertices for each face.

```python
tfg.geometry.representation.mesh.normals.gather_faces(
    vertices, indices, name=None
)
```



<!-- Placeholder for "Used in" -->


#### Note:

In the following, A1 to An are optional batch dimensions.



#### Args:


* <b>`vertices`</b>: A tensor of shape `[A1, ..., An, V, D]`, where `V` is the number
  of vertices and `D` the dimensionality of each vertex. The rank of this
  tensor should be at least 2.
* <b>`indices`</b>: A tensor of shape `[A1, ..., An, F, M]`, where `F` is the number of
  faces, and `M` is the number of vertices per face. The rank of this tensor
  should be at least 2.
* <b>`name`</b>: A name for this op. Defaults to "normals_gather_faces".


#### Returns:

A tensor of shape `[A1, ..., An, F, M, D]` containing the vertices of each
face.



#### Raises:


* <b>`ValueError`</b>: If the shape of `vertices` or `indices` is not supported.