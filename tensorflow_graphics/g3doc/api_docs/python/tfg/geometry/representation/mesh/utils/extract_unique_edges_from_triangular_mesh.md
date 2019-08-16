<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.geometry.representation.mesh.utils.extract_unique_edges_from_triangular_mesh" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.geometry.representation.mesh.utils.extract_unique_edges_from_triangular_mesh

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/geometry/representation/mesh/utils.py">View
source</a>

Extracts all the unique edges using the faces of a mesh.

``` python
tfg.geometry.representation.mesh.utils.extract_unique_edges_from_triangular_mesh(
    faces,
    directed_edges=False
)
```



<!-- Placeholder for "Used in" -->

#### Args:

* <b>`faces`</b>: A numpy.ndarray of shape [T, 3], where T is the number of triangular
  faces in the mesh. Each entry in this array describes the index of a
  vertex in the mesh.
* <b>`directed_edges`</b>: A boolean flag, whether to treat an edge as directed or
  undirected.  If (i, j) is an edge in the mesh and directed_edges is True,
  then both (i, j) and (j, i) are returned in the list of edges.
  If (i, j) is an edge in the mesh and directed_edges is False,
  then one of (i, j) or (j, i) is returned.



#### Returns:

A numpy.ndarray of shape [E, 2], where E is the number of edges in
the mesh.


For eg: given faces = [[0, 1, 2], [0, 1, 3]], then
  for directed_edges = False, one valid output is
    [[0, 1], [0, 2], [0, 3], [1, 2], [3, 1]]
  for directed_edges = True, one valid output is
    [[0, 1], [0, 2], [0, 3], [1, 0], [1, 2], [1, 3],
     [2, 0], [2, 1], [3, 0], [3, 1]]

#### Raises:

* <b>`ValueError`</b>: If `faces` is not a numpy.ndarray or if its shape is not
  supported.