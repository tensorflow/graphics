<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.geometry.representation.mesh.utils.get_degree_based_edge_weights" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.geometry.representation.mesh.utils.get_degree_based_edge_weights

Computes vertex degree based weights for edges of a mesh.

``` python
tfg.geometry.representation.mesh.utils.get_degree_based_edge_weights(
    edges,
    dtype=np.float32
)
```



Defined in [`geometry/representation/mesh/utils.py`](https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/geometry/representation/mesh/utils.py).

<!-- Placeholder for "Used in" -->

The degree (valence) of a vertex is number of edges incident on the vertex.
The weight for an edge $w_{ij}$ connecting vertex $v_i$ and vertex $v_j$
is defined as,
$$
w_{ij} = 1.0 / degree(v_i)
\sum_{j} w_{ij} = 1
$$

#### Args:

* <b>`edges`</b>: A numpy.ndarray of shape [E, 2],
  where E is the number of directed edges in the mesh.
* <b>`dtype`</b>: A numpy float data type. The output weights are of data type dtype.


#### Returns:

* <b>`weights`</b>: A dtype numpy.ndarray of shape [E,] denoting edge weights.


#### Raises:

* <b>`ValueError`</b>: If `edges` is not a numpy.ndarray or if its shape is not
  supported, or dtype is not a float type.