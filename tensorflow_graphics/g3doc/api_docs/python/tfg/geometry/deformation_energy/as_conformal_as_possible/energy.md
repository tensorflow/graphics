<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.geometry.deformation_energy.as_conformal_as_possible.energy" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.geometry.deformation_energy.as_conformal_as_possible.energy

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/geometry/deformation_energy/as_conformal_as_possible.py">View source</a>



Estimates an As Conformal As Possible (ACAP) fitting energy.

```python
tfg.geometry.deformation_energy.as_conformal_as_possible.energy(
    vertices_rest_pose, vertices_deformed_pose, quaternions, edges,
    vertex_weight=None, edge_weight=None, conformal_energy=True,
    aggregate_loss=True, name=None
)
```



<!-- Placeholder for "Used in" -->

For a given mesh in rest pose, this function evaluates a variant of the ACAP
[1] fitting energy for a batch of deformed meshes. The vertex weights and edge
weights are defined on the rest pose.

The method implemented here is similar to [2], but with an added free variable
  capturing a scale factor per vertex.

[1]: Yusuke Yoshiyasu, Wan-Chun Ma, Eiichi Yoshida, and Fumio Kanehiro.
"As-Conformal-As-Possible Surface Registration." Computer Graphics Forum. Vol.
33. No. 5. 2014.</br>
[2]: Olga Sorkine, and Marc Alexa.
"As-rigid-as-possible surface modeling". Symposium on Geometry Processing.
Vol. 4. 2007.

#### Note:

In the description of the arguments, V corresponds to
  the number of vertices in the mesh, and E to the number of edges in this
  mesh.



#### Note:

In the following, A1 to An are optional batch dimensions.



#### Args:


* <b>`vertices_rest_pose`</b>: A tensor of shape `[V, 3]` containing the position of
  all the vertices of the mesh in rest pose.
* <b>`vertices_deformed_pose`</b>: A tensor of shape `[A1, ..., An, V, 3]` containing
  the position of all the vertices of the mesh in deformed pose.
* <b>`quaternions`</b>: A tensor of shape `[A1, ..., An, V, 4]` defining a rigid
  transformation to apply to each vertex of the rest pose. See Section 2
  from [1] for further details.
* <b>`edges`</b>: A tensor of shape `[E, 2]` defining indices of vertices that are
  connected by an edge.
* <b>`vertex_weight`</b>: An optional tensor of shape `[V]` defining the weight
  associated with each vertex. Defaults to a tensor of ones.
* <b>`edge_weight`</b>: A tensor of shape `[E]` defining the weight of edges. Common
  choices for these weights include uniform weighting, and cotangent
  weights. Defaults to a tensor of ones.
* <b>`conformal_energy`</b>: A `bool` indicating whether each vertex is associated with
  a scale factor or not. If this parameter is True, scaling information must
  be encoded in the norm of `quaternions`. If this parameter is False, this
  function implements the energy described in [2].
* <b>`aggregate_loss`</b>: A `bool` defining whether the returned loss should be an
  aggregate measure. When True, the mean squared error is returned. When
  False, returns two losses for every edge of the mesh.
* <b>`name`</b>: A name for this op. Defaults to "as_conformal_as_possible_energy".


#### Returns:

When aggregate_loss is `True`, returns a tensor of shape `[A1, ..., An]`
containing the ACAP energies. When aggregate_loss is `False`, returns a
tensor of shape `[A1, ..., An, 2*E]` containing each term of the summation
described in the equation 7 of [2].



#### Raises:


* <b>`ValueError`</b>: if the shape of `vertices_rest_pose`, `vertices_deformed_pose`,
`quaternions`, `edges`, `vertex_weight`, or `edge_weight` is not supported.