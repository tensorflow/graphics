# Copyright 2020 The TensorFlow Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module implements TensorFlow As Rigid As Possible utility functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_graphics.geometry.transformation import quaternion
from tensorflow_graphics.math import vector
from tensorflow_graphics.util import export_api
from tensorflow_graphics.util import shape


def energy(vertices_rest_pose,
           vertices_deformed_pose,
           quaternions,
           edges,
           vertex_weight=None,
           edge_weight=None,
           conformal_energy=True,
           aggregate_loss=True,
           name=None):
  """Estimates an As Conformal As Possible (ACAP) fitting energy.

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

  Note:
    In the description of the arguments, V corresponds to
      the number of vertices in the mesh, and E to the number of edges in this
      mesh.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    vertices_rest_pose: A tensor of shape `[V, 3]` containing the position of
      all the vertices of the mesh in rest pose.
    vertices_deformed_pose: A tensor of shape `[A1, ..., An, V, 3]` containing
      the position of all the vertices of the mesh in deformed pose.
    quaternions: A tensor of shape `[A1, ..., An, V, 4]` defining a rigid
      transformation to apply to each vertex of the rest pose. See Section 2
      from [1] for further details.
    edges: A tensor of shape `[E, 2]` defining indices of vertices that are
      connected by an edge.
    vertex_weight: An optional tensor of shape `[V]` defining the weight
      associated with each vertex. Defaults to a tensor of ones.
    edge_weight: A tensor of shape `[E]` defining the weight of edges. Common
      choices for these weights include uniform weighting, and cotangent
      weights. Defaults to a tensor of ones.
    conformal_energy: A `bool` indicating whether each vertex is associated with
      a scale factor or not. If this parameter is True, scaling information must
      be encoded in the norm of `quaternions`. If this parameter is False, this
      function implements the energy described in [2].
    aggregate_loss: A `bool` defining whether the returned loss should be an
      aggregate measure. When True, the mean squared error is returned. When
      False, returns two losses for every edge of the mesh.
    name: A name for this op. Defaults to "as_conformal_as_possible_energy".

  Returns:
    When aggregate_loss is `True`, returns a tensor of shape `[A1, ..., An]`
    containing the ACAP energies. When aggregate_loss is `False`, returns a
    tensor of shape `[A1, ..., An, 2*E]` containing each term of the summation
    described in the equation 7 of [2].

  Raises:
    ValueError: if the shape of `vertices_rest_pose`, `vertices_deformed_pose`,
    `quaternions`, `edges`, `vertex_weight`, or `edge_weight` is not supported.
  """
  with tf.compat.v1.name_scope(name, "as_conformal_as_possible_energy", [
      vertices_rest_pose, vertices_deformed_pose, quaternions, edges,
      conformal_energy, vertex_weight, edge_weight
  ]):
    vertices_rest_pose = tf.convert_to_tensor(value=vertices_rest_pose)
    vertices_deformed_pose = tf.convert_to_tensor(value=vertices_deformed_pose)
    quaternions = tf.convert_to_tensor(value=quaternions)
    edges = tf.convert_to_tensor(value=edges)
    if vertex_weight is not None:
      vertex_weight = tf.convert_to_tensor(value=vertex_weight)
    if edge_weight is not None:
      edge_weight = tf.convert_to_tensor(value=edge_weight)

    shape.check_static(
        tensor=vertices_rest_pose,
        tensor_name="vertices_rest_pose",
        has_rank=2,
        has_dim_equals=(-1, 3))
    shape.check_static(
        tensor=vertices_deformed_pose,
        tensor_name="vertices_deformed_pose",
        has_rank_greater_than=1,
        has_dim_equals=(-1, 3))
    shape.check_static(
        tensor=quaternions,
        tensor_name="quaternions",
        has_rank_greater_than=1,
        has_dim_equals=(-1, 4))
    shape.compare_batch_dimensions(
        tensors=(vertices_deformed_pose, quaternions),
        last_axes=(-3, -3),
        broadcast_compatible=False)
    shape.check_static(
        tensor=edges, tensor_name="edges", has_rank=2, has_dim_equals=(-1, 2))
    tensors_with_vertices = [vertices_rest_pose,
                             vertices_deformed_pose,
                             quaternions]
    names_with_vertices = ["vertices_rest_pose",
                           "vertices_deformed_pose",
                           "quaternions"]
    axes_with_vertices = [-2, -2, -2]
    if vertex_weight is not None:
      shape.check_static(
          tensor=vertex_weight, tensor_name="vertex_weight", has_rank=1)
      tensors_with_vertices.append(vertex_weight)
      names_with_vertices.append("vertex_weight")
      axes_with_vertices.append(0)
    shape.compare_dimensions(
        tensors=tensors_with_vertices,
        axes=axes_with_vertices,
        tensor_names=names_with_vertices)
    if edge_weight is not None:
      shape.check_static(
          tensor=edge_weight, tensor_name="edge_weight", has_rank=1)
      shape.compare_dimensions(
          tensors=(edges, edge_weight),
          axes=(0, 0),
          tensor_names=("edges", "edge_weight"))

    if not conformal_energy:
      quaternions = quaternion.normalize(quaternions)
    # Extracts the indices of vertices.
    indices_i, indices_j = tf.unstack(edges, axis=-1)
    # Extracts the vertices we need per term.
    vertices_i_rest = tf.gather(vertices_rest_pose, indices_i, axis=-2)
    vertices_j_rest = tf.gather(vertices_rest_pose, indices_j, axis=-2)
    vertices_i_deformed = tf.gather(vertices_deformed_pose, indices_i, axis=-2)
    vertices_j_deformed = tf.gather(vertices_deformed_pose, indices_j, axis=-2)
    # Extracts the weights we need per term.
    weights_shape = vertices_i_rest.shape.as_list()[-2]
    if vertex_weight is not None:
      weight_i = tf.gather(vertex_weight, indices_i)
      weight_j = tf.gather(vertex_weight, indices_j)
    else:
      weight_i = weight_j = tf.ones(
          weights_shape, dtype=vertices_rest_pose.dtype)
    weight_i = tf.expand_dims(weight_i, axis=-1)
    weight_j = tf.expand_dims(weight_j, axis=-1)
    if edge_weight is not None:
      weight_ij = edge_weight
    else:
      weight_ij = tf.ones(weights_shape, dtype=vertices_rest_pose.dtype)
    weight_ij = tf.expand_dims(weight_ij, axis=-1)
    # Extracts the rotation we need per term.
    quaternion_i = tf.gather(quaternions, indices_i, axis=-2)
    quaternion_j = tf.gather(quaternions, indices_j, axis=-2)
    # Computes the energy.
    deformed_ij = vertices_i_deformed - vertices_j_deformed
    rotated_rest_ij = quaternion.rotate((vertices_i_rest - vertices_j_rest),
                                        quaternion_i)
    energy_ij = weight_i * weight_ij * (deformed_ij - rotated_rest_ij)
    deformed_ji = vertices_j_deformed - vertices_i_deformed
    rotated_rest_ji = quaternion.rotate((vertices_j_rest - vertices_i_rest),
                                        quaternion_j)
    energy_ji = weight_j * weight_ij * (deformed_ji - rotated_rest_ji)
    energy_ij_squared = vector.dot(energy_ij, energy_ij, keepdims=False)
    energy_ji_squared = vector.dot(energy_ji, energy_ji, keepdims=False)
    if aggregate_loss:
      average_energy_ij = tf.reduce_mean(
          input_tensor=energy_ij_squared, axis=-1)
      average_energy_ji = tf.reduce_mean(
          input_tensor=energy_ji_squared, axis=-1)
      return (average_energy_ij + average_energy_ji) / 2.0
    return tf.concat((energy_ij_squared, energy_ji_squared), axis=-1)

# API contains all public functions and classes.
__all__ = export_api.get_functions_and_classes()
