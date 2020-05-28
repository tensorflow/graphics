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
"""Tests for as_conformal_as_possible."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import flagsaver
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_graphics.geometry.deformation_energy import as_conformal_as_possible
from tensorflow_graphics.geometry.transformation import quaternion
from tensorflow_graphics.util import test_case


class AsConformalAsPossibleTest(test_case.TestCase):

  def test_energy_identity(self):
    """Checks that energy evaluated between the rest pose and itself is zero."""
    number_vertices = np.random.randint(3, 10)
    batch_size = np.random.randint(3)
    batch_shape = np.random.randint(1, 10, size=(batch_size)).tolist()
    vertices_rest_pose = np.random.uniform(size=(number_vertices, 3))
    vertices_deformed_pose = tf.broadcast_to(
        vertices_rest_pose, shape=batch_shape + [number_vertices, 3])
    quaternions = quaternion.from_euler(
        np.zeros(shape=batch_shape + [number_vertices, 3]))
    num_edges = int(round(number_vertices / 2))
    edges = np.zeros(shape=(num_edges, 2), dtype=np.int32)
    edges[..., 0] = np.linspace(
        0, number_vertices / 2 - 1, num_edges, dtype=np.int32)
    edges[..., 1] = np.linspace(
        number_vertices / 2, number_vertices - 1, num_edges, dtype=np.int32)

    energy = as_conformal_as_possible.energy(
        vertices_rest_pose=vertices_rest_pose,
        vertices_deformed_pose=vertices_deformed_pose,
        quaternions=quaternions,
        edges=edges,
        conformal_energy=False)

    self.assertAllClose(energy, tf.zeros_like(energy))

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_energy_jacobian_random(self):
    """Checks the correctness of the jacobian of energy."""
    number_vertices = np.random.randint(3, 10)
    batch_size = np.random.randint(3)
    batch_shape = np.random.randint(1, 10, size=(batch_size)).tolist()
    vertices_rest_pose_init = np.random.uniform(size=(number_vertices, 3))
    vertices_deformed_pose_init = np.random.uniform(size=batch_shape +
                                                    [number_vertices, 3])
    quaternions_init = np.random.uniform(size=batch_shape +
                                         [number_vertices, 4])
    num_edges = int(round(number_vertices / 2))
    edges = np.zeros(shape=(num_edges, 2), dtype=np.int32)
    edges[..., 0] = np.linspace(
        0, number_vertices / 2 - 1, num_edges, dtype=np.int32)
    edges[..., 1] = np.linspace(
        number_vertices / 2, number_vertices - 1, num_edges, dtype=np.int32)

    def conformal_energy(vertices_rest_pose, vertices_deformed_pose,
                         quaternions):
      return as_conformal_as_possible.energy(
          vertices_rest_pose=vertices_rest_pose,
          vertices_deformed_pose=vertices_deformed_pose,
          quaternions=quaternions,
          edges=edges,
          conformal_energy=True)

    def nonconformal_energy(vertices_rest_pose, vertices_deformed_pose,
                            quaternions):
      return as_conformal_as_possible.energy(
          vertices_rest_pose=vertices_rest_pose,
          vertices_deformed_pose=vertices_deformed_pose,
          quaternions=quaternions,
          edges=edges,
          conformal_energy=False)

    with self.subTest(name="conformal"):
      self.assert_jacobian_is_correct_fn(conformal_energy, [
          vertices_rest_pose_init, vertices_deformed_pose_init, quaternions_init
      ])

    with self.subTest(name="non_conformal"):
      self.assert_jacobian_is_correct_fn(nonconformal_energy, [
          vertices_rest_pose_init, vertices_deformed_pose_init, quaternions_init
      ])

  @parameterized.parameters(
      ((1, 3), (1, 3), (1, 4), (1, 2), (1,), (1,)),
      ((1, 3), (None, 1, 3), (None, 1, 4), (1, 2), (1,), (1,)),
      ((1, 3), (2, None, 1, 3), (2, None, 1, 4), (1, 2), (1,), (1,)),
  )
  def test_energy_not_raised(self, *shapes):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(
        as_conformal_as_possible.energy, shapes,
        [tf.float32, tf.float32, tf.float32, tf.int32, tf.float32, tf.float32])

  def test_energy_preset(self):
    """Checks that energy returns the expected value."""
    vertices_rest_pose = np.array(((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)))
    vertices_deformed_pose = 2.0 * vertices_rest_pose
    quaternions = quaternion.from_euler(
        np.zeros(shape=vertices_deformed_pose.shape))
    edges = ((0, 1),)

    all_weights_1_energy = as_conformal_as_possible.energy(
        vertices_rest_pose, vertices_deformed_pose, quaternions, edges)
    all_weights_1_gt = 4.0
    vertex_weights = np.array((2.0, 1.0))
    vertex_weights_energy = as_conformal_as_possible.energy(
        vertices_rest_pose=vertices_rest_pose,
        vertices_deformed_pose=vertices_deformed_pose,
        quaternions=quaternions,
        edges=edges,
        vertex_weight=vertex_weights)
    vertex_weights_gt = 10.0
    edge_weights = np.array((2.0,))
    edge_weights_energy = as_conformal_as_possible.energy(
        vertices_rest_pose=vertices_rest_pose,
        vertices_deformed_pose=vertices_deformed_pose,
        quaternions=quaternions,
        edges=edges,
        edge_weight=edge_weights)
    edge_weights_gt = 16.0

    with self.subTest(name="all_weights_1"):
      self.assertAllClose(all_weights_1_energy, all_weights_1_gt)
    with self.subTest(name="vertex_weights"):
      self.assertAllClose(vertex_weights_energy, vertex_weights_gt)
    with self.subTest(name="edge_weights"):
      self.assertAllClose(edge_weights_energy, edge_weights_gt)

  @parameterized.parameters(
      ("vertices_rest_pose must have exactly 3 dimensions in axis",
       (1, 2), (1, 3), (1, 4), (1, 2), (1,), (1,)),
      ("vertices_rest_pose must have a rank of 2",
       (2, 1, 2), (1, 3), (1, 4), (1, 2), (1,), (1,)),
      ("vertices_deformed_pose must have exactly 3 dimensions in axis",
       (1, 3), (1, 2), (1, 4), (1, 2), (1,), (1,)),
      ("must have the same number of dimensions",
       (1, 3), (2, 3), (1, 4), (1, 2), (1,), (1,)),
      ("quaternions must have exactly 4 dimensions in axis",
       (1, 3), (1, 3), (1, 5), (1, 2), (1,), (1,)),
      ("must have the same number of dimensions",
       (1, 3), (1, 3), (2, 4), (1, 2), (1,), (1,)),
      ("Not all batch dimensions are identical",
       (1, 3), (1, 3), (2, 1, 4), (1, 2), (1,), (1,)),
      ("edges must have exactly 2 dimensions in axis",
       (1, 3), (1, 3), (1, 4), (1, 3), (1,), (1,)),
      ("edges must have a rank of 2",
       (1, 3), (1, 3), (1, 4), (2, 1, 2), (1,), (1,)),
      ("must have the same number of dimensions",
       (1, 3), (1, 3), (1, 4), (1, 2), (2,), (1,)),
      ("must have the same number of dimensions",
       (1, 3), (1, 3), (1, 4), (1, 2), (1,), (2,)),
  )  # pyformat: disable
  def test_energy_raised(self, error_msg, *shapes):
    """Tests that the shape exceptions are raised."""
    self.assert_exception_is_raised(
        as_conformal_as_possible.energy, error_msg, shapes,
        [tf.float32, tf.float32, tf.float32, tf.int32, tf.float32, tf.float32])


if __name__ == "__main__":
  test_case.main()
