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
"""Test cases for Mesh wrapper."""

import tensorflow as tf

from tensorflow_graphics.projects.mesh_rcnn.structures import mesh
from tensorflow_graphics.util import test_case


class MeshTest(test_case.TestCase):
  """Test cases for Mesh wrapper."""

  def test_mesh_packing_on_single_mesh(self):
    """Tests packing and unpacking of meshes with a single mesh."""
    unit_cube_verts = tf.constant(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ], dtype=tf.int32)

    unit_cube_faces = tf.constant(
        [
            [0, 1, 2],
            [1, 3, 2],  # left face: 0, 1
            [2, 3, 6],
            [3, 7, 6],  # bottom face: 2, 3
            [0, 2, 6],
            [0, 6, 4],  # front face: 4, 5
            [0, 5, 1],
            [0, 4, 5],  # up face: 6, 7
            [6, 7, 5],
            [6, 5, 4],  # right face: 8, 9
            [1, 7, 3],
            [1, 5, 7],  # back face: 10, 11
        ],
        dtype=tf.int32
    )

    unit_cube_mesh = mesh.Meshes([unit_cube_verts], [unit_cube_faces])

    expected_flat_vertices = tf.reshape(unit_cube_verts, (-1, 3))
    expected_flat_faces = tf.reshape(unit_cube_faces, (-1, 3))

    expected_padded_verts = tf.expand_dims(unit_cube_verts, 0)
    expected_padded_faces = tf.expand_dims(unit_cube_faces, 0)

    self.assertAllEqual(expected_flat_vertices,
                        unit_cube_mesh.get_flattened()[0])
    self.assertAllEqual(expected_flat_faces, unit_cube_mesh.get_flattened()[1])

    self.assertAllEqual(expected_padded_verts, unit_cube_mesh.get_padded()[0])
    self.assertAllEqual(expected_padded_faces, unit_cube_mesh.get_padded()[1])

    self.assertAllEqual(unit_cube_verts, unit_cube_mesh.get_unpadded()[0][0])
    self.assertAllEqual(unit_cube_faces, unit_cube_mesh.get_unpadded()[1][0])

  def test_mesh_packing_on_batch(self):
    """Tests packing and unpacking of meshes in a batch."""

    short_verts = tf.reshape(tf.range(12.), (4, 3))
    long_verts = tf.reshape(tf.range(12., 36.), (8, 3))

    short_faces = tf.reshape(tf.range(12), (4, 3))
    long_faces = tf.reshape(tf.range(12, 36), (8, 3))

    vertices = [long_verts, short_verts]
    faces = [long_faces, short_faces]

    meshes = mesh.Meshes(vertices, faces)

    padded_short_verts = tf.pad(short_verts, [[0, 4], [0, 0]], 'CONSTANT', 0)
    padded_short_faces = tf.pad(short_faces, [[0, 4], [0, 0]], 'CONSTANT', 0)

    batched_verts = tf.stack([long_verts, padded_short_verts], axis=0)
    batched_faces = tf.stack([long_faces, padded_short_faces], axis=0)

    expected_flat_verts = tf.reshape(batched_verts, (16, 3))[:-4]
    expected_flat_faces = tf.reshape(batched_faces, (16, 3))[:-4]

    # Check flat representation
    self.assertAllEqual(expected_flat_verts, meshes.get_flattened()[0])
    self.assertAllEqual(expected_flat_faces, meshes.get_flattened()[1])

    # check padded representation
    self.assertAllEqual(batched_verts, meshes.get_padded()[0])
    self.assertAllEqual(batched_faces, meshes.get_padded()[1])

    # check unpadding
    self.assertAllEqual(vertices[0], meshes.get_unpadded()[0][0])
    self.assertAllEqual(faces[0], meshes.get_unpadded()[1][0])
    self.assertAllEqual(vertices[1], meshes.get_unpadded()[0][1])
    self.assertAllEqual(faces[1], meshes.get_unpadded()[1][1])

  def test_with_empty_lists(self):
    """Tests implementation with empty meshes."""

    with self.assertRaises(ValueError):
      _ = mesh.Meshes([tf.constant([], dtype=tf.float32)],
                      [tf.constant([], dtype=tf.float32)])
