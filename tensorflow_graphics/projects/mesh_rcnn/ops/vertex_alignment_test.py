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
"""
Test cases for vert align Op.
"""

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_graphics.projects.mesh_rcnn.ops.vertex_alignment import \
  vert_align
from tensorflow_graphics.projects.mesh_rcnn.structures.mesh import Meshes
from tensorflow_graphics.util import test_case


class VertAlignTest(test_case.TestCase):
  """Test Cases for vert align OP."""

  def test_vert_align_with_matching_querypoints(self):
    """Test on set of points that match exactly with pixels in the source
    image."""

    image = tf.reshape(tf.range(20.), (1, 4, 5, 1))
    verts = tf.constant([[-1.5, -1.5, 10],
                         [-1.5, 0.5, 10],
                         [-0.5, -1.5, 10],
                         [-0.5, 0.5, 10]], dtype=tf.float32)

    batched_verts = tf.expand_dims(verts, 0)
    intrinsics = tf.constant([[10, 0, 2.5], [0, 10, 2.5], [0, 0, 1]],
                             dtype=tf.float32)
    batched_intrinsics = tf.expand_dims(intrinsics, 0)
    expected_result = tf.constant([[6.], [8.], [11.], [13.]])
    result = vert_align(image, batched_verts, batched_intrinsics)

    self.assertAllClose(expected_result, result[0])

  def test_border_padding(self):
    """ Tests on set of vertices, where projection is outside of feature
    map."""

    image = tf.reshape(tf.reshape(tf.range(70.), (10, 7))[1:-1, 1:-1],
                       (1, 8, 5, 1))

    verts = tf.constant([[-5., -3.5, 10],
                         [-5, 3.5, 10],
                         [4., -3.5, 10],
                         [4, 3.5, 10]], dtype=tf.float32)
    batched_verts = tf.expand_dims(verts, 0)
    intrinsics = tf.constant([[10, 0, 5], [0, 10, 3.5], [0, 0, 1]],
                             dtype=tf.float32)
    batched_intrinsics = tf.expand_dims(intrinsics, 0)
    expected_result = tf.constant([[8.], [12.], [57.], [61.]])
    result = vert_align(image, batched_verts, batched_intrinsics)

    self.assertAllClose(expected_result, result[0])

  def test_multi_channel(self):
    """Tests VertAlign on multi-channel input."""
    image = tf.reshape(tf.range(60.), (1, 4, 5, 3))

    verts = tf.constant([[-1.5, -1.5, 10],
                         [-1.5, 0.5, 10],
                         [-0.5, -1.5, 10],
                         [-0.5, 0.5, 10]], dtype=tf.float32)
    batched_verts = tf.expand_dims(verts, 0)
    intrinsics = tf.constant([[10, 0, 2.5], [0, 10, 2.5], [0, 0, 1]],
                             dtype=tf.float32)
    batched_intrinsics = tf.expand_dims(intrinsics, 0)
    expected_result = tf.constant([image[0, 1, 1, :].numpy(),
                                   image[0, 1, 3, :].numpy(),
                                   image[0, 2, 1, :].numpy(),
                                   image[0, 2, 3, :].numpy()])

    result = vert_align(image, batched_verts, batched_intrinsics)

    self.assertAllClose(expected_result, result[0])

  def test_batch_different_sized_vertices(self):
    """Tests on batch of different sized vertices."""
    image = tf.reshape(tf.range(20.), (1, 4, 5, 1))
    images = tf.repeat(image, repeats=[2], axis=0)
    verts1 = tf.constant([[-1.5, -1.5, 10],
                          [-1.5, 0.5, 10],
                          [-0.5, -1.5, 10],
                          [-0.5, 0.5, 10]], dtype=tf.float32)

    verts2 = tf.constant([[-1.5, -1.5, 10],
                          [-1.5, 0.5, 10]], dtype=tf.float32)

    intrinsics = tf.constant([[10, 0, 2.5], [0, 10, 2.5], [0, 0, 1]],
                             dtype=tf.float32)
    intrinsics = tf.stack([intrinsics, intrinsics])

    input_mesh = Meshes([verts1, verts2],
                        [tf.ones((2, 3)), tf.ones((2, 3))])

    expected_result1 = tf.constant([[6.], [8.], [11.], [13.]])
    expected_result2 = tf.constant([[6.], [8.], [15.], [15.]])
    result = vert_align(images, input_mesh.get_padded()[0], intrinsics)

    self.assertAllClose(expected_result1, result[0])
    self.assertAllClose(expected_result2, result[1])

  def test_multiple_batch_dimensions_random_input(self):
    """Tests on random input with multiple batch dimenaions"""
    batch_dim_1 = 2
    batch_dim_2 = 3
    features = tf.random.normal((batch_dim_1, batch_dim_2, 5, 5, 3))
    vertices = tf.random.normal((batch_dim_1, batch_dim_2, 10, 3))
    intrinsics = tf.random.normal((batch_dim_1, batch_dim_2, 3, 3))

    result = vert_align(features, vertices, intrinsics)

    self.assertEqual([batch_dim_1, batch_dim_2, 10, 3], result.shape)

  @parameterized.parameters(
      ([2, 2, 2, 1], [4, 5, 3], [2, 3, 3],
       'Not all batch dimensions are identical'),
      ([4, 2, 2, 1], [2, 5, 3], [2, 3, 3],
       'Not all batch dimensions are identical'),
      ([2, 2, 2, 1], [2, 5, 3], [4, 3, 3],
       'Not all batch dimensions are identical'),
      ([2, 2, 2, 1], [2, 5, 3], [3],
       'intrinsics must have a rank greater than 1.'),
      ([2, 2], [2, 5, 3], [2, 3, 3],
       'features must have a rank greater than 2.'),
      ([2, 4, 4, 1], [2, 5, 1], [2, 3, 3],
       'vertices must have exactly 3 dimensions in axis -1'),
      ([2, 4, 4, 1], [2, 5, 3], [2, 2, 3],
       'intrinsics must have exactly 3 dimensions in axis -2'),
      ([2, 4, 4, 1], [2, 5, 3], [2, 3, 2],
       'intrinsics must have exactly 3 dimensions in axis -1'),
  )
  def test_raising_inputs(self,
                          feature_shape,
                          vertex_shape,
                          intrinsics_shape,
                          msg):
    """Tests if correct exceptions are raised when called with wrong
    input shapes."""
    features = tf.random.normal(feature_shape)
    vertices = tf.random.normal(vertex_shape)
    intrinsics = tf.random.normal(intrinsics_shape)

    with self.assertRaisesWithPredicateMatch(ValueError, msg):
      _ = vert_align(features, vertices, intrinsics)


if __name__ == '__main__':
  test_case.main()
