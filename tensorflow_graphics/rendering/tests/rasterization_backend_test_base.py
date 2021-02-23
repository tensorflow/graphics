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
"""Base class for rasterization backend tests."""

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_graphics.rendering import rasterization_backend
from tensorflow_graphics.util import test_case


class RasterizationBackendTestBase(test_case.TestCase):
  """Base class for CPU/GPU rasterization backend tests."""

  def setUp(self):
    super().setUp()
    self._backend = rasterization_backend.RasterizationBackends.OPENGL

  def _create_placeholders(self, shapes, dtypes):
    if tf.executing_eagerly():
      # If shapes is an empty list, we can continue with the test. If shapes
      # has None values, we shoud return.
      shapes = self._remove_dynamic_shapes(shapes)
      if shapes is None:
        return
    placeholders = self._create_placeholders_from_shapes(
        shapes=shapes, dtypes=dtypes)
    return placeholders

  @parameterized.parameters((
      ((2, 7, 3), (5, 3), (2, 4, 4)),
      (tf.float32, tf.int32, tf.float32),
  ))
  def test_rasterizer_rasterize_exception_not_raised(self, shapes, dtypes):
    """Tests that supported backends do not raise exceptions."""
    placeholders = self._create_placeholders(shapes, dtypes)
    try:
      rasterization_backend.rasterize(placeholders[0], placeholders[1],
                                      placeholders[2], (600, 800),
                                      self._backend)
    except Exception as e:  # pylint: disable=broad-except
      self.fail('Exception raised: %s' % str(e))

  @parameterized.parameters((
      ((1, 7, 3), (5, 3), (1, 4, 4)),
      (tf.float32, tf.int32, tf.float32),
  ))
  def test_rasterizer_return_correct_batch_shapes(self, shapes, dtypes):
    """Tests that supported backends return correct shape."""
    placeholders = self._create_placeholders(shapes, dtypes)
    frame_buffer = rasterization_backend.rasterize(placeholders[0],
                                                   placeholders[1],
                                                   placeholders[2], (600, 800),
                                                   self._backend)
    batch_size = shapes[0][0]
    self.assertEqual([batch_size],
                     frame_buffer.triangle_id.get_shape().as_list()[:-3])
    self.assertEqual([batch_size],
                     frame_buffer.foreground_mask.get_shape().as_list()[:-3])

  @parameterized.parameters(
      (((2, 7, 3), (5, 3), (2, 4, 4)),
       (tf.float32, tf.int32, tf.float32), 'Foobar'),
      (((2, 7, 3), (5, 3), (2, 4, 4)),
       (tf.float32, tf.int32, tf.float32), 'Opengl'),
  )
  def test_rasterizer_rasterize_exception_raised(self, shapes, dtypes, backend):
    """Tests that unsupported backends raise exceptions."""
    placeholders = self._create_placeholders(shapes, dtypes)
    with self.assertRaisesRegexp(KeyError, 'Backend is not supported'):
      rasterization_backend.rasterize(placeholders[0], placeholders[1],
                                      placeholders[2], (600, 800), backend)

  def test_rasterizer_all_vertices_visible(self):
    """Renders simple triangle and asserts that it is fully visible."""
    vertices = tf.convert_to_tensor([[[0, 0, 0], [10, 10, 0], [0, 10, 0]]],
                                    dtype=tf.float32)
    triangles = tf.convert_to_tensor([[0, 1, 2]], dtype=tf.int32)
    view_projection_matrix = tf.expand_dims(tf.eye(4), axis=0)
    frame_buffer = rasterization_backend.rasterize(vertices, triangles,
                                                   view_projection_matrix,
                                                   (100, 100), self._backend)
    self.assertAllEqual(frame_buffer.triangle_id.shape[:-1],
                        frame_buffer.vertex_ids.shape[:-1])
    # Assert that triangle is visible.
    self.assertAllLess(frame_buffer.triangle_id, 2)
    self.assertAllGreaterEqual(frame_buffer.triangle_id, 0)
    # Assert that all three vertices are visible.
    self.assertAllLess(frame_buffer.triangle_id, 3)
    self.assertAllGreaterEqual(frame_buffer.triangle_id, 0)
