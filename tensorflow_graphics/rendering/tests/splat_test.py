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
"""Tests for google3.research.vision.viscam.diffren.mesh.splat."""

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_graphics.rendering import rasterization_backend
from tensorflow_graphics.rendering import splat
from tensorflow_graphics.rendering import triangle_rasterizer
from tensorflow_graphics.rendering.tests import rasterization_test_utils
from tensorflow_graphics.util import test_case


@tf.function
def rasterize_image(vertices, triangles, color, camera_matrix, image_width,
                    image_height):
  rasterized = triangle_rasterizer.rasterize(
      vertices,
      triangles, {'color': color},
      camera_matrix, (image_width, image_height),
      backend=rasterization_backend.RasterizationBackends.CPU)
  return rasterized['color']


class SplatTest(test_case.TestCase):

  @parameterized.parameters(([1],), ([2],), ([1, 2, 3],))
  def test_batch_dimension_preserved(self, batch_shape):
    """Tests that the input batch dimension preserved."""
    (vertices, triangles, attributes_dictionary, _, _, view_projection_matrix,
     image_height, image_width
    ) = rasterization_test_utils.create_rasterizer_inputs(batch_shape)

    rgba = splat.rasterize_then_splat(vertices, triangles,
                                      attributes_dictionary,
                                      view_projection_matrix,
                                      (image_height, image_width),
                                      lambda x: x['attribute1'])

    tensor_batch_shape = rgba.shape.as_list()[:len(batch_shape)]
    self.assertEqual(
        list(batch_shape),
        tensor_batch_shape,
        msg='Output has batch shape {0}, but expected is {1}'.format(
            tensor_batch_shape, batch_shape))

  def test_rasterizes_correct_shape(self):
    """Tests that rasterize returns the expected result."""
    batch_shape = []
    (vertices, triangles, attributes_dictionary, _, _, view_projection_matrix,
     image_height, image_width
    ) = rasterization_test_utils.create_rasterizer_inputs(batch_shape)

    rgba = splat.rasterize_then_splat(vertices, triangles,
                                      attributes_dictionary,
                                      view_projection_matrix,
                                      (image_height, image_width),
                                      lambda x: x['attribute1'])

    self.assertAllEqual(rgba.shape, (image_height, image_width, 4))

  def test_two_triangle_layers(self):
    """Checks that two overlapping triangles are accumulated correctly."""
    image_width = 32
    image_height = 32

    vertices = tf.constant(
        [[[-0.2, -0.2, 0], [0.5, -0.2, 0], [0.5, 0.5, 0], [0.2, -0.2, 0.5],
          [-0.5, -0.2, 0.5], [-0.5, 0.5, 0.5]]],
        dtype=tf.float32)
    triangles = [[0, 1, 2], [3, 5, 4]]
    colors = tf.constant(
        [[[0, 1.0, 0, 1.0], [0, 1.0, 0, 1.0], [0, 1.0, 0, 1.0],
          [1.0, 0, 0, 1.0], [1.0, 0, 0, 1.0], [1.0, 0, 0, 1.0]]],
        dtype=tf.float32)

    composite, _, normalized_layers = splat.rasterize_then_splat(
        vertices,
        triangles, {'color': colors},
        rasterization_test_utils.get_identity_view_projection_matrix(),
        (image_height, image_width),
        lambda x: x['color'],
        num_layers=2,
        return_extra_buffers=True)

    baseline_image = rasterization_test_utils.load_baseline_image(
        'Two_Triangles_Splat_Composite.png')
    baseline_image = tf.image.resize(baseline_image,
                                     (image_height, image_width))
    images_near, error_message = rasterization_test_utils.compare_images(
        self, baseline_image, composite)
    self.assertTrue(images_near, msg=error_message)

    for i in range(3):
      baseline_image = rasterization_test_utils.load_baseline_image(
          'Two_Triangles_Splat_Layer_{}.png'.format(i))
      image = normalized_layers[:, i, ...]
      image = tf.image.resize(
          image, (512, 512), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
      images_near, error_message = rasterization_test_utils.compare_images(
          self, baseline_image, image)
      self.assertTrue(images_near, msg=error_message)

  def test_optimize_single_triangle(self):
    """Checks that the position of a triangle can be optimized correctly.

    The optimization target is a translated version of the same triangle.
    Naive rasterization produces zero gradient in this case, but
    rasterize-then-splat produces a useful gradient.
    """
    image_width = 32
    image_height = 32

    initial_vertices = tf.constant([[[0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0]]],
                                   dtype=tf.float32)
    target_vertices = tf.constant(
        [[[-0.25, 0, 0], [0.25, 0, 0], [0.25, 0.5, 0]]], dtype=tf.float32)
    triangles = [[0, 1, 2]]
    colors = tf.constant(
        [[[0, 0.8, 0, 1.0], [0, 0.8, 0, 1.0], [0, 0.8, 0, 1.0]]],
        dtype=tf.float32)
    camera_matrix = rasterization_test_utils.get_identity_view_projection_matrix(
    )

    @tf.function
    def render_splat(verts):
      return splat.rasterize_then_splat(
          verts,
          triangles, {'color': colors},
          camera_matrix, (image_height, image_width),
          lambda x: x['color'],
          num_layers=1)

    # Perform a few iterations of gradient descent.
    num_iters = 15
    var_verts = tf.Variable(initial_vertices)
    splat_loss_initial = 0.0
    for i in range(num_iters):
      with tf.GradientTape(persistent=True) as g:
        target_image = rasterize_image(target_vertices, triangles, colors,
                                       camera_matrix, image_width,
                                       image_height)[0, ...]
        rasterized_only_image = rasterize_image(var_verts, triangles, colors,
                                                camera_matrix, image_width,
                                                image_height)[0, ...]
        splat_image = render_splat(var_verts)

        rasterized_loss = tf.reduce_mean(
            (rasterized_only_image - target_image)**2)
        splat_loss = tf.reduce_mean((splat_image - target_image)**2)

      rasterized_grad = g.gradient(rasterized_loss, var_verts)
      splat_grad = g.gradient(splat_loss, var_verts)

      if i == 0:
        # Check that the rasterized-only gradient is zero, while the
        # rasterize-then-splat gradient is non-zero.
        self.assertAlmostEqual(tf.norm(rasterized_grad).numpy(), 0.0)
        self.assertGreater(tf.norm(splat_grad).numpy(), 0.01)
        splat_loss_initial = splat_loss

      # Apply the gradient.
      var_verts.assign_sub(splat_grad)

    # Check that gradient descent reduces the loss by at least 50%.
    opt_image = render_splat(var_verts)
    opt_loss = tf.reduce_mean((opt_image - target_image)**2)
    self.assertLess(opt_loss.numpy(), splat_loss_initial.numpy() * 0.5)


if __name__ == '__main__':
  test_case.main()
