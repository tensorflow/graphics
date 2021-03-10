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
"""Tests for the graph convolution layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
from six.moves import range
from six.moves import zip
import tensorflow as tf

import tensorflow_graphics.nn.layer.graph_convolution as gc_layer
from tensorflow_graphics.util import test_case


def _dense_to_sparse(data):
  """Convert a numpy array to a tf.SparseTensor."""
  indices = np.where(data)
  return tf.SparseTensor(
      np.stack(indices, axis=-1), data[indices], dense_shape=data.shape)


def _dummy_data(batch_size, num_vertices, num_channels):
  """Create inputs for feature_steered_convolution."""
  if batch_size > 0:
    data = np.zeros(
        shape=(batch_size, num_vertices, num_channels), dtype=np.float32)
    neighbors = _dense_to_sparse(
        np.tile(np.eye(num_vertices, dtype=np.float32), (batch_size, 1, 1)))
  else:
    data = np.zeros(shape=(num_vertices, num_channels), dtype=np.float32)
    neighbors = _dense_to_sparse(np.eye(num_vertices, dtype=np.float32))
  return data, neighbors


class GraphConvolutionTestFeatureSteeredConvolutionLayerTests(
    test_case.TestCase):

  @parameterized.parameters(
      (1, 1, 1, 1, 1, False),
      (4, 2, 3, None, 5, False),
      (1, 2, 3, 4, 5, True),
  )
  def test_feature_steered_convolution_layer_exception_not_raised_shapes(
      self, batch_size, num_vertices, in_channels, out_channels,
      num_weight_matrices, translation_invariant):
    """Check if the convolution parameters and output have correct shapes."""
    data, neighbors = _dummy_data(batch_size, num_vertices, in_channels)
    name_scope = "test"
    if tf.executing_eagerly():
      layer = gc_layer.FeatureSteeredConvolutionKerasLayer(
          translation_invariant=translation_invariant,
          num_weight_matrices=num_weight_matrices,
          num_output_channels=out_channels,
          name=name_scope)

    def _run_convolution():
      """Run the appropriate feature steered convolution layer."""
      if tf.executing_eagerly():
        try:
          output = layer(inputs=[data, neighbors], sizes=None)
        except Exception as e:  # pylint: disable=broad-except
          self.fail("Exception raised: %s" % str(e))
      else:
        try:
          output = gc_layer.feature_steered_convolution_layer(
              data=data,
              neighbors=neighbors,
              sizes=None,
              translation_invariant=translation_invariant,
              num_weight_matrices=num_weight_matrices,
              num_output_channels=out_channels,
              var_name=name_scope)
        except Exception as e:  # pylint: disable=broad-except
          self.fail("Exception raised: %s" % str(e))
      return output

    output = _run_convolution()
    output_shape = output.shape.as_list()
    out_channels = in_channels if out_channels is None else out_channels

    self.assertEqual(output_shape[-1], out_channels)
    self.assertAllEqual(output_shape[:-1], data.shape[:-1])

    def _get_var_shape(var_name):
      """Get the shape of a variable by name."""
      if tf.executing_eagerly():
        trainable_variables = layer.trainable_variables
        for tv in trainable_variables:
          if tv.name == name_scope + "/" + var_name + ":0":
            return tv.shape.as_list()
        raise ValueError("Variable not found.")
      else:
        with tf.compat.v1.variable_scope(name_scope, reuse=True):
          variable = tf.compat.v1.get_variable(
              var_name, initializer=tf.constant(0))
          return variable.shape.as_list()

    self.assertAllEqual(_get_var_shape("u"), [in_channels, num_weight_matrices])
    self.assertAllEqual(_get_var_shape("c"), [num_weight_matrices])
    self.assertAllEqual(_get_var_shape("b"), [out_channels])
    self.assertAllEqual(
        _get_var_shape("w"), [num_weight_matrices, in_channels, out_channels])
    if not translation_invariant:
      self.assertAllEqual(
          _get_var_shape("v"), [in_channels, num_weight_matrices])

  def test_feature_steered_convolution_layer_initializer(self):
    """Tests a custom variable initializer."""
    data = np.array(((1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0)))
    neighbors_indices = np.array(((0, 0), (0, 1), (0, 3),
                                  (1, 0), (1, 1), (1, 2),
                                  (2, 1), (2, 2), (2, 3),
                                  (3, 0), (3, 2), (3, 3)))
    neighbors = tf.SparseTensor(
        neighbors_indices, np.ones(shape=(12,)) / 3.0, dense_shape=(4, 4))
    initializer = tf.compat.v1.keras.initializers.zeros()

    if tf.executing_eagerly():
      layer = gc_layer.FeatureSteeredConvolutionKerasLayer(
          translation_invariant=False,
          initializer=initializer)
      output = layer(inputs=[data, neighbors], sizes=None)
    else:
      out = gc_layer.feature_steered_convolution_layer(
          data=data,
          neighbors=neighbors,
          sizes=None,
          translation_invariant=False,
          initializer=initializer)
      self.evaluate(tf.compat.v1.global_variables_initializer())
      output = self.evaluate(out)

    # All zeros initializer should result in all zeros output.
    self.assertAllEqual(output, np.zeros_like(data))

  def test_feature_steered_convolution_layer_training(self):
    """Test a simple training loop."""
    # Generate a small valid input for a simple training task.
    # Four corners of a square.
    data = np.array(((1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0)))
    neighbors_indices = np.array(((0, 0), (0, 1), (0, 3),
                                  (1, 0), (1, 1), (1, 2),
                                  (2, 1), (2, 2), (2, 3),
                                  (3, 0), (3, 2), (3, 3)))
    neighbors = tf.SparseTensor(
        neighbors_indices, np.ones(shape=(12,)) / 3.0, dense_shape=(4, 4))
    # Desired output is arbitrary.
    labels = np.reshape([-1.0, -0.5, 0.5, 1.0], (-1, 1))
    num_training_iterations = 5

    if tf.executing_eagerly():
      with tf.GradientTape(persistent=True) as tape:
        layer = gc_layer.FeatureSteeredConvolutionKerasLayer(
            translation_invariant=False,
            num_weight_matrices=1,
            num_output_channels=1)
        output = layer(inputs=[data, neighbors], sizes=None)
        loss = tf.nn.l2_loss(output - labels)

      trainable_variables = layer.trainable_variables
      for _ in range(num_training_iterations):
        grads = tape.gradient(loss, trainable_variables)
        tf.compat.v1.train.GradientDescentOptimizer(1e-4).apply_gradients(
            list(zip(grads, trainable_variables)))
    else:
      output = gc_layer.feature_steered_convolution_layer(
          data=data,
          neighbors=neighbors,
          sizes=None,
          translation_invariant=False,
          num_weight_matrices=1,
          num_output_channels=1)
      train_op = tf.compat.v1.train.GradientDescentOptimizer(1e-4).minimize(
          tf.nn.l2_loss(output - labels))
      with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.initialize_all_variables())
        for _ in range(num_training_iterations):
          sess.run(train_op)


class GraphConvolutionTestDynamicGraphConvolutionKerasLayerTests(
    test_case.TestCase):

  @parameterized.parameters(
      (1, 1, 1, 1, "weighted"),
      (4, 2, 3, 12, "max"),
      (1, 2, 3, 4, "max"),
  )
  def test_dynamic_graph_convolution_keras_layer_exception_not_raised_shapes(
      self, batch_size, num_vertices, in_channels, out_channels, reduction):
    """Check if the convolution parameters and output have correct shapes."""
    if not tf.executing_eagerly():
      return
    data, neighbors = _dummy_data(batch_size, num_vertices, in_channels)
    layer = gc_layer.DynamicGraphConvolutionKerasLayer(
        num_output_channels=out_channels,
        reduction=reduction)

    try:
      output = layer(inputs=[data, neighbors], sizes=None)
    except Exception as e:  # pylint: disable=broad-except
      self.fail("Exception raised: %s" % str(e))

    self.assertAllEqual((batch_size, num_vertices, out_channels), output.shape)

  @parameterized.parameters(
      (1, 1, 1, 1, "weighted"),
      (4, 2, 3, 12, "max"),
      (1, 2, 3, 4, "max"),
  )
  def test_dynamic_graph_convolution_keras_layer_zero_kernel(
      self, batch_size, num_vertices, in_channels, out_channels, reduction):
    """Tests convolution with an all-zeros kernel."""
    if not tf.executing_eagerly():
      return
    data, neighbors = _dummy_data(batch_size, num_vertices, in_channels)
    data = np.random.uniform(size=data.shape).astype(np.float32)
    layer = gc_layer.DynamicGraphConvolutionKerasLayer(
        num_output_channels=out_channels,
        reduction=reduction,
        use_bias=False,
        kernel_initializer=tf.compat.v1.keras.initializers.zeros())

    output = layer(inputs=[data, neighbors], sizes=None)

    self.assertAllEqual(
        output,
        np.zeros(shape=(batch_size, num_vertices, out_channels),
                 dtype=np.float32))

  @parameterized.parameters((1, 1, 1), (2, 3, 12), (2, 3, 4))
  def test_dynamic_graph_convolution_keras_layer_duplicate_features(
      self, num_vertices, in_channels, out_channels):
    """Tests convolution when all vertex features are identical."""
    if not tf.executing_eagerly():
      return
    data = np.random.uniform(size=(1, in_channels))
    data = np.tile(data, (num_vertices, 1))
    # Results should be independent of 'neighbors'.
    neighbors = np.maximum(np.random.randint(
        0, 2, size=(num_vertices, num_vertices)), np.eye(num_vertices))
    neighbors = _dense_to_sparse(neighbors)
    layer = gc_layer.DynamicGraphConvolutionKerasLayer(
        num_output_channels=out_channels,
        reduction="max")

    output = layer(inputs=[data, neighbors], sizes=None)

    output_tile = tf.tile(output[:1, :], (num_vertices, 1))

    self.assertAllEqual(output, output_tile)

  @parameterized.parameters("weighted", "max")
  def test_dynamic_graph_convolution_keras_layer_training(self, reduction):
    """Test a simple training loop."""
    if not tf.executing_eagerly():
      return
    # Generate a small valid input for a simple training task.
    # Four corners of a square.
    data = np.array(((1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0)))
    neighbors_indices = np.array(((0, 0), (0, 1), (0, 3),
                                  (1, 0), (1, 1), (1, 2),
                                  (2, 1), (2, 2), (2, 3),
                                  (3, 0), (3, 2), (3, 3)))
    neighbors = tf.SparseTensor(
        neighbors_indices, np.ones(shape=(12,)) / 3.0, dense_shape=(4, 4))
    # Desired output is arbitrary.
    labels = np.reshape([-1.0, -0.5, 0.5, 1.0], (-1, 1))
    num_training_iterations = 5

    with tf.GradientTape(persistent=True) as tape:
      layer = gc_layer.DynamicGraphConvolutionKerasLayer(
          num_output_channels=2,
          reduction=reduction)
      output = layer(inputs=[data, neighbors], sizes=None)
      loss = tf.nn.l2_loss(output - labels)

      trainable_variables = layer.trainable_variables
      for _ in range(num_training_iterations):
        grads = tape.gradient(loss, trainable_variables)
        tf.compat.v1.train.GradientDescentOptimizer(1e-4).apply_gradients(
            list(zip(grads, trainable_variables)))


if __name__ == "__main__":
  test_case.main()
