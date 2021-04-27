
# Copyright 2021 The TensorFlow Authors
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
"""Tests for 3D spatial transformer."""

# pylint: disable=invalid-name

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_graphics.nn.layer.spatial_transformer_3d import SpatialTransformer3D
from tensorflow_graphics.util import test_case

class SpatialTransformer3DTest(test_case.TestCase):

  @parameterized.parameters(
      ("bilinear", "border", (0), (1, 10, 10, 10, 1)),
      ("bilinear", "border", (1), (1, 10, 10, 10, 1)),
      ("bilinear", "border", (2), (1, 10, 10, 10, 1))
  )
  def test_spatial_transformer_3d_train(
      self, interp_method, padding_mode, seed, volume_shape):
    """Test a simple training loop."""

    params_spatial_transformer = dict(
        interp_method=interp_method, padding_mode=padding_mode)
    # source and target volumes are 3D-tensors of shape [A0, W, H, D, C]
    source = tf.random.normal(shape=volume_shape, dtype=tf.float32, seed=seed)
    target = source + tf.random.normal(
        shape=volume_shape, stddev=1e-3, dtype=tf.float32, seed=seed)
    # train with 5 iterations
    num_iters = 5
    # initialize weights with identity transformation slightly pertubed
    init_quaternion = tf.constant([0., 0., 0., 1., 0., 0., 0., 1., 1., 1.])
    init_quaternion = init_quaternion + tf.random.normal(
        shape=(10,), stddev=1e-6, dtype=tf.float32, seed=seed)
    init_weights_quaternion = [
        tf.zeros((5, 10), dtype=tf.float32), init_quaternion]

    if tf.executing_eagerly():
      # input and model definition
      inp_source = tf.keras.Input(shape=volume_shape[1:], dtype="float32")
      reshaped = tf.keras.layers.Flatten()(inp_source)
      reshaped = tf.expand_dims(reshaped, axis=-1)
      max_pooled = tf.keras.layers.MaxPool1D(
          pool_size=2, strides=200, padding="SAME")(reshaped)
      flattened = tf.keras.layers.Flatten()(max_pooled)
      transformation = tf.keras.layers.Dense(
          units=10
          , activation=None
          , weights=init_weights_quaternion)(flattened)
      output = SpatialTransformer3D(
          **params_spatial_transformer)([inp_source, transformation])
      model = tf.keras.models.Model(inputs=[inp_source], outputs=[output])
      # optimizer function
      optimizer = tf.keras.optimizers.SGD(learning_rate=1e-6)
      # training loop
      for _ in range(num_iters):
        with tf.GradientTape() as tape:
          loss = tf.nn.l2_loss(model([source]) - target)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        # Check that gradients has correct shape
        tf.debugging.assert_equal(len(grads), 2)
        tf.debugging.assert_equal(grads[0].shape, tf.TensorShape((5, 10)))
        tf.debugging.assert_equal(grads[1].shape, tf.TensorShape((10)))

  @parameterized.parameters(
      ("bilinear", "border", (0), (1, 10, 10, 10, 1)),
      ("nn", "border", (0), (1, 10, 10, 10, 1)),
      ("nn", "zeros", (0), (1, 10, 10, 10, 1)),
      ("nn", "min", (0), (1, 10, 10, 10, 1))
  )
  def test_spatial_transformer_3d_forward_methods(
      self, interp_method, padding_mode, seed, volume_shape):
    """Test a simple forward pass with different interpolation and padding methods."""

    params_spatial_transformer = dict(
        interp_method=interp_method, padding_mode=padding_mode)
    # source volume is 3D-tensors of shape [A0, W, H, D, C]
    source = tf.random.normal(shape=volume_shape, dtype=tf.float32, seed=seed)
    # resample with identity transformation slightly pertubed
    quaternion = tf.constant(
        [[0., 0., 0., 1., 0., 0., 0., 1., 1., 1.]], dtype=tf.float32)
    quaternion = quaternion + tf.random.normal(
        shape=(1, 10), stddev=1e-9, dtype=tf.float32, seed=seed)

    if tf.executing_eagerly():
      output = SpatialTransformer3D(
          **params_spatial_transformer, trainable=False)([source, quaternion])
      tf.debugging.assert_near(
          source
          , output
          , atol=1e-6
          , message="Source volume and output should be approximately equal. "
                    "{}".format(params_spatial_transformer))

  @parameterized.parameters(
      ("bilinear", "border", (0), (1, 10, 10, 10, 1)
       , tf.constant(
           [[0., 0., 0., 1., 0., 0., 0., 1., 1., 1.]]
           , dtype=tf.float32)),
      ("bilinear", "border", (0), (1, 10, 10, 10, 1)
       , tf.constant(
           [[0., 0., 0., 1., 0., 0., 0.]]
           , dtype=tf.float32)),
      ("bilinear", "border", (0), (1, 10, 10, 10, 1)
       , tf.constant(
           [[0., 0., 0., 1.]]
           , dtype=tf.float32)),
  )
  def test_spatial_transformer_3d_forward_transformations(
      self, interp_method, padding_mode, seed, volume_shape, quaternion):
    """Test a simple forward pass with different transformations."""

    params_spatial_transformer = dict(
        interp_method=interp_method, padding_mode=padding_mode)
    # source volume is 3D-tensors of shape [A0, W, H, D, C]
    source = tf.random.normal(shape=volume_shape, dtype=tf.float32, seed=seed)
    # resample with identity transformation slightly pertubed
    shape_quaternion = tf.size(quaternion)
    quaternion = quaternion + tf.random.normal(
        shape=(1, shape_quaternion)
        , stddev=1e-6
        , dtype=tf.float32
        , seed=seed)

    if tf.executing_eagerly():
      output = SpatialTransformer3D(
          **params_spatial_transformer, trainable=False)([source, quaternion])
      tf.debugging.assert_near(
          source
          , output
          , atol=1e-3
          , message="Source volume and output should be approximately equal. "
                    "{}".format(quaternion))

if __name__ == "__main__":
  test_case.main()
