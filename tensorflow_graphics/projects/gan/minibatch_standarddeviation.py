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
"""An implementation of Minibatch Standard Deviation."""

import tensorflow as tf


class MiniBatchStandardDeviationBase(tf.keras.layers.Layer):
  """Layer that computes the standard deviation across an axis of its inputs."""

  def __init__(self, *args, batch_axis=0, channel_axis=-1, **kwargs):
    """Initializes the MiniBatchStandardDeviationBase instance.

    Args:
      *args: Arguments that get forwarded to the super initializer.
      batch_axis: Integer or a list of one integer, the axis the average
        standard deviation should be computed over (typically the batch axis).
        For instance, after a `Conv2D` layer, set `batch_axis=0` in
        `MiniBatchStandardDeviation`.
      channel_axis: Integer or a list of one integer, the axis the resulting
        standard deviation should be appended to (typically the channel axis).
        For instance, after a `Conv2D` layer with
        `data_format="channels_first"`, set `channel_axis=1` in
        `MiniBatchStandardDeviation`.
      **kwargs: Keyword argument dictionary that gets forwarded to the super
        initializer.
    Call arguments:
      inputs: Input tensor (of any rank).
    Input shape: Arbitrary. Use the keyword argument `input_shape` (tuple of
      integers, does not include the samples axis) when using this layer as the
      first layer in a model.
    Output shape: Same shape as input except in the dimension of the
      channel_axis, which will be incremented by 1.
    Reference:
      - [Tero Karras et al., 2018](https://arxiv.org/abs/1710.10196).
    """
    super().__init__(*args, **kwargs)
    if isinstance(batch_axis, (list, tuple)) and (len(batch_axis) == 1):
      self.batch_axis = batch_axis[0]
    elif isinstance(batch_axis, int):
      self.batch_axis = batch_axis
    else:
      raise TypeError('Expected an int or a list/tuple with one int for the '
                      'argument \'batch_axis\', but received: %r' % batch_axis)

    if isinstance(channel_axis, (list, tuple)) and (len(channel_axis) == 1):
      self.channel_axis = channel_axis[0]
    elif isinstance(channel_axis, int):
      self.channel_axis = channel_axis
    else:
      raise TypeError('Expected an int or a list/tuple with one int for the '
                      'argument \'channel_axis\', but received: %r' %
                      channel_axis)

  def _calculate_mean_feature_standard_deviation(self, inputs, batch_axis):
    mean = tf.reduce_mean(inputs, axis=batch_axis, keepdims=True)
    mean_sq_diff = tf.reduce_mean(
        tf.square(inputs - mean), axis=batch_axis, keepdims=True)
    mean_stddev = tf.reduce_mean(tf.sqrt(mean_sq_diff), keepdims=True)
    return mean_stddev

  def call(self, inputs):

    input_shape = tf.shape(inputs)
    ndims = len(input_shape)

    if self.batch_axis < 0:
      self.batch_axis = ndims + self.batch_axis

    # Validate axis
    if self.batch_axis >= ndims:
      raise ValueError(
          f'Invalid batch_axis. Expected 0 <= batch_axis < inputs.rank (with '
          f'inputs.rank={ndims}). Received: layer.batch_axis={self.batch_axis}')

    if self.channel_axis < 0:
      self.channel_axis = ndims + self.channel_axis

    # Validate axis
    if self.channel_axis >= ndims:
      raise ValueError(
          f'Invalid channel_axis. Expected 0 <= channel_axis < inputs.rank (with '
          f'inputs.rank={ndims}). Received: layer.channel_axis={self.channel_axis}'
      )

    mean_stddev = self._calculate_mean_feature_standard_deviation(
        inputs, self.batch_axis)
    output_shape = tf.tensor_scatter_nd_update(input_shape,
                                               [[self.channel_axis]], [1])
    outputs = tf.broadcast_to(mean_stddev, output_shape)

    return tf.concat([inputs, outputs], axis=self.channel_axis)

  def get_config(self):
    config = {
        'batch_axis': self.batch_axis,
        'channel_axis': self.channel_axis,
    }
    base_config = super(MiniBatchStandardDeviationBase, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class SyncMiniBatchStandardDeviation(MiniBatchStandardDeviationBase):
  """Compute standard deviation over features synchronously across replicas.

  Computes the mini batch standard deviation over the features of the previous
  layer at each batch
  by synchronizing the global batch statistics across all devices that are
  training the model. For specific details about batch normalization please
  refer to the `tf.keras.layers.MiniBatchStandardDeviationBase` layer docs.

  If this layer is used when using tf.distribute strategy to train models
  across devices/workers, there will be an allreduce call to aggregate batch
  statistics across all replicas at every training step. Without tf.distribute
  strategy, this layer behaves as a regular
  `tf.keras.layers.MiniBatchStandardDeviation`
  layer.

  Example usage:

  ```python
  strategy = tf.distribute.MirroredStrategy()

  with strategy.scope():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(16))
    model.add(SyncMiniBatchStandardDeviation())
  ```
  """

  def _calculate_mean_feature_standard_deviation(self, inputs, batch_axis):

    with tf.name_scope('mean_feature_standard_deviation'):
      # The dynamic range of fp16 is too limited to support the collection of
      # sufficient statistics. As a workaround we simply perform the operations
      # on 32-bit floats before converting the mean and variance back to fp16
      inputs_32 = tf.cast(inputs,
                          tf.float32) if inputs.dtype == tf.float16 else inputs
      replica_ctx = tf.distribute.get_replica_context()
      if replica_ctx:
        local_sum = tf.reduce_sum(inputs_32, axis=batch_axis, keepdims=True)
        local_squared_sum = tf.reduce_sum(
            tf.square(inputs_32), axis=batch_axis, keepdims=True)
        batch_size = tf.cast(tf.shape(inputs_32)[batch_axis], tf.float32)
        # TODO(b/163099951): batch the all-reduces once we sort out the ordering
        # issue for NCCL. We don't have a mechanism to launch NCCL in the same
        # order in each replica nowadays, so we limit NCCL to batch all-reduces.
        y_sum = replica_ctx.all_reduce(tf.distribute.ReduceOp.SUM, local_sum)
        y_squared_sum = replica_ctx.all_reduce(tf.distribute.ReduceOp.SUM,
                                               local_squared_sum)
        global_batch_size = replica_ctx.all_reduce(tf.distribute.ReduceOp.SUM,
                                                   batch_size)

        mean = y_sum / global_batch_size
        y_squared_mean = y_squared_sum / global_batch_size
        # var = E(x^2) - E(x)^2
        variance = y_squared_mean - tf.square(mean)
        stddev = tf.sqrt(variance)
        mean_stddev = tf.reduce_mean(stddev, keepdims=True)

      else:
        mean_stddev = super()._calculate_mean_feature_standard_deviation(
            inputs, batch_axis)

      return mean_stddev


class MiniBatchStandardDeviation(MiniBatchStandardDeviationBase):
  """Layer that computes the standard deviation across an axis of its inputs."""
