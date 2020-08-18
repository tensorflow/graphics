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
"""Model Implementations."""
import tensorflow.compat.v1 as tf

from tensorflow_graphics.projects.nasa.lib import model_utils

tf.disable_eager_execution()


def get_model(hparams):
  return model_dict[hparams.model](hparams)


def nasa(hparams):
  """Construct the model function of NASA."""
  # Parse model parameters from global configurations.
  n_parts = hparams.n_parts
  n_dims = hparams.n_dims
  transform_dims = (n_dims + 1)**2  # Using homogeneous coordinates.
  lr = hparams.lr
  level_set = hparams.level_set
  label_w = hparams.label_w
  minimal_w = hparams.minimal_w
  sample_vert = hparams.sample_vert
  sample_bbox = hparams.sample_bbox

  def _model_fn(features, labels, mode, params=None):
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    batch_size = features['point'].shape[0]
    n_sample_frames = features['point'].shape[1]
    accum_size = batch_size * n_sample_frames

    if params == 'gen_mesh':
      latent_output = tf.constant([0, 0, 0], dtype=tf.float32)
      latent_holder = tf.placeholder(tf.float32, latent_output.shape)

    # Decode the tranformed shapes and compute the losses
    with tf.variable_scope('shape/decode', reuse=tf.AUTO_REUSE):
      transform = tf.reshape(features['transform'],
                             [accum_size, n_parts, transform_dims])
      joint = tf.reshape(features['joint'], [accum_size, n_parts, n_dims])
      points = features['point']
      n_points = tf.shape(points)[2]
      points = tf.reshape(points, [accum_size, n_points, n_dims])

      if is_training:
        labels = tf.reshape(features['label'], [accum_size, n_points, 1])
        predictions, parts = model_utils.nasa_indicator(
            points, transform, joint, hparams, need_transformation=True)
        indicator_loss = model_utils.compute_l2_indicator_loss(
            labels, predictions)

        minimal_loss = tf.reduce_mean(tf.square(parts[..., :sample_bbox, :]))

        part_points = tf.reshape(features['vert'], [accum_size, -1, n_dims])
        part_weight = tf.reshape(features['weight'], [accum_size, -1, n_parts])
        if sample_vert > 0:  # If 0, use all vertices.
          n_vert = part_points.shape[1]
          sample_indices = tf.random.uniform([accum_size, sample_vert],
                                             minval=0,
                                             maxval=n_vert,
                                             dtype=tf.int32)
          part_points = tf.gather(
              part_points, sample_indices, axis=1, batch_dims=1)
          part_weight = tf.gather(
              part_weight, sample_indices, axis=1, batch_dims=1)
        unused_var, pred_parts = model_utils.nasa_indicator(
            part_points, transform, joint, hparams, need_transformation=True)
        part_label = tf.argmax(part_weight, axis=-1)
        part_label = tf.one_hot(
            part_label, depth=n_parts, axis=-1, dtype=tf.float32) * level_set
        part_label = tf.expand_dims(
            tf.transpose(part_label, [0, 2, 1]), axis=-1)
        label_loss = model_utils.compute_l2_indicator_loss(
            part_label, pred_parts)
      else:
        n_points = tf.shape(features['point'])[2]
        points = tf.reshape(features['point'], [accum_size, n_points, n_dims])
        predictions, parts = model_utils.nasa_indicator(
            points,
            transform,
            joint,
            hparams,
            need_transformation=True,
            noise=labels)

    if params == 'gen_mesh':
      return latent_holder, latent_output, tf.concat(
          [parts, tf.expand_dims(predictions, axis=1)], axis=1)

    tf.summary.scalar('indicator', indicator_loss)
    loss = indicator_loss
    if label_w > 0:
      tf.summary.scalar('label', label_loss)
      indicator_loss += label_loss * label_w
    if minimal_w > 0:
      tf.summary.scalar('minimal', minimal_loss)
      indicator_loss += minimal_loss * minimal_w

    global_step = tf.train.get_or_create_global_step()
    optimizer = tf.train.AdamOptimizer(lr)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(
          indicator_loss, global_step=global_step, name='optimizer_shape')

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  return _model_fn


model_dict = {
    'nasa': nasa,
}
