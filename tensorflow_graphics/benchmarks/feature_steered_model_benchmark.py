"""
Benchmarking script for various feature_steered_convolution implementations.

Runs training operation on models from `notebooks/mesh_segmentation_demo.ipynb`
with differing convolution kwargs. Reported memory usage/timings are for the
entire model, not just the convolution implementations.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import tensorflow as tf

from tensorflow_graphics.geometry.convolution import utils
from tensorflow_graphics.nn.layer import graph_convolution as graph_conv
from tensorflow_graphics.notebooks import mesh_segmentation_dataio as dataio

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_integer(
    'num_filters', help='number of filters (M in paper, W in code)', default=8)
flags.DEFINE_bool(
    'mem_only', help='memory efficient implementations only', default=False)

path_to_data_zip = tf.keras.utils.get_file(
    'data.zip',
    origin='https://storage.googleapis.com/tensorflow-graphics/notebooks/mesh_segmentation/data.zip',
    extract=True)

test_data_files = [
    os.path.join(
        os.path.dirname(path_to_data_zip),
        'data/Dancer_test_sequence.tfrecords')
]

MODEL_PARAMS = {
    'num_filters': 8,
    'num_classes': 16,
    'encoder_filter_dims': [32, 64, 128],
    'learning_rate': 1e-3,
    'beta': 0.9,
    'adam_epsilon': 1e-8,
    'preprocess_neighbors': True
}


def mesh_encoder(
        batch_mesh_data, num_filters, output_dim, conv_layer_dims, conv_kwargs,
        preprocess_neighbors=True):
  """A mesh encoder using feature steered graph convolutions.

    The shorthands used below are
      `B`: Batch size.
      `V`: The maximum number of vertices over all meshes in the batch.
      `D`: The number of dimensions of input vertex features, D=3 if vertex
        positions are used as features.

  Args:
    batch_mesh_data: A mesh_data dict with following keys
      'vertices': A [B, V, D] `float32` tensor of vertex features, possibly
        0-padded.
      'neighbors': A [B, V, V] `float32` sparse tensor of edge weights.
      'num_vertices': A [B] `int32` tensor of number of vertices per mesh.
    num_filters: The number of weight matrices to be used in feature steered
      graph conv.
    output_dim: A dimension of output per vertex features.
    conv_layer_dims: A list of dimensions used in graph convolution layers.

  Returns:
    vertex_features: A [B, V, output_dim] `float32` tensor of per vertex
      features.
  """
  batch_vertices = batch_mesh_data['vertices']
  neighbors = batch_mesh_data['neighbors']
  num_vertices = batch_mesh_data['num_vertices']

  # Linear: N x D --> N x 16.
  vertex_features = tf.keras.layers.Conv1D(16, 1, name='lin16')(batch_vertices)

  if preprocess_neighbors:
    num_vertices_square = tf.stack((num_vertices, num_vertices), axis=-1)
    neighbors = utils.convert_to_block_diag_2d(neighbors, num_vertices_square)
    sizes = None
    vertex_features, unflatten = utils.flatten_batch_to_2d(
      vertex_features, num_vertices)
  else:
    sizes = num_vertices
    unflatten = None

  # graph convolution layers
  for dim in conv_layer_dims:
    with tf.variable_scope('conv_%d' % dim):
      vertex_features = graph_conv.feature_steered_convolution_layer(
          vertex_features,
          neighbors,
          sizes=sizes,
          num_weight_matrices=num_filters,
          num_output_channels=dim,
          **conv_kwargs)
    vertex_features = tf.nn.relu(vertex_features)

  if unflatten is not None:
    vertex_features = unflatten(vertex_features)
  # Linear: N x 128 --> N x 256.
  vertex_features = tf.keras.layers.Conv1D(
      256, 1, name='lin256')(
          vertex_features)
  vertex_features = tf.nn.relu(vertex_features)

  # Linear: N x 256 --> N x output_dim.
  vertex_features = tf.keras.layers.Conv1D(
      output_dim, 1, name='lin_output')(
          vertex_features)

  return vertex_features


def model_fn(features, labels, mode, params):
  """Returns a mesh segmentation model_fn for use with tf.Estimator."""
  logits = mesh_encoder(features, params['num_filters'], params['num_classes'],
                        params['encoder_filter_dims'],
                        params.get('conv_kwargs'),
                        params.get('preprocess_neighbors', True))
  predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
  outputs = {
      'vertices': features['vertices'],
      'triangles': features['triangles'],
      'num_vertices': features['num_vertices'],
      'num_triangles': features['num_triangles'],
      'predictions': predictions,
  }
  # For predictions, return the outputs.
  if mode == tf.estimator.ModeKeys.PREDICT:
    outputs['labels'] = features['labels']
    return tf.estimator.EstimatorSpec(mode=mode, predictions=outputs)
  # Loss
  # Weight the losses by masking out padded vertices/labels.
  vertex_ragged_sizes = features['num_vertices']
  mask = tf.sequence_mask(vertex_ragged_sizes, tf.shape(labels)[-1])
  loss_weights = tf.cast(mask, dtype=tf.float32)
  loss = tf.losses.sparse_softmax_cross_entropy(
      logits=logits, labels=labels, weights=loss_weights)
  # For training, build the optimizer.
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer(
        learning_rate=params['learning_rate'],
        beta1=params['beta'],
        epsilon=params['adam_epsilon'])
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(
          loss=loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # For eval, return eval metrics.
  eval_ops = {
      'mean_loss':
          tf.metrics.mean(loss),
      'accuracy':
          tf.metrics.accuracy(
              labels=labels, predictions=predictions, weights=loss_weights)
  }
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_ops)

test_io_params = {
    'is_training': False,
    'sloppy': False,
    'shuffle': True,
    'repeat': False
}
test_tfrecords = test_data_files


def run_benchmark(conv_kwargs, **kwargs):
    with tf.Graph().as_default():
        features, labels = dataio.create_input_from_dataset(
            dataio.create_dataset_from_tfrecords, test_tfrecords, test_io_params)
        params = MODEL_PARAMS.copy()
        params['conv_kwargs'] = conv_kwargs
        params.update(kwargs)
        spec = model_fn(features, labels, tf.estimator.ModeKeys.TRAIN, params)
        init = tf.compat.v1.global_variables_initializer()

        print('--------------')
        for k in sorted(conv_kwargs):
            print('{:10s}: {}'.format(k, conv_kwargs[k]))
        with tf.Session() as sess:
            sess.run(init)
            bm = tf.test.Benchmark()
            result = bm.run_op_benchmark(sess, spec.train_op)
    return result


def main(args):
  num_filters = flags.FLAGS.num_filters
  # v1_p2d is the original implementation
  fast = dict(memory_efficient=False)
  names, kwargs = zip(*(
      ('v1_p2d', dict(version='v1', segment_sum_impl='partition2d', **fast)),
      ('v1_sorted', dict(version='v1', segment_sum_impl='sorted', **fast)),
      ('v1_unsorted', dict(version='v1', segment_sum_impl='unsorted', **fast)),
      ('v1_p2d_mem', dict(version='v1', segment_sum_impl='partition2d')),
      ('v1_sorted_mem', dict(version='v1', segment_sum_impl='sorted')),
      ('v1_unsorted_mem', dict(version='v1', segment_sum_impl='unsorted')),
      ('v2', dict(version='v2')),  # will be same as one of the below
      ('v3', dict(version='v3', **fast)),
      ('v3_mem', dict(version='v3')),
  ))
  if FLAGS.mem_only:
    names, kwargs = zip(*(
        (name, kw) for name, kw in zip(names, kwargs) if 'mem' in name))
  times = []
  memories = []
  for kw in kwargs:
    result = run_benchmark(kw, num_filters=num_filters)
    times.append(result['wall_time'])
    memories.append(
          result['extras']['allocator_maximum_num_bytes_GPU_0_bfc'])

  print('*************')
  print('** SUMMARY **')
  print('*************')
  print('{:15s}: {}'.format('num_filters', num_filters))

  times = np.array(times)
  # ti = np.argmin(times)
  ti = 0
  tmin = times[ti]
  print('Baseline time: {}, {}s'.format(names[ti], tmin))
  print('rel times:')
  for name, time in zip(names, times):
      print('{:15s} {:.3f}'.format(name, time / tmin))
  memories = np.array(memories)
  # mi = np.argmin(memories)
  mi = 0
  mmin = memories[mi]
  print('Baseline memory: {}, {}mb'.format(names[mi], mmin / 1024**2))
  for name, memory in zip(names, memories):
      print('{:15s} {:.3f}'.format(name, memory / mmin))


if __name__ == '__main__':
  app.run(main)
