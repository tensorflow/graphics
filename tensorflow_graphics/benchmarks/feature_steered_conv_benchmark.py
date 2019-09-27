"""
Benchmarking script for various feature_steered_convolution implementations.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import numpy as np
import tensorflow as tf
import tensorflow_graphics.geometry.convolution.graph_convolution as gc
from tensorflow_graphics.geometry.convolution.tests.graph_convolution_test \
    import _random_data, _random_variables

from absl import app
from absl import flags

flags.DEFINE_integer('batch_size', 8, help='size of batch')
flags.DEFINE_integer('num_vertices', 500, help='number of vertices')
flags.DEFINE_integer('in_channels', 32, help='number of input channels')
flags.DEFINE_integer('out_channels', 32, help='number of output channels')
flags.DEFINE_integer('num_filters',
                     8,
                     help='number of filters (W, or M in paper)')
flags.DEFINE_float('sparsity', 0.25, help='sparsity of neighbors')
flags.DEFINE_bool('mem_only',
                  default=False,
                  help='memory efficient implementations only')

FLAGS = flags.FLAGS


def main(args):
  random_state = np.random.RandomState(123)

  data, neighbors = _random_data(FLAGS.batch_size,
                                 FLAGS.num_vertices,
                                 FLAGS.in_channels,
                                 padding=False,
                                 only_self_edges=False,
                                 sparsity=FLAGS.sparsity,
                                 random_state=random_state)
  sizes = None
  data = tf.convert_to_tensor(value=data, dtype=tf.float32)

  u, v, c, w, b = _random_variables(FLAGS.in_channels,
                                    FLAGS.out_channels,
                                    FLAGS.num_filters,
                                    random_state=random_state)

  # v1_p2d is the original implementation
  fast = dict(memory_efficient=False)
  bad = dict(transform_data_first=FLAGS.in_channels <= FLAGS.out_channels)
  names, kwargs = zip(*(
      ('v1_p2d', dict(version='v1', segment_sum_impl='partition2d', **fast)),
      ('v1_p2d_bad', dict(
        version='v1', segment_sum_impl='partition2d', **fast, **bad)),
      ('v1_sorted', dict(version='v1', segment_sum_impl='sorted', **fast)),
      ('v1_unsorted', dict(version='v1', segment_sum_impl='unsorted', **fast)),
      ('v1_p2d_mem', dict(version='v1', segment_sum_impl='partition2d')),
      ('v1_sorted_mem', dict(version='v1', segment_sum_impl='sorted')),
      ('v1_unsorted_mem', dict(version='v1', segment_sum_impl='unsorted')),
      ('v2', dict(version='v2')),
      ('v2_bad', dict(version='v2', **bad)),
      ('v3', dict(version='v3', **fast)),
      ('v3_mem', dict(version='v3')),
      ('v3_bad', dict(version='v3', **bad, **fast)),
      ('v3_mem_bad', dict(version='v3', **bad)),
  ))
  if FLAGS.mem_only:
    names, kwargs = zip(*(
        (name, kw) for name, kw in zip(names, kwargs) if 'mem' in name))

  vals = [
      gc.feature_steered_convolution(data, neighbors, sizes, u, v, c, w, b,
                                     **kw) for kw in kwargs
  ]
  grads = [
      tf.gradients(val, (data, neighbors.values, u, v, c, w, b)) for val in vals
  ]

  errs = [tf.reduce_max(tf.abs(val - vals[0])) for val in vals[1:]]

  with tf.Session() as sess:
    errs = sess.run(errs)

    times = []
    memories = []
    for name, v, g in zip(names, vals, grads):
      print('------------')
      print(name)
      bm = tf.test.Benchmark()
      result = bm.run_op_benchmark(sess, (v, g))

      times.append(result['wall_time'])
      memories.append(result['extras']['allocator_maximum_num_bytes_GPU_0_bfc'])

  print('*************')
  print('** SUMMARY **')
  print('*************')
  print('{:15s}: {}'.format('batch_size', FLAGS.batch_size))
  print('{:15s}: {}'.format('num_vertices', FLAGS.num_vertices))
  print('{:15s}: {}'.format('in_channels', FLAGS.in_channels))
  print('{:15s}: {}'.format('out_channels', FLAGS.out_channels))
  print('{:15s}: {}'.format('num_filters', FLAGS.num_filters))
  print('{:15s}: {}'.format('sparsity', FLAGS.sparsity))

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

  print('Errors w.r.t {}'.format(names[0]))
  for name, err in zip(names[1:], errs):
    print('{:10s}: {}'.format(name, err))


if __name__ == '__main__':
  app.run(main)
