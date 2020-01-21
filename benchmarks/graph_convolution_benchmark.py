from absl import logging
from absl import app, flags
import numpy as np
import tensorflow as tf
from tensorflow_graphics.nn.layer.graph_convolution import \
  FeatureSteeredConvolutionKerasLayer
from tensorflow_graphics.geometry.convolution.graph_convolution import \
  SparseImplementation

logging.info('Finished imports')

Lambda = tf.keras.layers.Lambda
Input = tf.keras.Input

flags.DEFINE_boolean('jit', default=False, help='use XLA jit compilation')
flags.DEFINE_boolean('sparse', default=False, help='use sparse implementation')
flags.DEFINE_boolean('sort', default=False, help='use sorted indices')
flags.DEFINE_boolean(
    'backward', default=False, help='benchmark forward and backward pass')
flags.DEFINE_integer(
    'burn_iters', default=10, help='number of burn in iterations')
flags.DEFINE_integer('nv', default=100000, help='number of vertices')
flags.DEFINE_integer('ne',
           default=-1,
           help='number of edges, -1 will result in using 10*nv')
flags.DEFINE_integer('min_iters',
           default=20,
           help='minimum number of iterations to benchmark')
flags.DEFINE_integer(
    'num_weight_matrices', default=8, help='number of weight matrices')
flags.DEFINE_integer(
    'num_output_channels', default=32, help='number of output channels')
flags.DEFINE_integer('num_layers', default=10, help='number of layers')


def summarize(result, print_fn=print):
  """
  Args:
    result: output of a tf.test.Benchmark.run_op_benchmark call.
    print_fn: print-like function.
  """
  print_fn('Wall time (ms): {}'.format(result['wall_time'] * 1000))
  gpu_mem = result['extras'].get('allocator_maximum_num_bytes_GPU_0_bfc', 0)
  print_fn('Memory (Mb):  {}'.format(gpu_mem / 1024**2))


def get_data(num_vertices, num_edges, sort=True):
  if num_edges == -1:
    num_edges = 10 * num_vertices
  vertices = np.random.uniform(size=(num_vertices, 3)).astype(np.float32)
  # replace=False below gives memory issues
  indices = np.random.choice(num_vertices**2, num_edges, replace=True)
  if sort:
    indices.sort()
  i, j = np.unravel_index(indices, (num_vertices, num_vertices))  # pylint: disable=unbalanced-tuple-unpacking

  counts = np.zeros((num_vertices,), dtype=np.int64)
  for ii in i:
    counts[ii] += 1
  weights = (1. / counts)[i].astype(np.float32)
  indices = np.stack((i, j), axis=-1)

  return vertices, indices, weights


def main(_):
  FLAGS = flags.FLAGS
  tf.config.optimizer.set_jit(FLAGS.jit)
  tf.keras.backend.clear_session()
  vertices, indices, weights = get_data(FLAGS.nv, FLAGS.ne, sort=FLAGS.sort)
  nv = vertices.shape[0]

  with tf.Graph().as_default():
    vertices = tf.constant(vertices, dtype=tf.float32)
    indices = tf.constant(indices, dtype=tf.int64)
    weights = tf.constant(weights, dtype=tf.float32)
    # batch size of 1
    vertices, indices, weights = (
        tf.expand_dims(t, axis=0) for t in (vertices, indices, weights))

    data = Input(tensor=vertices)
    indices = Input(tensor=indices)
    weights = Input(tensor=weights)
    inputs = (data, indices, weights)
    data, indices, weights = tuple(
      Lambda(tf.squeeze, arguments=dict(axis=0), name='squeeze{}'.format(i))(t)
      for i, t in enumerate(inputs))
    nv = Lambda(lambda x: tf.shape(x, out_type=tf.int64)[0])(data)

    neighbors = Lambda(
        lambda args: tf.SparseTensor(args[0], args[1], (args[2], args[2])))(
            [indices, weights, nv])

    for _ in range(FLAGS.num_layers):
      layer = FeatureSteeredConvolutionKerasLayer(
          sparse_impl=(
              SparseImplementation.SPARSE_MATMUL if FLAGS.sparse else
              SparseImplementation.GATHER_SUM),
          num_weight_matrices=FLAGS.num_weight_matrices,
          num_output_channels=FLAGS.num_output_channels)
      data = layer([data, neighbors])
      data = tf.nn.relu(data)

    pred = data
    output = Lambda(tf.expand_dims, arguments=dict(axis=0))(pred)
    model = tf.keras.Model(inputs=inputs, outputs=output)

    if FLAGS.backward:
      loss = tf.reduce_sum(pred)
      optimizer = tf.keras.optimizers.SGD()

      model_weights = model.trainable_weights

      grads = optimizer.get_gradients(loss, model_weights)
      grads_and_vars = tuple(zip(grads, model_weights))
      train_op = optimizer.apply_gradients(grads_and_vars)
    else:
      train_op = pred

    bm = tf.test.Benchmark()
    with tf.compat.v1.Session() as sess:
      logging.info('Initializing variables...')

      sess.run(tf.compat.v1.global_variables_initializer())

      logging.info('Starting benchmarking...')
      result = bm.run_op_benchmark(
          sess, train_op, burn_iters=FLAGS.burn_iters,
          min_iters=FLAGS.min_iters)
    summarize(result)


if __name__ == '__main__':
  app.run(main)
