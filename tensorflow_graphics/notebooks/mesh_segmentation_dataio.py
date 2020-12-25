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
"""Dataset Pipeline for mesh_segmentation_demo.ipynb.

 The shorthands used in parameter descriptions below are
    'B': Batch size.
    'E': Number of unique directed edges in a mesh.
    'V': Number of vertices in a mesh.
    'T': Number of triangles in a mesh.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow_graphics.geometry.convolution import utils as conv_utils
from tensorflow_graphics.geometry.representation.mesh import utils as mesh_utils
from tensorflow_graphics.util import shape

DEFAULT_IO_PARAMS = {
    'batch_size': 8,
    'shuffle_buffer_size': 100,
    'is_training': True,
    'parallel_threads': 5,
    'mean_center': True,
    'shuffle': None,
    'repeat': None,
}


def adjacency_from_edges(edges, weights, num_edges, num_vertices):
  """Returns a batched sparse 1-ring adj tensor from edge list tensor.

  Args:
    edges: [B, E, 2] `int32` tensor of edges, possibly 0 padded.
    weights: [B, E] `float32` tensor of edge weights, possibly 0 padded.
    num_edges: [B] `int32` tensor of number of valid edges per batch sample.
    num_vertices: [B] `int32` tensor of number of valid vertices per batch
      sample.

  Returns:
    adj: A batched SparseTensor of weighted adjacency graph, of
      dense_shape [B, V, V] where V is max(num_vertices)
  """
  edges = tf.convert_to_tensor(value=edges)
  weights = tf.convert_to_tensor(value=weights)
  num_edges = tf.convert_to_tensor(value=num_edges)
  num_vertices = tf.convert_to_tensor(value=num_vertices)

  if not edges.dtype.is_integer:
    raise TypeError("'edges' must have an integer type.")
  if not num_edges.dtype.is_integer:
    raise TypeError("'num_edges' must have an integer type.")
  if not num_vertices.dtype.is_integer:
    raise TypeError("'num_vertices' must have an integer type.")
  if not weights.dtype.is_floating:
    raise TypeError("'weights' must have a floating type.")

  shape.check_static(tensor=edges, tensor_name='edges', has_rank=3)
  shape.check_static(tensor=weights, tensor_name='weights', has_rank=2)
  shape.check_static(tensor=num_edges, tensor_name='num_edges', has_rank=1)
  shape.check_static(
      tensor=num_vertices, tensor_name='num_vertices', has_rank=1)
  shape.compare_dimensions(
      tensors=(edges, weights, num_edges, num_vertices),
      tensor_names=('edges', 'weights', 'num_edges', 'num_vertices'),
      axes=(-3, -2, -1, -1))
  shape.compare_dimensions(
      tensors=(edges, weights),
      tensor_names=('edges', 'weights'),
      axes=(-2, -1))

  batch_size = tf.shape(input=edges)[0]
  max_num_vertices = tf.reduce_max(input_tensor=num_vertices)
  max_num_edges = tf.shape(input=edges)[1]
  batch_col = tf.reshape(tf.range(batch_size, dtype=edges.dtype), [-1, 1, 1])
  batch_col = tf.tile(batch_col, [1, max_num_edges, 1])
  batch_edges = tf.concat([batch_col, edges], axis=-1)

  indices, _ = conv_utils.flatten_batch_to_2d(batch_edges, sizes=num_edges)
  values, _ = conv_utils.flatten_batch_to_2d(
      tf.expand_dims(weights, -1), sizes=num_edges)
  values = tf.squeeze(values)
  adjacency = tf.SparseTensor(
      indices=tf.cast(indices, tf.int64),
      values=values,
      dense_shape=[batch_size, max_num_vertices, max_num_vertices])
  adjacency = tf.sparse.reorder(adjacency)
  return adjacency


def get_weighted_edges(faces, self_edges=True):
  r"""Gets unique edges and degree weights from a triangular mesh.

  The shorthands used below are:
      `T`: The number of triangles in the mesh.
      `E`: The number of unique directed edges in the mesh.

  Args:
    faces: A [T, 3] `int32` numpy.ndarray of triangle vertex indices.
    self_edges: A `bool` flag. If true, then for every vertex 'i' an edge
      [i, i] is added to edge list.
  Returns:
    edges: A  [E, 2] `int32` numpy.ndarray of directed edges.
    weights: A [E] `float32` numpy.ndarray denoting edge weights.

    The degree of a vertex is the number of edges incident on the vertex,
    including any self-edges. The weight for an edge $w_{ij}$ connecting vertex
    $v_i$ and vertex $v_j$ is defined as,
    $$
    w_{ij} = 1.0 / degree(v_i)
    \sum_{j} w_{ij} = 1
    $$
  """
  edges = mesh_utils.extract_unique_edges_from_triangular_mesh(
      faces, directed_edges=True).astype(np.int32)
  if self_edges:
    vertices = np.expand_dims(np.unique(edges[:, 0]), axis=1)
    self_edges = np.concatenate((vertices, vertices), axis=1)
    edges = np.unique(np.concatenate((edges, self_edges), axis=0), axis=0)
  weights = mesh_utils.get_degree_based_edge_weights(edges, dtype=np.float32)
  return edges, weights


def _tfrecords_to_dataset(tfrecords,
                          parallel_threads,
                          shuffle,
                          repeat,
                          sloppy,
                          max_readers=16):
  """Creates a TFRecordsDataset that iterates over filenames in parallel.

  Args:
    tfrecords: A list of tf.Data.TFRecords filenames.
    parallel_threads: The `int` number denoting number of parallel worker
      threads.
    shuffle: The `bool` flag denoting whether to shuffle the dataset.
    repeat: The `bool` flag denoting whether to repeat the dataset.
    sloppy: The `bool` flag denoting if elements are produced in deterministic
      order.
    max_readers: The `int` number denoting the maximum number of input tfrecords
      to interleave from in parallel.

  Returns:
    A tf.data.TFRecordDataset
  """

  total_tfrecords = sum([len(tf.io.gfile.glob(f)) for f in tfrecords])
  num_readers = min(total_tfrecords, max_readers)
  dataset = tf.data.Dataset.list_files(tfrecords, shuffle=shuffle)
  if repeat:
    dataset = dataset.repeat()
  return dataset.apply(
      tf.data.experimental.parallel_interleave(
          tf.data.TFRecordDataset,
          num_readers,
          sloppy=sloppy,
          buffer_output_elements=parallel_threads,
          prefetch_input_elements=parallel_threads))


def _parse_tfex_proto(example_proto):
  """Parses the tfexample proto to a raw mesh_data dictionary.

  Args:
    example_proto: A tf.Example proto storing the encoded mesh data.

  Returns:
    A mesh data dictionary with the following fields:
      'num_vertices': The `int64` number of vertices in mesh.
      'num_triangles': The `int64` number of triangles in mesh.
      'vertices': A serialized tensor of vertex positions.
      'triangles': A serialized tensor of triangle vertex indices.
      'labels': A serialized tensor of per vertex class labels.
  """
  feature_description = {
      'num_vertices': tf.io.FixedLenFeature([], tf.int64, default_value=0),
      'num_triangles': tf.io.FixedLenFeature([], tf.int64, default_value=0),
      'vertices': tf.io.FixedLenFeature([], tf.string, default_value=''),
      'triangles': tf.io.FixedLenFeature([], tf.string, default_value=''),
      'labels': tf.io.FixedLenFeature([], tf.string, default_value=''),
  }
  return tf.io.parse_single_example(
      serialized=example_proto, features=feature_description)


def _parse_mesh_data(mesh_data, mean_center=True):
  """Parses a raw mesh_data dictionary read from tf examples.

  Args:
    mesh_data: A mesh data dictionary with serialized data tensors,
      as output from _parse_tfex_proto()
    mean_center: If true, centers the mesh vertices to mean(vertices).
  Returns:
     A mesh data dictionary with following fields:
      'num_vertices': The `int32` number of vertices in mesh.
      'num_triangles': The `int32` number of triangles in mesh.
      'num_edges': The `int32` number of unique directed edges in mesh.
      'vertices': A [V, 3] `float32` of vertex positions.
      'triangles': A [T, 3] `int32` tensor of triangle vertex indices.
      'labels': A [V] `int32` tensor of per vertex class labels.
      'edges': A [E, 2] `int32` tensor of unique directed edges in mesh.
      'edge_weights': A [E] `float32` tensor of vertex degree based edge
        weights.
  """
  labels = tf.io.parse_tensor(mesh_data['labels'], tf.int32)
  vertices = tf.io.parse_tensor(mesh_data['vertices'], tf.float32)
  triangles = tf.io.parse_tensor(mesh_data['triangles'], tf.int32)
  if mean_center:
    vertices = vertices - tf.reduce_mean(
        input_tensor=vertices, axis=0, keepdims=True)

  edges, weights = tf.py_function(
      func=lambda t: get_weighted_edges(t.numpy()),
      inp=[triangles],
      Tout=[tf.int32, tf.float32])

  num_edges = tf.shape(input=edges)[0]
  num_vertices = tf.cast(mesh_data['num_vertices'], tf.int32)
  num_triangles = tf.cast(mesh_data['num_triangles'], tf.int32)
  mesh_data = dict(
      vertices=vertices,
      labels=labels,
      triangles=triangles,
      edges=edges,
      edge_weights=weights,
      num_triangles=num_triangles,
      num_vertices=num_vertices,
      num_edges=num_edges)
  return mesh_data


def create_dataset_from_tfrecords(tfrecords, params):
  """Creates a mesh dataset given a list of tf records filenames.

  Args:
    tfrecords: A list of TFRecords filenames.
    params: A dictionary of IO paramaters, see DEFAULT_IO_PARAMS.
  Returns:
    A tf.data.Dataset, with each element a dictionary of batched mesh data
      with following fields:
      'vertices': A [B, V, 3] `float32` tensor of vertex positions, possibly
        0-padded.
      'triangles': A [B, T, 3] `int32` tensor of triangle vertex indices,
        possibly 0-padded
      'labels': A [B, V] `int32` tensor of per vertex class labels, possibly
        0-padded
      'edges': A [B, E, 2] `int32` tensor of unique directed edges in mesh,
        possibly 0-padded
      'edge_weights': A [B, E] `float32` tensor of vertex degree based edge
        weights, possibly 0-padded.
      'num_edges': A [B] `int32` tensor of number of unique directed edges in
        each mesh in the batch.
      'num_vertices':  A [B] `int32` tensor of number of vertices in each mesh
        in the batch.
      'num_triangles': A [B] `int32` tensor of number of triangles in each mesh
        in the batch.
  """

  def _set_default_if_none(param, param_dict, default_val):
    if param not in param_dict:
      return default_val
    else:
      return default_val if param_dict[param] is None else param_dict[param]

  is_training = params['is_training']
  shuffle = _set_default_if_none('shuffle', params, is_training)
  repeat = _set_default_if_none('repeat', params, is_training)
  sloppy = _set_default_if_none('sloppy', params, is_training)

  if not isinstance(tfrecords, list):
    tfrecords = [tfrecords]
  dataset = _tfrecords_to_dataset(tfrecords, params['parallel_threads'],
                                  shuffle, repeat, sloppy)
  dataset = dataset.map(_parse_tfex_proto, tf.data.experimental.AUTOTUNE)
  dataset = dataset.map(
      lambda x: _parse_mesh_data(x, mean_center=params['mean_center']),
      tf.data.experimental.AUTOTUNE)
  if repeat:
    dataset = dataset.repeat()
  if shuffle:
    dataset = dataset.shuffle(params['shuffle_buffer_size'])
  return dataset.padded_batch(
      params['batch_size'],
      padded_shapes={
          'vertices': [None, 3],
          'labels': [None],
          'triangles': [None, 3],
          'edges': [None, 2],
          'edge_weights': [None],
          'num_edges': [],
          'num_vertices': [],
          'num_triangles': [],
      },
      drop_remainder=is_training)


def create_input_from_dataset(dataset_fn, files, io_params):
  """Creates input function given dataset generator and input files.

  Args:
    dataset_fn: A dataset generator function.
    files: A list of TFRecords filenames.
    io_params: A dictionary of IO paramaters, see DEFAULT_IO_PARAMS.
  Returns:
    features: A dictionary of mesh data training features.
    labels: A [B] `int32` tensor of per vertex class labels.
  """
  for k in DEFAULT_IO_PARAMS:
    io_params[k] = io_params[k] if k in io_params else DEFAULT_IO_PARAMS[k]

  dataset = dataset_fn(files, io_params)
  mesh_data = tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()
  mesh_data['neighbors'] = adjacency_from_edges(mesh_data['edges'],
                                                mesh_data['edge_weights'],
                                                mesh_data['num_edges'],
                                                mesh_data['num_vertices'])

  max_num_verts = tf.reduce_max(input_tensor=mesh_data['num_vertices'])
  features = dict(
      vertices=tf.reshape(mesh_data['vertices'], [-1, max_num_verts, 3]),
      triangles=mesh_data['triangles'],
      neighbors=mesh_data['neighbors'],
      num_triangles=mesh_data['num_triangles'],
      num_vertices=mesh_data['num_vertices'])
  labels = mesh_data['labels']
  # Copy labels to features dictionary for estimator prediction mode.
  if not io_params['is_training']:
    features['labels'] = mesh_data['labels']
  return features, labels
