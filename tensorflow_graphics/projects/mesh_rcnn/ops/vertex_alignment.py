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
"""
Implementation of the vert align operation for Mesh R-CNN.

This operation is also called 'perceptual feature pooling' in Wang et al.

Mesh R-CNN uses bilinear interpolation and border padding. Thus, this
implementation does not provide other algorithms.

References:
  * Georgia Gkioxari, Jitendra Malik, & Justin Johnson. (2019). Mesh R-CNN.
  * Nanyang Wang, Yinda Zhang, Zhuwen Li, Yanwei Fu, Wei Liu, & Yu-Gang Jiang.
    (2018). Pixel2Mesh: Generating 3D Mesh Models from Single RGB Images.
"""

import tensorflow as tf


def vert_align(features,
               vertices):
  """
  Sample vertex features from a feature map.

  Args:
    features: float32 tensor of shape `[N, H, W, C]` representing image features
      from which to sample the features. N is a batch dimension.
    vertices: list of N float32 tensors of shape `[V, 3]` containing the vertex
      positions for which to sample the image features.

  Returns:
    float32 tensor of shape `[N, V, C]` containing sampled features per vertex.
  """

  features = tf.convert_to_tensor(features)

  if not all(v.shape.rank == 2 for v in vertices):
    raise ValueError('vertices should be 2 dimensional.')

  if not features._rank() == 4:
    raise ValueError('features must be a Tensor of rank 4.')

  grid = tf.reshape()


def pad_vertices(vertices):
  """
  Pads and stacks vertices into a tensor of shape `[N, V, 3]`
  Args:
    vertices: list of N float32 tensors of shape `[V, 3]` containing the vertex
      positions for which to sample the image features.

  Returns:
    float32 tensor of shape `[N, V, 3]` with the padded and stacked vertices.
  """
  max_num_vertices = len(max(vertices, key=len))
  padded_vertices = []
  for vert in vertices:
    if vert.shape[1] != 3:
      raise ValueError('All vertices must have shape [V, 3].')

    pad_length = max_num_vertices - vert.shape[0]
    pad = tf.zeros((pad_length, 3), dtype=tf.float32)
    padded_vertices.append(tf.concat([vert, pad], 0))

  return tf.stack(padded_vertices)
