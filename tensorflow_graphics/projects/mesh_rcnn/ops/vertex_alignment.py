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
Implementation of the VertAlign operation for Mesh R-CNN.

This operation is also called 'perceptual feature pooling' in Wang et al.

Given the 3D coordinate of a
vertex, this OP calculates its 2D projection on input image plane using camera
intrinsics, and then pool the feature from four nearby pixels using
bilinear interpolation.

References:
  * Georgia Gkioxari, Jitendra Malik, & Justin Johnson. (2019). Mesh R-CNN.
  * Nanyang Wang, Yinda Zhang, Zhuwen Li, Yanwei Fu, Wei Liu, & Yu-Gang Jiang.
    (2018). Pixel2Mesh: Generating 3D Mesh Models from Single RGB Images.
"""

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow_graphics.rendering.camera import perspective


def vert_align(features,
               vertices,
               intrinsics):
  """
  Sample vertex features from a feature map.

  Args:
    features: float32 tensor of shape `[N, H, W, C]` representing image features
      from which to sample the features. N is a batch dimension.
    vertices: list of N float32 tensors of shape `[V, 3]` containing the vertex
      positions for which to sample the image features.
    intrinsics: float32 tensor of shape `[N, 3, 3]` representing the intrinsic
      camera matrix.

  Returns:
    float32 tensor of shape `[N, V, C]` containing sampled features per vertex.
  """

  features = tf.convert_to_tensor(features)
  intrinsics = tf.convert_to_tensor(intrinsics)

  if not all([v.shape.rank == 2 for v in vertices]):
    raise ValueError('vertices should be 2 dimensional.')

  if not tf.rank(features) == 4:
    raise ValueError('features must of shape (N, H, W, C).')


  padded_grid, padding_offsets = pad_query_points(vertices)

  focal_length = tf.linalg.diag_part(intrinsics)[..., :2]
  principal_point = tf.squeeze(tf.gather(intrinsics, [2], axis=1)[:2])
  projected_vertices = perspective.project(padded_grid,
                                           focal_length,
                                           principal_point)
  sampled_features = tfa.image.interpolate_bilinear(features,
                                                    projected_vertices,
                                                    indexing='ij')

  return [s_feat[:padding_offsets[i]] for i, s_feat in
          enumerate(sampled_features)]


def pad_query_points(points):
  """
  Pads and stacks query points into a tensor of shape `[N, P, 3]`

  Args:
    points: list of N float32 tensors of shape `[P, 3]` containing the
      coordinates for which to sample the image features.

  Returns:
    float32 tensor of shape `[N, P, 3]` with the padded and stacked query points
    and a list of ints denoting the lengths of the unpadded arrays.
  """
  max_num_points = len(max(points, key=len))
  padded_vertices = []
  original_lengths = []
  for point in points:
    if point.shape[-1] != 3:
      raise ValueError('points.shape[-1] has to be 3.')

    original_lengths.append(point.shape[0])
    pad_length = max_num_points - point.shape[0]
    pad = tf.zeros((pad_length, 3), dtype=tf.float32)
    if pad_length == 0:
      padded_vertices.append(point)
    else:
      padded_vertices.append(tf.concat([point, pad], 0))

  return tf.stack(padded_vertices), original_lengths
