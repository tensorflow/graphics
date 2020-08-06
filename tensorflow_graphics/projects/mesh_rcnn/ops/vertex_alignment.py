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
