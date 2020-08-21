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
from tensorflow_graphics.util import shape


def _check_vert_align_inputs(features, vertices, intrinsics):
  """
  Validates shapes of the input tensors passed to vert align.

  Args:
    features: tensor with image features passed to vert align.
    vertices: tensor with vertices passed to vert align
    intrinsics: tensor with the intrinsic matrices passed to vert align.

  Raises:
    ValueError: if one of the input tensors has a wrong shape, rank or if the
      batch dimensions are different.

  """
  shape.compare_batch_dimensions([features, vertices, intrinsics],
                                 last_axes=[-4, -3, -3],
                                 tensor_names=['features', 'vertices', 'intrinsics'],
                                 broadcast_compatible=False)
  shape.check_static(features,
                     has_rank_greater_than=2,
                     tensor_name='features')
  shape.check_static(vertices,
                     has_rank_greater_than=1,
                     has_dim_equals=(-1, 3),
                     tensor_name='vertices')
  shape.check_static(intrinsics,
                     has_rank_greater_than=1,
                     has_dim_equals=[(-1, 3), (-2, 3)],
                     tensor_name='intrinsics')


def vert_align(features,
               vertices,
               intrinsics):
  """
  Sample vertex features from a feature map.

  Args:
    features: A float32 tensor of shape `[A1, ..., An, H, W, C]` representing
      image features from which to sample the features. A1, ..., An are optional
      batch dimensions.
    vertices: A float32 tensor of shape `[A1,..., An, V, 3]` containing the
      (padded) vertex positions for which to sample the image features.
    intrinsics: A float32 tensor of shape `[A1, ..., An, 3, 3]` representing the
      intrinsic camera matrices.

  Returns:
    A float32 tensor of shape `[A1, ..., An, V, C]` containing sampled features
    per vertex.
  """
  features = tf.convert_to_tensor(features)
  vertices = tf.convert_to_tensor(vertices)
  intrinsics = tf.convert_to_tensor(intrinsics)

  _check_vert_align_inputs(features, vertices, intrinsics)

  # flatten batch dimensions for interpolate_bilinear
  flat_features = tf.reshape(features, [-1] + features.shape[-3:].as_list())
  flat_vertices = tf.reshape(vertices, [-1] + vertices.shape[-2:].as_list())
  flat_intrinsics = tf.reshape(intrinsics, [-1, 3, 3])

  focal_length = tf.expand_dims(tf.linalg.diag_part(flat_intrinsics)[..., :2], -2)
  principal_point = tf.expand_dims(
      tf.squeeze(tf.gather(flat_intrinsics, [2], axis=-1)[:, :2]), -2)
  projected_vertices = perspective.project(flat_vertices,
                                           focal_length,
                                           principal_point)
  sampled_features = tfa.image.interpolate_bilinear(flat_features,
                                                    projected_vertices,
                                                    indexing='ij')

  return tf.reshape(sampled_features, vertices.shape[:-1].as_list() + [-1])
