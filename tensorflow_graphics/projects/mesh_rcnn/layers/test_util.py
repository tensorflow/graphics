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
"""Utility functions for testing of Mesh R-CNN layers."""

import tensorflow as tf

from tensorflow_graphics.projects.mesh_rcnn.structures.mesh import Meshes


def get_mesh_input_data(batch_size=1,
                        vertex_features_dim=128,
                        generate_vert_features=False):
  """Generates valid input data."""
  tf.random.set_seed(42)

  image_features = []
  intrinsics = []
  vertices = []
  faces = []
  for _ in range(batch_size):
    image_features.append(tf.reshape(tf.range(20.), (4, 5, 1)))
    intrinsics.append(tf.constant([[10, 0, 2.5], [0, 10, 2.5], [0, 0, 1]],
                                  dtype=tf.float32))
    vertices.append(tf.constant([[-1.5, -1.5, 10],
                                 [-1.5, 0.5, 10],
                                 [-0.5, -1.5, 10],
                                 [-0.5, 0.5, 10]], dtype=tf.float32))
    faces.append(tf.constant([[0, 1, 3], [0, 2, 3]], dtype=tf.int32))

  vert_features = None
  if generate_vert_features:
    vert_features = tf.random.normal((sum([v.shape[0] for v in vertices]),
                                      vertex_features_dim))

  return {
      'feature': tf.stack(image_features, 0),
      'mesh': Meshes(vertices, faces),
      'intrinsics': tf.stack(intrinsics, 0),
      'vertex_features': vert_features
  }


def calc_conv_out_spatial_shape(in_width,
                                in_height,
                                kernel_size=3,
                                stride=1,
                                padding=0):
  """Computes the output size of a Conv2D layer."""
  out_width = ((in_width - kernel_size + 2 * padding)/stride) + 1
  out_height = ((in_height - kernel_size + 2 * padding)/stride) + 1
  return int(out_width), int(out_height)


def calc_deconv_out_spatial_shape(in_width,
                                  in_height,
                                  kernel_size,
                                  stride,
                                  padding=0):
  """Computes the output size of a Conv2DTranspose layer."""
  out_width = stride * (in_width - 1) + kernel_size - 2 * padding
  out_height = stride * (in_height - 1) + kernel_size - 2 * padding
  return int(out_width), int(out_height)
