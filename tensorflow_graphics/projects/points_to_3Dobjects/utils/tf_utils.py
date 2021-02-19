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
"""Utility functions for TensorFlow."""

import numpy as np
import tensorflow as tf


def gaussian_blur(img, sigma=5):
  """Applies gaussian blur to the given image.

  Args:
    img: The input image.
    sigma: The gaussian kernel size.

  Returns:
    Gaussian blurred image.
  """
  kernel_size = 2 * sigma
  def gauss_kernel(channels, kernel_size, sigma):
    ax = tf.range(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
    xx, yy = tf.meshgrid(ax, ax)
    kernel = tf.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    kernel = kernel / tf.reduce_sum(kernel)
    kernel = tf.tile(kernel[..., tf.newaxis], [1, 1, channels])
    return kernel

  gaussian_kernel = gauss_kernel(tf.shape(img)[-1], kernel_size, sigma)
  gaussian_kernel = gaussian_kernel[..., tf.newaxis]
  blurred_image = tf.nn.depthwise_conv2d(tf.expand_dims(img, axis=0),
                                         gaussian_kernel, [1, 1, 1, 1],
                                         padding='SAME', data_format='NHWC')
  return tf.squeeze(blurred_image, axis=0)


def euler_from_rotation_matrix(matrix: tf.Tensor, axis: int) -> tf.float32:
  """Extracts the euler angle of a 3D rotation around axis.

  Args:
    matrix: The 3D input rotation matrix.
    axis: The rotation axis.

  Returns:
    The euler angle in radians around the specified axis.
  """
  tf.debugging.assert_integer(axis)
  tf.debugging.assert_less_equal(axis, 2)
  tf.debugging.assert_greater_equal(axis, 0)
  tf.debugging.assert_equal(matrix.shape, [3, 3])
  mask = np.ones((3, 3), dtype=bool)
  mask[axis, :] = False
  mask[:, axis] = False
  matrix2d = tf.reshape(tf.boolean_mask(matrix, mask), [2, 2])
  a = matrix2d[0, 1] if axis == 1 else matrix2d[1, 0]
  euler_angle = tf.math.atan2(a, matrix2d[0, 0])
  return euler_angle


def compute_dot(image_size: tf.Tensor,
                intrinsics: tf.Tensor,
                extrinsics: tf.Tensor,
                axis=1,
                image_intersection=(0.5, 0.75)) -> tf.Tensor:
  """Computes the intersection at the ground plane of a ray from the camera.

  Args:
    image_size: The size of the camera image.
    intrinsics: The camera intrinsics matrix. Shape: (3, 3)
    extrinsics: The camera extrinsics matrix. Shape: (3, 4)
    axis: The ground plane corresponds to the plane defined by axis = 0.
    image_intersection: The relative image position of the ray intersection.

  Returns:
    The intersection. Shape: (3, 1)
  """
  # Shoot ray through image pixel
  ray_2d = tf.cast(image_size, tf.float32) * [image_intersection[0],
                                              image_intersection[1],
                                              1/image_size[-1]]
  ray_2d = tf.reshape(ray_2d, [3, 1])

  # Invert intrinsics matrix K
  k = tf.reshape(intrinsics, [3, 3])
  k_inv = tf.linalg.inv(k)

  # Decompose extrinsics into rotation and translation, and inverte
  rt = tf.reshape(extrinsics, [3, 4])
  r = tf.gather(rt, [0, 1, 2], axis=1)
  t = tf.gather(rt, [3], axis=1)

  r_inv = tf.linalg.inv(r)
  t_inv = r_inv @ t * (-1)

  # Compute ray intersection with the ground plane along specified axis
  ray = r_inv @ k_inv @ ray_2d
  l = t_inv[axis] * -1 / ray[axis]  # determine lambda
  l = tf.expand_dims(l, -1)

  # this is the same
  dot = ray * l + t_inv
  return dot


def get_next_sample_dataset(dataset_iter):
  """Get next sample."""
  try:
    sample = next(dataset_iter)
  except (StopIteration, RuntimeError) as e:
    if "Can't copy Tensor with type" in str(e):
      sample = None
    elif isinstance(e, StopIteration):
      sample = None
    else:
      raise e
  return sample


def get_devices(gpu_ids):
  """Get device."""
  if gpu_ids is not None:
    gpu_ids = [f'/gpu:{gpu}' for gpu in gpu_ids.split(',')]
    cpu_ids = [
        f'/cpu:{x.name.split(":")[-1]}'
        for x in tf.config.list_physical_devices('CPU')
    ]
    device_ids = [*cpu_ids, *gpu_ids]
  else:
    device_ids = None
  return device_ids


def using_multigpu():
  multigpu = False
  if tf.distribute.has_strategy():
    strategy = tf.distribute.get_strategy()
    if strategy.num_replicas_in_sync > 1:
      multigpu = True
  return multigpu


def compute_batch_size(tensor_dict):
  """Compute batch size."""
  if using_multigpu():
    dummy_tensor = next(iter(tensor_dict.values())).values
    batch_size = 0
    for ii in range(len(dummy_tensor)):
      batch_size += tf.shape(dummy_tensor[ii])[0]
  else:
    dummy_tensor = next(iter(tensor_dict.values()))
    batch_size = tf.shape(dummy_tensor)[0]

  return batch_size
