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
"""Image functions."""
# python3
from cvx2 import latest as cv2
import numpy as np


def get_affine_transform(center, scale, rot, output_size, inverse=False):
  """Affine transform."""
  if not isinstance(scale, (np.ndarray, list)):
    scale = np.array([scale, scale], dtype=np.float32)

  dst_w, dst_h = output_size[0], output_size[1]

  rot_rad = np.pi * rot / 180
  src_dir = get_dir([0, scale[0] * -0.5], rot_rad)
  dst_dir = np.array([0, dst_w * -0.5], np.float32)

  src = np.zeros((3, 2), dtype=np.float32)
  dst = np.zeros((3, 2), dtype=np.float32)
  src[0, :], src[1, :] = center, center + src_dir
  dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
  dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

  src[2:, :] = get_3rd_point(src[0, :], src[1, :])
  dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

  if inverse:
    transform = cv2.getAffineTransform(np.float32(dst), np.float32(src))
  else:
    transform = cv2.getAffineTransform(np.float32(src), np.float32(dst))

  return transform


def get_3rd_point(point_1, point_2):
  tmp_point = point_1 - point_2
  return point_2 + np.array([-tmp_point[1], tmp_point[0]], dtype=np.float32)


def get_dir(point, rot_rad):
  sin_rot, cos_rot = np.sin(rot_rad), np.cos(rot_rad)

  result = [0, 0]
  result[0] = point[0] * cos_rot - point[1] * sin_rot
  result[1] = point[0] * sin_rot + point[1] * cos_rot

  return np.array(result)


def transform_points(points, center, scale, output_size, inverse=False):
  transform = get_affine_transform(
      center, scale, 0, output_size, inverse=inverse)

  new_points = np.concatenate([points, np.ones([points.shape[0], 1])], axis=1)
  points_transformed = np.dot(transform, new_points.T).T
  return points_transformed


def transform_predictions(points, center, scale, output_size):
  return transform_points(points, center, scale, output_size, inverse=True)

