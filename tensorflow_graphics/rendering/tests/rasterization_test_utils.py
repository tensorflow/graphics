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
"""Util functions for rasterization tests."""

import numpy as np

from tensorflow_graphics.geometry.transformation import look_at
from tensorflow_graphics.rendering.camera import perspective


def make_perspective_matrix(image_width=None, image_height=None):
  """Generates perspective matrix for a given image size.

  Args:
    image_width: int representing image width.
    image_height: int representing image height.

  Returns:
    Perspective matrix, tensor of shape [4, 4].

  Note: Golden tests require image size to be fixed and equal to the size of
  golden image examples. The rest of the camera parameters are set such that
  resulting image will be equal to the baseline image.
  """

  field_of_view = (40 * np.math.pi / 180,)
  near_plane = (0.01,)
  far_plane = (10.0,)
  return perspective.right_handed(field_of_view,
                                  (float(image_width) / float(image_height),),
                                  near_plane, far_plane)


def make_look_at_matrix(
    camera_origin=(0.0, 0.0, 0.0), look_at_point=(0.0, 0.0, 0.0)):
  """Shortcut util function to creat model-to-eye matrix for tests."""
  camera_up = (0.0, 1.0, 0.0)
  return look_at.right_handed(camera_origin, look_at_point, camera_up)
