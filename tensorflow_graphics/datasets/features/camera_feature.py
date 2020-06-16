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
# Lint as: python3
"""Camera feature."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_datasets import features

from tensorflow_graphics.datasets.features import pose_feature


class Camera(features.FeaturesDict):
  """`FeatureConnector` for camera calibration (extrinsic and intrinsic).

  During `_generate_examples`, the feature connector accepts as input:

    * `parameter_dict:` A dictionary containing the extrinsic and instrinsic
    parameters of the camera as:
      - 'rotation': 3x3 rotation matrix with float32 values.
      - 'translation': 3x1 translation vector.
      - 'f': focal length of the camera in mm (either single float32 value
      or tuple of float32 as (f_x, f_y).
      - 'optical_center': Optical center of the camera
      in pixel coordinates as tuple (c_x, c_y)
      Optional parameters:
      - 'skew': float32 denoting the skew of the camera axes.
      - 'aspect_ratio': float32 denoting the aspect_ratio,
      if single fixed focal length is provided.


  Output:
    A dictionary containing:

    * 'pose': A `tensorflow_graphics.datasets.features.Pose` FeatureConnector
    representing the 3D pose of the camera.
    * 'K': A `float32` tensor with shape `[3,3]` denoting the intrinsic matrix.

  Example:
    Default values for skew (s) and aspect_ratio(a) are 0 and 1, respectively.

    Full calibration matrix:
      K = [[ f_x,   s, c_x ],
           [   0, f_y, c_y ],
           [   0,   0,  1  ]]

    With same focal length:
      K = [[ f,  s, c_x ],
           [ 0, af, c_y ],
           [ 0,  0,  1  ]]
  """

  def __init__(self):
    super(Camera, self).__init__({
      'pose': pose_feature.Pose(),
      'K': features.Tensor(shape=(3, 3), dtype=tf.float32),
    })

  def encode_example(self, parameter_dict):
    """Convert the given parameters into a dict convertible to tf example."""
    REQUIRED_KEYS = ['R', 't', 'f', 'optical_center']
    if not all(key in parameter_dict for key in REQUIRED_KEYS):
      raise ValueError(f"Missing keys in provided dictionary! "
                       f"Expected {REQUIRED_KEYS}, "
                       f"but {parameter_dict.keys()} were given.")

    features_dict = {'pose': self._feature_dict['pose'].encode_example({
      'R': parameter_dict['R'],
      't': parameter_dict['t']
    })}
    aspect_ratio = 1
    skew = 0
    if 'aspect_ratio' in parameter_dict.keys():
      if not isinstance(parameter_dict['f'], float):
        raise ValueError("If aspect ratio is provided, "
                         "f needs to be a single float.")
      aspect_ratio = parameter_dict['aspect_ratio']

    if 'skew' in parameter_dict.keys():
      skew = parameter_dict['skew']

    features_dict['K'] = self._create_calibration_matrix(
      parameter_dict['f'],
      parameter_dict['optical_center'],
      aspect_ratio,
      skew
    )

    return super(Camera, self).encode_example(features_dict)

  def _create_calibration_matrix(self, f, optical_center, aspect_ratio=1,
                                 skew=0):
    """Constructs the 3x3 calibration matrix K.

    Args:
      f: Focal length of the camera. Either single float.32 value or tuple of
      float32 when different focal lengths for each axis are provided (f_x, f_y).
      optical_center: Tuple (c_x, c_y) containing the optical center
      of the camera in pixel coordinates.
      aspect_ratio: Optional parameter, if fixed focal length for both
      dimensions is used. Defaults to 1.
      skew: Optional parameter denoting the skew between the camera axes.

    Returns:
      float32 Tensor of shape [3,3] containing the upper triangular
      calibration matrix K.
    """
    if not isinstance(optical_center, tuple):
      raise ValueError("Optical center of camera needs "
                       "to be a tuple of (c_x, c_y).")

    if isinstance(f, tuple):
      f_x, f_y = f
    else:
      f_x = f
      f_y = aspect_ratio * f

    return np.asarray([[f_x, skew, optical_center[0]],
                       [0, f_y, optical_center[1]],
                       [0, 0, 1]
                       ], dtype=np.float32)
