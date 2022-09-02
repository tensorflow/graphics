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
"""Tests for tensorflow_graphics.datasets.features.camera_feature."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow_graphics.datasets.features import camera_feature


class CameraFeatureTest(tfds.testing.FeatureExpectationsTestCase):
  """Test Cases for Camera FeatureConnector."""

  def __get_camera_params(self):
    pose = {'R': np.eye(3).astype(np.float32),
            't': np.zeros(3).astype(np.float32)}
    f = 35.
    optical_center = (640 / 2, 480 / 2)

    return pose, f, optical_center

  def test_simple_camera(self):
    """Tests camera parameters with fixed focal length, no skew and no aspect ratio."""

    expected_pose, expected_f, expected_center = self.__get_camera_params()

    expected_intrinsics = np.asarray([[expected_f, 0, expected_center[0]],
                                      [0, expected_f, expected_center[1]],
                                      [0, 0, 1]], dtype=np.float32)

    expected_camera = {'pose': expected_pose, 'intrinsics': expected_intrinsics}
    inputs = {'f': expected_f, 'optical_center': expected_center,
              'pose': expected_pose}
    lookat_inputs = {
        'f': expected_f,
        'optical_center': expected_center,
        'pose': {
            'look_at': np.array([0, 0, -1], dtype=np.float32),
            'up': np.array([0, 1, 0], dtype=np.float32),
            'position': np.array([0, 0, 0], dtype=np.float32)
        }
    }

    raising_pose_entry = {
        'f': expected_f,
        'optical_center': expected_center,
        'pose': np.eye(4)
    }

    raising_pose_inputs = {
        'f': expected_f,
        'optical_center': expected_center,
        'pose': {'rot': np.eye(3), 'trans': np.zeros(3)}
    }
    raising_lookat_inputs = {
        'f': expected_f,
        'optical_center': expected_center,
        'pose': {
            'l': np.array([0, 0, -1], dtype=np.float32),
            'up': np.array([0, 1, 0], dtype=np.float32),
            'C': np.array([0, 0, 0], dtype=np.float32)
        }
    }

    self.assertFeature(
        feature=camera_feature.Camera(),
        shape={
            'pose': {
                'R': (3, 3),
                't': (3,)
            },
            'intrinsics': (3, 3)
        },
        dtype={
            'pose': {
                'R': tf.float32,
                't': tf.float32
            },
            'intrinsics': tf.float32
        },
        tests=[
            tfds.testing.FeatureExpectationItem(
                value=inputs,
                expected=expected_camera,
            ),
            tfds.testing.FeatureExpectationItem(
                value=lookat_inputs,
                expected=expected_camera
            ),
            tfds.testing.FeatureExpectationItem(
                value=raising_pose_inputs,
                raise_cls=ValueError,
                raise_msg='Wrong keys for pose feature provided'
            ),
            tfds.testing.FeatureExpectationItem(
                value=raising_lookat_inputs,
                raise_cls=ValueError,
                raise_msg='Wrong keys for pose feature provided'
            ),
            tfds.testing.FeatureExpectationItem(
                value=raising_pose_entry,
                raise_cls=ValueError,
                raise_msg='Pose needs to be a dictionary'
            ),
        ],
    )

  def test_camera_with_aspect_ratio_and_skew(self):
    """Tests camera parameters with fixed focal length, aspect_ratio and skew."""

    expected_pose, expected_f, expected_center = self.__get_camera_params()
    expected_aspect_ratio = expected_center[0] / expected_center[1]
    expected_skew = 0.6
    expected_intrinsics = np.asarray(
        [[expected_f, expected_skew, expected_center[0]],
         [0, expected_aspect_ratio * expected_f, expected_center[1]],
         [0, 0, 1]], dtype=np.float32)

    expected_camera = {'pose': expected_pose, 'intrinsics': expected_intrinsics}
    inputs = {'f': expected_f,
              'optical_center': expected_center,
              'skew': expected_skew,
              'aspect_ratio': expected_aspect_ratio,
              'pose': expected_pose}

    self.assertFeature(
        feature=camera_feature.Camera(),
        shape={
            'pose': {
                'R': (3, 3),
                't': (3,)
            },
            'intrinsics': (3, 3)
        },
        dtype={
            'pose': {
                'R': tf.float32,
                't': tf.float32
            },
            'intrinsics': tf.float32
        },
        tests=[
            tfds.testing.FeatureExpectationItem(
                value=inputs,
                expected=expected_camera,
            ),
        ],
    )

  def test_full_camera_calibration_matrix(self):
    """Tests camera parameters with different focal length per camera axis and skew."""

    expected_pose, _, expected_optical_center = self.__get_camera_params()
    expected_skew = 0.6
    expected_f = (35., 40.)
    expected_intrinsics = np.array(
        [[expected_f[0], expected_skew, expected_optical_center[0]],
         [0, expected_f[1], expected_optical_center[1]],
         [0, 0, 1]], dtype=np.float32)

    expected_camera = {'pose': expected_pose, 'intrinsics': expected_intrinsics}
    inputs = {'f': expected_f,
              'optical_center': expected_optical_center,
              'skew': expected_skew, 'pose': expected_pose}

    raising_inputs = {'f': expected_f,
                      'aspect_ratio': 1.5,
                      'optical_center': expected_optical_center,
                      'skew': expected_skew, 'pose': expected_pose}
    self.assertFeature(
        feature=camera_feature.Camera(),
        shape={
            'pose': {
                'R': (3, 3),
                't': (3,)
            },
            'intrinsics': (3, 3)
        },
        dtype={
            'pose': {
                'R': tf.float32,
                't': tf.float32
            },
            'intrinsics': tf.float32
        },
        tests=[
            tfds.testing.FeatureExpectationItem(
                value=inputs,
                expected=expected_camera,
            ),
            tfds.testing.FeatureExpectationItem(
                value=raising_inputs,
                raise_cls=ValueError,
                raise_msg='If aspect ratio is provided, f needs to '
                          'be a single float',
            ),
        ],
    )


if __name__ == '__main__':
  tfds.testing.test_main()
