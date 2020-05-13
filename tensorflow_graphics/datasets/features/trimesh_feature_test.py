# Copyright 2020 Google LLC
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
"""Tests for tensorflow_graphics.datasets.features.trimesh_feature."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
from tensorflow_graphics.datasets.features import trimesh_feature
import trimesh

_TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'test_data')


class TrimeshFeatureTest(tfds.testing.FeatureExpectationsTestCase):

  def test_trimesh(self):
    obj_file_path = os.path.join(_TEST_DATA_DIR, 'cube.obj')
    obj_mesh = trimesh.load(obj_file_path)
    expected_vertices = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0],
                                  [0.0, 1.0, 0.0], [0.0, 1.0, 1.0],
                                  [1.0, 0.0, 0.0], [1.0, 0.0, 1.0],
                                  [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    expected_faces = np.array(
        [[0, 6, 4], [0, 2, 6], [0, 3, 2], [0, 1, 3], [2, 7, 6], [2, 3, 7],
         [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [1, 5, 7], [1, 7, 3]],
        dtype=np.uint64)
    expected_trimesh = {'vertices': expected_vertices, 'faces': expected_faces}
    # Create a scene with two cubes.
    scene = trimesh.Scene()
    scene.add_geometry(obj_mesh)
    scene.add_geometry(obj_mesh)
    # The expected TriangleFeature for the scene.
    expected_scene_feature = {
        'vertices':
            np.tile(expected_vertices, [2, 1]).astype(np.float32),
        'faces':
            np.concatenate(
                [expected_faces, expected_faces + len(expected_vertices)],
                axis=0)
    }
    self.assertFeature(
        feature=trimesh_feature.TriangleMesh(),
        shape={
            'vertices': (None, 3),
            'faces': (None, 3)
        },
        dtype={
            'vertices': tf.float32,
            'faces': tf.uint64
        },
        tests=[
            # File path
            tfds.testing.FeatureExpectationItem(
                value=obj_file_path,
                expected=expected_trimesh,
            ),
            # Trimesh
            tfds.testing.FeatureExpectationItem(
                value=obj_mesh,
                expected=expected_trimesh,
            ),
            # Scene
            tfds.testing.FeatureExpectationItem(
                value=scene,
                expected=expected_scene_feature,
            ),
            # FeaturesDict
            tfds.testing.FeatureExpectationItem(
                value=expected_scene_feature,
                expected=expected_scene_feature,
            ),
            # Invalid type
            tfds.testing.FeatureExpectationItem(
                value=np.random.rand(80, 3),
                raise_cls=ValueError,
                raise_msg='obj should be either a Trimesh or a Scene',
            ),
        ],
    )


if __name__ == '__main__':
  tfds.testing.test_main()
