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
"""`tensorflow_graphics.datasets.features` API defining feature types."""

from tensorflow_graphics.datasets.features.camera_feature import Camera
from tensorflow_graphics.datasets.features.pose_feature import Pose
from tensorflow_graphics.datasets.features.trimesh_feature import TriangleMesh
from tensorflow_graphics.datasets.features.voxel_feature import VoxelGrid

__all__ = [
    "TriangleMesh",
    "VoxelGrid",
    "Camera",
    "Pose"
]
