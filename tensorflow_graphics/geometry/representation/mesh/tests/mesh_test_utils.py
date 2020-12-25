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
"""Helper routines for mesh unit tests.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def create_single_triangle_mesh():
  r"""Creates a single-triangle mesh, in the z=0 plane and facing +z.

  (0,1) 2
        |\
        | \
        |  \
  (0,0) 0---1 (1,0)

  Returns:
    vertices: A [3, 3] float array
    faces: A [1, 3] int array
  """
  vertices = np.array(
      ((0, 0, 0), (1, 0, 0), (0, 1, 0)), dtype=np.float32)
  faces = np.array(((0, 1, 2),), dtype=np.int32)
  return vertices, faces


def create_square_triangle_mesh():
  r"""Creates a square mesh, in the z=0 planse and facing +z.

    # (0,1) 2---3 (1,1)
    #       |\ /|
    #       | 4 |
    #       |/ \|
    # (0,0) 0---1 (1,0)

  Returns:
    vertices: A [5, 3] float array
    faces: A [4, 3] int array
  """
  vertices = np.array(
      ((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0.5, 0.5, 0)),
      dtype=np.float32)
  faces = np.array(
      ((0, 1, 4), (1, 3, 4), (3, 2, 4), (2, 0, 4)), dtype=np.int32)
  return vertices, faces
