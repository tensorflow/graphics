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
"""Mesh of the simplified tensorflow-graphics logo."""

import numpy as np

vertices = (
    (1, -1, -2),
    (1, -2, -2),
    (1, -2, 2),
    (1, -1, 2),
    (1, -1, 1),
    (1, 1, 1),
    (1, 1, 2),
    (1, 2, 2),
    (1, 2, -1),
    (1, 1, -1),
    (2, -1, -2),
    (2, -2, -2),
    (2, -2, 2),
    (2, -1, 2),
    (2, -1, 1),
    (2, 1, 1),
    (2, 1, 2),
    (2, 2, 2),
    (2, 2, -1),
    (2, 1, -1),
    (-1, -2, -2),
    (-1, -2, -1),
    (-1, 2, -1),
    (-1, 2, -2),
    (-1, 1, -1),
    (-1, 1, 2),
    (-1, 2, 2),
    (-2, -2, -2),
    (-2, -2, -1),
    (-2, 2, -1),
    (-2, 2, -2),
    (-2, 1, -1),
    (-2, 1, 2),
    (-2, 2, 2),
    (-1, -1, -2),
    (-1, -1, -1),
    (1, -1, -1),
    (1, -2, -1),
)
vertices = np.array(vertices)
vertices[:, 2] = -vertices[:, 2]

faces = (
    # Right piece
    (2, 1, 0),
    (3, 2, 0),
    (3, 4, 6),
    (5, 6, 4),
    (7, 6, 8),
    (9, 8, 6),
    (10, 11, 12),
    (12, 13, 10),
    (14, 13, 16),
    (16, 15, 14),
    (16, 17, 18),
    (18, 19, 16),
    (0, 1, 10),
    (1, 11, 10),
    (0, 14, 4),
    (14, 0, 10),
    (1, 2, 12),
    (1, 12, 11),
    (4, 14, 5),
    (5, 14, 15),
    (5, 19, 9),
    (19, 5, 15),
    (9, 19, 18),
    (18, 8, 9),
    (8, 18, 17),
    (17, 7, 8),
    (7, 12, 2),
    (12, 7, 17),

    # Left piece
    (20, 21, 22),
    (22, 23, 20),
    (22, 24, 25),
    (26, 22, 25),
    (28, 27, 29),
    (30, 29, 27),
    (31, 29, 32),
    (29, 33, 32),
    (20, 28, 21),
    (28, 20, 27),
    (20, 30, 27),
    (20, 23, 30),
    (23, 33, 30),
    (33, 23, 26),
    (26, 25, 33),
    (25, 32, 33),
    (25, 24, 32),
    (24, 31, 32),
    (24, 21, 31),
    (21, 28, 31),

    # central piece
    (1, 0, 20),
    (20, 0, 34),
    (0, 35, 34),
    (36, 35, 0),
    (36, 37, 21),
    (35, 36, 21),
    (1, 21, 37),
    (20, 21, 1),
)
faces = np.array(faces)

mesh = {'vertices': vertices, 'faces': faces}
