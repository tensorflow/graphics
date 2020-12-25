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
"""Mesh of a flat rectangular surface."""

import numpy as np

vertices = (
    (-1.8, 0.0, 0.0),
    (-1.6, 0.0, 0.0),
    (-1.4, 0.0, 0.0),
    (-1.2, 0.0, 0.0),
    (-1.0, 0.0, 0.0),
    (-0.8, 0.0, 0.0),
    (-0.6, 0.0, 0.0),
    (-0.4, 0.0, 0.0),
    (-0.2, 0.0, 0.0),
    (0.0, 0.0, 0.0),
    (0.2, 0.0, 0.0),
    (0.4, 0.0, 0.0),
    (0.6, 0.0, 0.0),
    (0.8, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (1.2, 0.0, 0.0),
    (1.4, 0.0, 0.0),
    (1.6, 0.0, 0.0),
    (1.8, 0.0, 0.0),
    (-1.8, 1.0, 0.0),
    (-1.6, 1.0, 0.0),
    (-1.4, 1.0, 0.0),
    (-1.2, 1.0, 0.0),
    (-1.0, 1.0, 0.0),
    (-0.8, 1.0, 0.0),
    (-0.6, 1.0, 0.0),
    (-0.4, 1.0, 0.0),
    (-0.2, 1.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.2, 1.0, 0.0),
    (0.4, 1.0, 0.0),
    (0.6, 1.0, 0.0),
    (0.8, 1.0, 0.0),
    (1.0, 1.0, 0.0),
    (1.2, 1.0, 0.0),
    (1.4, 1.0, 0.0),
    (1.6, 1.0, 0.0),
    (1.8, 1.0, 0.0),
)
vertices = np.array(vertices)

faces = (
    (0, 1, 19),
    (1, 20, 19),
    (1, 2, 20),
    (2, 21, 20),
    (2, 3, 21),
    (3, 22, 21),
    (3, 4, 22),
    (4, 23, 22),
    (4, 5, 23),
    (5, 24, 23),
    (5, 6, 24),
    (6, 25, 24),
    (6, 7, 25),
    (7, 26, 25),
    (7, 8, 26),
    (8, 27, 26),
    (8, 9, 27),
    (9, 28, 27),
    (9, 10, 28),
    (10, 29, 28),
    (10, 11, 29),
    (11, 30, 29),
    (11, 12, 30),
    (12, 31, 30),
    (12, 13, 31),
    (13, 32, 31),
    (13, 14, 32),
    (14, 33, 32),
    (14, 15, 33),
    (15, 34, 33),
    (15, 16, 34),
    (16, 35, 34),
    (16, 17, 35),
    (17, 36, 35),
    (17, 18, 36),
    (18, 37, 36),
)
faces = np.array(faces)

mesh = {'vertices': vertices, 'faces': faces}
