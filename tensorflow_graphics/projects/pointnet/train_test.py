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
"""Tests of pointnet module."""

import sys
import tensorflow as tf

from tensorflow_graphics.util import test_case


def make_fake_batch(dimension_1, dimension_2):
  points = tf.random.normal((dimension_1, dimension_2, 3))
  label = tf.random.uniform((dimension_1,), minval=0, maxval=40, dtype=tf.int32)
  return points, label


class TrainTest(test_case.TestCase):

  def test_train(self):
    sys.argv = ["train.py", "--dryrun", "--assert_gpu", "False"]
    from tensorflow_graphics.projects.pointnet import train  # pylint: disable=import-outside-toplevel, unused-import, g-import-not-at-top
    for _ in range(2):
      batch = make_fake_batch(32, 1024)
      train.train(batch)


if __name__ == "__main__":
  test_case.main()
