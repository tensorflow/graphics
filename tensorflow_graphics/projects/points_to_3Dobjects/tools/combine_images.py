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
"""Tool to combine  mulitple images into one."""

import os
import subprocess

from absl import app

path = ''


def main(_):

  names = os.listdir(os.path.join(path, 'img_3d_mean'))
  os.mkdir(os.path.join(path, 'combined_'))
  for name in names:
    in1 = os.path.join(path, 'img', name.split('_')[-1])
    in2 = os.path.join(path, 'img_2d_mean', name)
    in3 = os.path.join(path, 'img_3d_mean', name)
    cmd = ['convert', in1, in2, in3, '+append', os.path.join(path, 'combined_',
                                                             name)]
    subprocess.call(cmd)

if __name__ == '__main__':
  app.run(main)
