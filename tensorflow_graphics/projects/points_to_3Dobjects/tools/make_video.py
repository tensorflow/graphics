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
"""Make video."""
import os
import subprocess

from absl import app
from absl import flags
from absl import logging

FLAGS = flags.FLAGS
flags.DEFINE_string('logdir', '', 'Path to log directory.')


def images_to_video(input_dir, output_path, sequences, input_pattern,
                    framerate=5):
  """Convert images to vide."""
  for sequence_id in range(len(sequences)):
    start = str(sequences[sequence_id][0])
    end = str(sequences[sequence_id][1] - sequences[sequence_id][0])
    try:
      path = os.path.join(output_path, 'video_' + str(sequence_id) + '.gif')
      subprocess.run([
          'ffmpeg', '-y', '-framerate', str(framerate),
          '-start_number', start,
          '-i', '"' + input_dir + '/' + input_pattern + '"',
          '-frames:v', end, path], check=True)
    except ValueError:
      logging.info('Could not process video.')


def chairs():
  """Chair."""
  sequences = [[0, 25],
               [26, 54],
               [55, 80],
               [80, 112],
               [113, 139],
               [140, 155],
               [156, 190],
               [191, 218],
               [219, 247],
               [248, 277],
               [278, 304],
               [305, 332],
               [333, 361],
               [362, 389],
               [390, 416],
               [417, 446],
               [981, 1005]]
  input_dir = os.path.join(FLAGS.logdir, 'images')
  output_dir = os.path.join(FLAGS.logdir, 'videos')
  images_to_video(input_dir, output_dir, sequences, '%05d.png')


def triplets():
  sequences = [[0, 10-1],
               [10, 20-1],
               [20, 30-1],
               [30, 40-1]]
  input_dir = os.path.join(FLAGS.logdir, 'images')
  output_dir = os.path.join(FLAGS.logdir, 'videos')
  images_to_video(input_dir, output_dir, sequences, 'val-%05d.png', framerate=1)


def main(_):
  # chairs()
  triplets()


if __name__ == '__main__':
  app.run(main)
