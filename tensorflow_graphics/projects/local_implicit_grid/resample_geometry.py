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
"""Compute point samples from a mesh after normalizing its scale."""

import os

from absl import app
from absl import flags
import numpy as np
from tensorflow_graphics.projects.local_implicit_grid.core import point_utils as pu
import trimesh

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

flags.DEFINE_string('input_mesh', '',
                    'Input geometry file. Must be a trimesh supported type')
flags.DEFINE_string('output_ply', '', 'Samples points ply file.')
flags.DEFINE_float('sampling_density', 2e-4,
                   'Approx surface area based point sampling density.')

FLAGS = flags.FLAGS


def normalize_mesh(mesh, in_place=True):
  """Rescales vertex positions to lie inside unit cube."""
  scale = 1.0 / np.max(mesh.bounds[1, :] - mesh.bounds[0, :])
  centroid = mesh.centroid
  scaled_vertices = (mesh.vertices - centroid) * scale
  if in_place:
    scaled_mesh = mesh
    scaled_mesh.vertices = scaled_vertices
  else:
    scaled_mesh = mesh.copy()
    scaled_mesh.vertices = scaled_vertices
  scaled_mesh.fix_normals()
  return scaled_mesh


def sample_mesh(mesh):
  """Samples oriented points from a mesh."""
  num_samples = int(mesh.area / FLAGS.sampling_density)
  sample_pts, sample_face_ids = trimesh.sample.sample_surface(mesh, num_samples)
  sample_normals = mesh.face_normals[sample_face_ids]
  return sample_pts, sample_normals


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  if not FLAGS.input_mesh:
    raise IOError('--input_mesh must be specified.')
  if not FLAGS.output_ply:
    raise IOError('--output_ply must be specified.')

  mesh = trimesh.load(FLAGS.input_mesh)
  mesh = normalize_mesh(mesh)
  sample_pts, sample_normals = sample_mesh(mesh)
  print('Computed {} samples from mesh.'.format(sample_pts.shape[0]))
  print('Writing sampled points to {}'.format(FLAGS.output_ply))
  pu.write_point_ply(FLAGS.output_ply, sample_pts, sample_normals)


if __name__ == '__main__':
  app.run(main)
