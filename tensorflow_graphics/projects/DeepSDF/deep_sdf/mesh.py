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
""" NO COMMENT NOW"""


import time
import logging
import skimage.measure
import numpy as np

import tensorflow as tf
import plyfile
import deep_sdf.utils

# TODO:
# numpy version
# this is slower than torch.tensor(cpu)


def create_mesh(
    decoder, latent_vec, filename,
    n=256, max_batch=32 ** 3, offset=None, scale=None
):
  start = time.time()
  ply_filename = filename

  # decoder.eval()

  # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
  voxel_origin = [-1, -1, -1]
  voxel_size = 2.0 / (n - 1)

  overall_index = np.arange(0, n ** 3, 1, dtype=np.int64)
  samples = np.zeros([n ** 3, 4])

  # transform first 3 columns
  # to be the x, y, z index
  samples[:, 2] = overall_index % n
  samples[:, 1] = (overall_index / n) % n
  samples[:, 0] = ((overall_index / n) / n) % n

  # transform first 3 columns
  # to be the x, y, z coordinate
  samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
  samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
  samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

  num_samples = n ** 3

  # CHECK
  # samples.requires_grad = False

  head = 0

  while head < num_samples:
    sample_subset = tf.convert_to_tensor(samples[head: min(
        head + max_batch, num_samples), 0:3])

    samples[head: min(head + max_batch, num_samples), 3] = (
        tf.squeeze(
            deep_sdf.utils.decode_sdf(
                decoder, latent_vec, sample_subset), axis=1)
    ).numpy()
    head += max_batch

  sdf_values = samples[:, 3]
  sdf_values = sdf_values.reshape(n, n, n)

  end = time.time()
  print("sampling takes: %f" % (end - start))

  convert_sdf_samples_to_ply(
      sdf_values,
      voxel_origin,
      voxel_size,
      ply_filename + ".ply",
      offset,
      scale,
  )


def convert_sdf_samples_to_ply(
    numpy_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    offset=None,
    scale=None,
):
  """
  Convert sdf samples to .ply

  :param np_3d_sdf_tensor: a np.ndarray(float) of shape (n,n,n)
  :voxel_grid_origin: a list of three floats:\
    the bottom, left, down origin of the voxel grid
  :voxel_size: float, the size of the voxels
  :ply_filename_out: string, path of the filename to save to

  This function adapted from: https://github.com/RobotLocomotion/spartan
  """
  start_time = time.time()

  verts, faces, _, _ = skimage.measure.marching_cubes_lewiner(
      numpy_3d_sdf_tensor, level=0.0, spacing=[voxel_size] * 3
  )

  # transform from voxel coordinates to camera coordinates
  # note x and y are flipped in the output of marching_cubes
  mesh_points = np.zeros_like(verts)
  mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
  mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
  mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

  # apply additional offset and scale
  if scale is not None:
    mesh_points = mesh_points / scale
  if offset is not None:
    mesh_points = mesh_points - offset

  # try writing to the ply file

  num_verts = verts.shape[0]
  num_faces = faces.shape[0]

  verts_tuple = np.zeros((num_verts,),
                         dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

  for i in range(0, num_verts):
    verts_tuple[i] = tuple(mesh_points[i, :])

  faces_building = []
  for i in range(0, num_faces):
    faces_building.append(((faces[i, :].tolist(),)))
  faces_tuple = np.array(faces_building,
                         dtype=[("vertex_indices", "i4", (3,))])

  el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
  el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

  ply_data = plyfile.PlyData([el_verts, el_faces])
  logging.debug("saving mesh to %s", ply_filename_out)
  ply_data.write(ply_filename_out)

  logging.debug(
      "converting to ply format and writing to file took %s s",
      time.time() - start_time
  )
