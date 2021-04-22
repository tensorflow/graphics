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


import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import tensorflow as tf

from PIL import Image

import im2mesh.common as common


def generate_images(tensor):
  images = tf.cast(tensor * 255.0, tf.uint8)
  row = []
  for i in range(images.shape[0]):
    row.append(images[i])
  images = tf.concat(row, axis=1)
  images = tf.image.encode_png(images)
  return images


def visualize_data(data, data_type, out_file):
  r""" Visualizes the data with regard to its type.

  Args:
      data (tensor): batch of data
      data_type (string): data type (img, voxels or pointcloud)
      out_file (string): output file
  """
  if data_type == "img":
    image = tf.cast(data * 255.0, tf.uint8)
    image = image.numpy()
    # if tf.rank(data) == 4:
    #     row = []
    #     for i in range(image.shape[0]):
    #         row.append(image[i])
    #     image = tf.concat(row, axis=1)

    Image.fromarray(image).save(out_file)
    # image = tf.image.encode_png(image)
    # with open(out_file, "wb") as fd:
    #     fd.write(image)

  elif data_type == "voxels":
    visualize_voxels(data, out_file=out_file)
  elif data_type == "pointcloud":
    visualize_pointcloud(data, out_file=out_file)
  elif data_type is None or data_type == "idx":
    pass
  else:
    raise ValueError('Invalid data_type "%s"' % data_type)


def visualize_voxels(voxels, out_file=None, show=False):
  r""" Visualizes voxel data.

  Args:
      voxels (tensor): voxel data
      out_file (string): output file
      show (bool): whether the plot should be shown
  """
  # Use numpy
  voxels = np.asarray(voxels)
  # Create plot
  fig = plt.figure()
  ax = fig.gca(projection=Axes3D.name)
  voxels = voxels.transpose(2, 0, 1)

  ax.voxels(voxels, edgecolor="k")
  ax.set_xlabel("Z")
  ax.set_ylabel("X")
  ax.set_zlabel("Y")
  ax.view_init(elev=30, azim=45)
  if out_file is not None:
    plt.savefig(out_file)
  if show:
    plt.show()
  plt.close(fig)


def visualize_pointcloud(points, normals=None, out_file=None, show=False):
  r""" Visualizes point cloud data.

  Args:
      points (tensor): point data
      normals (tensor): normal data (if existing)
      out_file (string): output file
      show (bool): whether the plot should be shown
  """
  # Use numpy
  points = points.numpy()
  # Create plot
  fig = plt.figure()
  ax = fig.gca(projection=Axes3D.name)
  ax.scatter(points[:, 2], points[:, 0], points[:, 1])
  if normals is not None:
    ax.quiver(
        points[:, 2],
        points[:, 0],
        points[:, 1],
        normals[:, 2],
        normals[:, 0],
        normals[:, 1],
        length=0.1,
        color="k",
    )
  ax.set_xlabel("Z")
  ax.set_ylabel("X")
  ax.set_zlabel("Y")
  ax.set_xlim(-0.5, 0.5)
  ax.set_ylim(-0.5, 0.5)
  ax.set_zlim(-0.5, 0.5)
  ax.view_init(elev=30, azim=45)
  if out_file is not None:
    plt.savefig(out_file)
  if show:
    plt.show()
  plt.close(fig)


def visualise_projection(
    points, world_mat, camera_mat, img, output_file="out.png"
):
  r""" Visualizes the transformation and projection to image plane.

      The first points of the batch are transformed and projected to the
      respective image. After performing the relevant transformations, the
      visualization is saved in the provided output_file path.

  Arguments:
      points (tensor): batch of point cloud points
      world_mat (tensor): batch of matrices to rotate pc to camera-based
              coordinates
      camera_mat (tensor): batch of camera matrices to project to 2D image
              plane
      img (tensor): tensor of batch GT image files
      output_file (string): where the output should be saved
  """
  points_transformed = common.transform_points(points, world_mat)
  points_img = common.project_to_camera(points_transformed, camera_mat)
  pimg2 = points_img[0].numpy()
  image = img[0].numpy()
  plt.imshow(image)
  plt.plot(
      (pimg2[:, 0] + 1) * image.shape[1] / 2,
      (pimg2[:, 1] + 1) * image.shape[2] / 2,
      "x",
  )
  plt.savefig(output_file)
