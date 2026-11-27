# copyright 2020 the tensorflow authors
#
# licensed under the apache license, version 2.0 (the "license");
# you may not use this file except in compliance with the license.
# you may obtain a copy of the license at
#
#    https://www.apache.org/licenses/license-2.0
#
# unless required by applicable law or agreed to in writing, software
# distributed under the license is distributed on an "as is" basis,
# without warranties or conditions of any kind, either express or implied.
# see the license for the specific language governing permissions and
# limitations under the license.
""" NO COMMENT NOW"""


import os
import glob
import random
import tensorflow as tf
from PIL import Image
import numpy as np
import trimesh
from im2mesh.data.core import Field
from im2mesh.utils import binvox_rw


class IndexField(Field):
  ''' Basic index field.'''

  def load(self, model_path, idx, category):
    ''' Loads the index field.

    Args:
        model_path (str): path to model
        idx (int): ID of data point
        category (int): index of category
    '''
    return idx

  def check_complete(self, files):
    ''' Check if field is complete.

    Args:
        files: files
    '''
    return True


class CategoryField(Field):
  ''' Basic category field.'''

  def load(self, model_path, idx, category):
    ''' Loads the category field.

    Args:
        model_path (str): path to model
        idx (int): ID of data point
        category (int): index of category
    '''
    return category

  def check_complete(self, files):
    ''' Check if field is complete.

    Args:
        files: files
    '''
    return True


class ImagesField(Field):
  ''' Image Field.

  It is the field used for loading images.

  Args:
      folder_name (str): folder name
      transform (list): list of transformations applied to loaded images
      extension (str): image extension
      random_view (bool): whether a random view should be used
      with_camera (bool): whether camera data should be provided
  '''

  def __init__(self, folder_name, transform=None,
               extension='jpg', random_view=True, with_camera=False):
    self.folder_name = folder_name
    self.transform = transform
    self.extension = extension
    self.random_view = random_view
    self.with_camera = with_camera

  def load(self, model_path, idx, category):
    ''' Loads the data point.

    Args:
        model_path (str): path to model
        idx (int): ID of data point
        category (int): index of category
    '''
    folder = os.path.join(model_path, self.folder_name)
    files = glob.glob(os.path.join(folder, '*.%s' % self.extension))
    if self.random_view:
      idx_img = random.randint(0, len(files)-1)
    else:
      idx_img = 0
    filename = files[idx_img]

    image = Image.open(filename).convert('RGB')
    image = np.array(image).astype(np.float32)
    image = tf.convert_to_tensor(image, np.float32)

    # image_r = tf.io.read_file(filename)
    # image = tf.image.decode_image(image_r, channels=3, dtype=tf.float32)

    if self.transform is not None:
      image = self.transform(image)

    data = {
        None: image
    }

    if self.with_camera:
      camera_file = os.path.join(folder, 'cameras.npz')
      camera_dict = np.load(camera_file)
      rt = camera_dict['world_mat_%d' % idx_img].astype(np.float32)
      k = camera_dict['camera_mat_%d' % idx_img].astype(np.float32)
      data['world_mat'] = rt
      data['camera_mat'] = k

    return data

  def check_complete(self, files):
    ''' Check if field is complete.

    Args:
        files: files
    '''
    complete = (self.folder_name in files)
    # TODO: check camera
    return complete


# 3D Fields
class PointsField(Field):
  ''' Point Field.

  It provides the field to load point data. This is used for the points
  randomly sampled in the bounding volume of the 3D shape.

  Args:
      file_name (str): file name
      transform (list): list of transformations which will be applied to the
          points tensor
      with_transforms (bool): whether scaling and rotation data should be
          provided

  '''

  def __init__(self, file_name, transform=None, with_transforms=False,
               unpackbits=False):
    self.file_name = file_name
    self.transform = transform
    self.with_transforms = with_transforms
    self.unpackbits = unpackbits

  def load(self, model_path, idx, category):
    ''' Loads the data point.

    Args:
        model_path (str): path to model
        idx (int): ID of data point
        category (int): index of category
    '''
    file_path = os.path.join(model_path, self.file_name)

    points_dict = np.load(file_path)
    points = points_dict['points']
    # Break symmetry if given in float16:
    if points.dtype == np.float16:
      points = points.astype(np.float32)
      points += 1e-4 * np.random.randn(*points.shape)
    else:
      points = points.astype(np.float32)

    occupancies = points_dict['occupancies']
    if self.unpackbits:
      occupancies = np.unpackbits(occupancies)[:points.shape[0]]
    occupancies = occupancies.astype(np.float32)

    data = {
        None: points,
        'occ': occupancies,
    }

    if self.with_transforms:
      data['loc'] = points_dict['loc'].astype(np.float32)
      data['scale'] = points_dict['scale'].astype(np.float32)

    if self.transform is not None:
      data = self.transform(data)

    for key, v in data.items():
      data[key] = tf.convert_to_tensor(v, np.float32)

    return data


class VoxelsField(Field):
  ''' Voxel field class.

  It provides the class used for voxel-based data.

  Args:
      file_name (str): file name
      transform (list): list of transformations applied to data points
  '''

  def __init__(self, file_name, transform=None):
    self.file_name = file_name
    self.transform = transform

  def load(self, model_path, idx, category):
    ''' Loads the data point.

    Args:
        model_path (str): path to model
        idx (int): ID of data point
        category (int): index of category
    '''
    file_path = os.path.join(model_path, self.file_name)

    with open(file_path, 'rb') as f:
      voxels = binvox_rw.read_as_3d_array(f)
    voxels = voxels.data.astype(np.float32)

    if self.transform is not None:
      voxels = self.transform(voxels)

    voxels = tf.convert_to_tensor(voxels, np.float32)

    return voxels

  def check_complete(self, files):
    ''' Check if field is complete.

    Args:
        files: files
    '''
    complete = (self.file_name in files)
    return complete


class PointCloudField(Field):
  ''' Point cloud field.

  It provides the field used for point cloud data. These are the points
  randomly sampled on the mesh.

  Args:
      file_name (str): file name
      transform (list): list of transformations applied to data points
      with_transforms (bool): whether scaling and rotation dat should be
          provided
  '''

  def __init__(self, file_name, transform=None, with_transforms=False):
    self.file_name = file_name
    self.transform = transform
    self.with_transforms = with_transforms

  def load(self, model_path, idx, category):
    ''' Loads the data point.

    Args:
        model_path (str): path to model
        idx (int): ID of data point
        category (int): index of category
    '''
    file_path = os.path.join(model_path, self.file_name)

    pointcloud_dict = np.load(file_path)

    points = pointcloud_dict['points'].astype(np.float32)
    normals = pointcloud_dict['normals'].astype(np.float32)

    data = {
        None: points,
        'normals': normals,
    }

    if self.with_transforms:
      data['loc'] = pointcloud_dict['loc'].astype(np.float32)
      data['scale'] = pointcloud_dict['scale'].astype(np.float32)

    if self.transform is not None:
      data = self.transform(data)

    for key, v in data.items():
      data[key] = tf.convert_to_tensor(v, np.float32)

    return data

  def check_complete(self, files):
    ''' Check if field is complete.

    Args:
        files: files
    '''
    complete = (self.file_name in files)
    return complete


# NOTE: this will produce variable length output.
# You need to specify collate_fn to make it work with a data laoder
class MeshField(Field):
  ''' Mesh field.

  It provides the field used for mesh data. Note that, depending on the
  dataset, it produces variable length output, so that you need to specify
  collate_fn to make it work with a data loader.

  Args:
      file_name (str): file name
      transform (list): list of transforms applied to data points
  '''

  def __init__(self, file_name, transform=None):
    self.file_name = file_name
    self.transform = transform

  def load(self, model_path, idx, category):
    ''' Loads the data point.

    Args:
        model_path (str): path to model
        idx (int): ID of data point
        category (int): index of category
    '''
    file_path = os.path.join(model_path, self.file_name)

    mesh = trimesh.load(file_path, process=False)
    if self.transform is not None:
      mesh = self.transform(mesh)

    mesh.vertices = tf.convert_to_tensor(mesh.vertices, np.float32)
    mesh.faces = tf.convert_to_tensor(mesh.faces, np.uint32)

    data = {
        'verts': mesh.vertices,
        'faces': mesh.faces,
    }

    return data

  def check_complete(self, files):
    ''' Check if field is complete.

    Args:
        files: files
    '''
    complete = (self.file_name in files)
    return complete
