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
"""A thin wrapper around the trimesh library for loading triangle meshes."""

import os
import tensorflow as tf
import trimesh
from trimesh import Scene
from trimesh import Trimesh


# TODO(b/156115314): Revisit the library for loading the triangle meshes.
class GFileResolver(trimesh.visual.resolvers.Resolver):
  """A resolver using gfile for accessing other assets in the mesh directory."""

  def __init__(self, path):
    if tf.io.gfile.isdir(path):
      self.directory = path
    elif tf.io.gfile.exists(path):
      self.directory = os.path.dirname(path)
    else:
      raise ValueError('path is not a file or directory')

  def get(self, name):
    with tf.io.gfile.GFile(os.path.join(self.directory, name), 'rb') as f:
      data = f.read()
    return data


def load(file_obj, file_type=None, **kwargs):
  """Loads a triangle mesh from the given GFile/file path.

  Args:
    file_obj: A tf.io.gfile.GFile object or a string specifying the mesh file
      path.
    file_type: A string specifying the type of the file (e.g. 'obj', 'stl'). If
      not specified the file_type will be inferred from the file name.
    **kwargs: Additional arguments that should be passed to trimesh.load().

  Returns:
    A trimesh.Trimesh or trimesh.Scene.
  """

  if isinstance(file_obj, str):
    with tf.io.gfile.GFile(file_obj, 'r') as f:
      if file_type is None:
        file_type = trimesh.util.split_extension(file_obj)
      return trimesh.load(
          file_obj=f,
          file_type=file_type,
          resolver=GFileResolver(file_obj),
          **kwargs)

  if trimesh.util.is_file(file_obj):
    if not hasattr(file_obj, 'name') or not file_obj.name:
      raise ValueError(
          'file_obj must have attribute "name". Try passing the file name instead.'
      )
    if file_type is None:
      file_type = trimesh.util.split_extension(file_obj.name)
    return trimesh.load(
        file_obj=file_obj,
        file_type=file_type,
        resolver=GFileResolver(file_obj.name),
        **kwargs)

  raise ValueError('file_obj should be either a file object or a string')


__all__ = ['load', 'Trimesh', 'Scene']
