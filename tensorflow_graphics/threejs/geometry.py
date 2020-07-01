# Copyright 2020 The TensorFlow Authors, Derek Liu
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
""" The geometry module of the threejs renderer."""
import tensorflow_graphics.threejs as THREE


# TODO: integrate https://docs.blender.org/api/current/bmesh.html ?

class Geometry(THREE.Object3D):
  """https://threejs.org/docs/#api/en/core/BufferGeometry"""
  def __init__(self):
    super().__init__()

class BufferGeometry(Geometry):
  def __init__(self):
    super().__init__()
    self.index = None
    self.attributes = dict()

  def set_index(self, nparray):
    """https://threejs.org/docs/#api/en/core/BufferGeometry.setIndex"""
    # TODO: assertions and type checks
    self.index = nparray

  def set_attribute(self, name, attribute):
    # TODO: assertions and type checks
    self.attributes[name] = attribute

  def blender(self):
    # TODO: BufferGeometry might be used by things OTHER than meshes, how to resolve?
    import bpy
    bpy.ops.object.add(type="MESH")
    blender_object = super().blender()
    # TODO: add asserts on type expectations from from_pydata
    vertices = self.attributes["position"].array.tolist()
    faces = self.index.tolist()
    blender_object.data.from_pydata(vertices, [], faces)
    return blender_object


class BoxGeometry(Geometry):
  def __init__(self, width=1.0, height=1.0, depth=1.0):
    super().__init__()
    self.scale = (width, height, depth)

  def blender(self):
    import bpy
    bpy.ops.mesh.primitive_cube_add(location=self.position)
    return super().blender()