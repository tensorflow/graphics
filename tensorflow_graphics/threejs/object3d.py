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
"""
Note: we use quaternions as the default rotation representation for 3D objects.

See:
  https://docs.blender.org/api/current/bpy.types.Object.html?highlight=euler#bpy.types.Object.rotation_euler
  https://docs.blender.org/api/current/bpy.types.Object.html?highlight=euler#bpy.types.Object.rotation_mode
"""
import mathutils

class Object3D(object):
  def __init__(self, name=None):
    self.name = name
    self._position = mathutils.Vector((0,0,0)) # (UI: click mesh > Transform > Location)
    self.quaternion = mathutils.Quaternion() # (UI: click mesh > Transform > Rotation)
    # TODO .rotation and .quaternion fields are coupled; see https://github.com/mrdoob/three.js/blob/master/src/core/Object3D.js
    self.scale = (1,1,1) # (UI: click mesh > Transform > Scale)
    self.receive_shadow = False
    # self.cast_shadow = False
    self.parent = None  # TODO: parent management
    self.up = (0,1,0) 

  @property
  def position(self):
    return self._position

  @position.setter
  def position(self, val):
    """Sets the position given a list of 3 elements."""
    self._position = mathutils.Vector(val)

  def look_at(self, x, y, z):
    direction = mathutils.Vector((x,y,z)) - self.position
    # TODO: shouldn't we be using self.up here?
    self.quaternion = direction.to_track_quat('-Z', 'Y')

  def blender(self):
    """Call just after creation of an object."""
    import bpy
    self._blender_object = bpy.context.object
    if self.name: self._blender_object.name = self.name
    self._blender_object.rotation_euler = self.quaternion.to_euler()
    self._blender_object.location = self.position
    self._blender_object.scale = self.scale
    return self._blender_object
  