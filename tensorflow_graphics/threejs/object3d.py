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
import mathutils

class Object3D(object):

  def __init__(self, name=None):
    self.name = name
    self.parent = None
    self._position = mathutils.Vector((0,0,0))
    self.quaternion = mathutils.Quaternion().identity()
    self.up = (0,1,0)
    self.scale = (1,1,1)
    self.receive_shadow = False
    # self.cast_shadow = False

  @property
  def position(self):
    return self._position

  @position.setter
  def position(self, val):
    """Sets the position given a list of 3 elements."""
    self._position = mathutils.Vector(val)

  def look_at(self, x, y, z):
    direction = mathutils.Vector((x,y,z)) - self.position
    # TODO: we should be using self.up here?
    self.quaternion = direction.to_track_quat('-Z', 'Y')
    
  # TODO: add inheritable blender() method
  def blender(self):
    print("not implemented!")
