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
import tensorflow_graphics.threejs as THREE


def hex_to_rgba(hexint: int):
  b = hexint & 255 
  g = (hexint>>8) & 255
  r = (hexint>>16) & 255
  return (r/255.0, g/255.0, b/255.0, 1.0)


class Light(THREE.Object3D):
  def __init__(self, color=(0.1, 0.1, 0.1, 1.0), intensity=2):
    self.color = color
    self.intensity = intensity # TODO convert intensity into a property (check ranges)

  @property
  def color(self):
    return self._color

  @color.setter
  def color(self, val):
    self._color = hex_to_rgba(val)


class AmbientLight(Light):
  def __init__(self, color=(0.1, 0.1, 0.1, 1.0)):
    self.color = color

  def blender(self):
    import bpy
    bpy.data.scenes[0].world.use_nodes = True #< TODO: shouldn't use_nodes be moved to scene creator?
    bpy.data.scenes[0].world.node_tree.nodes["Background"].inputs['Color'].default_value = self.color


class DirectionalLight(Light):
  # directional light uses a source/target (instead of just a direction) as the
  # THREEJS backend uses this to create shadows via shadow-mapping

  def __init__(self, color, intensity=2, shadow_softness=.1):
    super().__init__(color, intensity)
    self.target = THREE.Object3D()  # defaults to origin
    self.intensity = intensity  #TODO: @property to check ranges
    self.shadow_softness = shadow_softness #TODO: @property to check ranges

  def blender(self):
    print("TODO directional")
    import bpy
    x = self.target.position.x
    y = self.target.position.y
    z = self.target.position.z
    self.look_at(x,y,z)
    rotation = self.quaternion.to_euler()
    bpy.ops.object.light_add(type='SUN', rotation=rotation, location=self.position)
    lamp = bpy.data.lights['Sun']
    lamp.use_nodes = True
    lamp.angle = self.shadow_softness
    lamp.node_tree.nodes["Emission"].inputs['Strength'].default_value = self.intensity
    return lamp