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

class Mesh(THREE.Object3D):
  def __init__(self, geometry, material):
    self.geometry = geometry
    self.material = material

  def blender(self):
    self.geometry.blender()
    self.material.blender()


# TODO: this object merges a plane with a transparent material, should be factored into two components
class InvisibleGround(Mesh):

  def __init__(self, location=(0, 0, 0), size=20, shadow_brightness=.7 ):
    self.location = location
    self.size = size
    self.shadow_brightness = shadow_brightness

  def blender(self):
    import bpy
    bpy.context.scene.cycles.film_transparent = True  # TODO: this seems more appropriate in scene setup?
    bpy.ops.mesh.primitive_plane_add(location=self.location, size=self.size)
    bpy.context.object.cycles.is_shadow_catcher = True

    # --- set material
    ground = bpy.context.object
    mat = bpy.data.materials.new('MeshMaterial')
    ground.data.materials.append(mat)
    mat.use_nodes = True
    tree = mat.node_tree
    tree.nodes["Principled BSDF"].inputs['Transmission'].default_value = self.shadow_brightness
    return ground
