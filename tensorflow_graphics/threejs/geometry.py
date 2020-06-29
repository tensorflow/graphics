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

class BoxGeometry(THREE.Object3D):
  def __init__(self, width=1.0, height=1.0, depth=1.0, width_segments=1, height_segments=1, depth_segments=1):
    super().__init__(self)
    self.width = width
    self.height = height
    self.depth = depth
    # TODO: properties are not used
    self.width_segments = width_segments
    self.height_segments = height_segments
    self.depth_segments = depth_segments


  def blender(self):
    import bpy
    bpy.ops.mesh.primitive_cube_add(location=self.position)
    self._blender_object = bpy.context.active_object
    # TODO: adding explicit names to elements might be nice? (Blender only)
    # self._blender_object.name = "mycube"


# TODO: should we add subdivision to the Geometry abstraction?
# def subdivision(mesh, level = 0):
# 	bpy.context.view_layer.objects.active = mesh
# 	bpy.ops.object.modifier_add(type='SUBSURF')
# 	mesh.modifiers["Subdivision"].render_levels = level # rendering subdivision level
# 	mesh.modifiers["Subdivision"].levels = level # subdivision level in 3D view

# TODO: straight from file loader
# meshPath = "spot.ply"
# location = (-0.3, 0.6, -0.04) # (UI: click mesh > Transform > Location)
# rotation = (90, 0,0) # (UI: click mesh > Transform > Rotation)
# scale = (1.5,1.5,1.5) # (UI: click mesh > Transform > Scale)
# blender.read_ply(meshPath, location, rotation, scale)