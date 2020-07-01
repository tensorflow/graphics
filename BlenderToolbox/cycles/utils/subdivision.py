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
import bpy 

def subdivision(mesh, level = 0):
	bpy.context.view_layer.objects.active = mesh
	bpy.ops.object.modifier_add(type='SUBSURF')
	mesh.modifiers["Subdivision"].render_levels = level # rendering subdivision level
	mesh.modifiers["Subdivision"].levels = level # subdivision level in 3D view

