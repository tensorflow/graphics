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

def invisibleGround(location = (0,0,0), groundSize = 20, shadowBrightness = 0.7):
	# initialize a ground for shadow
	bpy.context.scene.cycles.film_transparent = True
	bpy.ops.mesh.primitive_plane_add(location = location, size = groundSize)
	bpy.context.object.cycles.is_shadow_catcher = True

	# # set material
	ground = bpy.context.object
	mat = bpy.data.materials.new('MeshMaterial')
	ground.data.materials.append(mat)
	mat.use_nodes = True
	tree = mat.node_tree
	tree.nodes["Principled BSDF"].inputs['Transmission'].default_value = shadowBrightness
