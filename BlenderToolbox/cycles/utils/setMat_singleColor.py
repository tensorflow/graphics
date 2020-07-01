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

def setMat_singleColor(mesh, meshColor, AOStrength):
	mat = bpy.data.materials.new('MeshMaterial')
	mesh.data.materials.append(mat)
	mesh.active_material = mat
	mat.use_nodes = True
	tree = mat.node_tree

	# set principled BSDF
	tree.nodes["Principled BSDF"].inputs['Roughness'].default_value = 0.7
	tree.nodes["Principled BSDF"].inputs['Sheen Tint'].default_value = 0

	# add Ambient Occlusion
	tree.nodes.new('ShaderNodeAmbientOcclusion')
	tree.nodes.new('ShaderNodeGamma')
	tree.nodes.new('ShaderNodeMixRGB')
	tree.nodes["Mix"].blend_type = 'MULTIPLY'
	tree.nodes["Gamma"].inputs["Gamma"].default_value = AOStrength
	tree.nodes["Ambient Occlusion"].inputs["Distance"].default_value = 10.0
	tree.nodes["Gamma"].location.x -= 600

	# set color using Hue/Saturation node
	HSVNode = tree.nodes.new('ShaderNodeHueSaturation')
	HSVNode.inputs['Color'].default_value = meshColor.RGBA
	HSVNode.inputs['Saturation'].default_value = meshColor.S
	HSVNode.inputs['Value'].default_value = meshColor.V
	HSVNode.inputs['Hue'].default_value = meshColor.H
	HSVNode.location.x -= 200

	# set color brightness/contrast
	BCNode = tree.nodes.new('ShaderNodeBrightContrast')
	BCNode.inputs['Bright'].default_value = meshColor.B
	BCNode.inputs['Contrast'].default_value = meshColor.C
	BCNode.location.x -= 400

	# link all the nodes
	tree.links.new(HSVNode.outputs['Color'], BCNode.inputs['Color'])
	tree.links.new(BCNode.outputs['Color'], tree.nodes['Ambient Occlusion'].inputs['Color'])
	tree.links.new(tree.nodes["Ambient Occlusion"].outputs['Color'], tree.nodes['Mix'].inputs['Color1'])
	tree.links.new(tree.nodes["Ambient Occlusion"].outputs['AO'], tree.nodes['Gamma'].inputs['Color'])
	tree.links.new(tree.nodes["Gamma"].outputs['Color'], tree.nodes['Mix'].inputs['Color2'])
	tree.links.new(tree.nodes["Mix"].outputs['Color'], tree.nodes['Principled BSDF'].inputs['Base Color'])

	
