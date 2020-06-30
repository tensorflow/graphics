import bpy
from include import *

def setMat_transparentWithEdge(mesh, edgeThickness, edgeColor, meshColor, transparency, transmission):
	mesh.cycles_visibility.shadow = False
	mat = bpy.data.materials.new('MeshMaterial')
	mesh.data.materials.append(mat)
	mesh.active_material = mat
	mat.use_nodes = True
	tree = mat.node_tree

	# color node
	C = initColorNode(tree, meshColor)
	tree.links.new(C.outputs['Color'], tree.nodes['Principled BSDF'].inputs['Base Color'])

	# set principled BSDF
	tree.nodes["Principled BSDF"].inputs['Roughness'].default_value = 0.7
	tree.nodes["Principled BSDF"].inputs['Metallic'].default_value = 0.0
	tree.nodes["Principled BSDF"].inputs['Sheen Tint'].default_value = 0
	tree.nodes["Principled BSDF"].inputs['Transmission'].default_value = transmission

	# init transparent BSDF
	T = tree.nodes.new('ShaderNodeBsdfTransparent')

	# link to mix
	MIX = tree.nodes.new('ShaderNodeMixShader')
	MIX.inputs['Fac'].default_value = transparency
	MIX.location.x -= 200
	MIX.location.y -= 200
	tree.links.new(tree.nodes['Principled BSDF'].outputs[0], MIX.inputs[1])
	tree.links.new(T.outputs[0], MIX.inputs[2])

	# add edge wireframe
	WIRE = tree.nodes.new(type="ShaderNodeWireframe")
	WIRE.inputs[0].default_value = edgeThickness
	WIRE.location.x -= 600
	WIRE.location.y += 200

	WIRE_MAT = tree.nodes.new(type="ShaderNodeBsdfDiffuse")
	HSVNode = tree.nodes.new('ShaderNodeHueSaturation')
	HSVNode.inputs['Color'].default_value = edgeColor.RGBA
	HSVNode.inputs['Saturation'].default_value = edgeColor.S
	HSVNode.inputs['Value'].default_value = edgeColor.V
	HSVNode.inputs['Hue'].default_value = edgeColor.H
	HSVNode.location.x -= 200
	HSVNode.location.y += 200
	# set color brightness/contrast
	BCNode = tree.nodes.new('ShaderNodeBrightContrast')
	BCNode.inputs['Bright'].default_value = edgeColor.B
	BCNode.inputs['Contrast'].default_value = edgeColor.C
	BCNode.location.x -= 400
	BCNode.location.y += 200

	tree.links.new(HSVNode.outputs['Color'],BCNode.inputs['Color'])
	tree.links.new(BCNode.outputs['Color'],WIRE_MAT.inputs['Color'])

	MIX2 = tree.nodes.new('ShaderNodeMixShader')
	tree.links.new(WIRE.outputs[0], MIX2.inputs[0])
	tree.links.new(WIRE_MAT.outputs['BSDF'], MIX2.inputs[2])

	tree.links.new(MIX.outputs[0], MIX2.inputs[1])
	tree.links.new(MIX2.outputs['Shader'], tree.nodes['Material Output'].inputs['Surface'])


	

	
