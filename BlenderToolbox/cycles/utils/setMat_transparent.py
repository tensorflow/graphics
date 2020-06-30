import bpy
from include import *

def setMat_transparent(mesh, meshColor, transparency, transmission):
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

	tree.links.new(MIX.outputs[0], tree.nodes['Material Output'].inputs['Surface'])


	

	
