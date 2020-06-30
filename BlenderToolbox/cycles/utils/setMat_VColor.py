import bpy

def setMat_VColor(mesh, meshVColor):
	mat = bpy.data.materials.new('MeshMaterial')
	mesh.data.materials.append(mat)
	mesh.active_material = mat
	mat.use_nodes = True
	tree = mat.node_tree

	# read vertex attribute
	tree.nodes.new('ShaderNodeAttribute')
	tree.nodes[-1].attribute_name = "Col"
	HSVNode = tree.nodes.new('ShaderNodeHueSaturation')
	tree.links.new(tree.nodes["Attribute"].outputs['Color'], HSVNode.inputs['Color'])
	HSVNode.inputs['Saturation'].default_value = meshVColor.S
	HSVNode.inputs['Value'].default_value = meshVColor.V
	HSVNode.inputs['Hue'].default_value = meshVColor.H
	HSVNode.location.x -= 200

	# set color brightness/contrast
	BCNode = tree.nodes.new('ShaderNodeBrightContrast')
	BCNode.inputs['Bright'].default_value = meshVColor.B
	BCNode.inputs['Contrast'].default_value = meshVColor.C
	BCNode.location.x -= 400

	# set principled BSDF
	tree.nodes["Principled BSDF"].inputs['Roughness'].default_value = 1.0
	tree.nodes["Principled BSDF"].inputs['Sheen Tint'].default_value = 0
	tree.links.new(HSVNode.outputs['Color'], BCNode.inputs['Color'])
	tree.links.new(BCNode.outputs['Color'], tree.nodes['Principled BSDF'].inputs['Base Color'])
