import bpy

def setMat_VColorEdge(mesh, meshVColor, edgeThickness, edgeColor):
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

	# link VColor to principle BSDF
	tree.links.new(HSVNode.outputs['Color'], BCNode.inputs['Color'])
	tree.links.new(BCNode.outputs['Color'], tree.nodes['Principled BSDF'].inputs['Base Color'])

	# add edge wire frame
	wire = tree.nodes.new(type="ShaderNodeWireframe")
	wire.inputs[0].default_value = edgeThickness
	wire.location.x -= 600
	wire.location.y += 200
	mat_wire = tree.nodes.new(type="ShaderNodeBsdfDiffuse")
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
	tree.links.new(BCNode.outputs['Color'],mat_wire.inputs['Color'])


	MIX = tree.nodes.new('ShaderNodeMixShader')
	tree.links.new(wire.outputs[0], MIX.inputs[0])
	tree.links.new(mat_wire.outputs['BSDF'], MIX.inputs[2])
	tree.links.new(tree.nodes["Principled BSDF"].outputs['BSDF'], MIX.inputs[1])

	tree.links.new(MIX.outputs['Shader'], tree.nodes['Material Output'].inputs['Surface'])

	
