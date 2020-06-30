import bpy

def setMat_edge(mesh, \
				edgeThickness, \
				edgeColor, \
				meshColor = (0.7,0.7,0.7,1), \
				AOStrength = 1.0):
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
	tree.nodes["Gamma"].location.x -= 600
	tree.nodes["Ambient Occlusion"].inputs["Distance"].default_value = 10.0
	tree.nodes["Ambient Occlusion"].inputs["Color"].default_value = meshColor
	tree.links.new(tree.nodes["Ambient Occlusion"].outputs['Color'], tree.nodes['Mix'].inputs['Color1'])
	tree.links.new(tree.nodes["Ambient Occlusion"].outputs['AO'], tree.nodes['Gamma'].inputs['Color'])
	tree.links.new(tree.nodes["Gamma"].outputs['Color'], tree.nodes['Mix'].inputs['Color2'])
	tree.links.new(tree.nodes["Mix"].outputs['Color'], tree.nodes['Principled BSDF'].inputs['Base Color'])

	# add edge wireframe
	tree.nodes.new(type="ShaderNodeWireframe")
	wire = tree.nodes[-1]
	wire.inputs[0].default_value = edgeThickness
	wire.location.x -= 200
	wire.location.y -= 200
	tree.nodes.new(type="ShaderNodeBsdfDiffuse")
	mat_wire = tree.nodes[-1]
	HSVNode = tree.nodes.new('ShaderNodeHueSaturation')
	HSVNode.inputs['Color'].default_value = edgeColor.RGBA
	HSVNode.inputs['Saturation'].default_value = edgeColor.S
	HSVNode.inputs['Value'].default_value = edgeColor.V
	HSVNode.inputs['Hue'].default_value = edgeColor.H
	HSVNode.location.x -= 200
	# set color brightness/contrast
	BCNode = tree.nodes.new('ShaderNodeBrightContrast')
	BCNode.inputs['Bright'].default_value = edgeColor.B
	BCNode.inputs['Contrast'].default_value = edgeColor.C
	BCNode.location.x -= 400

	tree.links.new(HSVNode.outputs['Color'],BCNode.inputs['Color'])
	tree.links.new(BCNode.outputs['Color'],mat_wire.inputs['Color'])

	tree.nodes.new('ShaderNodeMixShader')
	tree.links.new(wire.outputs[0], tree.nodes['Mix Shader'].inputs[0])
	tree.links.new(mat_wire.outputs['BSDF'], tree.nodes['Mix Shader'].inputs[2])
	tree.links.new(tree.nodes["Principled BSDF"].outputs['BSDF'], tree.nodes['Mix Shader'].inputs[1])
	tree.links.new(tree.nodes["Mix Shader"].outputs['Shader'], tree.nodes['Material Output'].inputs['Surface'])
	
	