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
