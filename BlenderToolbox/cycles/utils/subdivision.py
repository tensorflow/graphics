import bpy 

def subdivision(mesh, level = 0):
	bpy.context.view_layer.objects.active = mesh
	bpy.ops.object.modifier_add(type='SUBSURF')
	mesh.modifiers["Subdivision"].render_levels = level # rendering subdivision level
	mesh.modifiers["Subdivision"].levels = level # subdivision level in 3D view

