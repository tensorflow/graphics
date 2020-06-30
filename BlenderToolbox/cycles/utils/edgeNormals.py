import bpy 

def edgeNormals(mesh, angle = 10):
	bpy.context.view_layer.objects.active = mesh
	bpy.ops.object.shade_smooth()
	mesh.data.use_auto_smooth = True
	mesh.data.auto_smooth_angle = angle * 3.14159 / 180

