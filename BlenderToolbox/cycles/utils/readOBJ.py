import bpy
import numpy as np
import bmesh

def readOBJ(filePath, location, rotation_euler, scale):
	x = rotation_euler[0] * 1.0 / 180.0 * np.pi 
	y = rotation_euler[1] * 1.0 / 180.0 * np.pi 
	z = rotation_euler[2] * 1.0 / 180.0 * np.pi 
	angle = (x,y,z)

	prev = []
	for ii in range(len(list(bpy.data.objects))):
		prev.append(bpy.data.objects[ii].name)
	bpy.ops.import_scene.obj(filepath=filePath, split_mode='OFF')
	after = []
	for ii in range(len(list(bpy.data.objects))):
		after.append(bpy.data.objects[ii].name)
	name = list(set(after) - set(prev))[0]
	mesh = bpy.data.objects[name]

	mesh.location = location
	mesh.rotation_euler = angle
	mesh.scale = scale
	bpy.context.view_layer.update()

	return mesh 