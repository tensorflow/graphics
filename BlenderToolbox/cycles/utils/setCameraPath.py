import bpy
import numpy as np

# Inputs:
#   pathRadius: radius of the circle camera path
#   pathHeight: height of the circle camera path
#   lookAtPos: look at position
#   focalLength: focal length of the camera
#   duration: duration of the video (turing 360 degree)
#   startAngle: staring angle of the camera (angle from x-axis)

def setCameraPath(
	pathRadius, pathHeight, lookAtPos, focalLength,\
	duration = 100, startAngle = 0):
	
	# path circle position
	circPos = tuple(np.add(lookAtPos, (0,0,pathHeight))) 

	# create circle path
	bpy.ops.curve.primitive_bezier_circle_add(enter_editmode=False, location=circPos)
	OBJ_circle = bpy.context.object
	OBJ_circle.scale = (pathRadius,pathRadius,pathRadius)

	# create look at position
	bpy.ops.object.empty_add(type='PLAIN_AXES', location=lookAtPos)
	OBJ_empty = bpy.context.object

	# compute initial camera position 
	camPos = (pathRadius*np.cos(startAngle/180.0*np.pi), pathRadius*np.sin(startAngle/180.0*np.pi), 0)
	camPos = tuple(np.add(camPos, circPos))

	# initialize camera
	bpy.ops.object.camera_add(location = camPos)
	OBJ_camera = bpy.context.object
	OBJ_camera.data.lens = focalLength

	# set camera to follow a path
	OBJ_camera.select_set(True)
	OBJ_circle.select_set(True)
	bpy.context.view_layer.objects.active = OBJ_circle
	bpy.ops.object.parent_set(type='FOLLOW')
	bpy.ops.object.select_all(action='DESELECT')
	OBJ_circle.data.path_duration = duration

	# set camera lookAt
	lookAt = OBJ_camera.constraints.new(type='TRACK_TO')
	lookAt.target = OBJ_empty
	lookAt.track_axis = 'TRACK_NEGATIVE_Z'
	lookAt.up_axis = 'UP_Y'

	return OBJ_camera
