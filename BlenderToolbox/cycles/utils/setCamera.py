import bpy
import math
import mathutils
from utils.lookAt import *

def setCamera(camLocation, lookAtLocation = (0,0,0), focalLength = 35):
	# initialize camera
	bpy.ops.object.camera_add(location = camLocation) # name 'Camera'
	cam = bpy.context.object
	cam.data.lens = focalLength
	loc = mathutils.Vector(lookAtLocation)
	lookAt(cam, loc)
	return cam