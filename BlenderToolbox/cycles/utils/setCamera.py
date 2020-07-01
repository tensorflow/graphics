# Copyright 2020 The TensorFlow Authors, Derek Liu
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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