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
import numpy as np
import os

def readPLY(filePath, location, rotation_euler, scale):
	# example input types:
	# - location = (0.5, -0.5, 0)
	# - rotation_euler = (90, 0, 0)
	# - scale = (1,1,1)
	x = rotation_euler[0] * 1.0 / 180.0 * np.pi 
	y = rotation_euler[1] * 1.0 / 180.0 * np.pi 
	z = rotation_euler[2] * 1.0 / 180.0 * np.pi 
	angle = (x,y,z)
	prev = []
	for ii in range(len(list(bpy.data.objects))):
		prev.append(bpy.data.objects[ii].name)
	bpy.ops.import_mesh.ply(filepath=filePath)
	after = []
	for ii in range(len(list(bpy.data.objects))):
		after.append(bpy.data.objects[ii].name)
	name = list(set(after) - set(prev))[0]
	# filePath = filePath.rstrip(os.sep) 
	# name = os.path.basename(filePath)
	# name = name.replace('.ply', '')
	mesh = bpy.data.objects[name]
	# print(list(bpy.data.objects))
	# mesh = bpy.data.objects[-1]
	mesh.location = location
	mesh.rotation_euler = angle
	mesh.scale = scale
	bpy.context.view_layer.update()
	return mesh 