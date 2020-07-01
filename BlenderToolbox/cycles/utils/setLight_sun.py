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

def setLight_sun(rotation_euler, strength, shadow_soft_size = 0.05):
	x = rotation_euler[0] * 1.0 / 180.0 * np.pi 
	y = rotation_euler[1] * 1.0 / 180.0 * np.pi 
	z = rotation_euler[2] * 1.0 / 180.0 * np.pi 
	angle = (x,y,z)
	bpy.ops.object.light_add(type = 'SUN', rotation = angle)
	lamp = bpy.data.lights['Sun']
	lamp.use_nodes = True
	# lamp.shadow_soft_size = shadow_soft_size # this is for older blender 2.8
	lamp.angle = shadow_soft_size

	lamp.node_tree.nodes["Emission"].inputs['Strength'].default_value = strength
	return lamp