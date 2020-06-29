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
""" Helloworld for 3D visualization in Tensorflow Graphics.

Inspired by https://threejs.org/docs/#manual/en/introduction/Creating-a-scene.

WARNING: this needs to be executed in either of two ways
  1) Via the python REPL inside blender as `blender --background --python helloworld.py`
  2) Via `pip install bpy` via https://wiki.blender.org/wiki/Building_Blender/Other/BlenderAsPyModule
"""

# TODO: tensorflow_graphics is not installed in Blender's REPL
import os, sys
sys.path.append("/Users/atagliasacchi/dev/graphics")

import tensorflow_graphics.threejs as THREE

# --- setup the renderer
renderer = THREE.BlenderRenderer()
renderer.set_size(640,480)
scene = THREE.Scene()

# --- setup the camera
camera = THREE.OrthographicCamera()
camera.zoom = .1
camera.position = (10, 10, 10)
camera.up = (0, 0, 1)
camera.look_at(0, 0, .5)
camera.update_projection_matrix()

# --- lights
amb_light = THREE.AmbientLight(color=0x404040)
scene.add(amb_light)
# --- sun
dir_light = THREE.DirectionalLight(color=0xffffff, intensity=2)
dir_light.position = (10, 5, 10)
dir_light.target.position = (0, 0, 0)
scene.add(dir_light)

# --- instantiate object
geometry = THREE.BoxGeometry(1, 1, 1)
geometry.position = (0, 0, 1.5)
material = THREE.MeshBasicMaterial({"color": 0xFF0000})
cube = THREE.Mesh(geometry, material)
scene.add(cube)

# --- transparent floow
ground = THREE.InvisibleGround()
scene.add(ground)

# --- render to PNG or save .blend file (according to extension)
renderer.render(scene, camera, path="helloworld.png")