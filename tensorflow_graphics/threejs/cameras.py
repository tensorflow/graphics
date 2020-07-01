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
"""
The cameras to be used by the rendering system

See: 
  https://threejs.org/docs/#api/en/cameras/Camera
  https://docs.blender.org/api/current/bpy.types.Camera.html
"""
import tensorflow_graphics.threejs as THREE

class Camera(THREE.Object3D):
  def __init__(self):
    super().__init__()
    self.zoom = 1

  def update_projection_matrix(self):
    # TODO: why is this needed by 3js? (assumption: performance)
    pass

class OrthographicCamera(Camera):
  def __init__(self, left=-1, right=+1, top=+1, bottom=-1, near=.1, far=2000):
    # TODO(Derek): how to set camera limits in blender?
    super().__init__()

  def blender(self):
    import bpy
    bpy.ops.object.camera_add() # name 'Camera'
    blender_object = super().blender()
    blender_object.data.type = 'ORTHO'
    blender_object.data.ortho_scale = 1.0 / self.zoom
    return blender_object 


class PerspectiveCamera(Camera):
  def __init__(self, fov=75, aspect=1, near=0.1, far=1000):
    super().__init__()
    raise NotImplementedError

  def blender(self):
    raise NotImplementedError
