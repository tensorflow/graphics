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

class Renderer(object):
  """Superclass of all renderers."""
  def __init__(self):
    self.set_size()

  def set_size(self, width: int=320, height: int=240):
    self.width = width
    self.height = height

class BlenderRenderer(Renderer):

  def __init__(self, numSamples = 128, exposure = 1.5, useBothCPUGPU = False):
    super().__init__()
    import bpy
    # because blender has a default scene on load...
    self.clear_scene()
    # use cycle
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.render.resolution_x = self.width
    bpy.context.scene.render.resolution_y = self.height
    bpy.context.scene.render.film_transparent = True
    # bpy.context.scene.cycles.film_transparent = True # TODO derek?
    bpy.context.scene.cycles.samples = numSamples
    bpy.context.scene.cycles.max_bounces = 6
    bpy.context.scene.cycles.film_exposure = exposure
    bpy.data.scenes[0].view_layers['View Layer']['cycles']['use_denoising'] = 1

    # set devices # TODO derek?
    cyclePref  = bpy.context.preferences.addons['cycles'].preferences
    cyclePref.compute_device_type = 'CUDA'
    for dev in cyclePref.devices:
      if dev.type == "CPU" and useBothCPUGPU is False:
        dev.use = False
      else:
        dev.use = True
    bpy.context.scene.cycles.device = 'GPU'

    for dev in cyclePref.devices:
      print (dev)
      print (dev.use)

  def clear_scene(self):
    import bpy
    bpy.ops.wm.read_homefile()
    bpy.ops.object.select_all(action = 'SELECT')
    bpy.ops.object.delete()

  def render(self, scene=None, camera=None, path=None):
    assert path.endswith(".blend") or path.endswith(".png")
    import bpy

    # converts objects to blender mode
    blender_camera = camera.blender()
    scene.blender()

    # creates blender file
    if path.endswith(".blend"):
      bpy.ops.wm.save_mainfile(filepath=path)
    
    # renders scene directly to file
    if path.endswith(".png"):
      bpy.data.scenes['Scene'].render.filepath = path
      bpy.data.scenes['Scene'].camera = blender_camera
      bpy.ops.render.render(write_still = True)


class WebGLRenderer():
  """This would be the colab-friendly renderer."""
  def __init__(self):
    raise NotImplementedError()