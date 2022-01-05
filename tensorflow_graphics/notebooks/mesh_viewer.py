# Copyright 2020 The TensorFlow Authors
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
"""Helper class for viewing 3D meshes in Colab demos.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Dict

import numpy as np
from tensorflow_graphics.notebooks import threejs_visualization

SEGMENTATION_COLORMAP = np.array(
    ((165, 242, 12), (89, 12, 89), (165, 89, 165), (242, 242, 165),
     (242, 165, 12), (89, 12, 12), (165, 12, 12), (165, 89, 242), (12, 12, 165),
     (165, 12, 89), (12, 89, 89), (165, 165, 89), (89, 242, 12), (12, 89, 165),
     (242, 242, 89), (165, 165, 165)),
    dtype=np.float32) / 255.0


class Viewer(object):
  """A ThreeJS based viewer class for viewing 3D meshes."""

  def _mesh_from_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
    """Creates a dictionary of ThreeJS mesh objects from numpy data."""
    if 'vertices' not in data or 'faces' not in data:
      raise ValueError('Mesh Data must contain vertices and faces')
    vertices = np.asarray(data['vertices'])
    faces = np.asarray(data['faces'])
    material = self.context.THREE.MeshLambertMaterial.new_object({
        'color': 0xfffacd,
        'vertexColors': self.context.THREE.NoColors,
        'side': self.context.THREE.DoubleSide,
    })
    mesh = {'vertices': vertices, 'faces': faces}
    if 'vertex_colors' in data:
      mesh['vertex_colors'] = np.asarray(data['vertex_colors'])
      material = self.context.THREE.MeshLambertMaterial.new_object({
          'color': 0xfffacd,
          'vertexColors': self.context.THREE.VertexColors,
          'side': self.context.THREE.DoubleSide,
      })
    mesh['material'] = material
    return mesh

  def __init__(self, source_mesh_data: Dict[str, Any]):
    context = threejs_visualization.build_context()
    self.context = context
    light1 = context.THREE.PointLight.new_object(0x808080)
    light1.position.set(10., 10., 10.)
    light2 = context.THREE.AmbientLight.new_object(0x808080)
    lights = (light1, light2)

    camera = threejs_visualization.build_perspective_camera(
        field_of_view=30, position=(0.0, 0.0, 4.0))

    mesh = self._mesh_from_data(source_mesh_data)
    geometries = threejs_visualization.triangular_mesh_renderer([mesh],
                                                                lights=lights,
                                                                camera=camera,
                                                                width=400,
                                                                height=400)

    self.geometries = geometries
