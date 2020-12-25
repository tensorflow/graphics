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
"""Three.js vizualization functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

# pylint: disable=g-import-not-at-top
try:
  from google.colab.output import _js_builder
  from google.colab.output import _publish
  from google.colab.output import eval_js
except ImportError:
  print(
      'Warning: To use the threejs_vizualization, please install the colabtools'
      ' package following the instructions detailed in the README at'
      ' github.com/tensorflow/graphics.',
      file=sys.stderr)
# pylint: enable=g-import-not-at-top


def _triangular_mesh_to_three_geometry(vertices, faces, vertex_colors=None):
  """Converts a triangular mesh to a Three.js BufferGeometry object.

  Args:
    vertices: a [V,3] numpy ndarray containing the position of the vertices
      composing the mesh. V denotes the number of vertices. Vertex positions are
      expected to be floating values.
    faces: a [F,3] numpy array describing the vertices contained in each face of
      the mesh. F denotes the number of faces. The values of that array are
      expected to be positive integer values.
    vertex_colors: a [V, 3] numpy array describing the RGB color of each vertex
      in the mesh. V denotes the number of vertices. Each channel in vertex
      colors is expected to be floating values in range [0, 1].

  Returns:
    A BufferGeometry object describing the geometry of the mesh and which can
    be consumed by Three.js.
  """
  context = _js_builder.Js(mode=_js_builder.PERSISTENT)
  vertices = context.Float32Array.new_object(vertices.ravel().tolist())
  faces = context.Uint32Array.new_object(faces.ravel().tolist())
  geometry = context.THREE.BufferGeometry.new_object()
  geometry.addAttribute('position',
                        context.THREE.BufferAttribute.new_object(vertices, 3))
  geometry.setIndex(context.THREE.BufferAttribute.new_object(faces, 1))
  geometry.computeVertexNormals()
  if vertex_colors is not None:
    vertex_colors = context.Float32Array.new_object(
        vertex_colors.ravel().tolist())
    geometry.addAttribute(
        'color', context.THREE.BufferAttribute.new_object(vertex_colors, 3))

  return geometry


def build_context():
  """Builds a javascript context."""
  threejs_url = 'https://www.gstatic.com/external_hosted/threejs-r98/'
  _publish.javascript(url=threejs_url + 'three.min.js')
  _publish.javascript(url=threejs_url + 'examples/js/controls/OrbitControls.js')
  return _js_builder.Js(mode=_js_builder.PERSISTENT)


def build_perspective_camera(field_of_view=60.0,
                             aspect_ratio=1.0,
                             near_plane=0.01,
                             far_plane=1000.0,
                             position=(0.0, 0.0, 5.0),
                             enable_zoom=False):
  """Builds a perspective camera.

  Args:
    field_of_view: the camera frustum vertical field of view.
    aspect_ratio: the camera frustum aspect ratio.
    near_plane: the camera frustum near plane.
    far_plane: the camera frustum far plane.
    position: the camera position.
    enable_zoom: whether to add a zoom functionalities to the camera controls.

  Returns:
    A Threejs camera with orbit controls.
  """
  context = build_context()
  camera = context.THREE.PerspectiveCamera.new_object(field_of_view,
                                                      aspect_ratio, near_plane,
                                                      far_plane)
  camera.position.set(*position)
  controls = context.THREE.OrbitControls.new_object(camera)
  controls.enableZoom = enable_zoom
  return camera


def triangular_mesh_renderer(meshes,
                             lights=None,
                             camera=None,
                             width=800,
                             height=600,
                             clear_color='rgb(0, 0, 0)'):
  """Function for simple mesh visualization using Three.js.

  Args:
    meshes: a list or tuple of dictionaries, each containing two required keys:
      'vertices', a [V,3] numpy ndarray of vertices, and 'faces' a [F,3] numpy
      ndarray of vertex indices that belong to each face of the mesh. V and F
      respectively correspond to the number of vertices and faces in any given
      mesh. In addition, following optional keys may be provided for each mesh:
        'vertex_colors', a [V, 3] numpy ndarray of vertex colors, and
        'material', a Three.js Material object. If a material is not provided,
        then a default Lambertian material is used.
    lights: a list of Three.js lights to add to the scene. If no light is
      provided, a point light is added to the scene at the position (0., 0.,
      5.).
    camera: a Three.js camera to add to the scene. If no camera is provided, a
      perspetive camera with orbit controls is added to the scene.
    width: the width of the rendering window.
    height: the height of the rendering window.
    clear_color: the color of the background of the rendering window.

  Returns:
    A list of Three.js BufferGeometry objects that can be used to manipulate the
    vertices of the meshes being rendered.
  """
  if not isinstance(meshes, (tuple, list)):
    meshes = [meshes]

  context = build_context()

  # Instantiate the renderer.
  renderer = context.THREE.WebGLRenderer.new_object({'antialias': True})
  renderer.setSize(width, height)
  renderer.setClearColor(context.THREE.Color.new_object(clear_color))
  renderer.clear()
  context.document.body.appendChild(renderer.domElement)

  # If no camera is supplied, create a default one.
  if camera is None:
    camera = context.THREE.PerspectiveCamera.new_object(
        75.,
        float(width) / float(height), .1, 1000.)
    camera.position.z = 5.
    controls = context.THREE.OrbitControls.new_object(camera)
    controls.enableZoom = False

  # Create the scene.
  scene = context.THREE.Scene.new_object()

  # Add lights to the scene.
  if lights is None:
    ambient_light = context.THREE.AmbientLight.new_object(0x404040)
    point_light = context.THREE.PointLight.new_object(0xffffff)
    point_light.position.set(0., 0., 5.)
    lights = (ambient_light, point_light)
  for light in lights:
    scene.add(light)

  # Build meshes and add then to the scene.
  default_material = context.THREE.MeshLambertMaterial.new_object({
      'color': 0x808080,
  })
  geometries = []
  for mesh in meshes:
    if 'vertex_colors' in mesh.keys() and mesh['vertex_colors'] is not None:
      vertex_colors = mesh['vertex_colors']
      default_material['vertexColors'] = context.THREE.VertexColors
    else:
      vertex_colors = None
      default_material['vertexColors'] = context.THREE.NoColors
    geometry = _triangular_mesh_to_three_geometry(mesh['vertices'],
                                                  mesh['faces'], vertex_colors)
    geometries.append(geometry)
    if 'material' in mesh.keys() and mesh['material'] is not None:
      scene.add(context.THREE.Mesh.new_object(geometry, mesh['material']))
    else:
      scene.add(context.THREE.Mesh.new_object(geometry, default_material))

  # Create animate function and call it to start rendering.
  rendering_loop_txt = """
			var animate = function () {
				requestAnimationFrame(animate);
				%s.render(%s, %s);
			};
			animate();
  """ % (renderer._js_value(), scene._js_value(), camera._js_value())  # pylint: disable=protected-access
  eval_js(rendering_loop_txt)
  return geometries
