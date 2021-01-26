/* Copyright 2020 The TensorFlow Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef THIRD_PARTY_PY_TENSORFLOW_GRAPHICS_RENDERING_KERNELS_RASTERIZE_TRIANGLES_IMPL_H_
#define THIRD_PARTY_PY_TENSORFLOW_GRAPHICS_RENDERING_KERNELS_RASTERIZE_TRIANGLES_IMPL_H_

#include "absl/base/integral_types.h"

// Determines the mode for face culling. Analogous to OpenGL's glCullFace
// parameters.
enum class FaceCullingMode { kNone = 0, kBack, kFront };

// Computes the triangle id, barycentric coordinates, and z-buffer at each pixel
// in the image.
//
// vertices: A flattened 2D array with 4*vertex_count elements.
//     Each contiguous triplet is the XYZW location of the vertex with that
//     triplet's id. The coordinates are assumed to be OpenGL-style clip-space
//     (i.e., post-projection, pre-divide), where X points right, Y points up,
//     Z points away.
// triangles: A flattened 2D array with 3*triangle_count elements.
//     Each contiguous triplet is the three vertex ids indexing into vertices
//     describing one triangle with clockwise winding.
// triangle_count: The number of triangles stored in the array triangles.
// num_layers: Number of surface layers to store at each pixel, esentially
//     depth-peeling (https://en.wikipedia.org/wiki/Depth_peeling).
// face_culling_mode: mode for culling back-facing triangles, front-facing
//     triangles, or none.
// triangle_ids: A flattened 2D array with num_layers*image_height*image_width
//     elements. At return, each pixel contains a triangle id in the range
//     [0, triangle_count). The id value is also 0 if there is no triangle
//     at the pixel. The barycentric_coordinates must be checked to
//     distinguish the two cases.
// z_buffer: A flattened 2D array with num_layers*image_height*image_width
//     elements. At return, contains the normalized device Z coordinates of the
//     rendered triangles.
// barycentric_coordinates: A flattened 3D array with
//     num_layers*image_height*image_width*3 elements. At return, contains the
//     triplet of barycentric coordinates at each pixel in the same vertex
//     ordering as triangles. If no triangle is present, all coordinates are 0.
//     May be nullptr if barycentric coordinates are not desired.
void RasterizeTrianglesImpl(const float* vertices, const int32* triangles,
                            int32 triangle_count, int32 image_width,
                            int32 image_height, int32 num_layers,
                            FaceCullingMode face_culling_mode,
                            int32* triangle_ids, float* z_buffer,
                            float* barycentric_coordinates);

#endif  // THIRD_PARTY_PY_TENSORFLOW_GRAPHICS_RENDERING_KERNELS_RASTERIZE_TRIANGLES_IMPL_H_
