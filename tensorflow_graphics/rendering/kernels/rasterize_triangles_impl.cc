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
#include "rasterize_triangles_impl.h"

#include <algorithm>
#include <cmath>

namespace {

using fixed_t = int64_t;

// Converts to fixed point with 16 fractional bits and 48 integer bits.
// TODO(fcole): fixed-point depth may be too shallow.
// The algorithm requires multiplying two of the xyzw clip-space coordinates
// together, summing, and then multiplying by an NDC pixel coordinate (three
// total multiplies). After three multiplications, the fractional part will be
// 48 bits, leaving 16 bits for the integer part. The NDC pixel coordinates
// are in (-1,1) so they need only 1 integer bit, so as long as the values of
// the inverse matrix are < 2^15, the fixed-point math should not overflow. This
// seems a bit dicey but so far all the tests I've tried pass.
constexpr int kFractionalBits = 16;

// Clamps value to the range [low, high]. Requires low <= high.
template <typename T>
static const T& Clamp(const T& value, const T& low, const T& high) {
  if (high < value) return high;
  if (value < low) return low;
  return value;
}

constexpr fixed_t ShiftPointLeft(fixed_t x) { return x << kFractionalBits; }

constexpr fixed_t ToFixedPoint(float f) {
  return static_cast<fixed_t>(f * ShiftPointLeft(1));
}

// Takes the minimum of a and b, rounds down, and converts to an integer in
// the range [low, high].
inline int ClampedIntegerMin(float a, float b, int low, int high) {
  const float value = std::floor(std::min(a, b));
  return static_cast<int>(
      Clamp(value, static_cast<float>(low), static_cast<float>(high)));
}

// Takes the maximum of a and b, rounds up, and converts to an integer in the
// range [low, high].
inline int ClampedIntegerMax(float a, float b, int low, int high) {
  const float value = std::ceil(std::max(a, b));
  return static_cast<int>(
      Clamp(value, static_cast<float>(low), static_cast<float>(high)));
}

// Return true if the near plane is between the eye and the clip-space point
// with the provided z and w.
inline bool IsClipPointVisible(float z, float w) { return w > 0 && z >= -w; }

// Computes the screen-space bounding box of the given clip-space triangle and
// stores it into [left, right, bottom, top], where left and bottom limits are
// inclusive while right and top are not.
// Returns true if the bounding box includes any screen pixels.
bool ComputeTriangleBoundingBox(float v0x, float v0y, float v0z, float v0w,
                                float v1x, float v1y, float v1z, float v1w,
                                float v2x, float v2y, float v2z, float v2w,
                                int image_width, int image_height, int* left,
                                int* right, int* bottom, int* top) {
  // If the triangle is entirely visible, project the vertices to pixel
  // coordinates and find the triangle bounding box enlarged to the nearest
  // integer and clamped to the image boundaries. If the triangle is not
  // entirely visible, intersect the edges that cross the near plane with the
  // near plane and use those to compute screen bounds instead.
  *left = image_width;
  *right = 0;
  *bottom = image_height;
  *top = 0;

  auto add_point = [&](float x, float y, float w) {
    const float px = 0.5f * (x / w + 1) * image_width;
    const float py = 0.5f * (y / w + 1) * image_height;
    *left = ClampedIntegerMin(*left, px, 0, image_width);
    *right = ClampedIntegerMax(*right, px, 0, image_width);
    *bottom = ClampedIntegerMin(*bottom, py, 0, image_height);
    *top = ClampedIntegerMax(*top, py, 0, image_height);
  };

  auto add_near_point = [&](float x0, float y0, float z0, float w0, float x1,
                            float y1, float z1, float w1) {
    const float denom = z0 - z1 + w0 - w1;
    if (denom != 0) {
      // Interpolate to near plane, where z/w == -1.
      const float t = (z0 + w0) / denom;
      const float x = x0 + t * (x1 - x0);
      const float y = y0 + t * (y1 - y0);
      const float w = w0 + t * (w1 - w0);
      add_point(x, y, w);
    }
  };

  const bool visible_v0 = IsClipPointVisible(v0z, v0w);
  const bool visible_v1 = IsClipPointVisible(v1z, v1w);
  const bool visible_v2 = IsClipPointVisible(v2z, v2w);
  if (visible_v0) {
    add_point(v0x, v0y, v0w);
    if (!visible_v1) add_near_point(v0x, v0y, v0z, v0w, v1x, v1y, v1z, v1w);
    if (!visible_v2) add_near_point(v0x, v0y, v0z, v0w, v2x, v2y, v2z, v2w);
  }
  if (visible_v1) {
    add_point(v1x, v1y, v1w);
    if (!visible_v2) add_near_point(v1x, v1y, v1z, v1w, v2x, v2y, v2z, v2w);
    if (!visible_v0) add_near_point(v1x, v1y, v1z, v1w, v0x, v0y, v0z, v0w);
  }
  if (visible_v2) {
    add_point(v2x, v2y, v2w);
    if (!visible_v0) add_near_point(v2x, v2y, v2z, v2w, v0x, v0y, v0z, v0w);
    if (!visible_v1) add_near_point(v2x, v2y, v2z, v2w, v1x, v1y, v1z, v1w);
  }

  const bool is_valid = (*right > *left) && (*top > *bottom);
  return is_valid;
}

// Computes a 3x3 matrix inverse without dividing by the determinant.
// Instead, makes an unnormalized matrix inverse with the correct sign
// by flipping the sign of the matrix if the determinant is negative.
// By leaving out determinant division, the rows of M^-1 only depend on two out
// of three of the columns of M; i.e., the first row of M^-1 only depends on the
// second and third columns of M, the second only depends on the first and
// third, etc. This means we can compute edge functions for two neighboring
// triangles independently and produce exactly the same numerical result up to
// the sign.
// See http://mathworld.wolfram.com/MatrixInverse.html
// Culling is accomplished by inspecting the sign of the determinant as in:
// "Incremental and Hierarchical Hilbert Order Edge Equation Polygon
// Rasterization," McCool, et al., 2001
void ComputeUnnormalizedMatrixInverse(
    const fixed_t a11, const fixed_t a12, const fixed_t a13, const fixed_t a21,
    const fixed_t a22, const fixed_t a23, const fixed_t a31, const fixed_t a32,
    const fixed_t a33, const FaceCullingMode culling_mode, fixed_t m_inv[9]) {
  m_inv[0] = a22 * a33 - a32 * a23;
  m_inv[1] = a13 * a32 - a33 * a12;
  m_inv[2] = a12 * a23 - a22 * a13;
  m_inv[3] = a23 * a31 - a33 * a21;
  m_inv[4] = a11 * a33 - a31 * a13;
  m_inv[5] = a13 * a21 - a23 * a11;
  m_inv[6] = a21 * a32 - a31 * a22;
  m_inv[7] = a12 * a31 - a32 * a11;
  m_inv[8] = a11 * a22 - a21 * a12;

  // If the culling mode is kBack, leave the sign of the matrix unchanged.
  // Transfer the sign of the determinant if mode is kNone. If mode is kFront,
  // just invert the matrix.
  if (culling_mode == FaceCullingMode::kNone ||
      culling_mode == FaceCullingMode::kFront) {
    // The first column of the unnormalized M^-1 contains intermediate values
    // for det(M).
    const float det = a11 * m_inv[0] + a12 * m_inv[3] + a13 * m_inv[6];
    const float multiplier = (culling_mode == FaceCullingMode::kNone)
                                 ? std::copysign(1.0, det)
                                 : -1.0;
    for (int i = 0; i < 9; ++i) {
      m_inv[i] *= multiplier;
    }
  }
}

// Computes the edge functions from M^-1 as described by Olano and Greer,
// "Triangle Scan Conversion using 2D Homogeneous Coordinates."
//
// This function combines equations (3) and (4). It first computes
// [a b c] = u_i * M^-1, where u_0 = [1 0 0], u_1 = [0 1 0], etc.,
// then computes edge_i = aX + bY + c
void ComputeEdgeFunctions(const float px, const float py,
                          const fixed_t m_inv[9], fixed_t values[3]) {
  const fixed_t px_i = ToFixedPoint(px);
  const fixed_t py_i = ToFixedPoint(py);
  for (int i = 0; i < 3; ++i) {
    const fixed_t a = m_inv[3 * i + 0];
    const fixed_t b = m_inv[3 * i + 1];
    const fixed_t c = m_inv[3 * i + 2];

    // Before summing, shift the point of c to align with the products of
    // multiplication.
    values[i] = a * px_i + b * py_i + ShiftPointLeft(c);
  }
}

// Determines whether the point p lies inside a triangle. Counts pixels exactly
// on an edge as inside the triangle, as long as the triangle is not degenerate.
// Degenerate (zero-area) triangles always fail the inside test.
bool PixelIsInsideTriangle(const fixed_t edge_values[3]) {
  // Check that the edge values are all non-negative and that at least one is
  // positive (triangle is non-degenerate).
  return (edge_values[0] >= 0 && edge_values[1] >= 0 && edge_values[2] >= 0) &&
         (edge_values[0] > 0 || edge_values[1] > 0 || edge_values[2] > 0);
}

}  // namespace

void RasterizeTrianglesImpl(const float* vertices, const int32_t* triangles,
                            int32_t triangle_count, int32_t image_width,
                            int32_t image_height, int32_t num_layers,
                            FaceCullingMode face_culling_mode,
                            int32_t* triangle_ids, float* z_buffer,
                            float* barycentric_coordinates) {
  const float half_image_width = 0.5f * image_width;
  const float half_image_height = 0.5f * image_height;
  fixed_t unnormalized_matrix_inverse[9];
  fixed_t b_over_w[3];
  int left, right, bottom, top;

  for (int32_t triangle_id = 0; triangle_id < triangle_count; ++triangle_id) {
    const int32_t v0_x_id = 4 * triangles[3 * triangle_id];
    const int32_t v1_x_id = 4 * triangles[3 * triangle_id + 1];
    const int32_t v2_x_id = 4 * triangles[3 * triangle_id + 2];

    const float v0x = vertices[v0_x_id];
    const float v0y = vertices[v0_x_id + 1];
    const float v0z = vertices[v0_x_id + 2];
    const float v0w = vertices[v0_x_id + 3];

    const float v1x = vertices[v1_x_id];
    const float v1y = vertices[v1_x_id + 1];
    const float v1z = vertices[v1_x_id + 2];
    const float v1w = vertices[v1_x_id + 3];

    const float v2x = vertices[v2_x_id];
    const float v2y = vertices[v2_x_id + 1];
    const float v2z = vertices[v2_x_id + 2];
    const float v2w = vertices[v2_x_id + 3];

    const bool is_valid = ComputeTriangleBoundingBox(
        v0x, v0y, v0z, v0w, v1x, v1y, v1z, v1w, v2x, v2y, v2z, v2w, image_width,
        image_height, &left, &right, &bottom, &top);

    // Ignore triangles that do not overlap with any screen pixels.
    if (!is_valid) continue;

    ComputeUnnormalizedMatrixInverse(
        ToFixedPoint(v0x), ToFixedPoint(v1x), ToFixedPoint(v2x),
        ToFixedPoint(v0y), ToFixedPoint(v1y), ToFixedPoint(v2y),
        ToFixedPoint(v0w), ToFixedPoint(v1w), ToFixedPoint(v2w),
        face_culling_mode, unnormalized_matrix_inverse);

    // Iterate over each pixel in the bounding box.
    for (int iy = bottom; iy < top; ++iy) {
      for (int ix = left; ix < right; ++ix) {
        const float px = ((ix + 0.5f) / half_image_width) - 1.0f;
        const float py = ((iy + 0.5f) / half_image_height) - 1.0f;

        ComputeEdgeFunctions(px, py, unnormalized_matrix_inverse, b_over_w);
        if (!PixelIsInsideTriangle(b_over_w)) {
          continue;
        }

        const float one_over_w = b_over_w[0] + b_over_w[1] + b_over_w[2];
        const float b0 = b_over_w[0] / one_over_w;
        const float b1 = b_over_w[1] / one_over_w;
        const float b2 = b_over_w[2] / one_over_w;

        // Since we computed an unnormalized w above, we need to recompute
        // a properly scaled clip-space w value and then divide clip-space z
        // by that.
        const float clip_z = b0 * v0z + b1 * v1z + b2 * v2z;
        const float clip_w = b0 * v0w + b1 * v1w + b2 * v2w;
        const float z = clip_z / clip_w;

        // Skip the pixel if it is beyond the near or far clipping plane.
        if (z < -1.0f || z > 1.0f) continue;

        // Insert into appropriate depth layer with insertion sort.
        float z_next = z;
        int32_t id_next = triangle_id;
        float b0_next = b0;
        float b1_next = b1;
        float b2_next = b2;
        const int pixel_idx0 = iy * image_width + ix;
        for (int layer = 0; layer < num_layers; ++layer) {
          const int pixel_idx = pixel_idx0 + image_height * image_width * layer;
          if (z_next < z_buffer[pixel_idx]) {
            std::swap(z_next, z_buffer[pixel_idx]);
            std::swap(id_next, triangle_ids[pixel_idx]);
            if (barycentric_coordinates != nullptr) {
              std::swap(b0_next, barycentric_coordinates[3 * pixel_idx + 0]);
              std::swap(b1_next, barycentric_coordinates[3 * pixel_idx + 1]);
              std::swap(b2_next, barycentric_coordinates[3 * pixel_idx + 2]);
            }
          }
          // Exit the loop early if the clear depth (z == 1) is reached.
          if (z_next == 1) break;
        }
      }
    }
  }
}
