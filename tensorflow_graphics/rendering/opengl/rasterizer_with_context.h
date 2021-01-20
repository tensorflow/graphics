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
#ifndef THIRD_PARTY_PY_TENSORFLOW_GRAPHICS_RENDERING_OPENGL_RASTERIZER_WITH_CONTEXT_H_
#define THIRD_PARTY_PY_TENSORFLOW_GRAPHICS_RENDERING_OPENGL_RASTERIZER_WITH_CONTEXT_H_

#include "egl_offscreen_context.h"
#include "rasterizer.h"
#include "cleanup.h"
#include "tensorflow/core/lib/core/status.h"

class RasterizerWithContext : public Rasterizer {
 public:
  ~RasterizerWithContext();

  // Creates an EGL offscreen context and a rasterizer holding a valid OpenGL
  // program and render buffers.
  //
  // Arguments:
  // * width: width of the render buffers.
  // * height: height of the render buffers.
  // * vertex_shader_source: source code of a GLSL vertex shader.
  // * geometry_shader_source: source code of a GLSL geometry shader.
  // * fragment_shader_source: source code of a GLSL fragment shader.
  // * rasterizer_with_context: if the method succeeds, this variable returns an
  // object
  //   storing an EGL offscreen context and a rasterizer.
  // * clear_red: red component used when clearing the color buffers.
  // * clear_green: green component used when clearing the color buffers.
  // * clear_blue: blue component used when clearing the color buffers.
  // * clear_alpha: alpha component used when clearing the color buffers.
  // * clear_depth: depth value used when clearing the depth buffer
  //
  // Returns:
  //   A boolean set to false if any error occured during the process, and set
  //   to true otherwise.
  static tensorflow::Status Create(
      int width, int height, const std::string& vertex_shader_source,
      const std::string& geometry_shader_source,
      const std::string& fragment_shader_source,
      std::unique_ptr<RasterizerWithContext>* rasterizer_with_context,
      float clear_red = 0.0f, float clear_green = 0.0f, float clear_blue = 0.0f,
      float clear_alpha = 1.0f, float clear_depth = 1.0f,
      bool enable_cull_face = false);

  // Rasterizes the scenes.
  //
  // Arguments:
  // * num_points: the number of vertices to render.
  // * result: if the method succeeds, a buffer that stores the rendering
  //   result.
  //
  // Returns:
  //   A boolean set to false if any error occured during the process, and set
  //   to true otherwise.
  tensorflow::Status Render(int num_points, absl::Span<float> result) override;
  tensorflow::Status Render(int num_points,
                            absl::Span<unsigned char> result) override;

  // Uploads data to a shader storage buffer.
  //
  // Arguments:
  // * name: name of the shader storage buffer.
  // * data: data to upload to the shader storage buffer.
  //
  // Returns:
  //   A boolean set to false if any error occured during the process, and set
  //   to true otherwise.
  template <typename T>
  tensorflow::Status SetShaderStorageBuffer(const std::string& name,
                                            absl::Span<const T> data);

  // Specifies the value of a uniform matrix.
  //
  // Note: The input matrix is expected to be in column-major format. Both glm
  //       and OpenGL store matrices in column major format.
  //
  // Arguments:
  // * name: name of the uniform.
  // * num_columns: number of columns in the matrix.
  // * num_rows: number of rows in the matrix.
  // * transpose: indicates whether the supplied matrix needs to be transposed.
  // * matrix: a buffer storing the matrix
  //
  // Returns:
  //   A boolean set to false if any error occured during the process, and set
  //   to true otherwise.
  template <typename T>
  tensorflow::Status SetUniformMatrix(const std::string& name, int num_columns,
                                      int num_rows, bool transpose,
                                      absl::Span<const T> matrix);

 private:
  RasterizerWithContext() = delete;
  RasterizerWithContext(
      std::unique_ptr<EGLOffscreenContext>&& egl_context,
      std::unique_ptr<gl_utils::Program>&& program,
      std::unique_ptr<gl_utils::RenderTargets>&& render_targets,
      float clear_red, float clear_green, float clear_blue, float clear_alpha,
      float clear_depth, bool enable_cull_face);
  RasterizerWithContext(const RasterizerWithContext&) = delete;
  RasterizerWithContext(RasterizerWithContext&&) = delete;
  RasterizerWithContext& operator=(const RasterizerWithContext&) = delete;
  RasterizerWithContext& operator=(RasterizerWithContext&&) = delete;

  std::unique_ptr<EGLOffscreenContext> egl_context_;
};

template <typename T>
tensorflow::Status RasterizerWithContext::SetShaderStorageBuffer(
    const std::string& name, absl::Span<const T> data) {
  TF_RETURN_IF_ERROR(egl_context_->MakeCurrent());
  auto context_cleanup =
      MakeCleanup([this]() { return this->egl_context_->Release(); });
  TF_RETURN_IF_ERROR(Rasterizer::SetShaderStorageBuffer(name, data));
  // context_cleanup calls EGLOffscreenContext::Release here.
  return tensorflow::Status::OK();
}

template <typename T>
tensorflow::Status RasterizerWithContext::SetUniformMatrix(
    const std::string& name, int num_columns, int num_rows, bool transpose,
    absl::Span<const T> matrix) {
  TF_RETURN_IF_ERROR(egl_context_->MakeCurrent());
  auto context_cleanup =
      MakeCleanup([this]() { return this->egl_context_->Release(); });
  TF_RETURN_IF_ERROR(Rasterizer::SetUniformMatrix(name, num_columns, num_rows,
                                                  transpose, matrix));
  // context_cleanup calls EGLOffscreenContext::Release here.
  return tensorflow::Status::OK();
}

// template <typename T>
// tensorflow::Status RasterizerWithContext::Render(
//     int num_points, absl::Span<T> result) {
//   TF_RETURN_IF_ERROR(egl_context_->MakeCurrent());
//   auto context_cleanup =
//       MakeCleanup([this]() { return this->egl_context_->Release(); });
//   TF_RETURN_IF_ERROR(Rasterizer::Render(num_points, result));
//   // context_cleanup calls EGLOffscreenContext::Release here.
//   return tensorflow::Status::OK();
// }

#endif  // THIRD_PARTY_PY_TENSORFLOW_GRAPHICS_RENDERING_OPENGL_RASTERIZER_WITH_CONTEXT_H_
