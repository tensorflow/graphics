/* Copyright 2019 Google LLC

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

#include "tensorflow_graphics/rendering/opengl/egl_offscreen_context.h"
#include "tensorflow_graphics/rendering/opengl/gl_utils.h"
#include "tensorflow_graphics/rendering/opengl/rasterizer.h"
#include "tensorflow_graphics/util/cleanup.h"

template <typename T>
class RasterizerWithContext : public Rasterizer<T> {
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
  // * clear_r: red component used when clearing the color buffers.
  // * clear_g: green component used when clearing the color buffers.
  // * clear_b: blue component used when clearing the color buffers.
  // * clear_depth: depth value used when clearing the depth buffer
  //
  // Returns:
  //   A boolean set to false if any error occured during the process, and set
  //   to true otherwise.
  static bool Create(
      int width, int height, const std::string& vertex_shader_source,
      const std::string& geometry_shader_source,
      const std::string& fragment_shader_source,
      std::unique_ptr<RasterizerWithContext<T>>* rasterizer_with_context,
      float clear_r = 0.0f, float clear_g = 0.0f, float clear_b = 0.0f,
      float clear_depth = 1.0f);

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
  bool Render(int num_points, absl::Span<T> result) override;

  // Uploads data to a shader storage buffer.
  //
  // Arguments:
  // * name: name of the shader storage buffer.
  // * data: data to upload to the shader storage buffer.
  //
  // Returns:
  //   A boolean set to false if any error occured during the process, and set
  //   to true otherwise.
  template <typename S>
  bool SetShaderStorageBuffer(const std::string& name,
                              absl::Span<const S> data);

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
  bool SetUniformMatrix(const std::string& name, int num_columns, int num_rows,
                        bool transpose, absl::Span<const T> matrix) override;

 private:
  RasterizerWithContext() = delete;
  RasterizerWithContext(
      std::unique_ptr<EGLOffscreenContext>&& egl_context,
      std::unique_ptr<gl_utils::Program>&& program,
      std::unique_ptr<gl_utils::RenderTargets<T>>&& render_targets,
      float clear_r, float clear_g, float clear_b, float clear_depth);
  RasterizerWithContext(const RasterizerWithContext&) = delete;
  RasterizerWithContext(RasterizerWithContext&&) = delete;
  RasterizerWithContext& operator=(const RasterizerWithContext&) = delete;
  RasterizerWithContext& operator=(RasterizerWithContext&&) = delete;

  std::unique_ptr<EGLOffscreenContext> egl_context_;
};

template <typename T>
RasterizerWithContext<T>::RasterizerWithContext(
    std::unique_ptr<EGLOffscreenContext>&& egl_context,
    std::unique_ptr<gl_utils::Program>&& program,
    std::unique_ptr<gl_utils::RenderTargets<T>>&& render_targets, float clear_r,
    float clear_g, float clear_b, float clear_depth)
    : egl_context_(std::move(egl_context)),
      Rasterizer<T>(std::move(program), std::move(render_targets), clear_r,
                    clear_g, clear_b, clear_depth) {}

template <typename T>
RasterizerWithContext<T>::~RasterizerWithContext() {
  // Destroy the rasterizer in the correct EGL context.
  bool status = egl_context_->MakeCurrent();
  if (status == false)
    std::cerr
        << "~RasterizerWithContext: failure to set the context as current."
        << std::endl;
  // Reset all the members of the Rasterizer parent class before destroying the
  // context.
  this->Reset();
  // egl_context_ is destroyed here, which calls
  // egl_offscreen_context::Release().
}

template <typename T>
bool RasterizerWithContext<T>::Create(
    int width, int height, const std::string& vertex_shader_source,
    const std::string& geometry_shader_source,
    const std::string& fragment_shader_source,
    std::unique_ptr<RasterizerWithContext<T>>* rasterizer_with_context,
    float clear_r, float clear_g, float clear_b, float clear_depth) {
  std::unique_ptr<gl_utils::Program> program;
  std::unique_ptr<gl_utils::RenderTargets<float>> render_targets;
  std::vector<std::pair<std::string, GLenum>> shaders;
  std::unique_ptr<EGLOffscreenContext> offscreen_context;

  TFG_RETURN_FALSE_IF_ERROR(EGLOffscreenContext::Create(&offscreen_context));
  offscreen_context->MakeCurrent();
  // No need to have a MakeCleanup here as EGLOffscreenContext::Release()
  // would be called on destruction of the offscreen_context object, which
  // would happen here if the whole creation process was not successful.

  shaders.push_back(std::make_pair(vertex_shader_source, GL_VERTEX_SHADER));
  shaders.push_back(std::make_pair(geometry_shader_source, GL_GEOMETRY_SHADER));
  shaders.push_back(std::make_pair(fragment_shader_source, GL_FRAGMENT_SHADER));
  TFG_RETURN_FALSE_IF_ERROR(gl_utils::Program::Create(shaders, &program));
  TFG_RETURN_FALSE_IF_ERROR(
      gl_utils::RenderTargets<T>::Create(width, height, &render_targets));
  offscreen_context->Release();
  *rasterizer_with_context =
      std::unique_ptr<RasterizerWithContext>(new RasterizerWithContext(
          std::move(offscreen_context), std::move(program),
          std::move(render_targets), clear_r, clear_g, clear_b, clear_depth));
  return true;
}

template <typename T>
bool RasterizerWithContext<T>::Render(int num_points, absl::Span<T> result) {
  TFG_RETURN_FALSE_IF_ERROR(egl_context_->MakeCurrent());
  auto context_cleanup =
      MakeCleanup([this]() { this->egl_context_->Release(); });
  TFG_RETURN_FALSE_IF_ERROR(Rasterizer<T>::Render(num_points, result));
  // context_cleanup calls EGLOffscreenContext::Release here.
  return true;
}

template <typename T>
template <typename S>
bool RasterizerWithContext<T>::SetShaderStorageBuffer(
    const std::string& name, absl::Span<const S> data) {
  TFG_RETURN_FALSE_IF_ERROR(egl_context_->MakeCurrent());
  auto context_cleanup =
      MakeCleanup([this]() { this->egl_context_->Release(); });
  TFG_RETURN_FALSE_IF_ERROR(Rasterizer<T>::SetShaderStorageBuffer(name, data));
  // context_cleanup calls EGLOffscreenContext::Release here.
  return true;
}

template <typename T>
bool RasterizerWithContext<T>::SetUniformMatrix(const std::string& name,
                                                int num_columns, int num_rows,
                                                bool transpose,
                                                absl::Span<const T> matrix) {
  TFG_RETURN_FALSE_IF_ERROR(egl_context_->MakeCurrent());
  auto context_cleanup =
      MakeCleanup([this]() { this->egl_context_->Release(); });
  TFG_RETURN_FALSE_IF_ERROR(Rasterizer<T>::SetUniformMatrix(
      name, num_columns, num_rows, transpose, matrix));
  // context_cleanup calls EGLOffscreenContext::Release here.
  return true;
}

#endif  // THIRD_PARTY_PY_TENSORFLOW_GRAPHICS_RENDERING_OPENGL_RASTERIZER_WITH_CONTEXT_H_
