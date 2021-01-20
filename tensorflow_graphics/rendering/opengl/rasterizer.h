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
#ifndef THIRD_PARTY_PY_TENSORFLOW_GRAPHICS_RENDERING_OPENGL_TESTS_RASTERIZER_H_
#define THIRD_PARTY_PY_TENSORFLOW_GRAPHICS_RENDERING_OPENGL_TESTS_RASTERIZER_H_

#include <memory>
#include <string>
#include <unordered_map>

#include "gl_program.h"
#include "gl_render_targets.h"
#include "gl_shader_storage_buffer.h"
#include "cleanup.h"

class RasterizerWithContext;

class Rasterizer {
 public:
  virtual ~Rasterizer();

  // Creates a Rasterizer holding a valid OpenGL program and render buffers.
  //
  // Note: the template argument defines the data type stored in the render
  // buffer. See the documentation of RenderTargets::Create for more details.
  //
  // Arguments:
  // * width: width of the render buffers.
  // * height: height of the render buffers.
  // * vertex_shader_source: source code of a GLSL vertex shader.
  // * geometry_shader_source: source code of a GLSL geometry shader.
  // * fragment_shader_source: source code of a GLSL fragment shader.
  // * rasterizer: if the method succeeds, this variable returns an object
  //   storing a ready to use rasterizer.
  //
  // Returns:
  //   A tensorflow::Status object storing tensorflow::Status::OK() on success,
  //   and an object of type tensorflow::errors otherwise.
  template <typename T>
  static tensorflow::Status Create(const int width, const int height,
                                   const std::string& vertex_shader_source,
                                   const std::string& geometry_shader_source,
                                   const std::string& fragment_shader_source,
                                   std::unique_ptr<Rasterizer>* rasterizer);

  // Creates a Rasterizer holding a valid OpenGL program and render buffers.
  //
  // Arguments:
  // * width: width of the render buffers.
  // * height: height of the render buffers.
  // * vertex_shader_source: source code of a GLSL vertex shader.
  // * geometry_shader_source: source code of a GLSL geometry shader.
  // * fragment_shader_source: source code of a GLSL fragment shader.
  // * clear_red: red component used when clearing the color buffers.
  // * clear_green: green component used when clearing the color buffers.
  // * clear_blue: blue component used when clearing the color buffers.
  // * clear_alpha: alpha component used when clearing the color buffers.
  // * clear_depth: depth value used when clearing the depth buffer.
  // * enable_cull_face: enable face culling.
  // * rasterizer: if the method succeeds, this variable returns an object
  //   storing a ready to use rasterizer.
  //
  // Returns:
  //   A tensorflow::Status object storing tensorflow::Status::OK() on success,
  //   and an object of type tensorflow::errors otherwise.
  template <typename T>
  static tensorflow::Status Create(const int width, const int height,
                                   const std::string& vertex_shader_source,
                                   const std::string& geometry_shader_source,
                                   const std::string& fragment_shader_source,
                                   float clear_red, float clear_green,
                                   float clear_blue, float clear_alpha,
                                   float clear_depth, bool enable_cull_face,
                                   std::unique_ptr<Rasterizer>* rasterizer);

  // Rasterizes the scenes.
  //
  // Arguments:
  // * num_points: the number of primitives to render.
  // * result: if the method succeeds, a buffer that stores the rendering
  //   result. This buffer must be of size 4 * width * height, where the values
  //   of width and height must at least match those used in when calling
  //   Create.
  //
  // Returns:
  //   A tensorflow::Status object storing tensorflow::Status::OK() on success,
  //   and an object of type tensorflow::errors otherwise.
  virtual tensorflow::Status Render(int num_points, absl::Span<float> result);
  virtual tensorflow::Status Render(int num_points,
                                    absl::Span<unsigned char> result);

  // Uploads data to a shader storage buffer.
  //
  // Arguments:
  // * name: name of the shader storage buffer.
  // * data: data to upload to the shader storage buffer.
  //
  // Returns:
  //   A tensorflow::Status object storing tensorflow::Status::OK() on success,
  //   and an object of type tensorflow::errors otherwise.
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
  //   A tensorflow::Status object storing tensorflow::Status::OK() on success,
  //   and an object of type tensorflow::errors otherwise.
  virtual tensorflow::Status SetUniformMatrix(const std::string& name,
                                              int num_columns, int num_rows,
                                              bool transpose,
                                              absl::Span<const float> matrix);

 private:
  Rasterizer() = delete;
  Rasterizer(std::unique_ptr<gl_utils::Program>&& program,
             std::unique_ptr<gl_utils::RenderTargets>&& render_targets,
             float clear_red, float clear_green, float clear_blue,
             float clear_alpha, float clear_depth, bool enable_cull_face);
  Rasterizer(const Rasterizer&) = delete;
  Rasterizer(Rasterizer&&) = delete;
  Rasterizer& operator=(const Rasterizer&) = delete;
  Rasterizer& operator=(Rasterizer&&) = delete;
  template <typename T>
  tensorflow::Status RenderImpl(int num_points, absl::Span<T> result);
  void Reset();

  std::unique_ptr<gl_utils::Program> program_;
  std::unique_ptr<gl_utils::RenderTargets> render_targets_;
  std::unordered_map<std::string,
                     std::unique_ptr<gl_utils::ShaderStorageBuffer>>
      shader_storage_buffers_;
  float clear_red_, clear_green_, clear_blue_, clear_alpha_, clear_depth_;
  bool enable_cull_face_;

  friend class RasterizerWithContext;
};

template <typename T>
tensorflow::Status Rasterizer::Create(const int width, const int height,
                                      const std::string& vertex_shader_source,
                                      const std::string& geometry_shader_source,
                                      const std::string& fragment_shader_source,
                                      std::unique_ptr<Rasterizer>* rasterizer) {
  return Create<T>(width, height, vertex_shader_source, geometry_shader_source,
                   fragment_shader_source, 0.0, 0.0, 0.0, 1.0, 1.0, false,
                   rasterizer);
}

template <typename T>
tensorflow::Status Rasterizer::Create(const int width, const int height,
                                      const std::string& vertex_shader_source,
                                      const std::string& geometry_shader_source,
                                      const std::string& fragment_shader_source,
                                      float clear_red, float clear_green,
                                      float clear_blue, float clear_alpha,
                                      float clear_depth, bool enable_cull_face,
                                      std::unique_ptr<Rasterizer>* rasterizer) {
  std::unique_ptr<gl_utils::Program> program;
  std::unique_ptr<gl_utils::RenderTargets> render_targets;
  std::vector<std::pair<std::string, GLenum>> shaders = {
      {vertex_shader_source, GL_VERTEX_SHADER},
      {geometry_shader_source, GL_GEOMETRY_SHADER},
      {fragment_shader_source, GL_FRAGMENT_SHADER}};

  TF_RETURN_IF_ERROR(gl_utils::Program::Create(shaders, &program));
  TF_RETURN_IF_ERROR(
      gl_utils::RenderTargets::Create<T>(width, height, &render_targets));

  *rasterizer = std::unique_ptr<Rasterizer>(new Rasterizer(
      std::move(program), std::move(render_targets), clear_red, clear_green,
      clear_blue, clear_alpha, clear_depth, enable_cull_face));
  return tensorflow::Status::OK();
}

template <typename T>
tensorflow::Status Rasterizer::RenderImpl(int num_points,
                                          absl::Span<T> result) {
  const GLenum kProperty = GL_BUFFER_BINDING;

  TFG_RETURN_IF_GL_ERROR(glDisable(GL_BLEND));
  TFG_RETURN_IF_GL_ERROR(glEnable(GL_DEPTH_TEST));
  if (enable_cull_face_) TFG_RETURN_IF_GL_ERROR(glEnable(GL_CULL_FACE));

  // Bind storage buffer to shader names
  for (const auto& buffer : shader_storage_buffers_) {
    const std::string& name = buffer.first;
    GLint slot;
    if (program_->GetResourceProperty(name, GL_SHADER_STORAGE_BLOCK, 1,
                                      &kProperty, 1,
                                      &slot) != tensorflow::Status::OK())
      // Buffer not found in program, so do nothing.
      continue;
    TF_RETURN_IF_ERROR(buffer.second->BindBufferBase(slot));
  }

  // Bind the program after the last call to SetUniform, since
  // SetUniform binds program 0.
  TF_RETURN_IF_ERROR(program_->Use());
  auto program_cleanup = MakeCleanup([this]() { return program_->Detach(); });

  TF_RETURN_IF_ERROR(render_targets_->BindFramebuffer());
  auto framebuffer_cleanup =
      MakeCleanup([this]() { return render_targets_->UnbindFrameBuffer(); });

  TFG_RETURN_IF_GL_ERROR(glViewport(0, 0, render_targets_->GetWidth(),
                                    render_targets_->GetHeight()));
  TFG_RETURN_IF_GL_ERROR(
      glClearColor(clear_red_, clear_green_, clear_blue_, clear_alpha_));
  TFG_RETURN_IF_GL_ERROR(glClearDepthf(clear_depth_));
  TFG_RETURN_IF_GL_ERROR(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));

  TFG_RETURN_IF_GL_ERROR(glDrawArrays(GL_POINTS, 0, num_points));

  TF_RETURN_IF_ERROR(render_targets_->CopyPixelsInto(result));

  // The program and framebuffer and released here.
  return tensorflow::Status::OK();
}

template <typename T>
tensorflow::Status Rasterizer::SetShaderStorageBuffer(
    const std::string& name, absl::Span<const T> data) {
  // If the buffer does not exist, create it.
  if (shader_storage_buffers_.count(name) == 0) {
    std::unique_ptr<gl_utils::ShaderStorageBuffer> shader_storage_buffer;
    TF_RETURN_IF_ERROR(
        gl_utils::ShaderStorageBuffer::Create(&shader_storage_buffer));
    // Insert the buffer in the storage.
    shader_storage_buffers_[name] = std::move(shader_storage_buffer);
  }
  // Upload the data to the shader storage buffer.
  TF_RETURN_IF_ERROR(shader_storage_buffers_.at(name)->Upload(data));

  return tensorflow::Status::OK();
}

#endif  // THIRD_PARTY_PY_TENSORFLOW_GRAPHICS_RENDERING_OPENGL_TESTS_RASTERIZER_H_
