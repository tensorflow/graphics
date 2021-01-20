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
#include "rasterizer.h"

Rasterizer::Rasterizer(
    std::unique_ptr<gl_utils::Program>&& program,
    std::unique_ptr<gl_utils::RenderTargets>&& render_targets, float clear_red,
    float clear_green, float clear_blue, float clear_alpha, float clear_depth,
    bool enable_cull_face)
    : program_(std::move(program)),
      render_targets_(std::move(render_targets)),
      clear_red_(clear_red),
      clear_green_(clear_green),
      clear_blue_(clear_blue),
      clear_alpha_(clear_alpha),
      clear_depth_(clear_depth),
      enable_cull_face_(enable_cull_face) {}

Rasterizer::~Rasterizer() {}

void Rasterizer::Reset() {
  program_.reset();
  render_targets_.reset();
  for (auto&& buffer : shader_storage_buffers_) buffer.second.reset();
}

tensorflow::Status Rasterizer::Render(int num_points,
                                      absl::Span<float> result) {
  return RenderImpl(num_points, result);
}

tensorflow::Status Rasterizer::Render(int num_points,
                                      absl::Span<unsigned char> result) {
  return RenderImpl(num_points, result);
}

tensorflow::Status Rasterizer::SetUniformMatrix(
    const std::string& name, int num_columns, int num_rows, bool transpose,
    absl::Span<const float> matrix) {
  if (size_t(num_rows * num_columns) != matrix.size())
    return TFG_INTERNAL_ERROR("num_rows * num_columns != matrix.size()");

  typedef void (*setter_fn)(GLint location, GLsizei count, GLboolean transpose,
                            const GLfloat* value);

  static const auto type_mapping =
      std::unordered_map<int, std::tuple<int, int, setter_fn>>({
          {GL_FLOAT_MAT2, std::make_tuple(2, 2, glUniformMatrix2fv)},
          {GL_FLOAT_MAT3, std::make_tuple(3, 3, glUniformMatrix3fv)},
          {GL_FLOAT_MAT4, std::make_tuple(4, 4, glUniformMatrix4fv)},
          {GL_FLOAT_MAT2x3, std::make_tuple(2, 3, glUniformMatrix2x3fv)},
          {GL_FLOAT_MAT2x4, std::make_tuple(2, 4, glUniformMatrix2x4fv)},
          {GL_FLOAT_MAT3x2, std::make_tuple(3, 2, glUniformMatrix3x2fv)},
          {GL_FLOAT_MAT3x4, std::make_tuple(3, 4, glUniformMatrix3x4fv)},
          {GL_FLOAT_MAT4x2, std::make_tuple(4, 2, glUniformMatrix4x2fv)},
          {GL_FLOAT_MAT4x3, std::make_tuple(4, 3, glUniformMatrix4x3fv)},
      });

  GLint uniform_type;
  GLenum property = GL_TYPE;

  TF_RETURN_IF_ERROR(program_->GetResourceProperty(
      name, GL_UNIFORM, 1, &property, 1, &uniform_type));

  // Is a resource active under that name?
  if (uniform_type == GLint(GL_INVALID_INDEX))
    return TFG_INTERNAL_ERROR("GL_INVALID_INDEX");

  auto type_info = type_mapping.find(uniform_type);
  if (type_info == type_mapping.end())
    return TFG_INTERNAL_ERROR("Unsupported type");
  if (std::get<0>(type_info->second) != num_columns ||
      std::get<1>(type_info->second) != num_rows)
    return TFG_INTERNAL_ERROR("Invalid dimensions");

  GLint uniform_location;
  property = GL_LOCATION;
  TF_RETURN_IF_ERROR(program_->GetResourceProperty(
      name, GL_UNIFORM, 1, &property, 1, &uniform_location));

  TF_RETURN_IF_ERROR(program_->Use());
  auto program_cleanup = MakeCleanup([this]() { return program_->Detach(); });

  // Specify the value of the uniform in the current program.
  TFG_RETURN_IF_GL_ERROR(std::get<2>(type_info->second)(
      uniform_location, 1, transpose ? GL_TRUE : GL_FALSE, matrix.data()));

  // Cleanup the program; no program is active at this point.
  return tensorflow::Status::OK();
}
