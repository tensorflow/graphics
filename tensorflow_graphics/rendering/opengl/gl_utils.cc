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
#include "tensorflow_graphics/rendering/opengl/gl_utils.h"

#include <memory>

#include "absl/strings/string_view.h"
#include "tensorflow_graphics/rendering/opengl/gl_macros.h"
#include "tensorflow_graphics/util/cleanup.h"

namespace gl_utils {

Program::Program(GLuint program_handle) : program_handle_(program_handle) {}

Program::~Program() { glDeleteProgram(program_handle_); }

bool Program::CompileShader(const std::string& shader_code,
                            const GLenum& shader_type, GLuint* shader_idx) {
  // Create an empty shader object.
  TFG_RETURN_FALSE_IF_GL_ERROR(*shader_idx = glCreateShader(shader_type));
  if (*shader_idx == 0) {
    std::cerr << "Error while creating the shader object." << std::endl;
    return false;
  }
  auto shader_cleanup = MakeCleanup(
      [shader_idx]() { glDeleteShader(*shader_idx); });

  // Set the source code in the shader object.
  auto shader_code_c_str = shader_code.c_str();
  TFG_RETURN_FALSE_IF_GL_ERROR(
      glShaderSource(*shader_idx, 1, &shader_code_c_str, nullptr));

  // Compile the shader.
  TFG_RETURN_FALSE_IF_GL_ERROR(glCompileShader(*shader_idx));

  GLint compilation_status;
  TFG_RETURN_FALSE_IF_GL_ERROR(
      glGetShaderiv(*shader_idx, GL_COMPILE_STATUS, &compilation_status));
  if (compilation_status != GL_TRUE) {
    GLsizei log_length;
    TFG_RETURN_FALSE_IF_GL_ERROR(
        glGetShaderiv(*shader_idx, GL_INFO_LOG_LENGTH, &log_length));

    std::vector<char> info_log(log_length + 1);
    TFG_RETURN_FALSE_IF_GL_ERROR(
        glGetShaderInfoLog(*shader_idx, log_length, nullptr, &info_log[0]));
    TFG_RETURN_FALSE_IF_GL_ERROR(glDeleteShader(*shader_idx));

    std::cerr << "Error while compiling the shader: "
              << std::string(&info_log[0]) << std::endl;
    return false;
  }
  shader_cleanup.release();
  return true;
}

bool Program::Create(const std::vector<std::pair<std::string, GLenum>>& shaders,
                     std::unique_ptr<Program>* program) {
  // Create an empty program object.
  GLuint program_handle;

  program_handle = glCreateProgram();
  if (program_handle == 0) {
    std::cerr << "Error while creating the program object." << std::endl;
    return false;
  }
  auto program_cleanup = MakeCleanup(
      [program_handle]() { glDeleteProgram(program_handle); });

  // Compile and attach the input shaders to the program.
  std::vector<Cleanup<std::function<void()>>> shader_cleanups;
  for (auto shader : shaders) {
    GLuint shader_idx;
    if (CompileShader(shader.first, shader.second, &shader_idx) == false)
      return false;
    std::function<void()> compile_cleanup = [shader_idx]() {
      glDeleteShader(shader_idx);
    };
    shader_cleanups.push_back(MakeCleanup(compile_cleanup));

    TFG_RETURN_FALSE_IF_GL_ERROR(glAttachShader(program_handle, shader_idx));
    std::function<void()> attach_cleanup = [program_handle, shader_idx]() {
      glDetachShader(program_handle, shader_idx);
    };
    shader_cleanups.push_back(MakeCleanup(attach_cleanup));
  }

  // Link the program to the executable that will run on the programmable
  // vertex/fragment processors.
  TFG_RETURN_FALSE_IF_GL_ERROR(glLinkProgram(program_handle));
  *program = std::unique_ptr<Program>(new Program(program_handle));

  program_cleanup.release();
  // The content of shader_cleanups needs cleanup and hence is not released; see
  // https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/glDeleteProgram.xhtml.
  return true;
}

bool Program::GetProgramResourceIndex(GLenum program_interface,
                                      absl::string_view resource_name,
                                      GLuint* resource_index) const {
  TFG_RETURN_FALSE_IF_GL_ERROR(*resource_index = glGetProgramResourceIndex(
                                   program_handle_, program_interface,
                                   resource_name.data()));
  return true;
}

bool Program::GetProgramResourceiv(GLenum program_interface,
                                   GLuint resource_index, int num_properties,
                                   const GLenum* properties,
                                   int num_property_value, GLsizei* length,
                                   GLint* property_value) const {
  TFG_RETURN_FALSE_IF_GL_ERROR(glGetProgramResourceiv(
      program_handle_, program_interface, resource_index, num_properties,
      properties, num_property_value, length, property_value));
  return true;
}

bool Program::GetResourceProperty(const std::string& resource_name,
                                  GLenum program_interface, int num_properties,
                                  const GLenum* properties,
                                  int num_property_value,
                                  GLint* property_value) {
  if (num_property_value != num_properties) return false;

  GLuint resource_index;
  // Query the index of the named resource within the program.
  TFG_RETURN_FALSE_IF_ERROR(GetProgramResourceIndex(
      program_interface, resource_name, &resource_index));

  // No resource is active under that name.
  if (resource_index == GL_INVALID_INDEX) return false;

  // Retrieve the value for the property.
  GLsizei length;
  TFG_RETURN_FALSE_IF_ERROR(GetProgramResourceiv(
      program_interface, resource_index, num_properties, properties,
      num_property_value, &length, property_value));
  if (length != num_properties) return false;
  return true;
}

bool Program::Use() const {
  TFG_RETURN_FALSE_IF_GL_ERROR(glUseProgram(program_handle_));
  return true;
}

bool Program::Detach() const {
  TFG_RETURN_FALSE_IF_GL_ERROR(glUseProgram(0));
  return true;
}

ShaderStorageBuffer::ShaderStorageBuffer(GLuint buffer) : buffer_(buffer) {}

ShaderStorageBuffer::~ShaderStorageBuffer() { glDeleteBuffers(1, &buffer_); }

bool ShaderStorageBuffer::Create(
    std::unique_ptr<ShaderStorageBuffer>* shader_storage_buffer) {
  GLuint buffer;

  // Generate one buffer object.
  TFG_RETURN_FALSE_IF_GL_ERROR(glGenBuffers(1, &buffer));
  *shader_storage_buffer =
      std::unique_ptr<ShaderStorageBuffer>(new ShaderStorageBuffer(buffer));
  return true;
}

bool ShaderStorageBuffer::BindBufferBase(GLuint index) const {
  TFG_RETURN_FALSE_IF_GL_ERROR(
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, index, buffer_));
  return true;
}

}  // namespace gl_utils
