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

#include "tensorflow_graphics/rendering/opengl/gl_macros.h"
#include "tensorflow/core/lib/gtl/cleanup.h"

namespace gl_utils {

Program::Program(GLuint program_handle) : program_handle_(program_handle) {}

Program::~Program() { glDeleteProgram(program_handle_); }

bool Program::CompileShader(const string& shader_code,
                            const GLenum& shader_type, GLuint* shader_idx) {
  // Create an empty shader object.
  RETURN_FALSE_IF_GL_ERROR(*shader_idx = glCreateShader(shader_type));
  if (*shader_idx == 0) {
    std::cerr << "Error while creating the shader object." << std::endl;
    return false;
  }
  auto shader_cleanup = tensorflow::gtl::MakeCleanup(
      [shader_idx]() { glDeleteShader(*shader_idx); });

  // Set the source code in the shader object.
  auto shader_code_c_str = shader_code.c_str();
  RETURN_FALSE_IF_GL_ERROR(
      glShaderSource(*shader_idx, 1, &shader_code_c_str, nullptr));

  // Compile the shader.
  RETURN_FALSE_IF_GL_ERROR(glCompileShader(*shader_idx));

  GLint compilation_status;
  RETURN_FALSE_IF_GL_ERROR(
      glGetShaderiv(*shader_idx, GL_COMPILE_STATUS, &compilation_status));
  if (compilation_status != GL_TRUE) {
    GLsizei log_length;
    RETURN_FALSE_IF_GL_ERROR(
        glGetShaderiv(*shader_idx, GL_INFO_LOG_LENGTH, &log_length));

    std::vector<char> info_log(log_length + 1);
    RETURN_FALSE_IF_GL_ERROR(
        glGetShaderInfoLog(*shader_idx, log_length, nullptr, &info_log[0]));
    RETURN_FALSE_IF_GL_ERROR(glDeleteShader(*shader_idx));

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
  auto program_cleanup = tensorflow::gtl::MakeCleanup(
      [program_handle]() { glDeleteProgram(program_handle); });

  // Compile and attach the input shaders to the program.
  std::vector<tensorflow::gtl::Cleanup<std::function<void()>>> shader_cleanups;
  for (auto shader : shaders) {
    GLuint shader_idx;
    if (CompileShader(shader.first, shader.second, &shader_idx) == false)
      return false;
    std::function<void()> compile_cleanup = [shader_idx]() {
      glDeleteShader(shader_idx);
    };
    shader_cleanups.push_back(tensorflow::gtl::MakeCleanup(compile_cleanup));

    RETURN_FALSE_IF_GL_ERROR(glAttachShader(program_handle, shader_idx));
    std::function<void()> attach_cleanup = [program_handle, shader_idx]() {
      glDetachShader(program_handle, shader_idx);
    };
    shader_cleanups.push_back(tensorflow::gtl::MakeCleanup(attach_cleanup));
  }

  // Link the program to the executable that will run on the programmable
  // vertex/fragment processors.
  RETURN_FALSE_IF_GL_ERROR(glLinkProgram(program_handle));
  *program = std::unique_ptr<Program>(new Program(program_handle));

  program_cleanup.release();
  // The content of shader_cleanups needs cleanup and hence is not released; see
  // https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/glDeleteProgram.xhtml.
  return true;
}

GLuint Program::GetHandle() const { return program_handle_; }

}  // namespace gl_utils
