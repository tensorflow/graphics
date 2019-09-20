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
#ifndef THIRD_PARTY_PY_TENSORFLOW_GRAPHICS_RENDERING_OPENGL_GL_UTILS_H_
#define THIRD_PARTY_PY_TENSORFLOW_GRAPHICS_RENDERING_OPENGL_GL_UTILS_H_

#include <memory>
#include <string>
#include <typeinfo>
#include <vector>

#include "GL/gl/include/GLES3/gl32.h"
#include "tensorflow_graphics/rendering/opengl/gl_macros.h"

namespace gl_utils {

class Program {
 public:
  ~Program();
  // Creates a program consisting of the supplied shaders. The program is also
  // linked to the executable that will run on the programmable vertex/fragment
  // processors.
  //
  // Arguments:
  // * shaders: a vector of shaders to compile and attach to the program. Each
  //   shader is composed of a string containing the code to compile, and a
  //   GLenum defining the type of the shader which must be one of
  //   GL_COMPUTE_SHADER, GL_VERTEX_SHADER, GL_TESS_CONTROL_SHADER,
  //   GL_TESS_EVALUATION_SHADER, GL_GEOMETRY_SHADER, or GL_FRAGMENT_SHADER.
  // * program: if the method succeeds, this variable returns an object storing
  //   a valid OpenGL program.
  //
  // Returns:
  //   A boolean set to false if any error occured during the process, and set
  //   to true otherwise.
  static bool Create(const std::vector<std::pair<std::string, GLenum>>& shaders,
                     std::unique_ptr<Program>* program);
  GLuint GetHandle() const;

 private:
  Program() = delete;
  explicit Program(GLuint program_handle);
  Program(const Program&) = delete;
  Program(Program&&) = delete;
  Program& operator=(const Program&) = delete;
  Program& operator=(Program&&) = delete;

  // Compiles a shader.
  //
  // Arguments:
  // * shader_code: string containing the shader code to compile.
  // * shader_type: type of shader to compile.
  // * shader_idx: a handle containing the successfully compiled shader.
  //
  // Returns:
  //   A boolean set to false if any error occured during the process, and set
  //   to true otherwise.
  static bool CompileShader(const string& shader_code,
                            const GLenum& shader_type, GLuint* shader_idx);

  GLuint program_handle_;
};

}  // namespace gl_utils

#endif  // THIRD_PARTY_PY_TENSORFLOW_GRAPHICS_RENDERING_OPENGL_GL_UTILS_H_
