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
#ifndef THIRD_PARTY_PY_TENSORFLOW_GRAPHICS_RENDERING_OPENGL_GL_PROGRAM_H_
#define THIRD_PARTY_PY_TENSORFLOW_GRAPHICS_RENDERING_OPENGL_GL_PROGRAM_H_

#include <GLES3/gl32.h>

#include "tensorflow/core/lib/core/status.h"

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
  //   A tensorflow::Status object storing tensorflow::Status::OK() on success,
  //   and an object of type tensorflow::errors otherwise.
  static tensorflow::Status Create(
      const std::vector<std::pair<std::string, GLenum>>& shaders,
      std::unique_ptr<Program>* program);

  // Sets the current rendering state to an invalid program object.
  tensorflow::Status Detach() const;

  // Queries the value of properties within the progam.
  //
  // Arguments:
  // * resource_name: name of the resource to query the properties of.
  // * program_interface: a token identifying the interface within program
  //   containing the resource named name. See
  //   https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/glGetProgramResourceIndex.xhtml
  //   for the list of possible values.
  // * num_properties: number of elements in 'properties'.
  // * properties: array of properties to get values for. See
  //   https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/glGetProgramResource.xhtml
  //   for the list of available properties.
  // * num_property_value: number of elements in 'property_value'.
  // * property_value: an array containing the value of the 'properties' in the
  //   resource 'resource_name'.
  //
  // Returns:
  //   A tensorflow::Status object storing tensorflow::Status::OK() on success,
  //   and an object of type tensorflow::errors otherwise.
  tensorflow::Status GetResourceProperty(const std::string& resource_name,
                                         GLenum program_interface,
                                         int num_properties,
                                         const GLenum* properties,
                                         int num_property_value,
                                         GLint* property_value);

  // Installs the program as part of current rendering state.
  tensorflow::Status Use() const;

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
  //   A tensorflow::Status object storing tensorflow::Status::OK() on success,
  //   and an object of type tensorflow::errors otherwise.
  static tensorflow::Status CompileShader(const std::string& shader_code,
                                          const GLenum& shader_type,
                                          GLuint* shader_idx);
  tensorflow::Status GetProgramResourceIndex(GLenum program_interface,
                                             absl::string_view resource_name,
                                             GLuint* resource_index) const;
  tensorflow::Status GetProgramResourceiv(
      GLenum program_interface, GLuint resource_index, int num_properties,
      const GLenum* properties, int num_property_value, GLsizei* length,
      GLint* property_value) const;

  GLuint program_handle_;
};

}  // namespace gl_utils

#endif  // THIRD_PARTY_PY_TENSORFLOW_GRAPHICS_RENDERING_OPENGL_GL_PROGRAM_H_
