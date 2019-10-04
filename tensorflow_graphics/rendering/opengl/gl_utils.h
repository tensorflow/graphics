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

#include <iostream>
#include <memory>
#include <ostream>
#include <string>
#include <typeinfo>
#include <vector>

#include <GLES3/gl32.h>
#include "absl/types/span.h"
#include "tensorflow_graphics/rendering/opengl/gl_macros.h"
#include "tensorflow_graphics/util/cleanup.h"

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
  // Returns:
  //   A boolean set to false if any error occured during the process, and set
  //   to true otherwise.
  bool GetResourceProperty(const std::string& resource_name,
                           GLenum program_interface, int num_properties,
                           const GLenum* properties, int num_property_value,
                           GLint* property_value);

  // Sets the current rendering state to an invalid program object.
  bool Detach() const;

  // Installs the program as part of current rendering state.
  bool Use() const;

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
  static bool CompileShader(const std::string& shader_code,
                            const GLenum& shader_type, GLuint* shader_idx);
  bool GetProgramResourceIndex(GLenum program_interface,
                               absl::string_view resource_name,
                               GLuint* resource_index) const;
  bool GetProgramResourceiv(GLenum program_interface, GLuint resource_index,
                            int num_properties, const GLenum* properties,
                            int num_property_value, GLsizei* length,
                            GLint* property_value) const;

  GLuint program_handle_;
};

// Class that creates a frame buffer to which a depth render buffer, and a color
// render and bound to. The template type correspond to the data type stored in
// the color render buffer. The supported template types are float and unsigned char.
template <typename T>
class RenderTargets {
 public:
  ~RenderTargets();

  // Binds the framebuffer to GL_FRAMEBUFFER.
  bool BindFramebuffer() const;

  // Creates a depth render buffer and a color render buffer. After
  // creation, these two render buffers are attached to the frame buffer.
  //
  // Arguments:
  // * width: width of the rendering buffers; must be smaller than
  // GL_MAX_RENDERBUFFER_SIZE.
  // * height: height of the rendering buffers; must be smaller than
  // GL_MAX_RENDERBUFFER_SIZE.
  // * render_targets: a valid and usable instance of this class.
  //
  // Returns:
  //   A boolean set to false if any error occured during the process, and set
  //   to true otherwise.
  static bool Create(GLsizei width, GLsizei height,
                     std::unique_ptr<RenderTargets<T>>* render_targets);

  // Returns the height of the internal render buffers.
  GLsizei GetHeight() const;

  // Returns the width of the internal render buffers.
  GLsizei GetWidth() const;

  // Reads pixels from the frame buffer.
  //
  // Arguments:
  // * buffer: the buffer where the read pixels are written to. Note that the
  // size of this buffer must be equal to 4 * width * height.
  //
  // Returns:
  //   A boolean set to false if any error occured during the process, and set
  //   to true otherwise.
  bool CopyPixelsInto(absl::Span<T> buffer) const;

  // Breaks the existing binding between the framebuffer object and
  // GL_FRAMEBUFFER.
  bool UnbindFrameBuffer() const;

 private:
  RenderTargets() = delete;
  RenderTargets(const GLsizei width, const GLsizei height,
                const GLuint color_buffer, const GLuint depth_buffer,
                const GLuint frame_buffer);
  RenderTargets(const RenderTargets&) = delete;
  RenderTargets(RenderTargets&&) = delete;
  RenderTargets& operator=(const RenderTargets&) = delete;
  RenderTargets& operator=(RenderTargets&&) = delete;
  static bool CreateValidInternalFormat(
      GLenum internalformat, GLsizei width, GLsizei height,
      std::unique_ptr<RenderTargets<T>>* render_targets);
  bool CopyPixelsIntoValidPixelType(GLenum pixel_type,
                                    absl::Span<T> buffer) const;

  GLsizei width_;
  GLsizei height_;
  GLuint color_buffer_;
  GLuint depth_buffer_;
  GLuint frame_buffer_;
};

template <typename T>
RenderTargets<T>::RenderTargets(const GLsizei width, const GLsizei height,
                                const GLuint color_buffer,
                                const GLuint depth_buffer,
                                const GLuint frame_buffer)
    : width_(width),
      height_(height),
      color_buffer_(color_buffer),
      depth_buffer_(depth_buffer),
      frame_buffer_(frame_buffer) {}

template <typename T>
RenderTargets<T>::~RenderTargets() {
  glDeleteRenderbuffers(1, &color_buffer_);
  glDeleteRenderbuffers(1, &depth_buffer_);
  glDeleteFramebuffers(1, &frame_buffer_);
}

template <typename T>
bool RenderTargets<T>::BindFramebuffer() const {
  TFG_RETURN_FALSE_IF_GL_ERROR(
      glBindFramebuffer(GL_FRAMEBUFFER, frame_buffer_));
  return true;
}

template <typename T>
bool RenderTargets<T>::Create(
    GLsizei width, GLsizei height,
    std::unique_ptr<RenderTargets<T>>* render_targets) {
  std::cerr << "Unsupported type " << typeid(T).name() << std::endl;
  return false;
}

template <>
inline bool RenderTargets<unsigned char>::Create(
    GLsizei width, GLsizei height,
    std::unique_ptr<RenderTargets<unsigned char>>* render_targets) {
  return CreateValidInternalFormat(GL_RGBA8, width, height, render_targets);
}

template <>
inline bool RenderTargets<float>::Create(
    GLsizei width, GLsizei height,
    std::unique_ptr<RenderTargets<float>>* render_targets) {
  return CreateValidInternalFormat(GL_RGBA32F, width, height, render_targets);
}

template <typename T>
bool RenderTargets<T>::CreateValidInternalFormat(
    GLenum internalformat, GLsizei width, GLsizei height,
    std::unique_ptr<RenderTargets>* render_targets) {
  GLuint color_buffer;
  GLuint depth_buffer;
  GLuint frame_buffer;

  // Generate one render buffer for color.
  TFG_RETURN_FALSE_IF_GL_ERROR(glGenRenderbuffers(1, &color_buffer));
  auto gen_color_cleanup = MakeCleanup(
      [color_buffer]() { glDeleteFramebuffers(1, &color_buffer); });
  // Bind the color buffer.
  TFG_RETURN_FALSE_IF_GL_ERROR(
      glBindRenderbuffer(GL_RENDERBUFFER, color_buffer));
  // Define the data storage, format, and dimensions of a render buffer
  // object's image.
  TFG_RETURN_FALSE_IF_GL_ERROR(
      glRenderbufferStorage(GL_RENDERBUFFER, internalformat, width, height));

  // Generate one render buffer for depth.
  TFG_RETURN_FALSE_IF_GL_ERROR(glGenRenderbuffers(1, &depth_buffer));
  auto gen_depth_cleanup = MakeCleanup(
      [depth_buffer]() { glDeleteFramebuffers(1, &depth_buffer); });
  // Bind the depth buffer.
  TFG_RETURN_FALSE_IF_GL_ERROR(
      glBindRenderbuffer(GL_RENDERBUFFER, depth_buffer));
  // Defines the data storage, format, and dimensions of a render buffer
  // object's image.
  TFG_RETURN_FALSE_IF_GL_ERROR(glRenderbufferStorage(
      GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, width, height));

  // Generate one frame buffer.
  TFG_RETURN_FALSE_IF_GL_ERROR(glGenFramebuffers(1, &frame_buffer));
  auto gen_frame_cleanup = MakeCleanup(
      [frame_buffer]() { glDeleteFramebuffers(1, &frame_buffer); });
  // Bind the frame buffer to both read and draw frame buffer targets.
  TFG_RETURN_FALSE_IF_GL_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, frame_buffer));
  // Attach the color buffer to the frame buffer.
  TFG_RETURN_FALSE_IF_GL_ERROR(glFramebufferRenderbuffer(
      GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, color_buffer));
  // Attach the depth buffer to the frame buffer.
  TFG_RETURN_FALSE_IF_GL_ERROR(glFramebufferRenderbuffer(
      GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_buffer));

  *render_targets = std::unique_ptr<RenderTargets<T>>(new RenderTargets(
      width, height, color_buffer, depth_buffer, frame_buffer));

  // Release all Cleanup objects.
  gen_color_cleanup.release();
  gen_depth_cleanup.release();
  gen_frame_cleanup.release();
  return true;
}

template <typename T>
GLsizei RenderTargets<T>::GetHeight() const {
  return height_;
}

template <typename T>
GLsizei RenderTargets<T>::GetWidth() const {
  return width_;
}

template <typename T>
bool RenderTargets<T>::CopyPixelsInto(absl::Span<T> buffer) const {
  std::cerr << "Unsupported type " << typeid(T).name() << std::endl;
  return false;
}

template <>
inline bool RenderTargets<float>::CopyPixelsInto(
    absl::Span<float> buffer) const {
  return CopyPixelsIntoValidPixelType(GL_FLOAT, buffer);
}

template <>
inline bool RenderTargets<unsigned char>::CopyPixelsInto(
    absl::Span<uint8> buffer) const {
  return CopyPixelsIntoValidPixelType(GL_UNSIGNED_BYTE, buffer);
}

template <typename T>
bool RenderTargets<T>::CopyPixelsIntoValidPixelType(
    GLenum pixel_type, absl::Span<T> buffer) const {
  if (buffer.size() != size_t(width_ * height_ * 4)) {
    std::cerr << "Buffer size is not equal to width * height * 4" << std::endl;
    return false;
  }

  TFG_RETURN_FALSE_IF_GL_ERROR(
      glReadPixels(0, 0, width_, height_, GL_RGBA, pixel_type, buffer.data()));
  return true;
}

template <typename T>
bool RenderTargets<T>::UnbindFrameBuffer() const {
  TFG_RETURN_FALSE_IF_GL_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, 0));
  return true;
}

// Class for creating and uploading data to storage buffers.
class ShaderStorageBuffer {
 public:
  ~ShaderStorageBuffer();
  bool BindBufferBase(GLuint index) const;
  static bool Create(
      std::unique_ptr<ShaderStorageBuffer>* shader_storage_buffer);

  // Uploads data to the buffer.
  template <typename T>
  bool Upload(absl::Span<T> data) const;

 private:
  ShaderStorageBuffer() = delete;
  ShaderStorageBuffer(GLuint buffer);
  ShaderStorageBuffer(const ShaderStorageBuffer&) = delete;
  ShaderStorageBuffer(ShaderStorageBuffer&&) = delete;
  ShaderStorageBuffer& operator=(const ShaderStorageBuffer&) = delete;
  ShaderStorageBuffer& operator=(ShaderStorageBuffer&&) = delete;

  GLuint buffer_;
};

template <typename T>
bool ShaderStorageBuffer::Upload(absl::Span<T> data) const {
  // Bind the buffer to the read/write storage for shaders.
  TFG_RETURN_FALSE_IF_GL_ERROR(glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer_));
  auto bind_cleanup = MakeCleanup(
      []() { glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); });
  // Create a new data store for the bound buffer and initializes it with the
  // input data.
  TFG_RETURN_FALSE_IF_GL_ERROR(glBufferData(GL_SHADER_STORAGE_BUFFER,
                                            data.size() * sizeof(T),
                                            data.data(), GL_DYNAMIC_COPY));
  // bind_cleanup is not released, leading the buffer to be unbound.
  return true;
}

}  // namespace gl_utils

#endif  // THIRD_PARTY_PY_TENSORFLOW_GRAPHICS_RENDERING_OPENGL_GL_UTILS_H_
