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
#ifndef THIRD_PARTY_PY_TENSORFLOW_GRAPHICS_RENDERING_OPENGL_GL_RENDER_TARGETS_H_
#define THIRD_PARTY_PY_TENSORFLOW_GRAPHICS_RENDERING_OPENGL_GL_RENDER_TARGETS_H_

#include <GLES3/gl32.h>

#include "absl/types/span.h"
#include "macros.h"
#include "cleanup.h"
#include "tensorflow/core/lib/core/status.h"

namespace gl_utils {

// Class that creates a frame buffer to which a depth render buffer, and a color
// render and bound to.
class RenderTargets {
 public:
  ~RenderTargets();

  // Binds the framebuffer to GL_FRAMEBUFFER.
  tensorflow::Status BindFramebuffer() const;

  // Creates a depth render buffer and a color render buffer. After
  // creation, these two render buffers are attached to the frame buffer.
  //
  // Note: The template type correspond to the data type stored in
  // the color render buffer. The supported template types are float and
  // unsigned char, which lead the internal format of the renderbuffer to be
  // GL_RGBA32F and GL_RGBA8 respectively.
  //
  // Arguments:
  // * width: width of the rendering buffers; must be smaller than
  // GL_MAX_RENDERBUFFER_SIZE.
  // * height: height of the rendering buffers; must be smaller than
  // GL_MAX_RENDERBUFFER_SIZE.
  // * render_targets: a valid and usable instance of this class.
  //
  // Returns:
  //   A tensorflow::Status object storing tensorflow::Status::OK() on success,
  //   and an object of type tensorflow::errors otherwise.
  template <typename T>
  static tensorflow::Status Create(
      GLsizei width, GLsizei height,
      std::unique_ptr<RenderTargets>* render_targets);

  // Returns the height of the internal render buffers.
  GLsizei GetHeight() const;

  // Returns the width of the internal render buffers.
  GLsizei GetWidth() const;

  // Reads pixels from the frame buffer.
  //
  // Note: if the type of T is not float, the buffer will contain values that
  // are transformed according to the formulas described in
  // https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/glReadPixels.xhtml.
  //
  // Arguments:
  // * buffer: the buffer where the read pixels are written to. Note that the
  // size of this buffer must be equal to 4 * width * height.
  //
  // Returns:
  //   A tensorflow::Status object storing tensorflow::Status::OK() on success,
  //   and an object of type tensorflow::errors otherwise.
  template <typename T>
  tensorflow::Status CopyPixelsInto(absl::Span<T> buffer) const;

  // Breaks the existing binding between the framebuffer object and
  // GL_FRAMEBUFFER.
  tensorflow::Status UnbindFrameBuffer() const;

 private:
  RenderTargets() = delete;
  RenderTargets(const GLsizei width, const GLsizei height,
                const GLuint color_buffer, const GLuint depth_buffer,
                const GLuint frame_buffer);
  RenderTargets(const RenderTargets&) = delete;
  RenderTargets(RenderTargets&&) = delete;
  RenderTargets& operator=(const RenderTargets&) = delete;
  RenderTargets& operator=(RenderTargets&&) = delete;
  static tensorflow::Status CreateValidInternalFormat(
      GLenum internalformat, GLsizei width, GLsizei height,
      std::unique_ptr<RenderTargets>* render_targets);

  template <typename T>
  tensorflow::Status CopyPixelsIntoValidPixelType(GLenum pixel_type,
                                                  absl::Span<T> buffer) const;

  GLsizei width_;
  GLsizei height_;
  GLuint color_buffer_;
  GLuint depth_buffer_;
  GLuint frame_buffer_;
};

template <typename T>
tensorflow::Status RenderTargets::Create(
    GLsizei width, GLsizei height,
    std::unique_ptr<RenderTargets>* render_targets) {
  return TFG_INTERNAL_ERROR("Unsupported type ", typeid(T).name());
}

template <>
inline tensorflow::Status RenderTargets::Create<unsigned char>(
    GLsizei width, GLsizei height,
    std::unique_ptr<RenderTargets>* render_targets) {
  return CreateValidInternalFormat(GL_RGBA8, width, height, render_targets);
}

template <>
inline tensorflow::Status RenderTargets::Create<float>(
    GLsizei width, GLsizei height,
    std::unique_ptr<RenderTargets>* render_targets) {
  return CreateValidInternalFormat(GL_RGBA32F, width, height, render_targets);
}

template <typename T>
tensorflow::Status RenderTargets::CopyPixelsInto(absl::Span<T> buffer) const {
  return TFG_INTERNAL_ERROR("Unsupported type ", typeid(T).name());
}

template <>
inline tensorflow::Status RenderTargets::CopyPixelsInto<float>(
    absl::Span<float> buffer) const {
  return CopyPixelsIntoValidPixelType(GL_FLOAT, buffer);
}

template <>
inline tensorflow::Status RenderTargets::CopyPixelsInto<unsigned char>(
    absl::Span<unsigned char> buffer) const {
  return CopyPixelsIntoValidPixelType(GL_UNSIGNED_BYTE, buffer);
}

template <typename T>
tensorflow::Status RenderTargets::CopyPixelsIntoValidPixelType(
    GLenum pixel_type, absl::Span<T> buffer) const {
  if (buffer.size() != size_t(width_ * height_ * 4))
    return TFG_INTERNAL_ERROR("Buffer size is not equal to width * height * 4");

  TFG_RETURN_IF_GL_ERROR(
      glReadPixels(0, 0, width_, height_, GL_RGBA, pixel_type, buffer.data()));
  return tensorflow::Status::OK();
}

}  // namespace gl_utils

#endif  // THIRD_PARTY_PY_TENSORFLOW_GRAPHICS_RENDERING_OPENGL_GL_RENDER_TARGETS_H_
