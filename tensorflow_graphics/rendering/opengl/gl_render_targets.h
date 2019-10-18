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
#ifndef THIRD_PARTY_PY_TENSORFLOW_GRAPHICS_RENDERING_OPENGL_GL_RENDER_TARGETS_H_
#define THIRD_PARTY_PY_TENSORFLOW_GRAPHICS_RENDERING_OPENGL_GL_RENDER_TARGETS_H_

#include <GLES3/gl32.h>

#include "tensorflow_graphics/rendering/opengl/macros.h"
#include "tensorflow_graphics/util/cleanup.h"
#include "tensorflow/core/lib/core/status.h"

namespace gl_utils {

// Class that creates a frame buffer to which a depth render buffer, and a color
// render and bound to. The template type correspond to the data type stored in
// the color render buffer. The supported template types are float and unsigned
// char.
template <typename T>
class RenderTargets {
 public:
  ~RenderTargets();

  // Binds the framebuffer to GL_FRAMEBUFFER.
  tensorflow::Status BindFramebuffer() const;

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
  //   A tensorflow::Status object storing tensorflow::Status::OK() on success,
  //   and an object of type tensorflow::errors otherwise.
  static tensorflow::Status Create(
      GLsizei width, GLsizei height,
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
  //   A tensorflow::Status object storing tensorflow::Status::OK() on success,
  //   and an object of type tensorflow::errors otherwise.
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
      std::unique_ptr<RenderTargets<T>>* render_targets);
  tensorflow::Status CopyPixelsIntoValidPixelType(GLenum pixel_type,
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
tensorflow::Status RenderTargets<T>::BindFramebuffer() const {
  TFG_RETURN_IF_GL_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, frame_buffer_));
  return tensorflow::Status::OK();
}

template <typename T>
tensorflow::Status RenderTargets<T>::Create(
    GLsizei width, GLsizei height,
    std::unique_ptr<RenderTargets<T>>* render_targets) {
  return TFG_INTERNAL_ERROR("Unsupported type ", typeid(T).name());
}

template <>
inline tensorflow::Status RenderTargets<unsigned char>::Create(
    GLsizei width, GLsizei height,
    std::unique_ptr<RenderTargets<unsigned char>>* render_targets) {
  return CreateValidInternalFormat(GL_RGBA8, width, height, render_targets);
}

template <>
inline tensorflow::Status RenderTargets<float>::Create(
    GLsizei width, GLsizei height,
    std::unique_ptr<RenderTargets<float>>* render_targets) {
  return CreateValidInternalFormat(GL_RGBA32F, width, height, render_targets);
}

template <typename T>
tensorflow::Status RenderTargets<T>::CreateValidInternalFormat(
    GLenum internalformat, GLsizei width, GLsizei height,
    std::unique_ptr<RenderTargets>* render_targets) {
  GLuint color_buffer;
  GLuint depth_buffer;
  GLuint frame_buffer;

  // Generate one render buffer for color.
  TFG_RETURN_IF_GL_ERROR(glGenRenderbuffers(1, &color_buffer));
  auto gen_color_cleanup =
      MakeCleanup([color_buffer]() { glDeleteFramebuffers(1, &color_buffer); });
  // Bind the color buffer.
  TFG_RETURN_IF_GL_ERROR(glBindRenderbuffer(GL_RENDERBUFFER, color_buffer));
  // Define the data storage, format, and dimensions of a render buffer
  // object's image.
  TFG_RETURN_IF_GL_ERROR(
      glRenderbufferStorage(GL_RENDERBUFFER, internalformat, width, height));

  // Generate one render buffer for depth.
  TFG_RETURN_IF_GL_ERROR(glGenRenderbuffers(1, &depth_buffer));
  auto gen_depth_cleanup =
      MakeCleanup([depth_buffer]() { glDeleteFramebuffers(1, &depth_buffer); });
  // Bind the depth buffer.
  TFG_RETURN_IF_GL_ERROR(glBindRenderbuffer(GL_RENDERBUFFER, depth_buffer));
  // Defines the data storage, format, and dimensions of a render buffer
  // object's image.
  TFG_RETURN_IF_GL_ERROR(glRenderbufferStorage(
      GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, width, height));

  // Generate one frame buffer.
  TFG_RETURN_IF_GL_ERROR(glGenFramebuffers(1, &frame_buffer));
  auto gen_frame_cleanup =
      MakeCleanup([frame_buffer]() { glDeleteFramebuffers(1, &frame_buffer); });
  // Bind the frame buffer to both read and draw frame buffer targets.
  TFG_RETURN_IF_GL_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, frame_buffer));
  // Attach the color buffer to the frame buffer.
  TFG_RETURN_IF_GL_ERROR(glFramebufferRenderbuffer(
      GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, color_buffer));
  // Attach the depth buffer to the frame buffer.
  TFG_RETURN_IF_GL_ERROR(glFramebufferRenderbuffer(
      GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_buffer));

  *render_targets = std::unique_ptr<RenderTargets<T>>(new RenderTargets(
      width, height, color_buffer, depth_buffer, frame_buffer));

  // Release all Cleanup objects.
  gen_color_cleanup.release();
  gen_depth_cleanup.release();
  gen_frame_cleanup.release();
  return tensorflow::Status::OK();
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
tensorflow::Status RenderTargets<T>::CopyPixelsInto(
    absl::Span<T> buffer) const {
  return TFG_INTERNAL_ERROR("Unsupported type ", typeid(T).name());
}

template <>
inline tensorflow::Status RenderTargets<float>::CopyPixelsInto(
    absl::Span<float> buffer) const {
  return CopyPixelsIntoValidPixelType(GL_FLOAT, buffer);
}

template <>
inline tensorflow::Status RenderTargets<unsigned char>::CopyPixelsInto(
    absl::Span<unsigned char> buffer) const {
  return CopyPixelsIntoValidPixelType(GL_UNSIGNED_BYTE, buffer);
}

template <typename T>
tensorflow::Status RenderTargets<T>::CopyPixelsIntoValidPixelType(
    GLenum pixel_type, absl::Span<T> buffer) const {
  if (buffer.size() != size_t(width_ * height_ * 4))
    return TFG_INTERNAL_ERROR("Buffer size is not equal to width * height * 4");

  TFG_RETURN_IF_GL_ERROR(
      glReadPixels(0, 0, width_, height_, GL_RGBA, pixel_type, buffer.data()));
  return tensorflow::Status::OK();
}

template <typename T>
tensorflow::Status RenderTargets<T>::UnbindFrameBuffer() const {
  TFG_RETURN_IF_GL_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, 0));
  return tensorflow::Status::OK();
}

}  // namespace gl_utils

#endif  // THIRD_PARTY_PY_TENSORFLOW_GRAPHICS_RENDERING_OPENGL_GL_RENDER_TARGETS_H_
