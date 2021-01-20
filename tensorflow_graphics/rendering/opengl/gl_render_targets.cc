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
#include "gl_render_targets.h"

#include <GLES3/gl32.h>

#include "macros.h"
#include "cleanup.h"
#include "tensorflow/core/lib/core/status.h"

namespace gl_utils {

RenderTargets::RenderTargets(const GLsizei width, const GLsizei height,
                             const GLuint color_buffer,
                             const GLuint depth_buffer,
                             const GLuint frame_buffer)
    : width_(width),
      height_(height),
      color_buffer_(color_buffer),
      depth_buffer_(depth_buffer),
      frame_buffer_(frame_buffer) {}

RenderTargets::~RenderTargets() {
  glDeleteRenderbuffers(1, &color_buffer_);
  glDeleteRenderbuffers(1, &depth_buffer_);
  glDeleteFramebuffers(1, &frame_buffer_);
}

tensorflow::Status RenderTargets::BindFramebuffer() const {
  TFG_RETURN_IF_GL_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, frame_buffer_));
  return tensorflow::Status::OK();
}

tensorflow::Status RenderTargets::CreateValidInternalFormat(
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

  *render_targets = std::unique_ptr<RenderTargets>(new RenderTargets(
      width, height, color_buffer, depth_buffer, frame_buffer));

  // Release all Cleanup objects.
  gen_color_cleanup.release();
  gen_depth_cleanup.release();
  gen_frame_cleanup.release();
  return tensorflow::Status::OK();
}

GLsizei RenderTargets::GetHeight() const { return height_; }

GLsizei RenderTargets::GetWidth() const { return width_; }

tensorflow::Status RenderTargets::UnbindFrameBuffer() const {
  TFG_RETURN_IF_GL_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, 0));
  return tensorflow::Status::OK();
}

}  // namespace gl_utils
