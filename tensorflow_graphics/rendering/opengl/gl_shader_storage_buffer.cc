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
#include "gl_shader_storage_buffer.h"

#include <GLES3/gl32.h>

#include "tensorflow/core/lib/core/status.h"

namespace gl_utils {

ShaderStorageBuffer::ShaderStorageBuffer(GLuint buffer) : buffer_(buffer) {}

ShaderStorageBuffer::~ShaderStorageBuffer() { glDeleteBuffers(1, &buffer_); }

tensorflow::Status ShaderStorageBuffer::Create(
    std::unique_ptr<ShaderStorageBuffer>* shader_storage_buffer) {
  GLuint buffer;

  // Generate one buffer object.
  TFG_RETURN_IF_EGL_ERROR(glGenBuffers(1, &buffer));
  *shader_storage_buffer =
      std::unique_ptr<ShaderStorageBuffer>(new ShaderStorageBuffer(buffer));
  return tensorflow::Status::OK();
}

tensorflow::Status ShaderStorageBuffer::BindBufferBase(GLuint index) const {
  TFG_RETURN_IF_EGL_ERROR(
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER, index, buffer_));
  return tensorflow::Status::OK();
}

}  // namespace gl_utils
