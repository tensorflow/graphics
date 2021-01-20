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
#ifndef THIRD_PARTY_PY_TENSORFLOW_GRAPHICS_RENDERING_OPENGL_GL_SHADER_STORAGE_BUFFER_H_
#define THIRD_PARTY_PY_TENSORFLOW_GRAPHICS_RENDERING_OPENGL_GL_SHADER_STORAGE_BUFFER_H_

#include <GLES3/gl32.h>

#include "absl/types/span.h"
#include "macros.h"
#include "cleanup.h"
#include "tensorflow/core/lib/core/status.h"

namespace gl_utils {

// Class for creating and uploading data to storage buffers.
class ShaderStorageBuffer {
 public:
  ~ShaderStorageBuffer();
  tensorflow::Status BindBufferBase(GLuint index) const;
  static tensorflow::Status Create(
      std::unique_ptr<ShaderStorageBuffer>* shader_storage_buffer);

  // Uploads data to the buffer.
  template <typename T>
  tensorflow::Status Upload(absl::Span<T> data) const;

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
tensorflow::Status ShaderStorageBuffer::Upload(absl::Span<T> data) const {
  // Bind the buffer to the read/write storage for shaders.
  TFG_RETURN_IF_GL_ERROR(glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer_));
  auto bind_cleanup =
      MakeCleanup([]() { glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); });
  // Create a new data store for the bound buffer and initializes it with the
  // input data.
  TFG_RETURN_IF_GL_ERROR(glBufferData(GL_SHADER_STORAGE_BUFFER,
                                      data.size() * sizeof(T), data.data(),
                                      GL_DYNAMIC_COPY));
  // bind_cleanup is not released, leading the buffer to be unbound.
  return tensorflow::Status::OK();
}

}  // namespace gl_utils

#endif  // THIRD_PARTY_PY_TENSORFLOW_GRAPHICS_RENDERING_OPENGL_GL_SHADER_STORAGE_BUFFER_H_
