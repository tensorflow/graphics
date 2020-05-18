/* Copyright 2020 Google LLC

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
#include "tensorflow_graphics/rendering/opengl/gl_shader_storage_buffer.h"

#include "gtest/gtest.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace {

TEST(GLUtilsTest, TestShaderStorageBuffer) {
  std::unique_ptr<gl_utils::ShaderStorageBuffer> shader_storage_buffer;

  TF_ASSERT_OK(gl_utils::ShaderStorageBuffer::Create(&shader_storage_buffer));
  std::vector<float> data{1.0f, 2.0f};
  TF_EXPECT_OK(shader_storage_buffer->Upload(absl::MakeSpan(data)));
}

TEST(GLUtilsTest, TestBindShaderStorageBuffer) {
  std::unique_ptr<gl_utils::ShaderStorageBuffer> shader_storage_buffer;

  std::vector<float> data{1.0f, 2.0f};
  TF_ASSERT_OK(gl_utils::ShaderStorageBuffer::Create(&shader_storage_buffer));
  TF_ASSERT_OK(shader_storage_buffer->Upload(absl::MakeSpan(data)));
  TF_EXPECT_OK(shader_storage_buffer->BindBufferBase(0));
}

}  // namespace
