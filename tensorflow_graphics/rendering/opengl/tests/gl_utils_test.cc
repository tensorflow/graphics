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
#include <string>

#include "gtest/gtest.h"
#include <GLES3/gl32.h>
#include "absl/types/span.h"
#include "tensorflow_graphics/rendering/opengl/egl_offscreen_context.h"

namespace {

const std::string kEmptyShaderCode =
    "#version 460\n"
    "void main() { }\n";

const std::string geometry_shader_code =
    "#version 460\n"
    "\n"
    "uniform mat4 view_projection_matrix;\n"
    "\n"
    "layout(points) in;\n"
    "layout(triangle_strip, max_vertices=2) out;\n"
    "\n"
    "void main() {\n"
    "  gl_Position = view_projection_matrix * vec4(1.0,2.0,3.0,4.0);\n"
    "\n"
    "  EmitVertex();\n"
    "  EmitVertex();\n"
    "  EndPrimitive();\n"
    "}\n";

TEST(ProgramTest, TestCompileInvalidShader) {
  std::unique_ptr<EGLOffscreenContext> context;
  std::unique_ptr<gl_utils::Program> program;
  const std::string kInvalidShaderCode =
      "#version 460\n"
      "uniform mat4 view_projection_matrix;\n"
      "void main() { syntax_error }\n";

  EXPECT_TRUE(EGLOffscreenContext::Create(&context));
  EXPECT_TRUE(context->MakeCurrent());

  std::vector<std::pair<std::string, GLenum>> shaders;
  shaders.push_back(
      std::pair<std::string, GLenum>(kInvalidShaderCode, GL_VERTEX_SHADER));
  EXPECT_FALSE(gl_utils::Program::Create(shaders, &program));
  EXPECT_TRUE(context->Release());
}

TEST(ProgramTest, TestCompileInvalidShaderType) {
  std::unique_ptr<EGLOffscreenContext> context;
  std::unique_ptr<gl_utils::Program> program;
  const std::string kInvalidShaderCode =
      "#version 460\n"
      "void main() { syntax_error }\n";

  EXPECT_TRUE(EGLOffscreenContext::Create(&context));
  EXPECT_TRUE(context->MakeCurrent());

  std::vector<std::pair<std::string, GLenum>> shaders;
  shaders.push_back(std::pair<std::string, GLenum>(kEmptyShaderCode, 0));
  EXPECT_FALSE(gl_utils::Program::Create(shaders, &program));
  EXPECT_TRUE(context->Release());
}

TEST(ProgramTest, TestCreateProgram) {
  std::unique_ptr<EGLOffscreenContext> context;
  std::unique_ptr<gl_utils::Program> program;

  EXPECT_TRUE(EGLOffscreenContext::Create(&context));
  EXPECT_TRUE(context->MakeCurrent());

  std::vector<std::pair<std::string, GLenum>> shaders;
  shaders.push_back(
      std::pair<std::string, GLenum>(kEmptyShaderCode, GL_VERTEX_SHADER));
  EXPECT_TRUE(gl_utils::Program::Create(shaders, &program));
  EXPECT_TRUE(context->Release());
}

TEST(ProgramTest, TestGetNonExistingResourceProperty) {
  std::unique_ptr<EGLOffscreenContext> context;
  std::unique_ptr<gl_utils::Program> program;
  GLint property_value;
  const GLenum kProperty = GL_TYPE;

  EXPECT_TRUE(EGLOffscreenContext::Create(&context));
  EXPECT_TRUE(context->MakeCurrent());

  std::vector<std::pair<std::string, GLenum>> shaders;
  shaders.push_back(
      std::pair<std::string, GLenum>(kEmptyShaderCode, GL_VERTEX_SHADER));
  EXPECT_TRUE(gl_utils::Program::Create(shaders, &program));
  EXPECT_FALSE(program->GetResourceProperty("resource_name", GL_UNIFORM, 1,
                                            &kProperty, 1, &property_value));
  EXPECT_TRUE(context->Release());
}

TEST(ProgramTest, TestGetExistingResourceProperty) {
  std::unique_ptr<EGLOffscreenContext> context;
  std::unique_ptr<gl_utils::Program> program;
  GLint property_value;
  GLenum kProperty = GL_TYPE;

  EXPECT_TRUE(EGLOffscreenContext::Create(&context));
  EXPECT_TRUE(context->MakeCurrent());

  std::vector<std::pair<std::string, GLenum>> shaders{
      std::make_pair(kEmptyShaderCode, GL_VERTEX_SHADER),
      std::make_pair(geometry_shader_code, GL_GEOMETRY_SHADER)};
  EXPECT_TRUE(gl_utils::Program::Create(shaders, &program));
  EXPECT_TRUE(program->GetResourceProperty("view_projection_matrix", GL_UNIFORM,
                                           1, &kProperty, 1, &property_value));
  EXPECT_TRUE(context->Release());
}

template <typename T>
class RenderTargetsInterfaceTest : public ::testing::Test {};
using valid_render_target_types = ::testing::Types<float, unsigned char>;
TYPED_TEST_CASE(RenderTargetsInterfaceTest, valid_render_target_types);

TYPED_TEST(RenderTargetsInterfaceTest, TestRenderClear) {
  std::unique_ptr<EGLOffscreenContext> context;
  const float kRed = 0.1;
  const float kGreen = 0.2;
  const float kBlue = 0.3;
  const float kAlpha = 1.0;
  const int kWidth = 10;
  const int kHeight = 5;
  float max_gl_value = 255.0f;

  if (typeid(TypeParam) == typeid(float)) max_gl_value = 1.0f;

  EXPECT_TRUE(EGLOffscreenContext::Create(&context));
  EXPECT_TRUE(context->MakeCurrent());

  std::unique_ptr<gl_utils::RenderTargets<TypeParam>> render_targets;
  EXPECT_TRUE(gl_utils::RenderTargets<TypeParam>::Create(kWidth, kHeight,
                                                         &render_targets));
  glClearColor(kRed, kGreen, kBlue, kAlpha);
  glClear(GL_COLOR_BUFFER_BIT);
  ASSERT_EQ(glGetError(), GL_NO_ERROR);
  std::vector<TypeParam> pixels(kWidth * kHeight * 4);
  EXPECT_TRUE(render_targets->CopyPixelsInto(absl::MakeSpan(pixels)));

  for (int index = 0; index < kWidth * kHeight; ++index) {
    ASSERT_EQ(pixels[index * 4], TypeParam(kRed * max_gl_value));
    ASSERT_EQ(pixels[index * 4 + 1], TypeParam(kGreen * max_gl_value));
    ASSERT_EQ(pixels[index * 4 + 2], TypeParam(kBlue * max_gl_value));
    ASSERT_EQ(pixels[index * 4 + 3], TypeParam(kAlpha * max_gl_value));
  }
  EXPECT_TRUE(context->Release());
}

TYPED_TEST(RenderTargetsInterfaceTest, TestReadFails) {
  std::unique_ptr<EGLOffscreenContext> context;
  const int kWidth = 10;
  const int kHeight = 5;

  EXPECT_TRUE(EGLOffscreenContext::Create(&context));
  EXPECT_TRUE(context->MakeCurrent());
  std::unique_ptr<gl_utils::RenderTargets<TypeParam>> render_targets;
  EXPECT_TRUE(gl_utils::RenderTargets<TypeParam>::Create(kWidth, kHeight,
                                                         &render_targets));
  std::vector<TypeParam> pixels(kWidth * kHeight * 3);
  EXPECT_FALSE(render_targets->CopyPixelsInto(absl::MakeSpan(pixels)));
}

TEST(GLUtilsTest, TestShaderStorageBuffer) {
  std::unique_ptr<gl_utils::ShaderStorageBuffer> shader_storage_buffer;

  EXPECT_TRUE(gl_utils::ShaderStorageBuffer::Create(&shader_storage_buffer));
  std::vector<float> data{1.0f, 2.0f};
  EXPECT_TRUE(shader_storage_buffer->Upload(absl::MakeSpan(data)));
}

TEST(GLUtilsTest, TestBindShaderStorageBuffer) {
  std::unique_ptr<gl_utils::ShaderStorageBuffer> shader_storage_buffer;

  std::vector<float> data{1.0f, 2.0f};
  EXPECT_TRUE(gl_utils::ShaderStorageBuffer::Create(&shader_storage_buffer));
  EXPECT_TRUE(shader_storage_buffer->Upload(absl::MakeSpan(data)));
  EXPECT_TRUE(shader_storage_buffer->BindBufferBase(0));
}

}  // namespace
