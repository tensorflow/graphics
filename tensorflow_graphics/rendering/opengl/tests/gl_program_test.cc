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
#include "tensorflow_graphics/rendering/opengl/gl_program.h"

#include "gtest/gtest.h"
#include "tensorflow_graphics/rendering/opengl/egl_offscreen_context.h"
#include "tensorflow/core/lib/core/status_test_util.h"

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

  TF_ASSERT_OK(EGLOffscreenContext::Create(&context));
  TF_ASSERT_OK(context->MakeCurrent());

  std::vector<std::pair<std::string, GLenum>> shaders;
  shaders.push_back(
      std::pair<std::string, GLenum>(kInvalidShaderCode, GL_VERTEX_SHADER));
  EXPECT_NE(gl_utils::Program::Create(shaders, &program),
            tensorflow::Status::OK());
  TF_EXPECT_OK(context->Release());
}

TEST(ProgramTest, TestCompileInvalidShaderType) {
  std::unique_ptr<EGLOffscreenContext> context;
  std::unique_ptr<gl_utils::Program> program;
  const std::string kInvalidShaderCode =
      "#version 460\n"
      "void main() { syntax_error }\n";

  TF_ASSERT_OK(EGLOffscreenContext::Create(&context));
  TF_ASSERT_OK(context->MakeCurrent());

  std::vector<std::pair<std::string, GLenum>> shaders;
  shaders.push_back(std::pair<std::string, GLenum>(kEmptyShaderCode, 0));
  EXPECT_NE(gl_utils::Program::Create(shaders, &program),
            tensorflow::Status::OK());
  TF_EXPECT_OK(context->Release());
}

TEST(ProgramTest, TestCreateProgram) {
  std::unique_ptr<EGLOffscreenContext> context;
  std::unique_ptr<gl_utils::Program> program;

  TF_ASSERT_OK(EGLOffscreenContext::Create(&context));
  TF_ASSERT_OK(context->MakeCurrent());

  std::vector<std::pair<std::string, GLenum>> shaders;
  shaders.push_back(
      std::pair<std::string, GLenum>(kEmptyShaderCode, GL_VERTEX_SHADER));
  TF_EXPECT_OK(gl_utils::Program::Create(shaders, &program));
  TF_EXPECT_OK(context->Release());
}

TEST(ProgramTest, TestGetNonExistingResourceProperty) {
  std::unique_ptr<EGLOffscreenContext> context;
  std::unique_ptr<gl_utils::Program> program;
  GLint property_value;
  const GLenum kProperty = GL_TYPE;

  TF_ASSERT_OK(EGLOffscreenContext::Create(&context));
  TF_ASSERT_OK(context->MakeCurrent());

  std::vector<std::pair<std::string, GLenum>> shaders;
  shaders.push_back(
      std::pair<std::string, GLenum>(kEmptyShaderCode, GL_VERTEX_SHADER));
  TF_ASSERT_OK(gl_utils::Program::Create(shaders, &program));
  EXPECT_NE(program->GetResourceProperty("resource_name", GL_UNIFORM, 1,
                                         &kProperty, 1, &property_value),
            tensorflow::Status::OK());
  TF_EXPECT_OK(context->Release());
}

TEST(ProgramTest, TestGetExistingResourceProperty) {
  std::unique_ptr<EGLOffscreenContext> context;
  std::unique_ptr<gl_utils::Program> program;
  GLint property_value;
  GLenum kProperty = GL_TYPE;

  TF_ASSERT_OK(EGLOffscreenContext::Create(&context));
  TF_ASSERT_OK(context->MakeCurrent());

  std::vector<std::pair<std::string, GLenum>> shaders{
      std::make_pair(kEmptyShaderCode, GL_VERTEX_SHADER),
      std::make_pair(geometry_shader_code, GL_GEOMETRY_SHADER)};
  TF_ASSERT_OK(gl_utils::Program::Create(shaders, &program));
  TF_EXPECT_OK(program->GetResourceProperty(
      "view_projection_matrix", GL_UNIFORM, 1, &kProperty, 1, &property_value));
  TF_EXPECT_OK(context->Release());
}

}  // namespace
