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

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "GL/gl/include/GLES3/gl32.h"
#include "tensorflow_graphics/rendering/opengl/egl_offscreen_context.h"

namespace {

const std::string kEmptyShaderCode =
    "#version 430\n"
    "void main() { }\n";

TEST(GLUtilsTest, TestCompileInvalidShader) {
  std::unique_ptr<EGLOffscreenContext> context;
  std::unique_ptr<gl_utils::Program> program;
  const std::string kInvalidShaderCode =
      "#version 430\n"
      "void main() { syntax_error }\n";

  EXPECT_TRUE(EGLOffscreenContext::Create(1, 1, &context));
  EXPECT_TRUE(context->MakeCurrent());

  std::vector<std::pair<std::string, GLenum>> shaders;
  shaders.push_back(
      std::pair<std::string, GLenum>(kInvalidShaderCode, GL_VERTEX_SHADER));
  EXPECT_FALSE(gl_utils::Program::Create(shaders, &program));
  EXPECT_TRUE(context->Release());
}

TEST(GLUtilsTest, TestCompileValidShaderType) {
  std::unique_ptr<EGLOffscreenContext> context;
  std::unique_ptr<gl_utils::Program> program;
  const std::string kInvalidShaderCode =
      "#version 430\n"
      "void main() { syntax_error }\n";

  EXPECT_TRUE(EGLOffscreenContext::Create(1, 1, &context));
  EXPECT_TRUE(context->MakeCurrent());

  std::vector<std::pair<std::string, GLenum>> shaders;
  shaders.push_back(std::pair<std::string, GLenum>(kEmptyShaderCode, 0));
  EXPECT_FALSE(gl_utils::Program::Create(shaders, &program));
  EXPECT_TRUE(context->Release());
}

TEST(GLUtilsTest, TestCreateProgram) {
  std::unique_ptr<EGLOffscreenContext> context;
  std::unique_ptr<gl_utils::Program> program;

  EXPECT_TRUE(EGLOffscreenContext::Create(1, 1, &context));
  EXPECT_TRUE(context->MakeCurrent());

  std::vector<std::pair<std::string, GLenum>> shaders;
  shaders.push_back(
      std::pair<std::string, GLenum>(kEmptyShaderCode, GL_VERTEX_SHADER));
  EXPECT_TRUE(gl_utils::Program::Create(shaders, &program));
  EXPECT_NE(0, program->GetHandle());
  EXPECT_TRUE(context->Release());
}

}  // namespace
