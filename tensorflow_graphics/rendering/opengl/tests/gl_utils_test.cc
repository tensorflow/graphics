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

#include <GLES3/gl32.h>
#include <memory>
#include <string>

#include "gtest/gtest.h"
#include "absl/types/span.h"
#include "tensorflow/core/lib/core/status_test_util.h"

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

}  // namespace
