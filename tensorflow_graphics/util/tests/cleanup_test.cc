#Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#include "third_party/py/tensorflow_graphics/util/cleanup.h"

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"

namespace {

static int global_value = -1;

void cleanup_function() { ++global_value; }

TEST(CleanupTest, TestCleanupLambda) {
  int value = 1;
  {
    auto cleaner = MakeCleanup([&value]() { ++value; });
  }
  EXPECT_EQ(value, 2);
}

TEST(CleanupTest, TestCleanupFunction) {
  global_value = 1;
  { auto cleaner = MakeCleanup(cleanup_function); }
  EXPECT_EQ(global_value, 2);
}

TEST(CleanupTest, TestReleaseLambda) {
  int value = 1;
  {
    auto cleaner = MakeCleanup([&value]() { ++value; });
    cleaner.Release();
  }
  EXPECT_EQ(value, 1);
}

TEST(CleanupTest, TestReleaseFunction) {
  global_value = 1;
  {
    auto cleaner = MakeCleanup(cleanup_function);
    cleaner.Release();
  }
  EXPECT_EQ(global_value, 1);
}

}  // namespace
