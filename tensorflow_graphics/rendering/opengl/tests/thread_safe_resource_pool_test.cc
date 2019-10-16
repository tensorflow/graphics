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
#include "tensorflow_graphics/rendering/opengl/thread_safe_resource_pool.h"

#include <array>
#include <memory>
#include <thread>

#include "gtest/gtest.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace {

constexpr int kDummyIncrements = 10;

class DummyClass {
 public:
  DummyClass() {
    for (int i = 0; i < kDummyIncrements; ++i) ++counter_;
    value_ = counter_;
  }

  int GetValue() { return value_; }
  void SetValue(int value) { value_ = value; }
  static void ResetCounter(){ counter_ = 0; }

 private:
  static int counter_;
  int value_;
};

int DummyClass::counter_ = 0;

static tensorflow::Status dummy_resource_creator(
    std::unique_ptr<DummyClass> *resource) {
  *resource = std::unique_ptr<DummyClass>(new DummyClass());
  return tensorflow::Status::OK();
}

TEST(ThreadSafeResourcePoolTest, TestSingleThread) {
  DummyClass::ResetCounter();
  constexpr int kPoolSize = 1;
  auto resource_pool = std::unique_ptr<ThreadSafeResourcePool<DummyClass>>(
      new ThreadSafeResourcePool<DummyClass>(dummy_resource_creator,
                                             kPoolSize));
  {
    std::unique_ptr<DummyClass> local_resource;
    TF_ASSERT_OK(resource_pool->AcquireResource(&local_resource));
    EXPECT_EQ(local_resource->GetValue(), kDummyIncrements);
    local_resource->SetValue(1);
    EXPECT_EQ(local_resource->GetValue(), 1);
  }
  std::unique_ptr<DummyClass> resource;
  TF_ASSERT_OK(resource_pool->AcquireResource(&resource));
  EXPECT_EQ(resource->GetValue(), 2 * kDummyIncrements);
}

TEST(ThreadSafeResourcePoolTest, TestPoolSize) {
  // we can acquire resource, return it and find the expected state.
  DummyClass::ResetCounter();
  constexpr int kPoolSize = 1;
  auto resource_pool = std::unique_ptr<ThreadSafeResourcePool<DummyClass>>(
      new ThreadSafeResourcePool<DummyClass>(dummy_resource_creator,
                                             kPoolSize));

  {
    std::unique_ptr<DummyClass> local_resource_1;
    std::unique_ptr<DummyClass> local_resource_2;
    TF_ASSERT_OK(resource_pool->AcquireResource(&local_resource_1));
    local_resource_1->SetValue(1);
    TF_ASSERT_OK(resource_pool->AcquireResource(&local_resource_2));
    local_resource_2->SetValue(2);
    TF_ASSERT_OK(resource_pool->ReturnResource(local_resource_1));
    TF_ASSERT_OK(resource_pool->ReturnResource(local_resource_2));
  }
  std::unique_ptr<DummyClass> resource;
  TF_ASSERT_OK(resource_pool->AcquireResource(&resource));
  EXPECT_EQ(resource->GetValue(), 1);
}

TEST(ThreadSafeResourcePoolTest, TestMultiThread) {
  DummyClass::ResetCounter();
  constexpr int kPoolSize = 1;
  auto resource_pool = std::shared_ptr<ThreadSafeResourcePool<DummyClass>>(
      new ThreadSafeResourcePool<DummyClass>(dummy_resource_creator,
                                             kPoolSize));
  constexpr int kNumThreads = 50;
  std::array<std::thread, kNumThreads> threads;
  std::array<std::unique_ptr<DummyClass>, kNumThreads> local_resources;

  // Launch all the threads.
  for (int i = 0; i < kNumThreads; ++i) {
    threads[i] =
        std::thread(&ThreadSafeResourcePool<DummyClass>::AcquireResource, resource_pool, &local_resources[i]);
  }

  // Wait for each thread to be done and check the value contained in the
  // resource.
  for (int i = 0; i < kNumThreads; ++i) {
    threads[i].join();
    EXPECT_EQ(local_resources[i]->GetValue() % kDummyIncrements, 0);
    TF_ASSERT_OK(resource_pool->ReturnResource(local_resources[i]));
  }

  // Check that we can re-acquire a previously created resource.
  TF_ASSERT_OK(resource_pool->AcquireResource(&local_resources[0]));
  EXPECT_TRUE(local_resources[0]->GetValue() <= kNumThreads * kDummyIncrements);
}

TEST(ThreadSafeResourcePoolTest, TestInvalidResourceCreator) {
  constexpr int kPoolSize = 1;
  std::function<tensorflow::Status(std::unique_ptr<DummyClass> *)>
      bad_resource_creator = [](std::unique_ptr<DummyClass> *) {
        return tensorflow::Status::OK();
      };
  auto resource_pool = std::unique_ptr<ThreadSafeResourcePool<DummyClass>>(
      new ThreadSafeResourcePool<DummyClass>(bad_resource_creator, kPoolSize));
  std::unique_ptr<DummyClass> resource;

  EXPECT_NE(resource_pool->AcquireResource(&resource),
            tensorflow::Status::OK());
}

TEST(ThreadSafeResourcePoolTest, TestReturnEmptyResource) {
  constexpr int kPoolSize = 1;
  auto resource_pool = std::unique_ptr<ThreadSafeResourcePool<DummyClass>>(
      new ThreadSafeResourcePool<DummyClass>(dummy_resource_creator,
                                             kPoolSize));
  std::unique_ptr<DummyClass> resource;

  EXPECT_NE(resource_pool->ReturnResource(resource), tensorflow::Status::OK());
}

}  // namespace
