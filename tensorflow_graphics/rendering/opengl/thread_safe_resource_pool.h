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
#ifndef THIRD_PARTY_PY_TENSORFLOW_GRAPHICS_RENDERING_OPENGL_THREAD_SAFE_RESOURCE_POOL_H_
#define THIRD_PARTY_PY_TENSORFLOW_GRAPHICS_RENDERING_OPENGL_THREAD_SAFE_RESOURCE_POOL_H_

#include <functional>
#include <iostream>
#include <memory>
#include <vector>

#include "absl/synchronization/mutex.h"
#include "macros.h"
#include "tensorflow/core/lib/core/status.h"

template <typename T>
class ThreadSafeResourcePool
    : public std::enable_shared_from_this<ThreadSafeResourcePool<T>> {
 public:
  // Arguments:
  // * resource_creator: an std::function that returns a unique_ptr on a
  // resource.
  // * maximum_pool_size: the maximum number of resources stored at once in the
  // pool.
  ThreadSafeResourcePool(
      std::function<tensorflow::Status(std::unique_ptr<T>*)> resource_creator,
      unsigned int maximum_pool_size = 5);

  // Acquires a unique_ptr on a resource.
  //
  // Arguments:
  // * resource: if the resource aquisition is successful, stores a pointer on
  // the acquired resource.
  //
  // Returns:
  //   A tensorflow::Status object storing tensorflow::Status::OK() on success,
  //   and an object of type tensorflow::errors otherwise.
  tensorflow::Status AcquireResource(std::unique_ptr<T>* resource);

  // Returns a resource to the pool. If the pool is full, the resource is
  // deleted.
  //
  // Arguments:
  // * resource: the resource to return to the pool.
  //
  // Returns:
  //   A tensorflow::Status object storing tensorflow::Status::OK() on success,
  //   and an object of type tensorflow::errors otherwise.
  tensorflow::Status ReturnResource(std::unique_ptr<T>& resource);

 private:
  unsigned int maximum_pool_size_;
  absl::Mutex mutex_;
  std::function<tensorflow::Status(std::unique_ptr<T>*)> resource_creator_;
  std::vector<std::unique_ptr<T>> resource_pool_;
};

template <typename T>
ThreadSafeResourcePool<T>::ThreadSafeResourcePool(
    const std::function<tensorflow::Status(std::unique_ptr<T>*)>
        resource_creator,
    const unsigned int maximum_pool_size)
    : maximum_pool_size_(maximum_pool_size),
      resource_creator_(resource_creator) {
  resource_pool_.reserve(maximum_pool_size);
}

template <typename T>
tensorflow::Status ThreadSafeResourcePool<T>::AcquireResource(
    std::unique_ptr<T>* resource) {
  absl::MutexLock lock(&mutex_);

  // Creates a new resource or get it from the pool.
  if (resource_pool_.empty()) {
    TF_RETURN_IF_ERROR(resource_creator_(resource));
    if (resource->get() == nullptr) {
      return TFG_INTERNAL_ERROR(
          "The resource creator returned an empty resource.");
    }
  } else {
    *resource = std::move(resource_pool_.back());
    resource_pool_.pop_back();
  }
  return tensorflow::Status::OK();
}

template <typename T>
tensorflow::Status ThreadSafeResourcePool<T>::ReturnResource(
    std::unique_ptr<T>& resource) {
  absl::MutexLock lock(&mutex_);

  if (resource.get() == nullptr)
    return TFG_INTERNAL_ERROR("Attempting to return an empty resource");

  // Adds the resource to the pool if not full, release it otherwise.
  if (resource_pool_.size() < maximum_pool_size_)
    resource_pool_.push_back(std::move(resource));
  else
    resource.reset();
  return tensorflow::Status::OK();
}

#endif  // THIRD_PARTY_PY_TENSORFLOW_GRAPHICS_RENDERING_OPENGL_THREAD_SAFE_RESOURCE_POOL_H_
