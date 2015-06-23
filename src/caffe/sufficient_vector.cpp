#include <cstring>

#include "caffe/sufficient_vector.hpp"

namespace caffe {

SufficientVector::SufficientVector(size_t a_size, size_t b_size) {
  a_size_ = a_size;
  b_size_ = b_size;
  a_.reset(new SyncedMemory(a_size_));
  b_.reset(new SyncedMemory(b_size_));
}

SufficientVector::~SufficientVector() {
  a_.reset();
  b_.reset();
}

const void* SufficientVector::cpu_a() const {
  CHECK(a_);
  return a_->cpu_data();
}
const void* SufficientVector::gpu_a() const {
  CHECK(a_);
  return a_->gpu_data();
}
const void* SufficientVector::cpu_b() const {
  CHECK(b_);
  return b_->cpu_data();
}
const void* SufficientVector::gpu_b() const {
  CHECK(b_);
  return b_->gpu_data();
}

void* SufficientVector::mutable_cpu_a() {
  CHECK(a_);
  return a_->mutable_cpu_data();
}
void* SufficientVector::mutable_gpu_a() {
  CHECK(a_);
  return a_->mutable_gpu_data();
}
void* SufficientVector::mutable_cpu_b() {
  CHECK(b_);
  return b_->mutable_cpu_data();
}
void* SufficientVector::mutable_gpu_b() {
  CHECK(b_);
  return b_->mutable_gpu_data();
}

}  // namespace caffe
