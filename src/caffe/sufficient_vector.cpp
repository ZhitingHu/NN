#include <cstring>

#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/sufficient_vector.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

SufficientVector::~SufficientVector() {
  CaffeFreeHost(a_);
  CaffeFreeHost(b_);
}

const void* SufficientVector::a() {
  return (const void*)a_;
}

const void* SufficientVector::b() {
  return (const void*)b_;
}

void* SufficientVector::mutable_a() {
  return a_;
}

void* SufficientVector::mutable_b() {
  return b_;
}

}  // namespace caffe
