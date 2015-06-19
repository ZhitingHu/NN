#ifndef CAFFE_SUFFICIENT_VECTOR_HPP_
#define CAFFE_SUFFICIENT_VECTOR_HPP_

#include <cstdlib>
#include <cstdio>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

/**
 * @brief 
 */
class SufficientVector {
 public:
  SufficientVector()
      : a_(NULL), b_(NULL), size_a_(0), size_b_(0) {}
  explicit SufficientVector(size_t size_a, size_t size_b)
      : a_(NULL), b_(NULL), size_a_(size_a), size_b_(size_b) {}
  ~SufficientVector();
  const void* a();
  const void* b();
  inline const size_t size_a() { return size_a_; }
  inline const size_t size_b() { return size_b_; }
  void* mutable_a();
  void* mutable_b();

 private:
  void to_cpu();
  // M = a_ x b_^{T}
  void* a_;
  void* b_;
  size_t size_a_;
  size_t size_b_;

  DISABLE_COPY_AND_ASSIGN(SufficientVector);
};  // class SufficientVector 

}  // namespace caffe

#endif  // CAFFE_SUFFICIENT_VECTOR_HPP_
