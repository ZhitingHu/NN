#ifndef CAFFE_SUFFICIENT_VECTOR_HPP_
#define CAFFE_SUFFICIENT_VECTOR_HPP_

#include <cstdlib>
#include <cstdio>

#include "caffe/proto/caffe.pb.h"
#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

/**
 * @brief 
 */
template <typename Dtype>
class SufficientVector {
 public:
  explicit SufficientVector(const size_t a_size, const size_t b_size,
      const int layer_id);
  explicit SufficientVector();
  ~SufficientVector();

  void Reshape(const size_t a_size, const size_t b_size);

  inline int layer_id() { return layer_id_; } 
  const Dtype* cpu_a() const;
  const Dtype* gpu_a() const;
  const Dtype* cpu_b() const;
  const Dtype* gpu_b() const;
  inline size_t a_size() const { return a_size_; }
  inline size_t b_size() const { return b_size_; }
  Dtype* mutable_cpu_a();
  Dtype* mutable_gpu_a();
  Dtype* mutable_cpu_b();
  Dtype* mutable_gpu_b();
  
  void FromProto(const SVProto& proto);
  void ToProto(SVProto* proto) const;

 private:
  // M = a_ x b_^{T}
  shared_ptr<SyncedMemory> a_;
  shared_ptr<SyncedMemory> b_;
  size_t a_size_;
  size_t b_size_;
  int layer_id_;
};  // class SufficientVector 

}  // namespace caffe

#endif  // CAFFE_SUFFICIENT_VECTOR_HPP_
