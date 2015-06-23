#include <cstring>
#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/sufficient_vector.hpp"
#include "caffe/sufficient_vector_queue.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

SufficientVectorQueue::~SufficientVectorQueue() {
  while(!sv_queue_.empty()) {
    delete sv_queue_.front();
    sv_queue_.pop();
  }
}

void SufficientVectorQueue::Add(SufficientVector* v) {
  std::unique_lock<std::mutex> lock(mtx_);
  sv_queue_.push(v);
}

bool SufficientVectorQueue::Get(SufficientVector* v) {
  std::unique_lock<std::mutex> lock(mtx_);
  if (sv_queue_.empty()) {
    return false;
  }
  // copy the front to v
  SufficientVector* front = sv_queue_.front();
  v = new SufficientVector(front->a_size(), front->b_size());
  caffe_gpu_memcpy(v->a_size(), front->cpu_a(), v->mutable_gpu_a());
  
  if (++read_count_ >= max_read_count_) {
    delete front;
    sv_queue_.pop();
    read_count_ = 0;
  }
  return true;
}

}  // namespace caffe
