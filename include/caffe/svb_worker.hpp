#ifndef CAFFE_SVBWORKER_HPP_
#define CAFFE_SVBWORKER_HPP_

#include <map>
#include <petuum_ps_common/include/host_info.hpp>
#include <petuum_ps_common/util/utils.hpp>
#include <petuum_ps_common/include/petuum_ps.hpp>

namespace caffe {

/**
 * @brief 
 */
class SVBWorker {
 public:
  SVBWorker();

  void Init();
  
  void Start();

 protected:
  void Connect();
  void Disconnect();
  void Send();
  void Receive();

  int client_id_;
  int port_;
  // mapping server ID to host info.
  std::map<int32_t, petuum::HostInfo> host_map_;
  petuum::CommBus* comm_bus_;
};

}  // namespace caffe

#endif  // CAFFE_SVBWORKER_HPP_
