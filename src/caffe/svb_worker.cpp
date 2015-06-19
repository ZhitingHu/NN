#include <cstdio>

#include <algorithm>
#include <string>
#include <vector>

#include "caffe/context.hpp"
#include "caffe/svb_worker.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

SVBWorker::SVBWorker() : comm_bus_(NULL) {
  Init();
}

void SVBWorker::Init() {
  util::Context& context = util::Context::get_instance();
  client_id_ = context.get_int32("client_id"); 
  LOG(INFO) << "svb " << client_id_;

  const string& host_file = context.get_string("hostfile");
  petuum::GetHostInfos(host_file, &host_map_);

  auto &host_info = host_map_[client_id_];
  port_ = std::stoi(host_info.port, 0) + 1000;
}

void SVBWorker::Connect() {
  auto &host_info = host_map_[client_id_];
  if (host_map_.size() >= 3) {
    comm_bus_ = new petuum::CommBus(client_id_, client_id_, host_map_.size());
    int ltype = (client_id_ == host_map_.size() - 1)
                ? petuum::CommBus::kNone : petuum::CommBus::kInterProc;
    petuum::CommBus::Config config(client_id_, ltype, host_info.ip +
                                   + ":" + std::to_string(port_));
    comm_bus_->ThreadRegister(config);

    /* This part of the code represents the handshake process to establish
       conenctions to all other clients. After this part, one client can send msgs
       to any other. */

    // client i connects to client [0, ..., i - 1]
    for (int i = 0; i < client_id_; ++i) {
      int conn_msg = client_id_;
      auto &other_host_info = host_map_[i];
      int other_port = std::stoi(other_host_info.port, 0) + 1000;
      comm_bus_->ConnectTo(
          i, other_host_info.ip + ":" + std::to_string(other_port),
          &conn_msg, sizeof(conn_msg));
    }

    // client i receives connection from [i + 1, n) (n is the total number of
    // clients);
    // client i also receives n messages each from a client
    // msgs from [0, i - 1] serves as confirmation of the connection
    // A client can broadcast only when it has 1) connected to all lower ones
    // and 2) received confirmation from them
    const int num_expected_conns = host_map_.size() - client_id_ - 1;
    int num_conns = 0;
    int num_other_msgs = 0;
    const int num_expected_confirms = client_id_;
    int num_confirms = 0;
    while (num_conns < num_expected_conns
           || num_confirms < num_expected_confirms) {
      zmq::message_t msg;
      int32_t sender_id;
      comm_bus_->RecvInterProc(&sender_id, &msg);
      int msg_val = *reinterpret_cast<int*>(msg.data());
      if (msg_val == sender_id) {
        num_conns++;
        LOG(INFO) << client_id_ <<  " received conn from " << sender_id
                << " msg = " << msg_val;
      } else {
        num_other_msgs++;
        LOG(INFO) << client_id_ <<  " received nonconn from " << sender_id
                << " msg = " << msg_val;
        if (sender_id < client_id_) num_confirms++;
      }
    }
  }

  // From now on, one client can send and receive from any other (not itself).
  LOG(INFO) << "Client " << client_id_ << " connected.";
}

void SVBWorker::Disconnect() {
  comm_bus_->ThreadDeregister();
  delete comm_bus_;
  LOG(INFO) << "Client " << client_id_ << " disconnected.";
}

// TODO
void SVBWorker::Send() {
  //for (int32_t dst = 0; dst < host_map_.size(); dst++) {
  //  int non_conn_msg = client_id_ + 100;
  //  if (dst == client_id_) continue; // cannot send to myself
  //  comm_bus_->SendInterProc(dst, &non_conn_msg, sizeof(non_conn_msg));
  //}
}

// TODO
void SVBWorker::Receive() {
  //while (num_other_msgs < host_map_.size() - 1) {
  //  zmq::message_t msg;
  //  int32_t sender_id;
  //  comm_bus_->RecvInterProc(&sender_id, &msg);
  //  int msg_val = *reinterpret_cast<int*>(msg.data());
  //  if (msg_val == sender_id) {
  //    LOG(FATAL) << "Error!";
  //  } else {
  //    num_other_msgs++;
  //    LOG(INFO) << client_id_ <<  " received nonconn from " << sender_id
  //            << " msg = " << msg_val;
  //  }
  //}
}

void SVBWorker::Start() {
  Connect();

  while(!util::Context::svb_completed()) {
    //Send();
    //Receive();
  }  
  
  Disconnect();
}

}  // namespace caffe
