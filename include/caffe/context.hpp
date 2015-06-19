#pragma once
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <boost/shared_ptr.hpp>

#include "leveldb/db.h"
#include "sufficient_vector.hpp"
namespace util {

// An extension of google flags. It is a singleton that stores 1) google flags
// and 2) other lightweight global flags. Underlying data structure is map of
// string and string, similar to google::CommandLineFlagInfo.
class Context {
public:
  static Context& get_instance();

  int get_int32(std::string key);
  double get_double(std::string key);
  bool get_bool(std::string key);
  std::string get_string(std::string key);

  void set(std::string key, int value);
  void set(std::string key, double value);
  void set(std::string key, bool value);
  void set(std::string key, std::string value);
  
  static int num_app_threads() { return get_instance().num_app_threads_; }
  static int num_rows_per_table() { return get_instance().num_rows_per_table_; }
  static bool svb_completed() { return get_instance().svb_completed_; }
  static void set_sev_completed() { get_instance().svb_completed_ = true; }
  static bool use_svb() { return get_instance().use_svb_; }
  static void set_use_svb() { get_instance().use_svb_ = true; }

  static std::vector<boost::shared_ptr<leveldb::DB> >& test_dbs() { 
    return get_instance().test_dbs_; 
  }
  static boost::shared_ptr<leveldb::DB>& global_db(int id) { 
    if (id == -1) {
      return get_instance().db_; 
    } else if (id >= 0) {
      CHECK_GT(get_instance().test_dbs_.size(), id);
      return get_instance().test_dbs_[id];
    } else {
      LOG(FATAL) << "Invalid net id " << id 
          << "; Valid net id is >= -1 and <= " 
          << get_instance().test_dbs_.size() - 1;
    }
  }

private:
  // Private constructor. Store all the gflags values.
  Context();
  // Underlying data structure
  std::unordered_map<std::string, std::string> ctx_;

  int num_app_threads_;
  int num_rows_per_table_;

  std::vector<std::vector<caffe::SufficientVector*> > send_buffer_;
  std::vector<std::vector<caffe::SufficientVector*> > recv_buffer_;
  bool use_svb_; // true of NN has inner_product layer(s)
  bool svb_completed_;

  // LEVELDB
  // different nets cannot use the same leveldb
  boost::shared_ptr<leveldb::DB> db_;
  std::vector<boost::shared_ptr<leveldb::DB> > test_dbs_;
};

}   // namespace util
