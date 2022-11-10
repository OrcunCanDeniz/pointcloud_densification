// Copyright 2021 Tier IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef POINTCLOUD_DENSIFICATION_HPP_
#define POINTCLOUD_DENSIFICATION_HPP_

#include <tf2_ros/buffer.h>
#include <eigen3/Eigen/Geometry>

#include <sensor_msgs/PointCloud2.h>

#include "config.hpp"

#include "cuda_utils.hpp"

#include <list>
#include <string>
#include <utility>

#define FINAL_FT_NUM 4 // XYZI


struct tf_time_t{
    float transform[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    float timelag = 0;
    bool last = 0;

    void setTf(float* data)
    {
      for(int i=0; i<16; i++)
      {
        transform[i] = data[i];
      }
    }
    void setNewest()
    {
      last=true;
    }
};

struct field_t
{
  int offset;
  size_t len;
  uint32_t count;

  field_t () {};
  field_t(const sensor_msgs::PointField& field_msg)
  {
    update(field_msg);
  }

  void update(const sensor_msgs::PointField& field_msg)
  {
    offset = field_msg.offset;
    count = field_msg.count;

    switch(field_msg.datatype)
    {
      case 1: 
        len = sizeof(int8_t);
        break;
      case 2:
        len = sizeof(uint8_t);
        break;
      case 3:
        len = sizeof(int16_t);
        break;
      case 4:
        len = sizeof(uint16_t);
        break;
      case 5:
        len = sizeof(int32_t);
        break;
      case 6:
        len = sizeof(uint32_t);
        break;
      case 7:
        len = sizeof(float);
        break;
      case 8:
        len = sizeof(double);
        break;
    }
  }
};

namespace centerpoint
{
class DensificationParam
{
public:
  DensificationParam(const std::string & world_frame_id, const unsigned int num_past_frames)
  : world_frame_id_(std::move(world_frame_id)),
    pc_cache_len_(num_past_frames + /*current frame*/ 1)
  {
  }

  std::string world_frame_id() const { return world_frame_id_; }
  inline unsigned int cache_len() const { return pc_cache_len_; }

private:
  std::string world_frame_id_;
  unsigned int pc_cache_len_{1};
};

struct Sweep
{
  // Struct to hold a Lidar sweep and other necessities for densification
  Sweep(uint8_t* src_ptr, double time, Eigen::Affine3f tf, float* dst_ptr): 
          src_data(src_ptr), time_secs(time), affine_past2world(tf), dst_data(dst_ptr){}
  uint8_t* src_data;
  float* dst_data;
  double time_secs;
  Eigen::Affine3f affine_past2world;
};

class PointCloudDensification
{
public:
  explicit PointCloudDensification(const DensificationParam & param);

  void registerSweep(const sensor_msgs::PointCloud2::ConstPtr & input_pointcloud_msg, const tf2_ros::Buffer & tf_buffer);

  inline double getCurrentTimestamp() const { return current_timestamp_; }
  inline Eigen::Affine3f getAffineWorldToCurrent() const { return affine_world2current_; }
  void densify();
  void init(const sensor_msgs::PointCloud2::ConstPtr& pc_msg, const int& pc_size);
  void format(const sensor_msgs::PointCloud2::ConstPtr& pc_msg);
  std::pair<uint8_t*, float*>  refreshCache();
  std::vector<float> getCloud();
  
  inline std::list<Sweep>::iterator getPointCloudCacheIter()
  {
    return pointcloud_cache_.begin();
  }
  
  inline bool isCacheEnd(std::list<Sweep>::iterator iter)
  {
    return iter == pointcloud_cache_.end();
  }
  
  inline bool initialized(){return is_init;}

  void appendPclCloud(std::array<float, Config::num_point_features> point);
  sensor_msgs::PointCloud2 getDenseCloud();
  void clearPclCloud();
  void dispatch( uint8_t* msg, float* dst, tf_time_t tf_str);

private:
  void enqueue(const sensor_msgs::PointCloud2::ConstPtr & msg, const Eigen::Affine3f & affine);
  void allocCuda(const int& point_step_bytes);
  void dequeue();

  DensificationParam param_;
  double current_timestamp_{0.0};
  Eigen::Affine3f affine_world2current_;
  int num_points_{0}; // number of points in each poincloud msg, constant for data from same sensors
  int point_step_{0}; // number of uint8_t elements belonging to single point in the array
  std::list<Sweep> pointcloud_cache_;

  std::map<std::string, field_t> field_to_dtype_m_;
  bool is_init{false};

  //CUDA stuff
  uint8_t* msgs_buffer_d; 
  float* dns_buffer_d;
  float* dst_h;
  std::vector<cudaStream_t> streams;
  
};

}  // namespace centerpoint

#endif  // POINTCLOUD_DENSIFICATION_HPP_
