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

#include <pcl_ros/transforms.h>
#include <pointcloud_densification.hpp>

#include <boost/optional.hpp>

#include <tf2_eigen/tf2_eigen.h>

#include <string>
#include <utility>

namespace
{
boost::optional<geometry_msgs::Transform> getTransform(
  const tf2_ros::Buffer & tf_buffer, const std::string & target_frame_id,
  const std::string & source_frame_id, const ros::Time & time)
{
  try {
    geometry_msgs::TransformStamped transform_stamped;
    transform_stamped = tf_buffer.lookupTransform(
      target_frame_id, source_frame_id, time, ros::Duration(0.5));
    return transform_stamped.transform;
  } catch (tf2::TransformException & ex) {
    ROS_WARN("%s", ex.what());
    return boost::none;
  }
}

Eigen::Affine3f transformToEigen(const geometry_msgs::Transform & t)
{
  Eigen::Affine3f a;
  a.matrix() = tf2::transformToEigen(t).matrix().cast<float>();
  return a;
}

}  // namespace

namespace centerpoint
{
PointCloudDensification::PointCloudDensification(const DensificationParam & param) : param_(param)
{}

void PointCloudDensification::init(const sensor_msgs::PointCloud2::ConstPtr& pc_msg, const int& pc_size)
{
  num_points_ = pc_size;
  point_step_ = pc_msg->point_step / sizeof(uint8_t);
  format(pc_msg); // get description of pc msg to decode the uint8 array
  allocCuda(pc_msg->point_step);
}

void PointCloudDensification::registerSweep(
  const sensor_msgs::PointCloud2::ConstPtr& pointcloud_msg, const tf2_ros::Buffer & tf_buffer)
{
  const auto header = pointcloud_msg->header;

  auto transform_world2current = getTransform(tf_buffer, header.frame_id, param_.world_frame_id(), header.stamp);
  if (!transform_world2current) { return; }

  auto affine_world2current = transformToEigen(transform_world2current.get());
  
  enqueue(pointcloud_msg, affine_world2current);
}

void PointCloudDensification::enqueue(
  const sensor_msgs::PointCloud2::ConstPtr& msg, const Eigen::Affine3f & affine_world2current)
{
  affine_world2current_ = affine_world2current;
  
  std::pair<uint8_t*, float*> free_buffers = refreshCache(); // return pointer for new data, drop latest data in cache
  current_timestamp_ = ros::Time(msg->header.stamp).toSec();
  CHECK_CUDA_ERROR(cudaMemcpy(free_buffers.first, msg->data.data(), num_points_ * point_step_ * sizeof(uint8_t), cudaMemcpyHostToDevice)); // load new data to gpu
  pointcloud_cache_.emplace_front(free_buffers.first, current_timestamp_,
                                      affine_world2current.inverse(), free_buffers.second); // add new sweep to front of the cache
  densify();
}

std::pair<uint8_t*, float*> PointCloudDensification::refreshCache()
{
  // get next available pointer(holding outdated data) to hold newly arrived data
  // drop latest pointcloud
  static int registered_pc_num{0};

  const int cyclic_idx = registered_pc_num % param_.cache_len();
  std::cout << cyclic_idx << std::endl;
  uint8_t* src_loc = msgs_buffer_d + cyclic_idx * num_points_ * point_step_;
  float* dst_loc = dns_buffer_d + cyclic_idx * num_points_ * FINAL_FT_NUM;

  if (pointcloud_cache_.size() == param_.cache_len())
  {
    pointcloud_cache_.pop_back();
  }
  registered_pc_num ++;
  return std::make_pair(src_loc, dst_loc);
}

void PointCloudDensification::densify()
{
  for (auto cache_iter = getPointCloudCacheIter(); !isCacheEnd(cache_iter); cache_iter++) 
  {
    tf_time_t tf_time;
    tf_time.timelag = static_cast<float>(getCurrentTimestamp() - cache_iter->time_secs);
    if (cache_iter != getPointCloudCacheIter()) // if not the last received cloud
    {
      auto affine_past2current = getAffineWorldToCurrent() * cache_iter->affine_past2world;
      float* tf_mat = affine_past2current.matrix().data();
      tf_time.setTf(tf_mat);
    } else {
      tf_time.setNewest();
    }
    dispatch(cache_iter->src_data, cache_iter->dst_data, tf_time); 
  }
  cudaDeviceSynchronize();
}

std::vector<float> PointCloudDensification::getCloud()
{
  CHECK_CUDA_ERROR(cudaMemcpy(dst_h, dns_buffer_d, 
                                  num_points_ * FINAL_FT_NUM  * sizeof(float) * param_.cache_len(),
                                  cudaMemcpyDeviceToHost));

  return std::vector<float>(dst_h, dst_h + num_points_ * FINAL_FT_NUM * param_.cache_len());
}

void PointCloudDensification::allocCuda(const int& point_step_bytes)
{                                             // point_step: how many bytes does a point need
  CHECK_CUDA_ERROR(cudaMalloc((void**)&msgs_buffer_d, num_points_ * point_step_bytes * param_.cache_len()));
  CHECK_CUDA_ERROR(cudaMalloc((void**)&dns_buffer_d, num_points_ * FINAL_FT_NUM * sizeof(float) * param_.cache_len()));
  CHECK_CUDA_ERROR(cudaMallocHost((void**)&dst_h, num_points_ * FINAL_FT_NUM * sizeof(float) * param_.cache_len()));

  for (int i=0; i<param_.cache_len(); i++)
  {
    streams.emplace_back();
  }
}

void PointCloudDensification::format(const sensor_msgs::PointCloud2::ConstPtr& pc_msg)
{
  for(const auto & elem: pc_msg->fields)
  {
    field_t field_meta_(elem);
    field_to_dtype_m_.insert({elem.name, field_meta_});
  }
}

void PointCloudDensification::dequeue()
{
  if (pointcloud_cache_.size() > param_.cache_len()) {
    pointcloud_cache_.pop_back();
  }
}

// void PointCloudDensification::appendPclCloud(std::array<float, Config::num_point_features> point)
// {
//   pcl::PointXYZI pclPoint;
//   pclPoint.x = point.at(0);
//   pclPoint.y = point.at(1);
//   pclPoint.z = point.at(2);
//   pclPoint.intensity = point.at(3);
//   dense_pcl_cloud.push_back(pclPoint);
// }

// sensor_msgs::PointCloud2 PointCloudDensification::getDenseCloud()
// {
//   sensor_msgs::PointCloud2 msg;
//   pcl::toROSMsg(dense_pcl_cloud, msg);
//   return msg;
// } 

// void PointCloudDensification::clearPclCloud()
// {
//   dense_pcl_cloud.clear();
// }

}  // namespace centerpoint
