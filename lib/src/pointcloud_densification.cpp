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

#include <pcl_conversions/pcl_conversions.h>
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
  
  uint8_t* next_buffer = refreshCache(); // return pointer for new data, drop latest data in cache
  
  cudaMemcpyAsync(next_buffer, msg->data.data(), num_points_ * point_step_ * sizeof(uint8_t), cudaMemcpyHostToDevice); // load new data to gpu
  pointcloud_cache_.emplace_front(next_buffer, ros::Time(msg->header.stamp).toSec(),
                                       affine_world2current.inverse()); // add new sweep to front of the cache
  densify();
}

uint8_t* PointCloudDensification::refreshCache()
{
  // get next available pointer(holding outdated data) to hold newly arrived data
  // drop latest pointcloud
  static int registered_pc_num{0};

  uint8_t* loc = msgs_buffer_d + (registered_pc_num % param_.cache_len()) 
                  * num_points_ * point_step_;

  if (pointcloud_cache_.size() == param_.cache_len())
  {
    pointcloud_cache_.pop_back();
  }
  registered_pc_num ++;
  return loc;
}

void PointCloudDensification::densify()
{
  for (auto cache_iter = getPointCloudCacheIter(); !isCacheEnd(cache_iter); cache_iter++) 
  {
      const float timelag = static_cast<float>(getCurrentTimestamp() - cache_iter->time_secs);
    // auto pc_msg = cache_iter->pointcloud_msg;
    if (cache_iter != getPointCloudCacheIter()) // if not the last received cloud
    {
      auto affine_past2current = getAffineWorldToCurrent() * cache_iter->affine_past2world;
      const float* tf_mat = affine_past2current.matrix().data();
      // kernel<<<>>>(data_ptr, transform, timelag, fields)
      // TODO: DISPATCH KERNEL THAT SETS TIMELAG AND TRANSFORMS
    } else {
      std::cout<< "Newest one " <<std::endl; //Here for compilation 
      // kernel<<<>>>(data_ptr, transform, timelag, fields)
      // TODO: DISPATCH KERNEL THAT ONLY SETS TIMELAG
    }
    // The latest pointcloud does not need to be transformed. Just reformat and add time_lag

    // std::cout << tf_mat[0] << " " << tf_mat[1] << " " << tf_mat[2] << " " << tf_mat[3] << std::endl;
    // std::cout << tf_mat[4] << " " << tf_mat[5] << " " << tf_mat[6] << " " << tf_mat[7] << std::endl;
    // std::cout << tf_mat[8] << " " << tf_mat[9] << " " << tf_mat[10] << " " << tf_mat[11] << std::endl;
    // std::cout << tf_mat[12] << " " << tf_mat[13] << " " << tf_mat[14] << " " << tf_mat[15] << std::endl;
    std::cout << "###############################" << std::endl;
  }
}

void PointCloudDensification::allocCuda(const int& point_step_bytes)
{                                             // point_step: how many bytes does a point need
  cudaMalloc((void**)&msgs_buffer_d, num_points_ * point_step_bytes * param_.cache_len());
  cudaMalloc((void**)&dns_buffer_d, num_points_ * 5 /*XYZIT*/ * sizeof(float) * param_.cache_len());

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

void PointCloudDensification::appendPclCloud(std::array<float, Config::num_point_features> point)
{
  pcl::PointXYZI pclPoint;
  pclPoint.x = point.at(0);
  pclPoint.y = point.at(1);
  pclPoint.z = point.at(2);
  pclPoint.intensity = point.at(3);
  dense_pcl_cloud.push_back(pclPoint);
}

sensor_msgs::PointCloud2 PointCloudDensification::getDenseCloud()
{
  sensor_msgs::PointCloud2 msg;
  pcl::toROSMsg(dense_pcl_cloud, msg);
  return msg;
} 

void PointCloudDensification::clearPclCloud()
{
  dense_pcl_cloud.clear();
}

}  // namespace centerpoint
