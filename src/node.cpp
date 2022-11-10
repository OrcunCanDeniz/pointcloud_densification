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

#include "lidar_centerpoint/node.hpp"

// #include <autoware_utils/geometry/geometry.hpp>
#include <config.hpp>
#include <pcl_ros/transforms.h>
#include <pcl_conversions/pcl_conversions.h>

#include <pointcloud_densification.hpp>

#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <memory>
#include <string>
#include <vector>

namespace centerpoint
{
LidarCenterPointNode::LidarCenterPointNode():  nh_(""), pnh_("~"), tf_listener_(tf_buffer_)
{
  pnh_.param<std::string>("densification_world_frame_id", densification_world_frame_id, "map");
  pnh_.param<int>("densification_num_past_frames", densification_num_past_frames, 1);

  ROS_INFO_STREAM("densification_world_frame_id: " << densification_world_frame_id);
  ROS_INFO_STREAM("densification_num_past_frames: " << densification_num_past_frames);
  DensificationParam densification_param(
    densification_world_frame_id, densification_num_past_frames);

  dns_p = std::make_unique<PointCloudDensification>(densification_param);

  pointcloud_sub_ = pnh_.subscribe("input/pointcloud", 1, &LidarCenterPointNode::pointCloudCallback, this);
  pointcloud_pub_ = pnh_.advertise<sensor_msgs::PointCloud2>("debug/pointcloud_densification", 1);

  ROS_WARN("Ready for inference.");
}

void LidarCenterPointNode::pointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr pc_msg)
{
  static const int pc_size = pc_msg->height * pc_msg->width;
  std::cout << "CB" << std::endl;

  if (!dns_p->initialized()) { dns_p->init(pc_msg, pc_size); }
  
  dns_p->registerSweep(pc_msg, tf_buffer_);

  std::vector<float> points_flat = dns_p->getCloud();

  pcl::PointCloud<pcl::PointXYZI> out_cloud;
  for(int i=0; i<points_flat.size()/FINAL_FT_NUM; i++)
  {
    pcl::PointXYZI point_;
    const int idx = i*FINAL_FT_NUM;
    point_.x = points_flat[idx];
    point_.y = points_flat[idx + 1];
    point_.z = points_flat[idx + 2];
    point_.intensity = points_flat[idx + 3];

    if (point_.x + point_.y + point_.z != 0)
    {
      out_cloud.push_back(point_); // when this is active only one pointcloud of points are passing thru
    }
    // if (idx>1310700)
    // {
    //   std::cout << point_.x << " "
    //             << point_.y << " "
    //             << point_.z << " "
    //             << point_.intensity << std::endl;
    // }
  }
  
  // std::cout << out_cloud.size() << std::endl; // 262144

  sensor_msgs::PointCloud2 out_msg;
  pcl::toROSMsg(out_cloud, out_msg);
  out_msg.header = pc_msg->header;
  pointcloud_pub_.publish(out_msg);

  // if (1 > 0) {
  //   sensor_msgs::PointCloud2 dense_msg = detector_ptr_->vg_ptr_->pd_ptr_->getDenseCloud();
  // }
}

}  // namespace centerpoint

