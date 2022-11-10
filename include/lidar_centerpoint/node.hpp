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

#ifndef LIDAR_CENTERPOINT__NODE_HPP_
#define LIDAR_CENTERPOINT__NODE_HPP_

#include <config.hpp>
#include <autoware_perception_msgs/DynamicObjectArray.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

#include "pointcloud_densification.hpp"

#include <ros/ros.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>

#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/point_cloud.h>

#include <memory>
#include <string>
#include <vector>

namespace centerpoint
{

class LidarCenterPointNode
{
public:
  LidarCenterPointNode();

private:
  void pointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr pc_msg);

  static uint8_t getSemanticType(const std::string & class_name);
  static bool isCarLikeVehicleLabel(const uint8_t label);

  ros::NodeHandle nh_{};
  ros::NodeHandle pnh_{};

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  ros::Subscriber pointcloud_sub_{};
  ros::Publisher objects_pub_{};
  ros::Publisher pointcloud_pub_{};
  
  int densification_num_past_frames;
  float score_threshold_{0.0};
  bool use_encoder_trt_{false};
  bool use_head_trt_{false};
  std::string trt_precision_;

  std::string encoder_onnx_path_;
  std::string encoder_engine_path_;
  std::string encoder_pt_path_;
  std::string head_onnx_path_;
  std::string head_engine_path_;
  std::string head_pt_path_;
  std::string densification_world_frame_id;

  std::vector<std::string> class_names_;
  bool rename_car_to_truck_and_bus_{false};

  std::unique_ptr<PointCloudDensification> dns_p;

};

}  // namespace centerpoint

#endif  // LIDAR_CENTERPOINT__NODE_HPP_
