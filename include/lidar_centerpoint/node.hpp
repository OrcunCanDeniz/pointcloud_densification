#ifndef LIDAR_CENTERPOINT__NODE_HPP_
#define LIDAR_CENTERPOINT__NODE_HPP_

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

#include "pointcloud_densification.hpp"

#include <ros/ros.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include <memory>
#include <string>
#include <vector>

namespace densifier
{

class DensifierNode
{
public:
  DensifierNode();

private:
  void pointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr pc_msg);

  static uint8_t getSemanticType(const std::string & class_name);
  static bool isCarLikeVehicleLabel(const uint8_t label);

  ros::NodeHandle nh_{};
  ros::NodeHandle pnh_{};

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  ros::Subscriber pointcloud_sub_{};
  ros::Publisher pointcloud_pub_{};
  
  int densification_num_past_frames;
  std::string densification_world_frame_id;


  std::unique_ptr<PointCloudDensification> dns_p;

};

}  // namespace densifier

#endif  // LIDAR_CENTERPOINT__NODE_HPP_
