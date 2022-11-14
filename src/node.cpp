#include "lidar_centerpoint/node.hpp"
#include <pointcloud_densification.hpp>

#include <pcl_conversions/pcl_conversions.h>

#include <memory>
#include <string>
#include <vector>

namespace densifier
{
DensifierNode::DensifierNode():  nh_(""), pnh_("~"), tf_listener_(tf_buffer_)
{
  pnh_.param<std::string>("densification_world_frame_id", densification_world_frame_id, "map");
  pnh_.param<int>("densification_num_past_frames", densification_num_past_frames, 2);

  ROS_INFO_STREAM("densification_world_frame_id: " << densification_world_frame_id);
  ROS_INFO_STREAM("densification_num_past_frames: " << densification_num_past_frames);
  DensificationParam densification_param(densification_world_frame_id, densification_num_past_frames);

  dns_p = std::make_unique<PointCloudDensification>(densification_param);

  pointcloud_sub_ = pnh_.subscribe("/lidar/concatenated/pointcloud", 1, &DensifierNode::pointCloudCallback, this);
#if DEBUG_OUT
  pointcloud_pub_ = pnh_.advertise<sensor_msgs::PointCloud2>("debug/pointcloud_densification", 1);
#endif
  ROS_WARN("Ready");
}

void DensifierNode::pointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr pc_msg)
{
  static const int pc_size = pc_msg->height * pc_msg->width;

  if (!dns_p->initialized()) { dns_p->init(pc_msg, pc_size); }
  
  dns_p->registerSweep(pc_msg, tf_buffer_);

#if DEBUG_OUT
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
    out_cloud.push_back(point_); // when this is active only one pointcloud of points are passing thru
  }

  sensor_msgs::PointCloud2 out_msg;
  pcl::toROSMsg(out_cloud, out_msg);
  out_msg.header = pc_msg->header;
  pointcloud_pub_.publish(out_msg);
#endif
}

}  // namespace densifier

