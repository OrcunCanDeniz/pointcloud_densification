
#include "lidar_centerpoint/node.hpp"

int main(int argc, char ** argv)
{
  ros::init(argc, argv, "lidar_centerpointv2");
  densifier::DensifierNode node;
  ros::spin();
  return 0;
}