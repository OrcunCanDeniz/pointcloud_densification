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

namespace densifier
{
PointCloudDensification::PointCloudDensification(const DensificationParam & param) : param_(param)
{}

void PointCloudDensification::init(const sensor_msgs::PointCloud2::ConstPtr& pc_msg, const int& pc_size)
{
  num_points_ = pc_size;
  point_step_ = pc_msg->point_step / sizeof(uint8_t);
  format(pc_msg); // get description of pc msg to decode the uint8 array
  allocCuda(pc_msg->point_step);
  is_init = true;
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
  static int registered_pc_num{0};
  const int cyclic_idx = registered_pc_num % param_.cache_len();

  affine_world2current_ = affine_world2current;
  
  std::pair<uint8_t*, float*> free_buffers = getBuffers(cyclic_idx); // return pointer for new data, drop latest data in cache
  current_timestamp_ = ros::Time(msg->header.stamp).toSec();
  toCuda((uint8_t*)msg->data.data(), free_buffers.first, free_buffers.second);
  pointcloud_cache_.emplace_front(free_buffers.first, current_timestamp_, affine_world2current.inverse(),
                                      cyclic_idx, free_buffers.second); // add new sweep to front of the cache
  registered_pc_num ++;
  densify();
}

std::pair<uint8_t*, float*> PointCloudDensification::getBuffers(const int cyclic_idx)
{
  // get next available pointer(holding outdated data) to hold newly arrived data
  // drop latest pointcloud

  uint8_t* src_loc = msgs_buffer_d + cyclic_idx * num_points_ * point_step_;
  float* dst_loc = dns_buffer_d + cyclic_idx * num_points_ * FINAL_FT_NUM;

  if (pointcloud_cache_.size() == param_.cache_len())
  {
    pointcloud_cache_.pop_back();
  }
  return std::make_pair(src_loc, dst_loc);
}

void PointCloudDensification::densify()
{
  for (auto cache_iter = std::next(getPointCloudCacheIter()); !isCacheEnd(cache_iter); cache_iter++) 
  {
    tf_time_t tf_time;
    tf_time.timelag = static_cast<float>(getCurrentTimestamp() - cache_iter->time_secs);
    auto affine_past2current = getAffineWorldToCurrent() * cache_iter->affine_past2world;
    tf_time.setTf(affine_past2current.matrix().data());
    launch_transform_set_time(cache_iter->src_data, cache_iter->dst_data, tf_time, cache_iter->id); 
  }
}

std::vector<float> PointCloudDensification::getCloud()
{
  CHECK_CUDA_ERROR(cudaMemcpy(dst_h, dns_buffer_d, 
                                  num_points_ * FINAL_FT_NUM  * sizeof(float) * param_.cache_len(),
                                  cudaMemcpyDeviceToHost));

  return std::vector<float>(dst_h, dst_h + num_points_ * FINAL_FT_NUM * param_.cache_len());
}

void PointCloudDensification::allocCuda(const int& point_step_bytes)
{ // point_step: how many bytes a point needs
  CHECK_CUDA_ERROR(cudaMalloc((void**)&msgs_buffer_d, num_points_ * point_step_bytes * param_.cache_len()));
  CHECK_CUDA_ERROR(cudaMalloc((void**)&dns_buffer_d, num_points_ * FINAL_FT_NUM * sizeof(float) * param_.cache_len()));
  CHECK_CUDA_ERROR(cudaMemset(dns_buffer_d, 0, num_points_ * FINAL_FT_NUM * sizeof(float) * param_.cache_len()));
  #if DEBUG_OUT
    CHECK_CUDA_ERROR(cudaMallocHost((void**)&dst_h, num_points_ * FINAL_FT_NUM * sizeof(float) * param_.cache_len()));
  #endif
  
  for (int i=0; i<num_streams; i++)
  {
    cudaStream_t stream_;
    cudaStreamCreate(&stream_);
    streams.push_back(stream_);
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

void PointCloudDensification::syncCuda()
{
  for (int i=0; i<num_streams; i++)
  {
    cudaStreamSynchronize(streams.at(i));
  }
}

}  // namespace densifier
