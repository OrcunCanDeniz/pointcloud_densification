#include "pointcloud_densification.hpp"

inline __device__ float3 atPC2(uint8_t* msg, const int point_step, const int x_offset, const int y_offset, 
                                const int z_offset, const int point_idx)
{
    // Read XYZ for a point, from the byte array of sensor_msgs/PointCloud2
    const int src_point_offs = point_idx * point_step;

    const float* x_ptr = reinterpret_cast<float*>(msg + src_point_offs + x_offset);
    const float* y_ptr = reinterpret_cast<float*>(msg + src_point_offs + y_offset);
    const float* z_ptr = reinterpret_cast<float*>(msg + src_point_offs + z_offset);
    return make_float3(*x_ptr, *y_ptr, *z_ptr);
}

inline __device__ float3 transform(const float tf[16], const float3 point) // 4*4 x 4*1
{
    const float x_ = point.x * tf[0] +  point.y * tf[1] + point.z * tf[2] + tf[3]; 
    const float y_ = point.x * tf[4] +  point.y * tf[5] + point.z * tf[6] + tf[7];
    const float z_ = point.x * tf[8] +  point.y * tf[9] + point.z * tf[10] + tf[11];
    const float w_ = point.x * tf[12] +  point.y * tf[13] + point.z * tf[14] + tf[15];

    return make_float3(x_/w_, y_/w_, z_/w_);
}

__global__ void transform_set_time_kernel(uint8_t* msg, const int point_step, const int num_points, const int x_offset, 
                            const int y_offset, const int z_offset, tf_time_t tf_time, float* dst)
{
    const int thread_ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_ix > num_points) {return;} //bound check
    
    const int dst_point_offs = thread_ix * FINAL_FT_NUM;
    
    const float3 point = atPC2(msg, point_step, x_offset, y_offset, z_offset, thread_ix); //decode PointCloud2 msg data
    const float3 transformed_point = transform(tf_time.transform, point);

    dst[dst_point_offs] = transformed_point.x;
    dst[dst_point_offs + 1] = transformed_point.y;
    dst[dst_point_offs + 2] = transformed_point.z;
    dst[dst_point_offs + 3] = tf_time.timelag;
}

__global__ void set_time_kernel(uint8_t* msg, const int point_step, const int num_points, const int x_offset, 
    const int y_offset, const int z_offset, tf_time_t tf_time, float* dst)
{
    const int thread_ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_ix > num_points) {return;} 

    const int dst_point_offs = thread_ix * FINAL_FT_NUM;

    const float3 point = atPC2(msg, point_step, x_offset, y_offset, z_offset, thread_ix); 

    dst[dst_point_offs] = point.x;
    dst[dst_point_offs + 1] = point.y;
    dst[dst_point_offs + 2] = point.z;
    dst[dst_point_offs + 3] = tf_time.timelag;
}

namespace centerpoint
{
    void PointCloudDensification::dispatch(uint8_t* msg, float* dst, tf_time_t tf_time)
    {
        dim3 block(1024);
        dim3 grid(ceil(num_points_/1024));
        
        if (tf_time.last)
        { // do not transform latest pointcloud 
            set_time_kernel<<<grid, block>>>(msg, point_step_, num_points_, field_to_dtype_m_["x"].offset, 
                                                field_to_dtype_m_["y"].offset, field_to_dtype_m_["z"].offset,
                                                tf_time, dst);
        } else {
            transform_set_time_kernel<<<grid, block>>>(msg, point_step_, num_points_, field_to_dtype_m_["x"].offset, 
                                            field_to_dtype_m_["y"].offset, field_to_dtype_m_["z"].offset,
                                            tf_time, dst);
        }
    }
} // namespace centerpoint