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
    const float x_ = point.x * tf[0] +  point.y * tf[4] + point.z * tf[8] + tf[12]; 
    const float y_ = point.x * tf[1] +  point.y * tf[5] + point.z * tf[9] + tf[13];
    const float z_ = point.x * tf[2] +  point.y * tf[6] + point.z * tf[10] + tf[14];
    return make_float3(x_, y_, z_);
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
    const int y_offset, const int z_offset, const int points_in_chunk, float* dst)
{
    const int thread_ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_ix > points_in_chunk) {return;} 

    const int dst_point_offs = thread_ix * FINAL_FT_NUM;

    const float3 point = atPC2(msg, point_step, x_offset, y_offset, z_offset, thread_ix); 

    dst[dst_point_offs] = point.x;
    dst[dst_point_offs + 1] = point.y;
    dst[dst_point_offs + 2] = point.z;
    dst[dst_point_offs + 3] = 0.0f;
}

namespace densifier
{
    void PointCloudDensification::launch_transform_set_time(uint8_t* msg, float* dst, tf_time_t tf_time, int& idx)
    {
        const dim3 block(1024);
        dim3 grid(ceil(num_points_/1024));
        
        transform_set_time_kernel<<<grid, block, 0, streams.back()>>>(msg, point_step_, num_points_, field_to_dtype_m_["x"].offset, 
                                        field_to_dtype_m_["y"].offset, field_to_dtype_m_["z"].offset,
                                        tf_time, dst);
    }

    void PointCloudDensification::toCuda(uint8_t* msg_src, uint8_t* msg_dst, float* out_d)
    {
        // do not transform latest pointcloud
        const dim3 block(1024);
        const int max_points_in_chunk = ceil(num_points_ / num_streams-1);
        const int num_chunks = ceil(num_points_/max_points_in_chunk);
        int remaining_points = num_points_;
        // split input into batches then process those batches 
        for (int i=0; i<num_chunks; i++)
        {
            int points_in_chunk{0};
            if (remaining_points >= max_points_in_chunk)
            {
                points_in_chunk = max_points_in_chunk;
                remaining_points -= points_in_chunk;
            } else {
                points_in_chunk = remaining_points;
            }

            const dim3 grid(ceil(points_in_chunk/1024));
            const int batch_step_msg =  i * max_points_in_chunk * point_step_;
            const int batch_step_dst =  i * max_points_in_chunk * FINAL_FT_NUM;
            uint8_t* msg_h = msg_src + batch_step_msg;
            uint8_t* msg_d = msg_dst + batch_step_msg;
            float* dst_loc = out_d + batch_step_dst;
            // load new data to gpu
            CHECK_CUDA_ERROR(cudaMemcpyAsync(msg_d, msg_h, points_in_chunk * point_step_ , cudaMemcpyHostToDevice, streams.at(i))); 
            set_time_kernel<<<grid, block, 0, streams.at(i)>>>(msg_d, point_step_, num_points_, field_to_dtype_m_["x"].offset, 
                                                                field_to_dtype_m_["y"].offset, field_to_dtype_m_["z"].offset,
                                                                points_in_chunk, dst_loc);
        }
    }
} // namespace densifier