
# ifndef OPENCL_UTILS_CHK
# define OPENCL_UTILS_CHK

//#include "RunCL.hpp"
//#include <CL/opencl.hpp>

using namespace std;

void cl_mem_swap_ptr(cl_mem buf1, cl_mem buf2);

void _clSetKernelArg(cl_kernel kernel,  cl_uint arg_index,  size_t arg_size, const void* arg_value, string fname);

void _clEnqueueWriteBuffer(
    cl_command_queue    command_queue,
    cl_mem              buffer,
    cl_bool             blocking_write,
    size_t              offset,
    size_t              size,
    const void*         ptr,
    cl_uint             num_events_in_wait_list,
    const cl_event*     event_wait_list,
    cl_event*           event
);


void _clEnqueueFillBuffer(
    cl_command_queue    command_queue,
    cl_mem              buffer,
    const void*         pattern,
    size_t              pattern_size,
    size_t              offset,
    size_t              size,
    cl_uint             num_events_in_wait_list,
    const cl_event*     event_wait_list,
    cl_event*           event
);


void _clCreateBuffer(
    cl_context          context,
    cl_mem_flags        flags,
    size_t              size,
    void*               host_ptr,
    cl_int*             errcode_ret
);


void _clReleaseMemObject(cl_mem memobj);

void _clReleaseKerne(cl_kernel kernel);

string checkerror(int input);

# endif





