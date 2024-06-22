#include "opencl_utils.hpp"

void cl_mem_swap_ptr(cl_mem buf1, cl_mem buf2){
    cl_mem temp_mem = buf1;
	buf1 = buf2;
	buf2 = temp_mem;
}


void _clSetKernelArg(cl_kernel kernel,  cl_uint arg_index,  size_t arg_size, const void* arg_value){
	cl_int  res;
	res 	= clSetKernelArg( kernel, arg_index,  arg_size, arg_value);
	if(res	!=CL_SUCCESS){	
		cout<<"\n_clSetKernelArg  "<<kernel<<",   arg_index="<<arg_index<<"   res = "<<checkerror(res)<<"\n"<<flush;
		exit_(res);
	}
}


void _cl_int clEnqueueWriteBuffer(
	cl_command_queue 	command_queue,
	cl_mem 				buffer,
	cl_bool 			blocking_write,
	size_t 				offset,
	size_t 				size,
	const void* 		ptr,
	cl_uint 			num_events_in_wait_list,
	const cl_event* 	event_wait_list,
	cl_event* 			event
)
{
	cl_int 		status;
	status 		= clEnqueueWriteBuffer( command_queue, buffer, blocking_write, offset, size, ptr, num_events_in_wait_list, event_wait_list, event);
	if (status 	!= CL_SUCCESS)	{ 
		cout << "\n_clEnqueueWriteBuffer  buffer="<<buffer<<",  status = " << checkerror(status) << "Error1: failed to enqueue\n" << endl;
		exit_(status);
	}	
	clFlush(uload_queue); 
	status = clFinish(uload_queue);
	if (status 	!= CL_SUCCESS)	{ 
		cout << "\n_clEnqueueWriteBuffer  buffer="<<buffer<<",  status = " << checkerror(status) << "Error2: failed to finish\n" << endl;
		exit_(status);
	}	
}


void _clEnqueueFillBuffer(
	cl_command_queue 	command_queue,
	cl_mem 				buffer,
	const void* 		pattern,
	size_t 				pattern_size,
	size_t 				offset,
	size_t 				size,
	cl_uint 			num_events_in_wait_list,
	const cl_event* 	event_wait_list,
	cl_event* 			event
)
{
	cl_int 		status;
	status = clEnqueueFillBuffer( command_queue, buffer, pattern, pattern_size, offset, size, num_events_in_wait_list, event_wait_list, event);	
	if (status != CL_SUCCESS)	{ 
		cout << "\n_clEnqueueFillBuffer    buffer="<<buffer<<",  status = " << checkerror(status) << "Error1: failed to enqueue\n" << endl;
		exit_(status);
	}	
	clFlush(uload_queue); 
	status = clFinish(uload_queue);
	if (status != CL_SUCCESS)	{ 
		cout << "\n_clEnqueueFillBuffer    buffer="<<buffer<<",  status = " << checkerror(status) << "Error2: failed to finish\n" << endl;
		exit_(status);
	}	
}


void _clCreateBuffer(
    cl_context          context,
    cl_mem_flags        flags,
    size_t              size,
    void*               host_ptr,
    cl_int*             errcode_ret,
	clmem 				memobj 
)
{
	memobj = clCreateBuffer(m_context, CL_MEM_READ_ONLY  						, mm_size_bytes_C4,  		0, &res);			
	
	if(res!=CL_SUCCESS){
		cout<<"\n_clCreateBuffer  buffer="<<memobj<<",   error="<<checkerror(res)<<"\n"<<flush;
		exit_(res);
	}
}


void _clReleaseMemObject(clmem 	memobj)
{
	cl_int 			status;
	status 			= clReleaseMemObject( memobj );
	
	if (status != CL_SUCCESS)	{ 
		cout << "\n_clReleaseMemObject("<< memobj <<",   status = " << checkerror(status) <<"\n"<<flush; 
	}
}


void _clReleaseKerne(cl_kernel kernel)
{
	cl_int 			status;
	status = clReleaseKernel(cvt_color_space_linear_kernel);	
	
	if (status != CL_SUCCESS)	{ 
		cout << "\n_clReleaseKerne("<<kernel<<") 	status = " << checkerror(status) <<"\n"<<flush; 
	}
}

// TODO
// Need to put all buffers, kernels and command queues into a set of c++ dictionaries.
// Use loops to release them.
// Use a local alias cl_kernel kern = kDict[..kernelname..] to set arguments etc..
// Have name lists to instantiate  kernels and command queues.
