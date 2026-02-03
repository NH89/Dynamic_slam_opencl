#include "RunCL.hpp"

void RunCL::_clEnqueueNDRangeKernel(	// TODO will become obsolete when all kernels use newest patch system.
	cl_command_queue _queue,
	cl_kernel        kernel,
	cl_uint          work_dim,
	const size_t *   global_work_offset,
	const size_t *   global_work_size,
	const size_t *   local_work_size,
	string           fname
){
	_cl_flush_finish(_queue, fname);
	cl_event ev;
	cl_int res;
	res = clEnqueueNDRangeKernel(
		_queue,
		kernel,
		work_dim,
		global_work_offset,
		global_work_size,
		local_work_size,
		0,
		NULL,
		&ev
	);
	if (res != CL_SUCCESS)	{ cout << "\n"<<fname<<"  res = " << checkerror(res) <<"\n"<<flush; exit_(res);}

	_cl_flush_finish(_queue,  fname);
}


void RunCL::_cl_flush_finish(cl_command_queue	_queue,  string fname){
	cl_int status;
	status = clFlush(_queue); 				if (status != CL_SUCCESS)	{ cout << "\n"<<fname<<"  clFlush(m_queue) status = " << checkerror(status) <<"\n"<<flush; 	exit_(status);}
	status = clFinish(_queue); 				if (status != CL_SUCCESS)	{ cout << "\n"<<fname<<"  clFinish(m_queue)="<<status<<" "<<checkerror(status)<<"\n"<<flush; exit_(status);}
}


int RunCL::waitForEventAndRelease(cl_event *event){
	int local_verbosity_threshold = V_RUNCL_WAITFOREVENTANDRELEASE;//verbosity_mp["RunCL::waitForEventAndRelease"];
											if(verbosity>local_verbosity_threshold) cout << "\nwaitForEventAndRelease_chk0, event="<<event<<" *event="<<*event << flush;
	cl_int status = CL_SUCCESS;
	status	= clWaitForEvents(1, event); 	if (status != CL_SUCCESS) { cout << "\nclWaitForEvents status=" << status << ", " <<  checkerror(status) <<"\n" << flush; exit_(status); }
	status	= clReleaseEvent(*event); 		if (status != CL_SUCCESS) { cout << "\nclReleaseEvent status="  << status << ", " <<  checkerror(status) <<"\n" << flush; exit_(status); }
	return status;
}


void RunCL::cl_mem_swap_ptr(cl_mem buf1, cl_mem buf2){
    cl_mem temp_mem = buf1;
	buf1 = buf2;
	buf2 = temp_mem;
}


void RunCL::_clSetKernelArg(cl_kernel kernel,  cl_uint arg_index,  size_t arg_size, const void* arg_value, string fname){
	cl_int  res;
	res 	= clSetKernelArg( kernel, arg_index,  arg_size, arg_value);
											if(res	!=CL_SUCCESS){	std::cout<< "\n"<<fname << "_clSetKernelArg  "<<kernel<<",   arg_index="<<arg_index<<"   res = "<<checkerror(res)<<"\n"<<flush;	exit_(res); }
}


void RunCL::_clEnqueueWriteBuffer(
	cl_command_queue 	command_queue,
	cl_mem 				buffer,
	cl_bool 			blocking_write,
	size_t 				offset,
	size_t 				size,
	const void* 		ptr,
	string 				fname
){
	cl_int 		status;
	cl_event 	event;
	status 		= clEnqueueWriteBuffer( command_queue, buffer, blocking_write, offset, size, ptr,  0, NULL, &event);
											if (status 	!= CL_SUCCESS)	{ cout << "\n"<<fname << "_clEnqueueWriteBuffer  buffer="<<buffer<<",  status = " << checkerror(status) << "Error1: failed to enqueue\n" << endl; exit_(status); }
	_cl_flush_finish( command_queue,  fname);
}


void RunCL::_clEnqueueFillBuffer(
	cl_command_queue 	command_queue,
	cl_mem 				buffer,
	const void* 		pattern,
	size_t 				pattern_size,
	size_t 				offset,
	size_t 				size,
	string 				fname
){
	cl_int 		status;
	cl_event 	event;
	status 		= clEnqueueFillBuffer( command_queue, buffer, pattern, pattern_size, offset, size, 0, NULL, &event);
											if (status != CL_SUCCESS)	{ cout << "\n"<<fname << "_clEnqueueFillBuffer    buffer="<<buffer<<",  status = " << checkerror(status) << "Error1: failed to enqueue\n" << endl;  exit_(status); }
	_cl_flush_finish(command_queue, fname);
}


void RunCL::_clEnqueueCopyBuffer(
		cl_command_queue command_queue,
		cl_mem src_buffer,
		cl_mem dst_buffer,
		size_t src_offset,
		size_t dst_offset,
		size_t size,
		string fname
){
	cl_int 		status;
	cl_event 	event;
	status = clEnqueueCopyBuffer(
		command_queue,
		src_buffer,
		dst_buffer,
		src_offset,
		dst_offset,
		size,
		0,
		NULL,
		&event
	);

	if (status 	!= CL_SUCCESS)	{
		cout << "\n"<<fname << ":_clEnqueueCopyBuffer(..)"
		<<"  src_buffer="		<< src_buffer
		<<",  dst_buffer="		<< dst_buffer
		<<",  src_offset="		<< src_offset
		<<",  dst_offset="		<< dst_offset
		<<",  size="			<< size
		<<",  status = " 		<< checkerror(status)
		<< "Error1: failed to enqueue\n" << endl;
		exit_(status);
	}

	_cl_flush_finish( command_queue,  fname);
}


void RunCL::_clCreateBuffer(
    cl_context          context,
    cl_mem_flags        flags,
    size_t              size,
    void*               host_ptr,
	cl_mem 				memobj,
	string 				fname
){
	cl_int		res;
	memobj		= clCreateBuffer( context, flags, size,	host_ptr, &res );

	if(res!=CL_SUCCESS){
		cout<<"\n"			<<fname << "_clCreateBuffer(..)"
		<<"    buffer="		<<memobj
		<<",   error="		<<checkerror(res)<<"\n"<<flush;
		exit_(res);
	}
}


void RunCL::_clReleaseMemObject(cl_mem 	memobj)
{
	cl_int 		status;
	status 		= clReleaseMemObject( memobj );
											if (status != CL_SUCCESS)	{ cout << "\n_clReleaseMemObject("<< memobj <<",   status = " << checkerror(status) <<"\n"<<flush; }
}


void RunCL::_clReleaseKerne(cl_kernel kernel)
{
	cl_int 		status;
	status 		= clReleaseKernel(cvt_color_space_linear_kernel);
											if (status != CL_SUCCESS)	{ cout << "\n_clReleaseKerne("<<kernel<<") 	status = " << checkerror(status) <<"\n"<<flush; }
}

// TODO
// Need to put all buffers, kernels and command queues into a set of c++ dictionaries.
// Use loops to release them.
// Use a local alias cl_kernel kern = kDict[..kernelname..] to set arguments etc..
// Have name lists to instantiate  kernels and command queues.


string  RunCL::checkerror(int input) {
		int errorCode = input;
		switch (errorCode) {
		case -9999:											return "Illegal read or write to a buffer";		// NVidia error code
		case CL_DEVICE_NOT_FOUND:							return "CL_DEVICE_NOT_FOUND";
		case CL_DEVICE_NOT_AVAILABLE:						return "CL_DEVICE_NOT_AVAILABLE";
		case CL_COMPILER_NOT_AVAILABLE:						return "CL_COMPILER_NOT_AVAILABLE";
		case CL_MEM_OBJECT_ALLOCATION_FAILURE:				return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
		case CL_OUT_OF_RESOURCES:							return "CL_OUT_OF_RESOURCES";
		case CL_OUT_OF_HOST_MEMORY:							return "CL_OUT_OF_HOST_MEMORY";
		case CL_PROFILING_INFO_NOT_AVAILABLE:				return "CL_PROFILING_INFO_NOT_AVAILABLE";
		case CL_MEM_COPY_OVERLAP:							return "CL_MEM_COPY_OVERLAP";
		case CL_IMAGE_FORMAT_MISMATCH:						return "CL_IMAGE_FORMAT_MISMATCH";
		case CL_IMAGE_FORMAT_NOT_SUPPORTED:					return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
		case CL_BUILD_PROGRAM_FAILURE:						return "CL_BUILD_PROGRAM_FAILURE";
		case CL_MAP_FAILURE:								return "CL_MAP_FAILURE";
		case CL_MISALIGNED_SUB_BUFFER_OFFSET:				return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
		case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:	return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
		case CL_INVALID_VALUE:								return "CL_INVALID_VALUE";
		case CL_INVALID_DEVICE_TYPE:						return "CL_INVALID_DEVICE_TYPE";
		case CL_INVALID_PLATFORM:							return "CL_INVALID_PLATFORM";
		case CL_INVALID_DEVICE:								return "CL_INVALID_DEVICE";
		case CL_INVALID_CONTEXT:							return "CL_INVALID_CONTEXT";
		case CL_INVALID_QUEUE_PROPERTIES:					return "CL_INVALID_QUEUE_PROPERTIES";
		case CL_INVALID_COMMAND_QUEUE:						return "CL_INVALID_COMMAND_QUEUE";
		case CL_INVALID_HOST_PTR:							return "CL_INVALID_HOST_PTR";
		case CL_INVALID_MEM_OBJECT:							return "CL_INVALID_MEM_OBJECT";
		case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:			return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
		case CL_INVALID_IMAGE_SIZE:							return "CL_INVALID_IMAGE_SIZE";
		case CL_INVALID_SAMPLER:							return "CL_INVALID_SAMPLER";
		case CL_INVALID_BINARY:								return "CL_INVALID_BINARY";
		case CL_INVALID_BUILD_OPTIONS:						return "CL_INVALID_BUILD_OPTIONS";
		case CL_INVALID_PROGRAM:							return "CL_INVALID_PROGRAM";
		case CL_INVALID_PROGRAM_EXECUTABLE:					return "CL_INVALID_PROGRAM_EXECUTABLE";
		case CL_INVALID_KERNEL_NAME:						return "CL_INVALID_KERNEL_NAME";
		case CL_INVALID_KERNEL_DEFINITION:					return "CL_INVALID_KERNEL_DEFINITION";
		case CL_INVALID_KERNEL:								return "CL_INVALID_KERNEL";
		case CL_INVALID_ARG_INDEX:							return "CL_INVALID_ARG_INDEX";
		case CL_INVALID_ARG_VALUE:							return "CL_INVALID_ARG_VALUE";
		case CL_INVALID_ARG_SIZE:							return "CL_INVALID_ARG_SIZE";
		case CL_INVALID_KERNEL_ARGS:						return "CL_INVALID_KERNEL_ARGS";
		case CL_INVALID_WORK_DIMENSION:						return "CL_INVALID_WORK_DIMENSION";
		case CL_INVALID_WORK_GROUP_SIZE:					return "CL_INVALID_WORK_GROUP_SIZE";
		case CL_INVALID_WORK_ITEM_SIZE:						return "CL_INVALID_WORK_ITEM_SIZE";
		case CL_INVALID_GLOBAL_OFFSET:						return "CL_INVALID_GLOBAL_OFFSET";
		case CL_INVALID_EVENT_WAIT_LIST:					return "CL_INVALID_EVENT_WAIT_LIST";
		case CL_INVALID_EVENT:								return "CL_INVALID_EVENT";
		case CL_INVALID_OPERATION:							return "CL_INVALID_OPERATION";
		case CL_INVALID_GL_OBJECT:							return "CL_INVALID_GL_OBJECT";
		case CL_INVALID_BUFFER_SIZE:						return "CL_INVALID_BUFFER_SIZE";
		case CL_INVALID_MIP_LEVEL:							return "CL_INVALID_MIP_LEVEL";
		case CL_INVALID_GLOBAL_WORK_SIZE:					return "CL_INVALID_GLOBAL_WORK_SIZE";
#if CL_HPP_MINIMUM_OPENCL_VERSION >= 200
		case CL_INVALID_DEVICE_QUEUE:						return "CL_INVALID_DEVICE_QUEUE";
		case CL_INVALID_PIPE_SIZE:							return "CL_INVALID_PIPE_SIZE";
#endif
		default:											return "unknown error code";
		}
}
