#include "../RunCL/RunCL.hpp"

void RunCL::rho_sq(uint out_block_size, uint iter, uint layer  ){
	string fname = "RunCL::rho_sq( ..)";
	const int se3_dof  = 6;
	float zero  = 0;
	_clEnqueueFillBuffer( uload_queue, SE3_rho_map_mem, 	&zero, sizeof( float), 0, 		  2*mm_size_bytes_C1, 	fname);
	_clEnqueueFillBuffer( uload_queue, SE3_weight_map_mem, 	&zero, sizeof( float), 0, se3_dof*2*mm_size_bytes_C1, 	fname);
	_clEnqueueFillBuffer( uload_queue, SE3_incr_map_mem, 	&zero, sizeof( float), 0, se3_dof*2*mm_size_bytes_C1, 	fname);

	_clSetKernelArg( rho_sq_kernel, 2, sizeof( uint),   						&out_block_size,	 									fname);		//__private		uint 		out_block_size,			//1

	_clSetKernelArg( rho_sq_kernel, 3, sizeof( cl_mem), 						&mipmap_buf, 											fname);		//__constant	uint8*		mipmap_params,			//2
	_clSetKernelArg( rho_sq_kernel, 4, sizeof( cl_mem), 						&uint_param_buf, 										fname);		//__constant	uint*		uint_params,			//3
	_clSetKernelArg( rho_sq_kernel, 5, sizeof( cl_mem), 						&fp32_param_buf, 										fname);		//__constant	float*		fp32_params,			//4
	//output
	_clSetKernelArg( rho_sq_kernel,20, sizeof( cl_mem), 						&SE3_rho_map_mem, 										fname);		//__global		float2* 	Rho_,					//20	// { sum rho^2 ,  count of valid pixels used } Writen to dense patches.
	_clSetKernelArg( rho_sq_kernel,21, sizeof( float)*local_work_size,			NULL, 													fname);		//__local		float2*		local_rho				//21	// float2 local_rho[ local_work_size/2 ]  hence sizeof( float)*local_work_size.

	_clSetKernelArg( rho_sq_kernel,22, sizeof( cl_mem), 						&SE3_weight_map_mem,									fname);		//__global		float4* 	weights_map,			//22
	_clSetKernelArg( rho_sq_kernel,23, sizeof( float)*local_work_size*se3_dof,	NULL,													fname);		//__local		float4* 	local_weights,			//23	// float4 local_weights[ local_work_size/2 ]  hence sizeof( float)*local_work_size*4 NB only used as a message between threads.

	_clSetKernelArg( rho_sq_kernel,24, sizeof( cl_mem), 						&SE3_incr_map_mem,										fname);		//__global 		float4*		SE3_incr_map_,			//24
	_clSetKernelArg( rho_sq_kernel,25, sizeof( float)*local_work_size*se3_dof,	NULL,													fname);		//__local 		float4*		local_SE3_incr			//25

	const uint			patch_size 			= 32;																						//TODO set global patch size from device parameters		// generally:  device_work_size_multiple = patch_size * integer,   eg 32, 64, 128
	uint				read_rows			= MipMap[layer * 8 + MiM_READ_ROWS] ;
	uint				read_cols			= MipMap[layer * 8 + MiM_READ_COLS] ;
	uint				rows_blocks			= read_rows/patch_size  + (fmod(read_rows, patch_size) != 0) ;
	uint				cols_blocks			= read_cols/patch_size  + (fmod(read_cols, patch_size) != 0) ;
	uint				cols_per_row		= cols_blocks * patch_size;

	size_t				threads_required 	= rows_blocks * cols_blocks * patch_size ;
	size_t				threads_to_launch	= (threads_required/device_work_size_multiple  +  ( fmod( threads_required, device_work_size_multiple ) != 0 ) )  * device_work_size_multiple ; 	// smallest integer multiple of "device_work_size_multiple" >= threads_required;

	cl_kernel 			kernel_to_call		= rho_sq_kernel;
	cl_command_queue 	queue_to_call		= m_queue;
	cl_int				res, status;
	cl_event			ev;

	_clSetKernelArg( rho_sq_kernel, 0, sizeof( uint),   				&layer,	 												fname);		//__private		uint 		layer,					//0
	_clSetKernelArg( rho_sq_kernel, 1, sizeof( uint),   				&cols_per_row,											fname);		//__private		uint 		cols_per_row,			//1

	res		= clEnqueueNDRangeKernel(queue_to_call, kernel_to_call, 1, 0, &threads_to_launch, &device_work_size_multiple, 0, NULL, &ev); 	// run mipmap_float4_kernel, NB wait for own previous iteration.
}


void RunCL::update_SE3( uint layer, float delta_theta, float delta )									// NB good for images upto 640x480 layer zero, above that need a patch kernel approach to ensure each DoF fits in 1 workgroup. see device_work_size_multiple
{
	string fname = "RunCL::se3_rho_sq( ..)";
	cl_kernel		kernel 				= update_SE3_kernel;																			//NB call just one workgroup to sum the whole image maps from the patch kernel.
	const uint		patch_size 			= 32;																							//TODO set global patch size from device parameters // generally:  device_work_size_multiple = patch_size * integer,   eg 32, 64, 128
	const uint		SE3_DoF				= 6;
	uint			read_rows			= MipMap[layer * 8 + MiM_READ_ROWS] ;
	uint			read_cols			= MipMap[layer * 8 + MiM_READ_COLS] ;
	uint			rows_blocks			= read_rows/patch_size  + (fmod(read_rows, patch_size) != 0) ;									// num rows in the fully reduced map
	uint			cols_blocks			= read_cols/patch_size  + (fmod(read_cols, patch_size) != 0) ;									// num cols in the fully reduced map

	uint			threads_per_DoF		= powf(2,ceil( log2((float)cols_blocks) )); // 10 layer 1 =>  pown(2,ciel(log2(10.0f) ))=16; 6*16=96.      // * rows_blocks  ;//	8x10=80 layer1 => 96 threads to launch?			// num pixels in fully reduced map. Req per SE3 DoF.
	uint 			DoF_per_workgroup	= device_work_size_multiple / threads_per_DoF;
	size_t			threads_required	= (device_work_size_multiple * SE3_DoF) / DoF_per_workgroup;		// NB device_work_size_multiple is usually a poer of 2, DoF_per_workgroup will also be a power of 2.
	uint 			workgroups_required	= ceil( (float)threads_required / device_work_size_multiple );
	size_t			threads_to_launch	= workgroups_required  *  device_work_size_multiple;

	uint 			row_offset			= rows_blocks + 4;																								//2
	uint			thread_offset		= threads_per_DoF;																								//3     2^n  > pixels in fully reduced patch
	uint			mm_cols				= uint_params[MM_COLS];																							//4

	//private
	_clSetKernelArg( kernel, 0, sizeof( uint),							&cols_blocks,				fname);								//__private	uint		cols,					//0
	_clSetKernelArg( kernel, 1, sizeof( uint),							&rows_blocks,				fname);								//__private	uint 		rows,					//1
	_clSetKernelArg( kernel, 2, sizeof( uint),							&row_offset,				fname);								//__private	uint 		row_offset,				//2
	_clSetKernelArg( kernel, 3, sizeof( uint),							&thread_offset,				fname);								//__private	uint		thread_offset,			//3
	_clSetKernelArg( kernel, 4, sizeof( uint),							&mm_cols,					fname);								//__private	uint		mm_cols,				//4
	//global
	_clSetKernelArg( kernel, 7, sizeof( cl_mem),						&SE3_rho_map_mem,			fname);								//__global	float2*		Rho_,					//7		// { sum rho^2 ,  count of valid pixels used } Writen to dense patches.
	_clSetKernelArg( kernel, 8, sizeof( cl_mem),						&SE3_weight_map_mem,		fname);								//__global	float2*		weights_map,			//8
	_clSetKernelArg( kernel, 9, sizeof( cl_mem),						&SE3_incr_map_mem,			fname);								//__global	float2*		SE3_incr_map_,			//9

	cl_command_queue	queue_to_call		= m_queue;
	cl_int				res, status;
	cl_event			ev;

	res 	= clEnqueueNDRangeKernel(queue_to_call, kernel, 1, 0, &threads_to_launch, &device_work_size_multiple, 0, NULL, &ev); 	// run mipmap_float4_kernel, NB wait for own previous iteration.
}
