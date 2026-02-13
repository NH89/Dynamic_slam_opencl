#include "RunCL.hpp"

void RunCL::update_depth( uint layer){
	string fname = "RunCL::update_depth(..)";
	int local_verbosity_threshold = V_RUNCL_UPDATE_DEPTH;																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::mipmap_depthmap(..)_chk0"<<flush;}
	cl_kernel		kernel		= update_depth_kernel;

	float	reduction		;
	uint	lookup_table_offset;
	uint	out_block_size;

	uint	read_offset_;
	uint	stop_offset;
	uint	layer_pixels;
	uint	read_cols_;
	uint	read_rows_;

	uint	mm_cols;
	uint	mm_rows;
	uint	mm_pixels;

	uint	ST3_offset;
	uint	ST3_u_step;
	uint	ST3_v_step;

	// constant buffers
	cl_float16*	inv_k2k;
	cl_float4*	st3;
	cl_float4*	lookup_table;
	cl_float4*	SE3_grad_map;

	//Inputs:
	_clSetKernelArg( kernel, 2, sizeof(float),						&reduction,											fname);		// __private	const float	reduction,				//0		//	i.e. 2^layer		= base_cols/read_cols_;		//  NB these __private args couldbe a single __constant uint* buffer, uploaded at the start of the loop. //
	_clSetKernelArg( kernel, 2, sizeof(uint),						&lookup_table_offset,								fname);		// __private	const uint	lookup_table_offset,	//1															//  Likewise could list the order of img_ and vel_ buffers with a __constant uint* buffer				 //
	_clSetKernelArg( kernel, 2, sizeof(uint),						&out_block_size,									fname);		// __private	const uint	out_block_size,			//2

	_clSetKernelArg( kernel, 2, sizeof(uint),						&read_offset_,										fname);		// __private	const uint	read_offset_,			//3					= mipmap_params_[MiM_READ_OFFSET];
	_clSetKernelArg( kernel, 2, sizeof(uint),						&stop_offset,										fname);		// __private	const uint	stop_offset,			//4					= layer_offset + (read_rows_ -1) * mm_cols + read_cols_	;	// bottom right corner of source image layer
	_clSetKernelArg( kernel, 2, sizeof(uint),						&layer_pixels,										fname);		// __private	const uint	layer_pixels,			//5					= mipmap_params_[MiM_PIXELS];
	_clSetKernelArg( kernel, 2, sizeof(uint),						&read_cols_,										fname);		// __private	const uint	read_cols_,				//6					= mipmap_params_[MiM_READ_COLS];
	_clSetKernelArg( kernel, 2, sizeof(uint),						&read_rows_,										fname);		// __private	const uint	read_rows_,				//7					= mipmap_params_[MiM_READ_ROWS];

	_clSetKernelArg( kernel, 2, sizeof(uint),						&mm_cols,											fname);		// __private	const uint	mm_cols,				//8					= uint_params[MM_COLS];
	_clSetKernelArg( kernel, 2, sizeof(uint),						&mm_rows,											fname);		// __private	const uint	mm_rows,				//9					= uint_params[MM_ROWS];
	_clSetKernelArg( kernel, 2, sizeof(uint),						&mm_pixels,											fname);		// __private	const uint	mm_pixels,				//10					= uint_params[MM_PIXELS];
	// 																//layer_offset/mm_cols;
	_clSetKernelArg( kernel, 2, sizeof(uint),						&ST3_offset,										fname);		// __private	const uint	ST3_offset,				//11					= ST3_offset3.s0;	//SE3_out_step_1 * (num_SE3_DoF + 1);// + layer_offset; __private	uint3		ST3_offset3,			//4
	_clSetKernelArg( kernel, 2, sizeof(uint),						&ST3_u_step,										fname);		// __private	const uint	ST3_u_step,				//12					= ST3_offset3.s1;	// step between elements of the Hessian matrix
	_clSetKernelArg( kernel, 2, sizeof(uint),						&ST3_v_step,										fname);		// __private	const uint	ST3_v_step,				//13					= ST3_offset3.s2;

	_clSetKernelArg( kernel, 2, sizeof(uint),						&inv_k2k,											fname);		// __constant	float16*	inv_k2k,				//14		// transforms for 4 past frames,  k2k_buf
	_clSetKernelArg( kernel, 2, sizeof(uint),						&st3,												fname);		// __constant	float4*		st3,					//15		// array of pose transforms to the set previous frames
	_clSetKernelArg( kernel, 2, sizeof(uint),						&lookup_table,										fname);		// __constant 	float4*		lookup_table,			//16		// should ideally be a constant.
	_clSetKernelArg( kernel, 2, sizeof(uint),						&SE3_grad_map,										fname);		// __constant 	float4*		SE3_grad_map,			//17		// _cur_frame

	_clSetKernelArg( kernel, 2, sizeof(cl_mem),						&current_frames[current_frames_idx[0]].img_buf,		fname);		// __global		float4*		img_cur,				//18		// multiple past frames. NB retain frames at powers of 2, and vary starting power plus num franes.
	_clSetKernelArg( kernel, 2, sizeof(cl_mem),						&current_frames[current_frames_idx[1]].img_buf,		fname);		// __global		float4*		img_past_0,				//19
	_clSetKernelArg( kernel, 2, sizeof(cl_mem),						&current_frames[current_frames_idx[2]].img_buf,		fname);		// __global		float4*		img_past_1,				//20
	_clSetKernelArg( kernel, 2, sizeof(cl_mem),						&current_frames[current_frames_idx[3]].img_buf,		fname);		// __global		float4*		img_past_2,				//21
	_clSetKernelArg( kernel, 2, sizeof(cl_mem),						&current_frames[current_frames_idx[4]].img_buf,		fname);		// __global		float4*		img_past_3,				//22

	_clSetKernelArg( kernel, 2, sizeof(cl_mem),						&depth_mem,											fname);		// __global		float*		depth_map,				//23	// current frame depth, now stored as inv_depth

	_clSetKernelArg( kernel, 2, sizeof(cl_mem),						&current_frames[current_frames_idx[0]].r_vel_buf,	fname);		// __global		float4*		vel_cur,				//24	// multiple past frames.
	_clSetKernelArg( kernel, 2, sizeof(cl_mem),						&current_frames[current_frames_idx[1]].r_vel_buf,	fname);		// __global		float4*		vel_past_0,				//25	// TO DO, relative velocity not used yet. Will use it to modify depth map with timestep for past frames.
	_clSetKernelArg( kernel, 2, sizeof(cl_mem),						&current_frames[current_frames_idx[2]].r_vel_buf,	fname);		// __global		float4*		vel_past_1,				//26
	_clSetKernelArg( kernel, 2, sizeof(cl_mem),						&current_frames[current_frames_idx[3]].r_vel_buf,	fname);		// __global		float4*		vel_past_2,				//27
	_clSetKernelArg( kernel, 2, sizeof(cl_mem),						&current_frames[current_frames_idx[4]].r_vel_buf,	fname);		// __global		float4*		vel_past_3,				//28

	// //outputs
	_clSetKernelArg( kernel, 2, sizeof(cl_mem), 					&SE3_rho_map_mem,									fname);		// __global		float2*		Rho_,					//29	// { sum rho^2 ,  count of valid pixels used } Writen to dense patches.
	_clSetKernelArg( kernel, 2, sizeof(cl_float2)*local_work_size,	NULL,												fname);		// __local		float2*		local_rho,				//30	// float2 local_rho[ local_work_size/2 ]  hence sizeof( float)*local_work_size.

	_clSetKernelArg( kernel, 2, sizeof(cl_mem), 					&depth_incr_mem,									fname);		// __global		float2*		inv_depth_incr,			//31
	_clSetKernelArg( kernel, 2, sizeof(cl_float2)*local_work_size, 	NULL,												fname);		// __local		float*		local_depth_incr,		//32
	_clSetKernelArg( kernel, 2, sizeof(uint), 						NULL,												fname);		// __local		float2*		local_J_inv_d			//33




}
