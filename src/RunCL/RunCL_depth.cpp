#include "RunCL.hpp"

void RunCL::update_depth( uint out_block_size, uint layer){
	string fname = "RunCL::update_depth(..)";
	int local_verbosity_threshold 		= V_RUNCL_UPDATE_DEPTH;																		if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::update_depth(..)_chk0"<<flush;}
	cl_kernel		kernel				= update_depth_kernel;

	float	reduction					= pow(2,layer);
	uint	lookup_table_offset			= patch_lookup_table_offset[layer];
	//uint	out_block_size;

	uint	read_offset_				= MipMap[layer*8 + MiM_READ_OFFSET];
	uint	layer_pixels				= MipMap[layer*8 + MiM_PIXELS];
	uint	read_cols_					= MipMap[layer*8 + MiM_READ_COLS];
	uint	read_rows_					= uint_params[MM_PIXELS];

	uint	mm_cols						= uint_params[MM_COLS];
	uint	mm_rows						= uint_params[MM_ROWS];
	uint	mm_pixels					= uint_params[MM_PIXELS];

	uint	layer_offset				= MipMap[layer*8 + MiM_READ_OFFSET];
	uint	stop_offset					= layer_offset + (read_rows_ -1) * mm_cols + read_cols_;

	uint	ST3_offset					= patch_ST3_hessian_start_idx[layer][0][0];
	uint	ST3_u_step					= patch_ST3_hessian_start_idx[layer][0][1] - ST3_offset;
	uint	ST3_v_step					= patch_ST3_hessian_start_idx[layer][1][0] - ST3_offset;

	// constant buffers uploaded

	//Inputs:
	_clSetKernelArg( kernel, 2, sizeof(float),						&reduction,											fname);		// __private	const float	reduction,				//0		//	i.e. 2^layer		= base_cols/read_cols_;		//  NB these __private args couldbe a single __constant uint* buffer, uploaded at the start of the loop. //
	_clSetKernelArg( kernel, 2, sizeof(uint),						&lookup_table_offset,								fname);		// __private	const uint	lookup_table_offset,	//1															//  Likewise could list the order of img_ and vel_ buffers with a __constant uint* buffer				 //
	_clSetKernelArg( kernel, 2, sizeof(uint),						&out_block_size,									fname);		// __private	const uint	out_block_size,			//2

	_clSetKernelArg( kernel, 2, sizeof(uint),						&read_offset_,										fname);		// __private	const uint	read_offset_,			//3		= mipmap_params_[MiM_READ_OFFSET];
	_clSetKernelArg( kernel, 2, sizeof(uint),						&stop_offset,										fname);		// __private	const uint	stop_offset,			//4		= layer_offset + (read_rows_ -1) * mm_cols + read_cols_	;	// bottom right corner of source image layer
	_clSetKernelArg( kernel, 2, sizeof(uint),						&layer_pixels,										fname);		// __private	const uint	layer_pixels,			//5		= mipmap_params_[MiM_PIXELS];
	_clSetKernelArg( kernel, 2, sizeof(uint),						&read_cols_,										fname);		// __private	const uint	read_cols_,				//6		= mipmap_params_[MiM_READ_COLS];
	_clSetKernelArg( kernel, 2, sizeof(uint),						&read_rows_,										fname);		// __private	const uint	read_rows_,				//7		= mipmap_params_[MiM_READ_ROWS];

	_clSetKernelArg( kernel, 2, sizeof(uint),						&mm_cols,											fname);		// __private	const uint	mm_cols,				//8		= uint_params[MM_COLS];
	_clSetKernelArg( kernel, 2, sizeof(uint),						&mm_rows,											fname);		// __private	const uint	mm_rows,				//9		= uint_params[MM_ROWS];
	_clSetKernelArg( kernel, 2, sizeof(uint),						&mm_pixels,											fname);		// __private	const uint	mm_pixels,				//10	= uint_params[MM_PIXELS];

	_clSetKernelArg( kernel, 2, sizeof(uint),						&ST3_offset,										fname);		// __private	const uint	ST3_offset,				//11	= ST3_offset3.s0;	//SE3_out_step_1 * (num_SE3_DoF + 1);// + layer_offset; __private	uint3		ST3_offset3,			//4
	_clSetKernelArg( kernel, 2, sizeof(uint),						&ST3_u_step,										fname);		// __private	const uint	ST3_u_step,				//12	= ST3_offset3.s1;	// step between elements of the Hessian matrix
	_clSetKernelArg( kernel, 2, sizeof(uint),						&ST3_v_step,										fname);		// __private	const uint	ST3_v_step,				//13	= ST3_offset3.s2;

	_clSetKernelArg( kernel, 2, sizeof(uint),						&cur_frames_k2kbuf,									fname);		// __constant	float16*	inv_k2k,				//14		// transforms for 4 past frames,  k2k_buf
	_clSetKernelArg( kernel, 2, sizeof(uint),						&cur_frames_st3buf,									fname);		// __constant	float4*		st3,					//15		// array of pose transforms to the set previous frames
	_clSetKernelArg( kernel, 2, sizeof(uint),						&patch_lookup_table_buf,							fname);		// __constant 	float4*		lookup_table,			//16		// should ideally be a constant.
	_clSetKernelArg( kernel, 2, sizeof(uint),						&SE3_map_mem,										fname);		// __constant 	float4*		SE3_map,				//17		// _cur_frame

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

	_clSetKernelArg( kernel, 2, sizeof(cl_mem), 					&depth_mem_temp,									fname);		// __global		float2*		inv_depth_incr,			//31
	_clSetKernelArg( kernel, 2, sizeof(cl_float2)*local_work_size, 	NULL,												fname);		// __local		float*		local_depth_incr,		//32
	_clSetKernelArg( kernel, 2, sizeof(uint), 						NULL,												fname);		// __local		float2*		local_J_inv_d			//33

	size_t	threads_to_launch	= block_size;						// NB could be a problem on AMD GPUs with minmum 64 threads, not 32.
	size_t	local_work_size_	= block_size;

	_clEnqueueNDRangeKernel(										// NB depth iteration is internal to the kernel within the layer.  Regularization and propagation to next layer requires further kernels.
		m_queue,				//cl_command_queue _queue,
		kernel,					//cl_kernel        kernel,
		1,						//cl_uint          work_dim,
		0,						//const size_t *   global_work_offset,
		&threads_to_launch,		//const size_t *   global_work_size,
		&local_work_size_,		//const size_t *   local_work_size,
		fname					//string           fname
	);
if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::update_depth()_finished #############################################################"<<flush;
																																	int offset			=	MipMap[layer*8 +  MiM_READ_OFFSET   ];
																																	int rows			=	MipMap[layer*8 +  MiM_READ_ROWS   ];
																																	int size_bytes		= rows * mm_width * 4*sizeof(float) ;

																																	cv::Mat temp_mat 	= cv::Mat::zeros (rows, mm_width, CV_32FC4);
																																	cout<<"\n offset 	= "<<offset<<",  rows ="<<rows<<flush;

																																	// read 1st elem of Jacobian to verify kernel summation.
																																	ReadOutput(temp_mat.data, SE3_grad_map_mem, size_bytes, offset*4*sizeof(float)   );// , 0/*offset*/

																																	cl_float4 sum_J1	= {{0.0f}};
																																	cl_float4 sum_H11	= {{0.0f}};

																																	for(int row=0; row<temp_mat.rows; row++){
																																		for(int col=0; col<temp_mat.cols; col++){

																																			sum_J1.w 	+= temp_mat.at<cl_float4>(row,col).w;
																																			sum_J1.x 	+= temp_mat.at<cl_float4>(row,col).x;
																																			sum_J1.y 	+= temp_mat.at<cl_float4>(row,col).y;
																																			sum_J1.z 	+= temp_mat.at<cl_float4>(row,col).z;

																																			sum_H11.w 	+= pow(temp_mat.at<cl_float4>(row,col).w, 2);
																																			sum_H11.x 	+= pow(temp_mat.at<cl_float4>(row,col).x, 2);
																																			sum_H11.y 	+= pow(temp_mat.at<cl_float4>(row,col).y, 2);
																																			sum_H11.z 	+= pow(temp_mat.at<cl_float4>(row,col).z, 2);
																																		}
																																	}
																																	cout<<"\n\n##### layer = "<<layer<<", SE3_grad_map_mem sum_J1 = "<<sum_J1.w<<", "<<sum_J1.x<<", "<<sum_J1.y<<", "<<sum_J1.z
																																													<<",    sum_H11 = "<< sum_H11.w<<", "<<sum_H11.x<<", "<<sum_H11.y<<", "<<sum_H11.z
																																	<<endl<<endl<<flush;
																																}
}


void RunCL::propagate_depth_next_layer(uint layer){
	string fname = "RunCL::mipmap_depthmap(..)";
	int local_verbosity_threshold = V_RUNCL_MIPMAP_DEPTHMAP;																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::mipmap_depthmap(..)_chk0"<<flush;}
	cl_kernel		kernel		= enlarge_layer_float_kernel;

	size_t local_size = local_work_size;																								// set kernel args
	//      __private	 uint layer, set in mipmap_call_kernel(..) below																__private	 uint	    layer,		    //0
	_clSetKernelArg( kernel, 0, sizeof(int), 						&reduction);
    _clSetKernelArg( kernel, 1, sizeof(cl_mem), 					&mipmap_buf,		fname);								//__constant uint8*		mipmap_params,	//1
	_clSetKernelArg( kernel, 2, sizeof(cl_mem), 					&uint_param_buf,	fname);								//__constant uint*		uint_params,	//3
	_clSetKernelArg( kernel, 3, sizeof(cl_mem), 					&depth_mem,			fname);								//__global   float*		img,			//4
	_clSetKernelArg( kernel, 4, (local_size+4) *5*sizeof(float), 	NULL,				fname);								//__local    float*		local_img_patch //5


	 kernel, m_queue, true

																																		if(verbosity>local_verbosity_threshold) {
																																			cout<<"\n\nRunCL::mipmap_depthmap(..)_chk3 Finished all loops."<<flush;
																																			stringstream ss;	ss << dataset_frame_num << "_mipmap_depthmap";
																																			cv::Size new_Image_size = cv::Size(mm_width, mm_height);
																																			//size_t   new_size_bytes = mm_width * mm_height * 4*4;
																																			ss << "_raw_";
																																			stringstream ss_path;	ss_path << "depth_GT";
																																			DownloadAndSave( depthmap_,   	ss.str(),   paths.at(ss_path.str()),   	mm_size_bytes_C1,   mm_Image_size,   CV_32FC1, 	false , fp32_params[MAX_INV_DEPTH]);

																																			cout << "\n  (local_size+4) *5*4* sizeof(float) = "<<  (local_size+4) *5*4* sizeof(float) << " ,   (local_size+4) = " <<  (local_size+4) << endl << flush;
																																		}
																																		if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::mipmap_depthmap(..)_chk4 Finished:#######################################################"<<flush;}

}
