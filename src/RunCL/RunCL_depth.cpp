#include "RunCL.hpp"

void RunCL::update_depth( uint out_block_size, uint layer){
	string	fname 						= "RunCL::update_depth(..)";
	int 	local_verbosity_threshold 	= V_RUNCL_UPDATE_DEPTH;																if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::update_depth(..)_chk0"<<flush;}
	cl_kernel	kernel					= update_depth_kernel;

	//uint	frame_coun					= dataset_frame_num;
	float	reduction					= pow(2,layer);
	uint	lookup_table_offset			= patch_lookup_table_offset[layer];
	//uint	out_block_size;

	uint	read_offset_				= MipMap[layer*8 + MiM_READ_OFFSET];
	uint	layer_pixels				= MipMap[layer*8 + MiM_PIXELS];
	uint	read_cols_					= MipMap[layer*8 + MiM_READ_COLS];
	uint	read_rows_					= MipMap[layer*8 + MiM_READ_ROWS];	//uint_params[MM_PIXELS];

	uint	mm_cols						= uint_params[MM_COLS];
	uint	mm_rows						= uint_params[MM_ROWS];
	uint	mm_pixels					= uint_params[MM_PIXELS];

	uint	layer_offset				= MipMap[layer*8 + MiM_READ_OFFSET];
	uint	stop_offset					= layer_offset + (read_rows_ -1) * mm_cols + read_cols_;

	uint	ST3_offset					= patch_ST3_hessian_start_idx[layer][0][0];
	uint	ST3_u_step					= patch_ST3_hessian_start_idx[layer][0][1] - ST3_offset;
	uint	ST3_v_step					= patch_ST3_hessian_start_idx[layer][1][0] - ST3_offset;

	size_t	threads_to_launch			= patch_num_threads[layer];
	size_t	local_work_size_			= block_size;									// Could be changed to an integer multiple, i.e. use "RunCL::local_work_size", beware numbers not multiples of out_block_size.
	uint	local_mem_size				= (block_size * local_work_size_)/(pow( out_block_size, 2) ); // i.e. one entry per 4x4 pixel patch.

	// Zero output buffers
	float minus_one_f					=-1.0f;
	uint depth_iter_per_layer			= 3;
	uint cols							= MipMap[ (layer+2)*8 + MiM_READ_COLS];
	uint rows							= (MipMap[ (layer+2)*8 + MiM_READ_ROWS] + 1)	*  depth_iter_per_layer;
	size_t	depthUpdate_bytes			= cols * rows * sizeof(cl_float2);


	_clEnqueueFillBuffer( uload_queue, SE3_rho_map_mem, &minus_one_f, sizeof(float), 0, depthUpdate_bytes, fname   );
	_clEnqueueFillBuffer( uload_queue, depth_mem_temp,  &minus_one_f, sizeof(float), 0, depthUpdate_bytes, fname   );


	// constant buffers uploaded

	//Inputs:
	_clSetKernelArg( kernel, 0, sizeof(uint),						&frame_count,										fname);		// __private	const uint	frame_count				//0
	_clSetKernelArg( kernel, 1, sizeof(float),						&reduction,											fname);		// __private	const float	reduction,				//1		//	i.e. 2^layer		= base_cols/read_cols_;		//  NB these __private args couldbe a single __constant uint* buffer, uploaded at the start of the loop. //
	_clSetKernelArg( kernel, 2, sizeof(uint),						&lookup_table_offset,								fname);		// __private	const uint	lookup_table_offset,	//2															//  Likewise could list the order of img_ and vel_ buffers with a __constant uint* buffer				 //
	_clSetKernelArg( kernel, 3, sizeof(uint),						&out_block_size,									fname);		// __private	const uint	out_block_size,			//3

	_clSetKernelArg( kernel, 4, sizeof(uint),						&read_offset_,										fname);		// __private	const uint	read_offset_,			//4		= mipmap_params_[MiM_READ_OFFSET];
	_clSetKernelArg( kernel, 5, sizeof(uint),						&stop_offset,										fname);		// __private	const uint	stop_offset,			//5		= layer_offset + (read_rows_ -1) * mm_cols + read_cols_	;	// bottom right corner of source image layer
	_clSetKernelArg( kernel, 6, sizeof(uint),						&layer_pixels,										fname);		// __private	const uint	layer_pixels,			//6		= mipmap_params_[MiM_PIXELS];
	_clSetKernelArg( kernel, 7, sizeof(uint),						&read_cols_,										fname);		// __private	const uint	read_cols_,				//7		= mipmap_params_[MiM_READ_COLS];
	_clSetKernelArg( kernel, 8, sizeof(uint),						&read_rows_,										fname);		// __private	const uint	read_rows_,				//8		= mipmap_params_[MiM_READ_ROWS];

	_clSetKernelArg( kernel, 9, sizeof(uint),						&mm_cols,											fname);		// __private	const uint	mm_cols,				//9		= uint_params[MM_COLS];
	_clSetKernelArg( kernel, 10, sizeof(uint),						&mm_rows,											fname);		// __private	const uint	mm_rows,				//10		= uint_params[MM_ROWS];
	_clSetKernelArg( kernel, 11, sizeof(uint),						&mm_pixels,											fname);		// __private	const uint	mm_pixels,				//11	= uint_params[MM_PIXELS];

	_clSetKernelArg( kernel, 12, sizeof(uint),						&ST3_offset,										fname);		// __private	const uint	ST3_offset,				//12	= ST3_offset3.s0;	//SE3_out_step_1 * (num_SE3_DoF + 1);// + layer_offset; __private	uint3		ST3_offset3,			//4
	_clSetKernelArg( kernel, 13, sizeof(uint),						&ST3_u_step,										fname);		// __private	const uint	ST3_u_step,				//13	= ST3_offset3.s1;	// step between elements of the Hessian matrix
	_clSetKernelArg( kernel, 14, sizeof(uint),						&ST3_v_step,										fname);		// __private	const uint	ST3_v_step,				//14	= ST3_offset3.s2;

	_clSetKernelArg( kernel, 15, sizeof(cl_mem),					&cur_frames_k2kbuf,									fname);		// __constant	float16*	inv_k2k,				//15		// transforms for 4 past frames,  k2k_buf
	_clSetKernelArg( kernel, 16, sizeof(cl_mem),					&cur_frames_st3buf,									fname);		// __constant	float4*		st3,					//16		// array of pose transforms to the set previous frames
	_clSetKernelArg( kernel, 17, sizeof(cl_mem),					&patch_lookup_table_buf,							fname);		// __constant 	float4*		lookup_table,			//17		// should ideally be a constant.
	_clSetKernelArg( kernel, 18, sizeof(cl_mem),					&SE3_map_mem,										fname);		// __constant 	float4*		SE3_map,				//18		// _cur_frame

	_clSetKernelArg( kernel, 19, sizeof(cl_mem),					&ST3_img_grad_mem,									fname);		// __constant 	float4*		SE3_map,				//18		// _cur_frame

	_clSetKernelArg( kernel, 20, sizeof(cl_mem),					&current_frames[current_frames_idx[0]].img_buf,		fname);		// __global		float4*		img_cur,				//19		// multiple past frames. NB retain frames at powers of 2, and vary starting power plus num franes.
	_clSetKernelArg( kernel, 21, sizeof(cl_mem),					&current_frames[current_frames_idx[1]].img_buf,		fname);		// __global		float4*		img_past_0,				//20
	_clSetKernelArg( kernel, 22, sizeof(cl_mem),					&current_frames[current_frames_idx[2]].img_buf,		fname);		// __global		float4*		img_past_1,				//21
	_clSetKernelArg( kernel, 23, sizeof(cl_mem),					&current_frames[current_frames_idx[3]].img_buf,		fname);		// __global		float4*		img_past_2,				//22
	_clSetKernelArg( kernel, 24, sizeof(cl_mem),					&current_frames[current_frames_idx[4]].img_buf,		fname);		// __global		float4*		img_past_3,				//23

	_clSetKernelArg( kernel, 25, sizeof(cl_mem),					&depth_mem,											fname);		// __global		float*		depth_map,				//24	// current frame depth, now stored as inv_depth

	_clSetKernelArg( kernel, 26, sizeof(cl_mem),					&current_frames[current_frames_idx[0]].r_vel_buf,	fname);		// __global		float4*		vel_cur,				//25	// multiple past frames.
	_clSetKernelArg( kernel, 27, sizeof(cl_mem),					&current_frames[current_frames_idx[1]].r_vel_buf,	fname);		// __global		float4*		vel_past_0,				//26	// TO DO, relative velocity not used yet. Will use it to modify depth map with timestep for past frames.
	_clSetKernelArg( kernel, 28, sizeof(cl_mem),					&current_frames[current_frames_idx[2]].r_vel_buf,	fname);		// __global		float4*		vel_past_1,				//27
	_clSetKernelArg( kernel, 29, sizeof(cl_mem),					&current_frames[current_frames_idx[3]].r_vel_buf,	fname);		// __global		float4*		vel_past_2,				//28
	_clSetKernelArg( kernel, 30, sizeof(cl_mem),					&current_frames[current_frames_idx[4]].r_vel_buf,	fname);		// __global		float4*		vel_past_3,				//29

	// //outputs
	_clSetKernelArg( kernel, 31, sizeof(cl_mem), 									&SE3_rho_map_mem,					fname);		// __global		float2*		Rho_,					//30	// { sum rho^2 ,  count of valid pixels used } Writen to dense patches.
	_clSetKernelArg( kernel, 32, sizeof(cl_float2)*local_mem_size,					NULL,								fname);		// __local		float2*		local_rho,				//31	// float2 local_rho[ local_work_size/2 ]  hence sizeof( float)*local_work_size.

	_clSetKernelArg( kernel, 33, sizeof(cl_mem), 									&depth_mem_temp,					fname);		// __global		float2*		inv_depth_incr,			//32
	_clSetKernelArg( kernel, 34, sizeof(float)*local_mem_size,						NULL,								fname);		// __local		float*		local_depth_incr,		//33
	_clSetKernelArg( kernel, 35, sizeof(cl_float2)*local_mem_size,					NULL,								fname);		// __local		float2*		local_J_inv_d			//34

																																if( verbosity>local_verbosity_threshold) {
																																	cout<<"\n\nRunCL::update_depth()_chk1"<<
																																	"\nthreads_to_launch   = "<<threads_to_launch<<
																																	"\nlocal_work_size_    = "<<local_work_size_<<
																																	"\nlocal_mem_size      = "<<local_mem_size<<
																																	"\nread_offset_        = "<<read_offset_<<
																																	"\nlayer_offset        = "<<layer_offset<<
																																	"\nlookup_table_offset = "<<lookup_table_offset<<
																																	"\nlayer_pixels        = "<<layer_pixels<<
																																	"\nread_cols_          = "<<read_cols_<<
																																	"\nread_rows_          = "<<read_rows_<<
																																	endl<<flush;
																																}

	_clEnqueueNDRangeKernel(										// NB depth iteration is internal to the kernel within the layer.  Regularization and propagation to next layer requires further kernels.
		m_queue,				//cl_command_queue _queue,
		kernel,					//cl_kernel        kernel,
		1,						//cl_uint          work_dim,
		0,						//const size_t *   global_work_offset,
		&threads_to_launch,		//const size_t *   global_work_size,
		&local_work_size_,		//const size_t *   local_work_size,
		fname					//string           fname
	);
																																if( verbosity>local_verbosity_threshold) {
																																	cout<<"\n\nRunCL::update_depth()_finished #############################################################"<<flush;
																																	uint depth_iter_per_layer	= min(frame_count-1, num_current_frames);	//num_current_frames;  3; //
																																	DownloadAndSaveDepthUpdate( layer, depth_iter_per_layer  );
																																	/*
																																	int offset			=	MipMap[layer*8 +  MiM_READ_OFFSET   ];
																																	int rows			=	MipMap[layer*8 +  MiM_READ_ROWS   ];
																																	int size_bytes		= rows * mm_width * 4*sizeof(float) ;

																																	cv::Mat temp_mat 	= cv::Mat::zeros (rows, mm_width, CV_32FC4);
																																	cout<<"\nlayer = "<<layer<<", offset 	= "<<offset<<",  rows ="<<rows<<flush;

																																	// read 1st elem of Jacobian to verify kernel summation.
																																	ReadOutput(temp_mat.data, SE3_grad_map_mem, size_bytes, offset*4*sizeof(float)   );// , 0  //offset

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
																																			sum_H11.z 	+= pow(temp_mat.at<cl_float4>(row,col).z, 2);
																																		}
																																	}
																																	cout<<"\n\n##### layer = "<<layer<<", SE3_grad_map_mem sum_J1 = "<<sum_J1.w<<", "<<sum_J1.x<<", "<<sum_J1.y<<", "<<sum_J1.z
																																													<<",    sum_H11 = "<< sum_H11.w<<", "<<sum_H11.x<<", "<<sum_H11.y<<", "<<sum_H11.z
																																	<<endl<<endl<<flush;
																																	*/
																																}
}


void RunCL::update_depth_2( uint out_block_size, uint layer){
	string	fname 						= "RunCL::update_depth(..)";
	int 	local_verbosity_threshold 	= V_RUNCL_UPDATE_DEPTH;																if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::update_depth(..)_chk0"<<flush;}
	cl_kernel	kernel					= update_depth_2_kernel;

	float	reduction					= pow(2,layer);
	uint	lookup_table_offset			= patch_lookup_table_offset[layer];

	uint	read_offset_				= MipMap[layer*8 + MiM_READ_OFFSET];
	uint	layer_pixels				= MipMap[layer*8 + MiM_PIXELS];
	uint	read_cols_					= MipMap[layer*8 + MiM_READ_COLS];
	uint	read_rows_					= MipMap[layer*8 + MiM_READ_ROWS];	//uint_params[MM_PIXELS];

	uint	mm_cols						= uint_params[MM_COLS];
	float	inv_depth_step				= fp32_params[MAX_INV_DEPTH] / ((float)NUM_DEPTH_STEPS);

	uint	layer_offset				= MipMap[layer*8 + MiM_READ_OFFSET];
	uint	stop_offset					= layer_offset + (read_rows_ -1) * mm_cols + read_cols_;

	size_t	threads_to_launch			= patch_num_threads[layer];
	size_t	local_work_size_			= block_size;									// Could be changed to an integer multiple, i.e. use "RunCL::local_work_size", beware numbers not multiples of out_block_size.
	uint	local_mem_size				= (block_size * local_work_size_)/(pow( out_block_size, 2) ); // i.e. one entry per 4x4 pixel patch.

	// Zero output buffers
	float minus_one_f					=-1.0f;
	uint depth_iter_per_layer			= 3;
	uint cols							= MipMap[ (layer+2)*8 + MiM_READ_COLS];
	uint rows							= (MipMap[ (layer+2)*8 + MiM_READ_ROWS] + 1)	*  depth_iter_per_layer;
	size_t	depthUpdate_bytes			= cols * rows * sizeof(cl_float2);

	float default_inv_depth				= 0.07f;	// half the max inv depth, i.e. twice the min depth.

	_clEnqueueFillBuffer( uload_queue, SE3_rho_map_mem, &minus_one_f, sizeof(float), 0, depthUpdate_bytes, fname   );
	_clEnqueueFillBuffer( uload_queue, depth_mem_temp,  &minus_one_f, sizeof(float), 0, depthUpdate_bytes, fname   );

	if (layer==4) { _clEnqueueFillBuffer( uload_queue, depth_mem,		&default_inv_depth, sizeof(float), 0, mm_size_bytes_C1,  fname   ); }	// TODO  remove this, temporary for testing tracking and mapping given GT poses.

	// constant buffers uploaded

	//Inputs:
	_clSetKernelArg( kernel, 0, sizeof(uint),						&frame_count,										fname);		// __private	const uint	frame_count				//0
	_clSetKernelArg( kernel, 1, sizeof(float),						&reduction,											fname);		// __private	const float	reduction,				//1		//	i.e. 2^layer		= base_cols/read_cols_;		//  NB these __private args couldbe a single __constant uint* buffer, uploaded at the start of the loop. //
	_clSetKernelArg( kernel, 2, sizeof(uint),						&lookup_table_offset,								fname);		// __private	const uint	lookup_table_offset,	//2															//  Likewise could list the order of img_ and vel_ buffers with a __constant uint* buffer				 //
	_clSetKernelArg( kernel, 3, sizeof(uint),						&out_block_size,									fname);		// __private	const uint	out_block_size,			//3

	_clSetKernelArg( kernel, 4, sizeof(uint),						&read_offset_,										fname);		// __private	const uint	read_offset_,			//4		= mipmap_params_[MiM_READ_OFFSET];
	_clSetKernelArg( kernel, 5, sizeof(uint),						&stop_offset,										fname);		// __private	const uint	stop_offset,			//5		= layer_offset + (read_rows_ -1) * mm_cols + read_cols_	;	// bottom right corner of source image layer
	_clSetKernelArg( kernel, 6, sizeof(uint),						&layer_pixels,										fname);		// __private	const uint	layer_pixels,			//6		= mipmap_params_[MiM_PIXELS];
	_clSetKernelArg( kernel, 7, sizeof(uint),						&read_cols_,										fname);		// __private	const uint	read_cols_,				//7		= mipmap_params_[MiM_READ_COLS];
	_clSetKernelArg( kernel, 8, sizeof(uint),						&read_rows_,										fname);		// __private	const uint	read_rows_,				//8		= mipmap_params_[MiM_READ_ROWS];

	_clSetKernelArg( kernel, 9, sizeof(uint),						&mm_cols,											fname);		// __private	const uint	mm_cols,				//9		= uint_params[MM_COLS];
	_clSetKernelArg( kernel, 10, sizeof(float),						&inv_depth_step,									fname);		// __private	const uint	mm_cols,				//9		= uint_params[MM_COLS];

	_clSetKernelArg( kernel, 11, sizeof(cl_mem),					&cur_frames_k2kbuf,									fname);		// __constant	float16*	inv_k2k,				//15		// transforms for 4 past frames,  k2k_buf
	_clSetKernelArg( kernel, 12, sizeof(cl_mem),					&patch_lookup_table_buf,							fname);		// __constant 	float4*		lookup_table,			//17		// should ideally be a constant.

	_clSetKernelArg( kernel, 13, sizeof(cl_mem),					&current_frames[current_frames_idx[0]].img_buf,		fname);		// __global		float4*		img_cur,				//19		// multiple past frames. NB retain frames at powers of 2, and vary starting power plus num franes.
	_clSetKernelArg( kernel, 14, sizeof(cl_mem),					&current_frames[current_frames_idx[1]].img_buf,		fname);		// __global		float4*		img_past_0,				//20
	_clSetKernelArg( kernel, 15, sizeof(cl_mem),					&current_frames[current_frames_idx[2]].img_buf,		fname);		// __global		float4*		img_past_1,				//21
	_clSetKernelArg( kernel, 16, sizeof(cl_mem),					&current_frames[current_frames_idx[3]].img_buf,		fname);		// __global		float4*		img_past_2,				//22
	_clSetKernelArg( kernel, 17, sizeof(cl_mem),					&current_frames[current_frames_idx[4]].img_buf,		fname);		// __global		float4*		img_past_3,				//23

	_clSetKernelArg( kernel, 18, sizeof(cl_mem),					&depth_mem,											fname);		// __global		float*		depth_map,				//24	// current frame depth, now stored as inv_depth

	_clSetKernelArg( kernel, 19, sizeof(cl_mem),					&current_frames[current_frames_idx[0]].r_vel_buf,	fname);		// __global		float4*		vel_cur,				//25	// multiple past frames.
	_clSetKernelArg( kernel, 20, sizeof(cl_mem),					&current_frames[current_frames_idx[1]].r_vel_buf,	fname);		// __global		float4*		vel_past_0,				//26	// TO DO, relative velocity not used yet. Will use it to modify depth map with timestep for past frames.
	_clSetKernelArg( kernel, 21, sizeof(cl_mem),					&current_frames[current_frames_idx[2]].r_vel_buf,	fname);		// __global		float4*		vel_past_1,				//27
	_clSetKernelArg( kernel, 22, sizeof(cl_mem),					&current_frames[current_frames_idx[3]].r_vel_buf,	fname);		// __global		float4*		vel_past_2,				//28
	_clSetKernelArg( kernel, 23, sizeof(cl_mem),					&current_frames[current_frames_idx[4]].r_vel_buf,	fname);		// __global		float4*		vel_past_3,				//29

	// //outputs
	_clSetKernelArg( kernel, 24, sizeof(cl_mem), 					&SE3_rho_map_mem,									fname);		// __global		float2*		Rho_,					//30	// { sum rho^2 ,  count of valid pixels used } Writen to dense patches.
	_clSetKernelArg( kernel, 25, sizeof(cl_float2)*local_mem_size,	NULL,												fname);		// __local		float2*		local_rho,				//31	// float2 local_rho[ local_work_size/2 ]  hence sizeof( float)*local_work_size.

	_clSetKernelArg( kernel, 26, sizeof(cl_mem), 					&depth_mem_temp,									fname);		// __global		float2*		inv_depth_incr,			//32

																																if( verbosity>local_verbosity_threshold) {
																																	cout<<"\n\nRunCL::update_depth()_chk1"<<
																																	"\nthreads_to_launch   = "<<threads_to_launch<<
																																	"\nlocal_work_size_    = "<<local_work_size_<<
																																	"\nlocal_mem_size      = "<<local_mem_size<<
																																	"\nread_offset_        = "<<read_offset_<<
																																	"\nlayer_offset        = "<<layer_offset<<
																																	"\nlookup_table_offset = "<<lookup_table_offset<<
																																	"\nlayer_pixels        = "<<layer_pixels<<
																																	"\nread_cols_          = "<<read_cols_<<
																																	"\nread_rows_          = "<<read_rows_<<
																																	endl<<flush;
																																}

	_clEnqueueNDRangeKernel(										// NB depth iteration is internal to the kernel within the layer.  Regularization and propagation to next layer requires further kernels.
		m_queue,				//cl_command_queue _queue,
		kernel,					//cl_kernel        kernel,
		1,						//cl_uint          work_dim,
		0,						//const size_t *   global_work_offset,
		&threads_to_launch,		//const size_t *   global_work_size,
		&local_work_size_,		//const size_t *   local_work_size,
		fname					//string           fname
	);
																																if( verbosity>local_verbosity_threshold) {
																																	cout<<"\n\nRunCL::update_depth()_finished #############################################################"<<flush;
																																	uint depth_iter_per_layer	= 1; //num_current_frames;
																																	DownloadAndSaveDepthUpdate( layer, depth_iter_per_layer  );
																																	/*
																																	int offset			=	MipMap[layer*8 +  MiM_READ_OFFSET   ];
																																	int rows			=	MipMap[layer*8 +  MiM_READ_ROWS   ];
																																	int size_bytes		= rows * mm_width * 4*sizeof(float) ;

																																	cv::Mat temp_mat 	= cv::Mat::zeros (rows, mm_width, CV_32FC4);
																																	cout<<"\nlayer = "<<layer<<", offset 	= "<<offset<<",  rows ="<<rows<<flush;

																																	// read 1st elem of Jacobian to verify kernel summation.
																																	ReadOutput(temp_mat.data, SE3_grad_map_mem, size_bytes, offset*4*sizeof(float)   );// , 0  //offset

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
																																	*/
																																}
}

void RunCL::propagate_depth_next_layer(uint write_layer ){	// layer = write layer
	string 	fname						= "RunCL::mipmap_depthmap(..)";
	int 	local_verbosity_threshold	= V_RUNCL_PROPAGATE_DEPTH_NEXT_LAYER;												if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::propagate_depth_next_layer(..)_chk0"<<
																																",   write_layer = "<<write_layer<<flush; }
	cl_kernel		kernel				= enlarge_layer_float_kernel;

	uint	lookup_table_read_offset	= patch_lookup_table_offset[write_layer+1];
	uint	write_offset				= MipMap[write_layer*8 + MiM_READ_OFFSET];
	uint	buf_width					= uint_params[MM_COLS];
	uint	patch_height				= patch_size;

	uint	read_cols_					= MipMap[ write_layer*8 + MiM_READ_COLS];
	uint	read_rows_					= MipMap[ write_layer*8 + MiM_READ_ROWS];  //uint_params[MM_PIXELS];

	uint	stop_offset					= write_offset + (read_rows_ -1) * buf_width + read_cols_;
																															if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL:propagate_depth_next_layer(..) chk1" <<
																																"\nlookup_table_read_offset 	= "		<<lookup_table_read_offset<<
																																"\nlookup_table_write_offset 	= "		<<write_offset<<
																																"\nbuf_width                	= "		<<buf_width<<
																																"\npatch_height             	= "		<<patch_height<<
																																"\nstop_offset              	= "		<<stop_offset<<
																																"\npatch_lookup_table_buf   	= "		<<patch_lookup_table_buf<<
																																"\ndepth_mem                	= "		<<depth_mem<<
																																endl<<flush;
																															}
	_clSetKernelArg( kernel, 0, sizeof(int),						&lookup_table_read_offset,							fname);		// __private	const uint	lookup_table_read_offset,	//0
	_clSetKernelArg( kernel, 1, sizeof(int),						&write_offset,										fname);		// __private	const uint	lookup_table_write_offset,	//1
	_clSetKernelArg( kernel, 2, sizeof(int),						&buf_width,											fname);		// __private	uint		buf_width,					//2		mm_cols, i.e. width of the buffer holding the image pyramid
	_clSetKernelArg( kernel, 3, sizeof(int),						&patch_height,										fname);		// __private	uint		patch_height,				//3
	_clSetKernelArg( kernel, 4, sizeof(int),						&stop_offset,										fname);		// __private	uint		stop_offset,				//4
	_clSetKernelArg( kernel, 5, sizeof(cl_mem),						&patch_lookup_table_buf,							fname);		// __constant 	float4*		lookup_table,				//5
	_clSetKernelArg( kernel, 6, sizeof(cl_mem),						&depth_mem,											fname);		// __global 	float*		img							//6

	size_t	threads_to_launch	= patch_num_threads[write_layer+1];
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
																															if(verbosity>local_verbosity_threshold) {
																																cout<<"\n\nRunCL::propagate_depth_next_layer(..)_chk3 Finished all loops."<<flush;
																																stringstream ss;	ss << dataset_frame_num << "_propagate_depth_next_layer";
																																cv::Size new_Image_size = cv::Size(mm_width, mm_height);
																																ss << "_raw_";
																																stringstream ss_path;	ss_path << "depth_mem";
																																DownloadAndSave( depth_mem,   	ss.str(),   paths.at(ss_path.str()),   	mm_size_bytes_C1,   mm_Image_size,   CV_32FC1, 	false , fp32_params[MAX_INV_DEPTH]);
																															}
																															if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::propagate_depth_next_layer(..)_chk4 Finished:#######################################################"<<flush;}
}
