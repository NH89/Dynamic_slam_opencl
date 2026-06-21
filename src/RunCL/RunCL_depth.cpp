#include "RunCL.hpp"

void RunCL::update_depth_2( uint out_block_size, uint layer){
	string		fname 						= "RunCL::update_depth(..)";
	int 		local_verbosity_threshold 	= V_RUNCL_UPDATE_DEPTH;																if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::update_depth_2(..)_chk0"<<flush;}
	cl_kernel	kernel						= update_depth_2_kernel;

	float		reduction					= pow(2,layer);
	uint		lookup_table_offset			= patch_lookup_table_offset[layer];

	uint		read_offset_				= MipMap[layer*8 + MiM_READ_OFFSET];
	uint		layer_pixels				= MipMap[layer*8 + MiM_PIXELS];
	uint		read_cols_					= MipMap[layer*8 + MiM_READ_COLS];
	uint		read_rows_					= MipMap[layer*8 + MiM_READ_ROWS];
	uint		mm_cols						= uint_params[MM_COLS];

	uint		write_offset				= depthmap_params[layer].DM_DATA_OFFSET; 						//patch_depthmap_offset[ layer];
	uint		dm_win_cols					= depthmap_params[layer].DM_WIN_COLS;
	uint		dm_data_rows				= depthmap_params[layer].DM_DATA_ROWS;
	uint		dm_data_stop				= write_offset + dm_win_cols * dm_data_rows;

	float		inv_depth_step				= fp32_params[MAX_INV_DEPTH] / ((float)NUM_DEPTH_STEPS);

	uint		layer_offset				= MipMap[layer*8 + MiM_READ_OFFSET];
	uint		stop_offset					= layer_offset + (read_rows_ -1) * mm_cols + read_cols_;	cout<<"\nstop_offset("<<stop_offset<<")= layer_offset("<<layer_offset<<") + (read_rows_("<<read_rows_<<")_ -1) * mm_cols("<<mm_cols<<") + read_cols_("<<read_cols_<<")_"<<flush;

	size_t		threads_to_launch			= patch_num_threads[layer];
	size_t		local_work_size_			= block_size;									// Could be changed to an integer multiple, i.e. use "RunCL::local_work_size", beware numbers not multiples of out_block_size.
	uint		local_mem_size				= (block_size * local_work_size_)/(pow( out_block_size, 2) ); // i.e. one entry per 4x4 pixel patch.

	// constant buffers uploaded

	//Inputs:
	_clSetKernelArg( kernel, 0, sizeof(uint),						&frame_count,											fname);		// __private	const uint	frame_count				//0
	_clSetKernelArg( kernel, 1, sizeof(float),						&reduction,												fname);		// __private	const float	reduction,				//1		//	i.e. 2^layer		= base_cols/read_cols_;		//  NB these __private args couldbe a single __constant uint* buffer, uploaded at the start of the loop. //
	_clSetKernelArg( kernel, 2, sizeof(uint),						&lookup_table_offset,									fname);		// __private	const uint	lookup_table_offset,	//2															//  Likewise could list the order of img_ and vel_ buffers with a __constant uint* buffer				 //
	_clSetKernelArg( kernel, 3, sizeof(uint),						&out_block_size,										fname);		// __private	const uint	out_block_size,			//3

	_clSetKernelArg( kernel, 4, sizeof(uint),						&read_offset_,											fname);		// __private	const uint	read_offset_,			//4		= mipmap_params_[MiM_READ_OFFSET];
	_clSetKernelArg( kernel, 5, sizeof(uint),						&stop_offset,											fname);		// __private	const uint	stop_offset,			//5		= layer_offset + (read_rows_ -1) * mm_cols + read_cols_	;	// bottom right corner of source image layer
	_clSetKernelArg( kernel, 6, sizeof(uint),						&layer_pixels,											fname);		// __private	const uint	layer_pixels,			//6		= mipmap_params_[MiM_PIXELS];
	_clSetKernelArg( kernel, 7, sizeof(uint),						&read_cols_,											fname);		// __private	const uint	read_cols_,				//7		= mipmap_params_[MiM_READ_COLS];
	_clSetKernelArg( kernel, 8, sizeof(uint),						&read_rows_,											fname);		// __private	const uint	read_rows_,				//8		= mipmap_params_[MiM_READ_ROWS];
	_clSetKernelArg( kernel, 9, sizeof(uint),						&mm_cols,												fname);		// __private	const uint	mm_cols,				//9		= uint_params[MM_COLS];

	_clSetKernelArg( kernel, 10, sizeof(uint),						&write_offset,											fname);		// __private	const uint	mm_cols,				//9		= uint_params[MM_COLS];
	_clSetKernelArg( kernel, 11, sizeof(uint),						&dm_win_cols,											fname);		// __private	const uint	mm_cols,				//9		= uint_params[MM_COLS];
	_clSetKernelArg( kernel, 12, sizeof(uint),						&dm_data_rows,											fname);		// __private	const uint	mm_cols,				//9		= uint_params[MM_COLS];
	_clSetKernelArg( kernel, 13, sizeof(uint),						&dm_data_stop,											fname);		// __private	const uint	mm_cols,				//9		= uint_params[MM_COLS];

	_clSetKernelArg( kernel, 14, sizeof(float),						&inv_depth_step,										fname);		// __private	const uint	mm_cols,				//9		= uint_params[MM_COLS];

	_clSetKernelArg( kernel, 15, sizeof( cl_mem),					&current_frames[current_frames_idx[1]].k2k_buf_from_0,	fname);		//__constant	float16*	inv_k2k,				//6		// transforms for 4 past frames
	_clSetKernelArg( kernel, 16, sizeof( cl_mem),					&current_frames[current_frames_idx[2]].k2k_buf_from_0,	fname);		//__constant	float16*	inv_k2k,				//6		// transforms for 4 past frames
	_clSetKernelArg( kernel, 17, sizeof( cl_mem),					&current_frames[current_frames_idx[3]].k2k_buf_from_0,	fname);		//__constant	float16*	inv_k2k,				//6		// transforms for 4 past frames
	_clSetKernelArg( kernel, 18, sizeof( cl_mem),					&current_frames[current_frames_idx[4]].k2k_buf_from_0,	fname);		//__constant	float16*	inv_k2k,				//6		// transforms for 4 past frames

	_clSetKernelArg( kernel, 19, sizeof(cl_mem),					&patch_lookup_table_buf,								fname);		// __constant 	uint4*		lookup_table,			//17		// should ideally be a constant.

	_clSetKernelArg( kernel, 20, sizeof(cl_mem),					&current_frames[current_frames_idx[0]].img_buf,			fname);		// __global		float4*		img_cur,				//19		// multiple past frames. NB retain frames at powers of 2, and vary starting power plus num franes.
	_clSetKernelArg( kernel, 21, sizeof(cl_mem),					&current_frames[current_frames_idx[1]].img_buf,			fname);		// __global		float4*		img_past_0,				//20
	_clSetKernelArg( kernel, 22, sizeof(cl_mem),					&current_frames[current_frames_idx[2]].img_buf,			fname);		// __global		float4*		img_past_1,				//21
	_clSetKernelArg( kernel, 23, sizeof(cl_mem),					&current_frames[current_frames_idx[3]].img_buf,			fname);		// __global		float4*		img_past_2,				//22
	_clSetKernelArg( kernel, 24, sizeof(cl_mem),					&current_frames[current_frames_idx[4]].img_buf,			fname);		// __global		float4*		img_past_3,				//23

	_clSetKernelArg( kernel, 25, sizeof(cl_mem),					&current_frames[current_frames_idx[0]].r_vel_buf,		fname);		// __global		float4*		vel_cur,				//25	// multiple past frames.
	_clSetKernelArg( kernel, 26, sizeof(cl_mem),					&current_frames[current_frames_idx[1]].r_vel_buf,		fname);		// __global		float4*		vel_past_0,				//26	// TO DO, relative velocity not used yet. Will use it to modify depth map with timestep for past frames.
	_clSetKernelArg( kernel, 27, sizeof(cl_mem),					&current_frames[current_frames_idx[2]].r_vel_buf,		fname);		// __global		float4*		vel_past_1,				//27
	_clSetKernelArg( kernel, 28, sizeof(cl_mem),					&current_frames[current_frames_idx[3]].r_vel_buf,		fname);		// __global		float4*		vel_past_2,				//28
	_clSetKernelArg( kernel, 29, sizeof(cl_mem),					&current_frames[current_frames_idx[4]].r_vel_buf,		fname);		// __global		float4*		vel_past_3,				//29

	// //outputs
	_clSetKernelArg( kernel, 30, sizeof(cl_mem), 					&SE3_rho_map_mem,										fname);		// __global		float2*		Rho_,					//30	// { sum rho^2 ,  count of valid pixels used } Writen to dense patches.
	_clSetKernelArg( kernel, 31, sizeof(cl_float2)*local_mem_size,	NULL,													fname);		// __local		float2*		local_rho,				//31	// float2 local_rho[ local_work_size/2 ]  hence sizeof( float)*local_work_size.

	_clSetKernelArg( kernel, 32, sizeof(cl_mem), 					&depth_mem_temp,										fname);		// __global		float2*		inv_depth_incr,			//32

																																if( verbosity>local_verbosity_threshold) {
																																	cout<<"\n\nRunCL::update_depth_2()_chk1"<<
																																	"\nthreads_to_launch   = "<<threads_to_launch<<
																																	"\nlocal_work_size_    = "<<local_work_size_<<
																																	"\nlocal_mem_size      = "<<local_mem_size<<
																																	"\nread_offset_        = "<<read_offset_<<
																																	"\nstop_offset         = "<<stop_offset<<
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
																																	cout<<"\nRunCL::update_depth_2() chk_1   write_offset = "<< write_offset<<",   mm_size_bytes_C1="<< mm_size_bytes_C1<<flush;
																																	uint win_offset 	= depthmap_params[layer].DM_WIN_OFFSET;
																																	DownloadAndSaveDepthUpdate( layer, win_offset, win_offset, fname );		//  write_offset, depthmap_params[layer].DM_WIN_OFFSET
																																																			//  depth_save_offset[layer]
																																	cout<<"\n\nRunCL::update_depth_2()_finished #############################################################"<<flush;
																																}
}


void RunCL::regularize_depth(uint write_layer ){
	string		fname						= "RunCL::regularize_depth(..)";
	int			local_verbosity_threshold	= V_RUNCL_REGULARIZE_DEPTH;												if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::regularize_depth(..)_chk0"<<
																																",   write_layer = "<<write_layer<<flush; }
	cl_kernel	kernel						= regularize_depth_kernel;

	uint		lookup_table_read_offset	= patch_lookup_table_offset[ write_layer +2];		// i.e. read a 4x reduction of the current layer. Used to generate (u,v) pixel coords, and to colourize point cloud.
	uint		buf_width					= uint_params[MM_COLS];								// For reading img_grad_mem

	uint		depth_width					= depthmap_params[write_layer].DM_WIN_COLS;			//patch_depthmap_width[  write_layer ];
	uint		depth_read_offset			= depthmap_params[write_layer].DM_WIN_OFFSET;		//patch_depthmap_offset[ write_layer ];
																														// NB these are pixel offsets. The buffer has mm_size_bytes_C1. The depth patches are 4x4, so 16x reduced, but float2.
	uint		write_offset				= depthmap_params[write_layer].DM_DATA_OFFSET	+  depthmap_params[ max_mipmap_layers-1].DM_WIN_OFFSET;
	uint		prev_layer_dm_offset		= depthmap_params[write_layer+1].DM_DATA_OFFSET +  depthmap_params[ max_mipmap_layers-1].DM_WIN_OFFSET;
	uint		prev_layer_depth_width		= depthmap_params[write_layer+1].DM_WIN_COLS;
	//uint		regluarized_dm_offset		= depthmap_params[ max_mipmap_layers-1].DM_WIN_OFFSET ;
																														// depth_read_offset + depth_save_offset[ max_mipmap_layers-1];
																														// Place to store the regularized depth maps. NB these maps (pixels+margins) are packed densely in the buffer, _not_ as a mipmap.
	uint		patch_height				= patch_size;
	uint		read_cols_					= MipMap[ write_layer*8 + MiM_READ_COLS];
	uint		stop_offset					= write_offset + depth_width * read_cols_;

	//
	uint		dm_win_cols					= depthmap_params[ write_layer].DM_WIN_COLS;
	uint		dm_data_rows				= depthmap_params[ write_layer].DM_DATA_ROWS;
	uint		dm_data_stop				= write_offset + dm_win_cols * dm_data_rows;

																															if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::regularize_depth(..) chk1"<<
																																"\nlookup_table_read_offset 	= "		<<lookup_table_read_offset<<
																																"\nbuf_width                	= "		<<buf_width<<

																																"\ndepth_width              	= "		<<depth_width<<
																																"\ndepth_read_offset        	= "		<<depth_read_offset<<
																																"\nwrite_offset             	= "		<<write_offset<<

																																"\npatch_height             	= "		<<patch_height<<
																																"\nread_cols_               	= "		<<read_cols_<<
																																"\nstop_offset              	= "		<<stop_offset<<

																																"\ndm_data_stop             	= "		<<dm_data_stop<<
																																endl<<flush;
																															}
	_clSetKernelArg( kernel, 0, sizeof(int),						&lookup_table_read_offset,							fname);		// __private	const uint	lookup_table_read_offset,	//0
	_clSetKernelArg( kernel, 1, sizeof(int),						&depth_read_offset,									fname);		// __private	const uint	lookup_table_write_offset,	//1
	_clSetKernelArg( kernel, 2, sizeof(int),						&write_offset,										fname);		// __private	const uint	lookup_table_write_offset,	//2
	_clSetKernelArg( kernel, 3, sizeof(int),						&depth_width,										fname);		// __private	const uint	depth_width					//3
	_clSetKernelArg( kernel, 4, sizeof(int),						&buf_width,											fname);		// __private	uint		buf_width,					//2		mm_cols, i.e. width of the buffer holding the image pyramid
	_clSetKernelArg( kernel, 5, sizeof(int),						&patch_height,										fname);		// __private	uint		patch_height,				//3
	_clSetKernelArg( kernel, 6, sizeof(int),						&dm_data_stop,										fname);		// __private	uint		stop_offset,				//4
	_clSetKernelArg( kernel, 7, sizeof(cl_mem),						&patch_lookup_table_buf,							fname);		// __constant 	uint4*		lookup_table,				//5
	_clSetKernelArg( kernel, 8, sizeof(cl_mem),						&img_edge_mem,										fname);		// __global 	float2*		img							//6
	_clSetKernelArg( kernel, 9, sizeof(cl_mem),						&depth_mem_temp,									fname);		// __global 	float2*		img							//7
	_clSetKernelArg( kernel, 10, sizeof(int),						&prev_layer_dm_offset,								fname);		// __private	uint		stop_offset,				//4
	_clSetKernelArg( kernel, 11, sizeof(int),						&prev_layer_depth_width,							fname);		// __private	uint		stop_offset,				//4

	//_clSetKernelArg( kernel, 10, sizeof(int),						&regluarized_dm_offset,								fname);		// __private	uint		stop_offset,				//4

	size_t	threads_to_launch	= patch_num_threads[write_layer+2];
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
																																cout<<"\n\nRunCL::regularize_depth(..)_chk3"<<flush;
																																uint rho_save_offset 	= depthmap_params[ write_layer].DM_WIN_OFFSET;
																																uint depth_save_offset_ = depthmap_params[ write_layer].DM_WIN_OFFSET  + depthmap_params[ max_mipmap_layers-1].DM_WIN_OFFSET ;	//DM_WIN_OFFSET
																																// depth_save_offset[ write_layer] + depth_save_offset[ max_mipmap_layers-1];

																																DownloadAndSaveDepthUpdate( write_layer, rho_save_offset, depth_save_offset_, fname );

																																cout<<"\n write_layer = "<< write_layer
																																	<<"\n rho_save_offset   = depthmap_params[ write_layer].DM_WIN_OFFSET = "<< rho_save_offset
																																	<<"\n depth_read_offset = depthmap_params[ write_layer].DM_WIN_OFFSET = "<< depth_read_offset
																																	<<"\n"
																																	<<"\n depthmap_params[write_layer].DM_DATA_OFFSET                     = "<< depthmap_params[write_layer].DM_DATA_OFFSET
																																	<<"\n depthmap_params[ max_mipmap_layers-1].DM_WIN_OFFSET             = "<< depthmap_params[ max_mipmap_layers-1].DM_WIN_OFFSET
																																	<<"\n"
																																	<<"\n write_offset      = depthmap_params[ write_layer].DM_DATA_OFFSET  + depthmap_params[ max_mipmap_layers-1].DM_WIN_OFFSET = "<< write_offset
																																	<<"\n depth_save_offset_= depthmap_params[ write_layer].DM_WIN_OFFSET   + depthmap_params[ max_mipmap_layers-1].DM_WIN_OFFSET = "<< depth_save_offset_
																																	<<flush;

																																// stringstream ss;	ss << dataset_frame_num << "_regularize_depth";
																																// cv::Size new_Image_size = cv::Size(mm_width, mm_height);
																																// ss << "_raw_";
																																// stringstream ss_path;	ss_path << "depth_mem";
																																// DownloadAndSave( depth_mem_temp,   	ss.str(),   paths.at(ss_path.str()),   	mm_size_bytes_C1,   mm_Image_size,   CV_32FC1, 	false , fp32_params[MAX_INV_DEPTH]);
																															}
																															if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::regularize_depth(..)_chk4 Finished:#######################################################"<<flush;}
}


void RunCL::propagate_depth_next_layer(uint write_layer ){	// layer = write layer
	string 	fname						= "RunCL::propagate_depth_next_layer(..)";
	int 	local_verbosity_threshold	= V_RUNCL_PROPAGATE_DEPTH_NEXT_LAYER;												if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::propagate_depth_next_layer(..)_chk0"<<
																																",   write_layer = "<<write_layer<<flush; }
	cl_kernel		kernel				= enlarge_layer_float_kernel;
	const frame		*frame_0			= &current_frames[						current_frames_idx[0] ];
	const cl_mem 	depth_mem_			= frame_0->depth_buf;

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
																																"\ndepth_mem_                	= "		<<depth_mem_<<
																																endl<<flush;
																															}
	_clSetKernelArg( kernel, 0, sizeof(int),						&lookup_table_read_offset,							fname);		// __private	const uint	lookup_table_read_offset,	//0
	_clSetKernelArg( kernel, 1, sizeof(int),						&write_offset,										fname);		// __private	const uint	lookup_table_write_offset,	//1
	_clSetKernelArg( kernel, 2, sizeof(int),						&buf_width,											fname);		// __private	uint		buf_width,					//2		mm_cols, i.e. width of the buffer holding the image pyramid
	_clSetKernelArg( kernel, 3, sizeof(int),						&patch_height,										fname);		// __private	uint		patch_height,				//3
	_clSetKernelArg( kernel, 4, sizeof(int),						&stop_offset,										fname);		// __private	uint		stop_offset,				//4
	_clSetKernelArg( kernel, 5, sizeof(cl_mem),						&patch_lookup_table_buf,							fname);		// __constant 	float4*		lookup_table,				//5
	_clSetKernelArg( kernel, 6, sizeof(cl_mem),						&depth_mem_,										fname);		// __global 	float2*		img							//6

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
																																uint offset_depth_bytes	=0;
																																DownloadAndSave_2Channel( depth_mem_,   	ss.str(),   paths.at(ss_path.str()),   	2*mm_size_bytes_C1,   mm_Image_size,   CV_32FC2, 	false , fp32_params[MAX_INV_DEPTH], offset_depth_bytes);
																															}
																															if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::propagate_depth_next_layer(..)_chk4 Finished:#######################################################"<<flush;}
}


void RunCL::use_inferred_depthmap(uint write_layer ){
	string 	fname							= "RunCL::use_inferred_depthmap(..)";
	int 	local_verbosity_threshold		= V_RUNCL_USE_INFERRED_DEPTH;													if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::use_inferred_depthmap(..)_chk0"<<
																																",   write_layer = "<<write_layer<<flush; }
	cl_kernel		kernel					= use_inferred_depthmap_kernel;
	const frame		*frame_0				= &current_frames[						current_frames_idx[0] ];
	const cl_mem	depth_mem_				= frame_0->depth_buf;

	const uint	lookup_table_read_offset	= patch_lookup_table_offset[write_layer+2];																	//0
	const uint	read_offset					= depthmap_params[write_layer].DM_DATA_OFFSET	+  depthmap_params[ max_mipmap_layers-1].DM_WIN_OFFSET;		//1					//MipMap[write_layer*8 + MiM_READ_OFFSET];			//1
	uint		depth_in_width				= depthmap_params[write_layer].DM_WIN_COLS;																	//2					//MipMap[ (2+write_layer)*8 + MiM_READ_COLS];		//2

	uint		write_offset				= MipMap[ write_layer*8 + MiM_READ_OFFSET ];																//3
	uint		depth_width_out				= uint_params[MM_COLS];																						//4
	uint		patch_height				= patch_size;																								//4

	uint		dm_win_cols					= depthmap_params[ write_layer].DM_WIN_COLS;
	uint		dm_data_rows				= depthmap_params[ write_layer].DM_DATA_ROWS;
	uint		stop_offset					= read_offset + dm_win_cols * (dm_data_rows+2);																//5
																															cout << "\nstop_offset="<<stop_offset<<"	= read_offset ("<<read_offset<<") + (dm_win_cols "<<dm_win_cols <<" -1) * dm_data_rows("<<dm_data_rows<<");"<<flush;

	_clSetKernelArg( kernel, 0, sizeof(int),						&lookup_table_read_offset,	fname);		// __private	const uint	lookup_table_read_offset,	//0
	_clSetKernelArg( kernel, 1, sizeof(int),						&read_offset,				fname);		// __private	const uint	write_offset,				//1
	_clSetKernelArg( kernel, 2, sizeof(int),						&depth_in_width,			fname);		// __private	uint		depth_in_width,				//2

	_clSetKernelArg( kernel, 3, sizeof(int),						&write_offset,				fname);		// __private	uint		write_offset,				//3
	_clSetKernelArg( kernel, 4, sizeof(int),						&depth_width_out,			fname);		// __private	uint		depth_width_out,			//4

	_clSetKernelArg( kernel, 5, sizeof(int),						&patch_height,				fname);		// __private	uint		patch_height,				//5
	_clSetKernelArg( kernel, 6, sizeof(int),						&stop_offset,				fname);		// __private	uint		stop_offset,				//6

	_clSetKernelArg( kernel, 7, sizeof(cl_mem),						&patch_lookup_table_buf,	fname);		// __constant 	uint4*		lookup_table,				//7

	_clSetKernelArg( kernel, 8, sizeof(cl_mem),						&depth_mem_temp,			fname);		// __global		float2*		temp_depth,					//8
	_clSetKernelArg( kernel, 9, sizeof(cl_mem),						&depth_mem_,				fname);		// __global		float2*		depth_map					//9

	size_t	threads_to_launch	= patch_num_threads[write_layer+2];
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
																																cout<<"\n\nRunCL::use_inferred_depthmap(..)_chk1 Finished all loops."<<flush;
																																stringstream ss;				ss << dataset_frame_num << "_use_inferred_depthmap_";
																																cv::Size new_Image_size 		= cv::Size(mm_width, mm_height);
																																ss << "_raw_";
																																stringstream ss_path;			ss_path << "depth_mem";
																																uint offset_depth_bytes			=0;

																																cv::Size depth_mem_temp_size	= { mm_Image_size.width, mm_Image_size.height/2 };
																																DownloadAndSave_2Channel( 		depth_mem_,   	ss.str(),   paths.at(ss_path.str()),   	mm_size_bytes_C1,  depth_mem_temp_size /*mm_Image_size*/,   CV_32FC2, 	false , fp32_params[MAX_INV_DEPTH], offset_depth_bytes);

																																//DownloadAndSave_2Channel( 		  depth_mem_temp,  ss.str( ), paths.at( "depth_mem_temp"),	mm_size_bytes_C1, depth_mem_temp_size,	CV_32FC2, false , fp32_params[MAX_INV_DEPTH], offset_depth_bytes);
																																/*
																																////// save point clouds
																																cv::Size depthUpdate_size(		  depthmap_params[write_layer].DM_WIN_COLS*4 ,  depthmap_params[write_layer].DM_WIN_ROWS*4);	// cols, rows );
																																size_t	 depthUpdate_bytes		= depthmap_params[write_layer].DM_WIN_BYTES*4*4; 												//cols * rows 	* sizeof(cl_float2);

																																uint 	rho_save_offset			= depthmap_params[ write_layer].DM_WIN_OFFSET;
																																//uint depth_save_offset_		= depthmap_params[ write_layer].DM_WIN_OFFSET  + depthmap_params[ max_mipmap_layers-1].DM_WIN_OFFSET ;	//DM_WIN_OFFSET

																																size_t	offset_rho_bytes		= rho_save_offset		* sizeof(cl_float2);
																																size_t	offset_depth_bytes_		= write_offset			* sizeof(cl_float2);
																																float 	scale 					= 1.0f;

																																cout << "\nRunCL::use_inferred_depthmap(..)_chk1.5  "
																																<<   "\ndepthUpdate_bytes("		<<depthUpdate_bytes
																																<<"), \ndepthUpdate_size("		<<depthUpdate_size
																																<<"), \noffset_rho_bytes("		<<offset_rho_bytes
																																<<"), \noffset_depth_bytes_("	<<offset_depth_bytes_
																																<<"), \nwrite_layer("			<<write_layer
																																<<"), \nscale("					<<scale
																																<<"), \nwrite_offset("			<<write_offset
																																<<")"<<flush;

																																//Save_pcd_depth( 			depth_mem, 		SE3_rho_map_mem, 	paths.at( "depth_mem"), 	 depthUpdate_bytes,   depthUpdate_size, offset_rho_bytes, 	offset_depth_bytes_, 		write_layer, 		scale );

																																//DownloadAndSave_2Channel( depth_mem,   	ss.str(),			paths.at(ss_path.str()),   	2*mm_size_bytes_C1,   mm_Image_size,   CV_32FC2, 	false , fp32_params[MAX_INV_DEPTH], offset_depth_bytes);
																															//	Save_pcd_depth( 			depth_mem,   	SE3_rho_map_mem,	paths.at(ss_path.str()),   	 depthUpdate_bytes,   depthUpdate_size,   offset_rho_bytes, offset_depth_bytes_, 		write_layer, 		scale );

																																//DownloadAndSave_2Channel( depth_mem_temp, ss.str( ), 			paths.at( "depth_mem_temp"), depthUpdate_bytes,   depthUpdate_size,	CV_32FC2, 			show, max_range,			offset_depth_bytes );
																																//Save_pcd_depth( 			depth_mem_temp, SE3_rho_map_mem, 	paths.at( "depth_mem_temp"), depthUpdate_bytes,   depthUpdate_size, offset_rho_bytes, 	offset_depth_bytes, 		layer, 				scale );


																																Save_pcd_depth( 			depth_mem,   	SE3_rho_map_mem,	paths.at(ss_path.str()),   	 mm_size_bytes_C1,    depth_mem_temp_size, offset_rho_bytes, offset_depth_bytes, 		write_layer, 		scale );
																																*/
																															}
																															if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::use_inferred_depthmap(..)_chk2 Finished:#######################################################"<<flush;}
}



void RunCL::use_GT_depthmap(uint write_layer ){
	string 	fname							= "RunCL::use_GT_depthmap(..)";
	int 	local_verbosity_threshold		= V_RUNCL_PROPAGATE_DEPTH_NEXT_LAYER;											if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::use_GT_depthmap(..)_chk0"<<
																																",   write_layer = "<<write_layer<<flush; }
	cl_kernel 	kernel						= use_GT_depthmap_kernel;
	const frame		*frame_0				= &current_frames[						current_frames_idx[0] ];
	const cl_mem	depth_mem_				= frame_0->depth_buf;

	const uint	lookup_table_read_offset	=	patch_lookup_table_offset[write_layer];				//0
	const uint	write_offset				=	MipMap[write_layer*8 + MiM_READ_OFFSET];			//1

	uint		depth_width_out				=	uint_params[MM_COLS];								//3
	uint		patch_height				=	patch_size;											//4

	uint		read_cols_					= MipMap[ write_layer*8 + MiM_READ_COLS];
	uint		read_rows_					= MipMap[ write_layer*8 + MiM_READ_ROWS];
	uint		stop_offset					= write_offset + (read_rows_ -1) * depth_width_out + read_cols_;			//6


	_clSetKernelArg( kernel, 0, sizeof(int),						&lookup_table_read_offset,	fname);		// __private	const uint	lookup_table_read_offset,	//0
	_clSetKernelArg( kernel, 1, sizeof(int),						&write_offset,				fname);		// __private	const uint	write_offset,				//1

	_clSetKernelArg( kernel, 2, sizeof(int),						&depth_width_out,			fname);		// __private	uint		depth_width_out,			//2
	_clSetKernelArg( kernel, 3, sizeof(int),						&patch_height,				fname);		// __private	uint		patch_height,				//3
	_clSetKernelArg( kernel, 4, sizeof(int),						&stop_offset,				fname);		// __private	uint		stop_offset,				//4

	_clSetKernelArg( kernel, 5, sizeof(cl_mem),						&patch_lookup_table_buf,	fname);		// __constant 	uint4*		lookup_table,				//5

	_clSetKernelArg( kernel, 6, sizeof(cl_mem),						&depth_mem_GT,				fname);		// __global		float*		temp_depth,					//6
	_clSetKernelArg( kernel, 7, sizeof(cl_mem),						&depth_mem_,					fname);		// __global		float2*		depth_map					//7

	size_t	threads_to_launch	= patch_num_threads[write_layer];
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


}
