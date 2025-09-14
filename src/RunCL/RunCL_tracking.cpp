#include "RunCL.hpp"

void RunCL::precom_param_maps( float SE3_k2k[6*16]){ //  Compute maps of pixel motion for each SE3 DoF, and camera params // Derived from RunCL::mipmap
	string fname = "RunCL::precom_param_maps( ..)";
	int local_verbosity_threshold = V_RUNCL_PRECOM_PARAM_MAPS;
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::precom_param_maps( float SE3_k2k[6*16])_chk_0 "<<flush;}
	cv::Mat depth		= cv::Mat::ones ( mm_height, mm_width, CV_32FC1);																	// NB must recompute translation maps at run time. NB parallax motion is proportional to inv depth.
	float mid_depth 	= ( fp32_params[MAX_INV_DEPTH] + fp32_params[MIN_INV_DEPTH])/2.0;                                                   // TODO fix : depthmap not used as a kernel arg. NB want to match scale of depth range, but ? parallax may vary.
	depth 				*= mid_depth;

	_clEnqueueWriteBuffer( uload_queue, SE3_k2kbuf,	CL_FALSE, 0, 6*16*sizeof( float), SE3_k2k,		fname);
	_clEnqueueWriteBuffer( uload_queue, depth_mem_temp,	CL_FALSE, 0, mm_size_bytes_C1,	 depth.data,	fname);

	//      __private	 uint layer, set in mipmap_call_kernel( ..) below                                                                      __private	 uint	    layer,		//0
    _clSetKernelArg( comp_param_maps_kernel, 1, sizeof( cl_mem),	&mipmap_buf, fname);														//__constant uint*	mipmap_params,	//1
	_clSetKernelArg( comp_param_maps_kernel, 2, sizeof( cl_mem), 	&uint_param_buf, fname);													//__global 	uint*	uint_params		//2
	_clSetKernelArg( comp_param_maps_kernel, 3, sizeof( cl_mem), 	&SE3_k2kbuf, fname);														//__global 	float* 	k2k,			//3
	_clSetKernelArg( comp_param_maps_kernel, 4, sizeof( cl_mem), 	&SE3_map_mem, fname);														//__global 	float* 	SE3_map,		//4
																																			if( verbosity>local_verbosity_threshold) {cout<<"\nRunCL::precom_param_maps( float SO3_k2k[6*16])_chk_1 "<<flush;}
	// SE3_map_mem, k_map_mem, dist_map_mem;
	mipmap_call_kernel( comp_param_maps_kernel, m_queue );
																																			if( verbosity>local_verbosity_threshold) {
																																													cout<<"\n\nRunCL::precom_param_maps( float SO3_k2k[6*16])_output "<<flush;
																																													for ( int i=0; i<1; i++) { // TODO x & y for all 6 SE3 DoF
																																														stringstream ss;	ss << dataset_frame_num << "_SE3_map";
																																														DownloadAndSave_2Channel_volume( SE3_map_mem, ss.str( ), paths.at( "SE3_map_mem"), mm_size_bytes_C1*2, mm_Image_size, CV_32FC2, false, -1.0, 6 /*SE3, 6DoF */);
																																													}
																																													cout<<"\nRunCL::precom_param_maps( float SE3_k2k[6*16])_chk.. Finished "<<flush;
																																			}
}

void RunCL::update_tracking_depthmap( cl_mem depthmap_){
	string fname = "RunCL::update_tracking_depthmap( cl_mem depthmap_)";
	int local_verbosity_threshold = V_RUNCL_UPDATE_TRACKING_DEPTHMAP;

	_clEnqueueCopyBuffer( m_queue,  depthmap_, keyframe_depth_mem, 0, 0, mm_size_bytes_C1, fname);
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::update_tracking_depthmap( ..)_finished ."<<flush;}
}

void RunCL::update_current_frame_depth_mem( cl_mem depthmap_){
	string fname = "update_current_frame_tracking_depthmap( cl_mem depthmap_ )";
	int local_verbosity_threshold = V_RUNCL_CURRENT_FRAME_DEPTH_MAP;

	_clEnqueueCopyBuffer( m_queue,  depthmap_, depth_mem, 0, 0, mm_size_bytes_C1, fname);
																																			if(verbosity>local_verbosity_threshold){
																																				stringstream 	ss;
																																				ss << "__update_tracking_depthmap" << (keyFrameCount*1000 + costvol_frame_num);
																																				float max_range_ = 0.0f;
																																				DownloadAndSave( depth_mem,   	ss.str(),   paths.at("depth_mem"),   	mm_size_bytes_C1,   mm_Image_size,   CV_32FC1, 	false , max_range_ );	cout << "\nDownloadAndSave (.. depth_mem_GT ..)\n"<<flush;
																																			}
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::update_current_frame_tracking_depthmap( ..)_finished ."<<flush;}
}

/*
// void RunCL::initialize_tracking_depthmap( float initial_depth){
// 	string fname = "RunCL::initialize_tracking_depthmap( float initial_depth)";
// 	int local_verbosity_threshold = V_RUNCL_INITIALIZE_TRACKING_DEPTHMAP;
//
// 	_clEnqueueFillBuffer( uload_queue, depth_mem, 	&initial_depth, sizeof( float), 0, mm_size_bytes_C1, 	fname);
// 																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::initialize_tracking_depthmap( ..)_finished ."<<flush;}
// }
*/

void RunCL::update_k2k_buf( float k2k_3_16_[16],		float pose_arry[16] ) {
	string fname = "RunCL::update_k2k_buf( ..)";
	int local_verbosity_threshold = V_RUNCL_UPDATE_K2K_BUF;
	_clEnqueueWriteBuffer( uload_queue, 	k2kbuf,		CL_FALSE, 0, 16*sizeof( float), k2k_3_16_,  	fname);
	_clEnqueueWriteBuffer( uload_queue, 	pose_buf,	CL_FALSE, 0, 16*sizeof( float), pose_arry, 		fname);

}

void RunCL::rho_sq(uint out_block_size, uint iter, uint layer, float delta_theta, float delta   ){	// To be launched with 1 thread per col for 32x32 patches, and an integer multiple of 32 threads.
																	// Needs 16 elements of local mem per 32x32 patch, to pass data between threads in recursive square reduction.
																	// Needs 32 elem array of private mem per thread.
																	// Writes answer to SE3_rho_map_mem, BUT as float2
	string fname					= "RunCL::rho_sq( ..)";
	int local_verbosity_threshold	= V_RUNCL_RHO_SQ;
	cl_kernel	kernel 				= rho_sq_kernel;
	//const int se3_dof				= 6;
	cl_float2	delta_SE3			= {{delta_theta, delta}};
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::rho_sq( ..)_chk0 .##################################################################"<<flush;
																																				PRINT_FLOAT_16( fp32_k2keyframe, cpu array);
																																				cout<<endl<<endl
																																					<<"   dataset_frame_num="	<<dataset_frame_num
																																					<<",  out_block_size="		<<out_block_size
																																					<<",  iter="				<<iter
																																					<<",  layer="				<<layer
																																					<< flush;
																																				float pose_ary[16];
																																				ReadOutput( (uchar*)pose_ary, pose_buf, sizeof(float)*16, 0);		// ReadOutput(uchar* outmat, cl_mem buf_mem, size_t data_size, size_t offset/*=0*/)
																																				PRINT_FLOAT_16(pose_ary, gpu buf);

																																				float k2kbuf_ary[16];
																																				ReadOutput( (uchar*)k2kbuf_ary, k2kbuf, sizeof(float)*16, 0);		// ReadOutput(uchar* outmat, cl_mem buf_mem, size_t data_size, size_t offset/*=0*/)
																																				PRINT_FLOAT_16(k2kbuf_ary, gpu buf);
																																			}
	const uint			patch_size					= 32;																					// TODO set global patch size from device parameters // generally: device_work_size_multiple = patch_size * integer, eg 32, 64, 128
																									if (fmod(device_work_size_multiple, patch_size)!=0)   { cout <<"\nRunCL::rho_sq( ..)  Error: fmod(device_work_size_multiple, patch_size) != 0 \n"<<flush;exit_(0);}

	for (uint i=0; i<5 ;i++){
		cout<<"\n\n## current_frames[ current_frames_idx["<<i<<"] ].frame_num = "<< current_frames[ current_frames_idx[i] ].frame_num << flush;
		PRINT_FLOAT_16( current_frames[ current_frames_idx[i] ].pose,		);
		PRINT_MATX44F(	current_frames[ current_frames_idx[i] ].pose_gt,	);
		PRINT_FLOAT_16( current_frames[ current_frames_idx[i] ].invk2k_gt,	);
	}

	const float zero  = 0;
	//_clEnqueueWriteBuffer( uload_queue, k2kbuf, CL_FALSE, 0, /*local_num_samples**/16*sizeof( float), identity_flt16 /*k2k_3_16_[start_sample_idx]*/, fname); // TODO  temporary debug, sets k2k to identity.

	_clEnqueueFillBuffer( uload_queue, SE3_rho_map_mem, 	&zero, sizeof( float), 0, 			  2*mm_size_bytes_C1, 	fname);				//_clEnqueueWriteBuffer( uload_queue, k2kbuf, CL_FALSE, 0, local_num_samples*16*sizeof( float), k2k_3_16_[start_sample_idx], fname);
	_clEnqueueFillBuffer( uload_queue, SE3_weight_map_mem, 	&zero, sizeof( float), 0, num_SE3_DoF*2*mm_size_bytes_C1, 	fname);
	_clEnqueueFillBuffer( uload_queue, SE3_incr_map_mem, 	&zero, sizeof( float), 0, num_SE3_DoF*2*mm_size_bytes_C1, 	fname);
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::rho_sq( ..)_chk_1 "<<flush;}
	size_t kernel_workgroup_size;
	cl_int k_wg_info =  clGetKernelWorkGroupInfo(
							kernel,								//cl_kernel kernel,
							deviceId,							//cl_device_id device,
							CL_KERNEL_WORK_GROUP_SIZE,			//cl_kernel_work_group_info param_name,
							sizeof(kernel_workgroup_size),		//size_t param_value_size,
							&kernel_workgroup_size,				//void* param_value,
							NULL								//size_t* param_value_size_ret
						);
	size_t  device_max_workitem_sizes[3];
	cl_int device_info_1 = clGetDeviceInfo(
							deviceId,							//cl_device_id device,
							CL_DEVICE_MAX_WORK_ITEM_SIZES,		//cl_device_info param_name,
							sizeof(device_max_workitem_sizes),	//size_t param_value_size,
							device_max_workitem_sizes,			//void* param_value,
							NULL								//size_t* param_value_size_ret
	);
	cl_uint device_max_compute_units;
	cl_int device_info_2 = clGetDeviceInfo(
							deviceId,							//cl_device_id device,
							CL_DEVICE_MAX_COMPUTE_UNITS,		//cl_device_info param_name,
							sizeof(device_max_compute_units),	//size_t param_value_size,
							&device_max_compute_units,			//void* param_value,
							NULL								//size_t* param_value_size_ret
	);
	size_t	max_workgroup_size	= min(kernel_workgroup_size, device_max_workitem_sizes[0] );
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::rho_sq( ..)_chk_2 "<<flush;
																																				// cout << "\n"
																																				// 	<<",  device_max_compute_units="				<<device_max_compute_units
																																				// 	<<",  max_workgroup_size="						<<max_workgroup_size
																																				// 	<<",  kernel_workgroup_size="					<<kernel_workgroup_size
																																				// 	<<",  device_max_workitem_sizes[0,1,2]={"		<<device_max_workitem_sizes[0]<<",  "
																																				// 													<<device_max_workitem_sizes[1]<<",  "
																																				// 													<<device_max_workitem_sizes[2]<<"}"
																																				// 	<< flush;
																																			}
	//uint				reduction 					= layer;
	uint				read_rows					= MipMap[layer * 8 + MiM_READ_ROWS] ;
	uint				read_cols					= MipMap[layer * 8 + MiM_READ_COLS] ;
	uint				rows_blocks					= ceil( (float)  read_rows / patch_size );
	uint				cols_blocks					= ceil( (float)  read_cols / patch_size );
	uint				cols_per_row				= cols_blocks  * patch_size;
	uint				patches_required			= cols_blocks  * rows_blocks;

	uint				patches_per_compute_uint	= ceil( (float)patches_required / device_max_compute_units ) ;
	uint 				blocks_per_k_wg_size		= max_workgroup_size			/ device_work_size_multiple;
						patches_per_compute_uint	= min( patches_per_compute_uint,  blocks_per_k_wg_size );

	uint				blocks_required				= ceil( (float)patches_required / patches_per_compute_uint );
	size_t				local_work_size_[1] 		= { patches_per_compute_uint	* patch_size };
	size_t				threads_to_launch 			= blocks_required 				* local_work_size_[0];									// TODO precompute an array for this function. ? where to store
																																			// ? Have a subclass and object for each kernel ?
																																			if( verbosity>local_verbosity_threshold-3) {cout<<"\n\nRunCL::rho_sq( ..)_chk_3 "<<flush;
																																				cout <<"\n"
																																					//<<",  reduction="<<reduction
																																					//<<",  num_threads[reduction]="					<<num_threads[reduction]
																																					//<<",  num_threads[reduction] / patch_size="		<<num_threads[reduction] / patch_size
																																					<<",  patches_required="						<<patches_required
																																					<<",  patches_per_compute_uint="				<<patches_per_compute_uint
																																					<<",  blocks_required="							<<blocks_required
																																					<<",  threads_to_launch="						<<threads_to_launch
																																					<<",  local_work_size="							<<local_work_size
																																					<<"},  local_work_size_[1]="					<<local_work_size_[0]
																																					<< endl<<flush;
																																				for (uint reduction = 0; reduction < 8  ; reduction ++){
																																					cout << "\n reduction = "						<< reduction
																																						<<"  num_threads[reduction] = "				<< num_threads[reduction]
																																						<< endl<<flush;
																																				}
																																				cout<<"\ntracking_num_samples*2*mm_size_bytes_C4="	<<tracking_num_samples*2*mm_size_bytes_C4
																																					<<"     24 * mm_size_bytes_C1="					<<24 * mm_size_bytes_C1
																																					<< endl<<flush;
																																			}
	//input integers
	_clSetKernelArg( kernel, 0, sizeof( uint),									&layer,	 												fname);		//__private		uint 		layer,					//0
	_clSetKernelArg( kernel, 1, sizeof( uint),									&cols_per_row,											fname);		//__private		uint 		cols_per_row,			//1
	_clSetKernelArg( kernel, 2, sizeof( uint),									&out_block_size,										fname);		//__private		uint 		out_block_size,			//2
	_clSetKernelArg( kernel, 3, sizeof( cl_float2),								&delta_SE3,												fname);		//__private		float2		delta_SE3,				//3

	_clSetKernelArg( kernel, 4, sizeof( cl_mem), 								&mipmap_buf,											fname);		//__constant	uint8*		mipmap_params,			//4
	_clSetKernelArg( kernel, 5, sizeof( cl_mem), 								&uint_param_buf,										fname);		//__constant	uint*		uint_params,			//5
	_clSetKernelArg( kernel, 6, sizeof( cl_mem), 								&fp32_param_buf,										fname);		//__constant	float*		fp32_params,			//6
	_clSetKernelArg( kernel, 7, sizeof( cl_mem), 								&k2kbuf,												fname);		//__constant	float16*	inv_k2k,				//7		// transforms for 4 past frames

	_clSetKernelArg( kernel, 8, sizeof( cl_mem),								&current_frames[current_frames_idx[0]].img_buf,			fname);		//__global		float4*		img_cur,				//8		// multiple past frames. NB retain frames at powers of 2, and vary starting power plus num franes.
	_clSetKernelArg( kernel, 9, sizeof( cl_mem), 								&current_frames[current_frames_idx[1]].img_buf,			fname);		//__global		float4*		img_past_0,				//9
	_clSetKernelArg( kernel,10, sizeof( cl_mem), 								&current_frames[current_frames_idx[2]].img_buf,			fname);		//__global		float4*		img_past_1,				//10
	_clSetKernelArg( kernel,11, sizeof( cl_mem), 								&current_frames[current_frames_idx[3]].img_buf,			fname);		//__global		float4*		img_past_2,				//11
	_clSetKernelArg( kernel,12, sizeof( cl_mem), 								&current_frames[current_frames_idx[4]].img_buf,			fname);		//__global		float4*		img_past_3,				//12
																																																				// NB GT_depth loaded to depth_mem by void RunCL::loadFrameData( ..)
	_clSetKernelArg( kernel,13, sizeof( cl_mem), 								&depth_mem,												fname);		//__global		float* 		depth_map,				//13	// current frame depth, now stored as inv_depth
	_clSetKernelArg( kernel,14, sizeof( cl_mem), 								&g1mem,													fname);		//__global		float8* 	g1p,					//14	// current frame g1mem
	_clSetKernelArg( kernel,15, sizeof( cl_mem), 								&SE3_grad_map_mem,										fname);		//__global 		float8*		SE3_grad_map_cur_frame,	//15

	_clSetKernelArg( kernel,16, sizeof( cl_mem), 								&current_frames[current_frames_idx[0]].r_vel_buf,		fname);		//__global		float4*		img_cur,				//16	// multiple past frames.
	_clSetKernelArg( kernel,17, sizeof( cl_mem), 								&current_frames[current_frames_idx[1]].r_vel_buf,		fname);		//__global		float4*		img_past_0,				//17
	_clSetKernelArg( kernel,18, sizeof( cl_mem), 								&current_frames[current_frames_idx[2]].r_vel_buf,		fname);		//__global		float4*		img_past_1,				//18
	_clSetKernelArg( kernel,19, sizeof( cl_mem), 								&current_frames[current_frames_idx[3]].r_vel_buf,		fname);		//__global		float4*		img_past_2,				//19
	_clSetKernelArg( kernel,20, sizeof( cl_mem), 								&current_frames[current_frames_idx[4]].r_vel_buf,		fname);		//__global		float4*		img_past_3,				//20
	//output
	_clSetKernelArg( kernel,21, sizeof( cl_mem), 								&SE3_rho_map_mem, 										fname);		//__global		float2* 	Rho_,					//21	// { sum rho^2 ,  count of valid pixels used } Writen to dense patches.
	_clSetKernelArg( kernel,22, sizeof( cl_float2)*local_work_size,				NULL, 													fname);		//__local		float2*		local_rho				//22	// float2 local_rho[ local_work_size/2 ]  hence sizeof( float)*local_work_size.

	_clSetKernelArg( kernel,23, sizeof( cl_mem), 								&SE3_incr_map_mem,										fname);		//__global 		float4*		SE3_incr_map_,			//23
	_clSetKernelArg( kernel,24, sizeof( cl_float2)*local_work_size*num_SE3_DoF,	NULL,													fname);		//__local 		float4*		local_SE3_incr			//24
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::rho_sq( ..)_chk_4 .  "<<flush;}

	cl_command_queue	queue_to_call		= m_queue;
	cl_int				res, status;
	cl_event			ev;
																									auto step_0 = high_resolution_clock::now();
	res 	= clEnqueueNDRangeKernel(queue_to_call, kernel, 1, 0, &threads_to_launch, local_work_size_, 0, NULL, &ev);
																									if (res    != CL_SUCCESS)	{ cout << "\nres = " << checkerror(res) <<"\n"<<flush; exit_(res);}
	status 	= clFlush(queue_to_call);																if (status != CL_SUCCESS)	{ cout << "\nRunCL::rho_sq( ..) call_kernel( cl_kernel "<<kernel<<",  clFlush(queue_to_call) status  = "		<<status<<" "<< checkerror(status) <<"\n"<<flush; exit_(status);}
																									auto step_1 = high_resolution_clock::now();
	status 	= clWaitForEvents (1, &ev);																if (status != CL_SUCCESS)	{ cout << "\nRunCL::rho_sq( ..) call_kernel( cl_kernel "<<kernel<<") final,  clWaitForEventsh(1, &ev) ="		<<status<<" "<<checkerror(status)  <<"\n"<<flush; exit_(status);}
																									auto step_2 = high_resolution_clock::now();
	clReleaseEvent(ev);
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::rho_sq( ..)_chk_5 . "<<\
																																				"Execution time = "<<  duration_cast<microseconds>(step_1 - step_0).count() \
																																				<<" , "<<duration_cast<microseconds>(step_2 - step_1).count() <<flush;
																																			}
																																			if( verbosity>local_verbosity_threshold -3) {cout<<"\n\nRunCL::rho_sq( ..)_chk_6 ."<<flush;

																																				stringstream ss;
																																				ss << "_ds-framenum"<<dataset_frame_num<<"_img_layer"<<layer<<"_iter"<<iter<<"_out_bock_size"<<out_block_size<<"_rho_sq()";
																																				stringstream ss_path;
																																				bool show				= false;
																																				float max_range			= -1;
																																				uint vol_layers			= 1;
																																				bool exception_tiff 	= false;
																																				bool display			= false;	//cout << "\nRunCL::rho_sq( ..)_chk_7   display="<< display<< endl << flush;
																																				bool old_tiff			= tiff;
																																				tiff					= true;
																																				DownloadAndSave_2Channel_volume(  SE3_rho_map_mem,		ss.str( ), paths.at( "SE3_rho_map_mem"),	2*mm_size_bytes_C1,   mm_Image_size,	CV_32FC2, show, max_range,	1);

																																				float max_range_ = 0;
																																				DownloadAndSave( depth_mem,   	ss.str(),   paths.at("depth_mem"),   	mm_size_bytes_C1,   mm_Image_size,   CV_32FC1, 	false , max_range_ );	cout << "\nDownloadAndSave (.. depth_mem_GT ..)\n"<<flush;

																																				for(int i=0; i<5; i++){
																																					ss<<"_"<<i;
																																					DownloadAndSave_3Channel( current_frames[current_frames_idx[i]].img_buf,  ss.str(),   paths.at("imgmem"),   	mm_size_bytes_C4,   mm_Image_size,   CV_32FC4, 	false , max_range_);
																																				}

																																				//DownloadAndSave_2Channel_volume(  SE3_weight_map_mem,	ss.str( ), paths.at( "SE3_weight_map_mem"),	2*mm_size_bytes_C1,   mm_Image_size,	CV_32FC2, show, max_range,	vol_layers);
																																				//DownloadAndSave_2Channel_volume(  SE3_incr_map_mem,		ss.str( ), paths.at( "SE3_incr_map_mem"),	2*mm_size_bytes_C1,   mm_Image_size,	CV_32FC2, show, max_range,	vol_layers);
																																				tiff = old_tiff;

																																				cout<<"\n\nRunCL::rho_sq( ..) finished ########################################################################"<<endl<< flush;
																																			}
}

void RunCL::reduce_patch_Rho ( uint out_block_size, uint iter, uint layer )									// NB good for images upto 640x480 layer zero, above that need a patch kernel approach to ensure each DoF fits in 1 workgroup. see device_work_size_multiple
{
	string fname = "RunCL::reduce_patch_Rho( ..)";
	int local_verbosity_threshold = V_RUNCL_REDUCE_PATCH_RHO;
																																		if( verbosity>local_verbosity_threshold) {cout<<"\nRunCL::reduce_patch_Rho( ..)_chk_1 _____________________"<<flush;}
																																			// if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::update_SE3( ..)_chk0 .##################################################################"<<flush;
																																			// 	float pose_update_ary[6];
																																			// 	ReadOutput( (uchar*)pose_update_ary, pose_update_buf, sizeof(float)*6, 0);	//ReadOutput(uchar* outmat, cl_mem buf_mem, size_t data_size, size_t offset/*=0*/)
																																			// 	cout<<"\n pose_update_ary = {"; for (int i=0; i<6; i++){ cout<<pose_update_ary[i]<<", "; }cout<<"}"<<flush;
																																			// }
	cl_kernel		kernel 				= reduce_patch_Rho_kernel;																			//NB call just one workgroup to sum the whole image maps from the patch kernel.
	const uint		patch_size 			= 32;																								//TODO set global patch size from device parameters // generally:  device_work_size_multiple = patch_size * integer,   eg 32, 64, 128
	const uint		SE3_DoF				= 6;
	uint			read_rows			= MipMap[layer * 8 + MiM_READ_ROWS] ;
	uint			read_cols			= MipMap[layer * 8 + MiM_READ_COLS] ;																// NB the largest (layer 0) read_cols, is the unreduced size of the input image. (here 640x480)
	uint			rows_blocks			= read_rows/patch_size ; //ceil( (float) read_rows/patch_size );								//0 // num rows in the fully reduced map  640/32=20 => 32 cols_blocs.  32*6=192 which would fit IFF groupsize >=256.
	uint			cols_blocks			= ceil( (float) read_cols/patch_size );															//1 // num cols in the fully reduced map. 1920x1080 1920/32=60 => 64 cols_blocs
																											// threads_per_DoF must be the first 2^n > cols per SE3 patch.
	uint			threads_per_DoF		= powf(2,ceil( log2((float)cols_blocks) )); 						// 10 layer 1 =>  pown(2,ciel(log2(10.0f) ))=16; 6*16=96.      // * rows_blocks  ;//	8x10=80 layer1 => 96 threads to launch?		// num pixels in fully reduced map. Req per SE3 DoF.
	uint 			DoF_per_workgroup	= device_work_size_multiple / threads_per_DoF;
	size_t			threads_required	= (device_work_size_multiple * SE3_DoF) / DoF_per_workgroup;		// NB device_work_size_multiple is usually a poer of 2, DoF_per_workgroup will also be a power of 2.
	uint 			workgroups_required	= ceil( (float)threads_required / device_work_size_multiple );
	size_t			threads_to_launch	= workgroups_required  *  device_work_size_multiple;

	uint 			row_offset			= rows_blocks + 4;																				//2
	uint			thread_offset		= threads_per_DoF;																				//3     2^n  > pixels in fully reduced patch
	uint			mm_cols				= uint_params[MM_COLS];																			//4
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::reduce_patch_Rho( ..)_chk_2 . "<<flush;
																																			cout<<"\nRunCL::reduce_patch_Rho(..)"\
																																				<<",  layer="						<<layer\
																																				<<",  thread_offset="				<<thread_offset\
																																				<<",  cols_blocks="					<<cols_blocks\
																																				<<",  rows_blocks="					<<rows_blocks\
																																				<<",  threads_required="			<<threads_required\
																																				<<",  workgroups_required="			<<workgroups_required\
																																				<<",  threads_to_launch="			<<threads_to_launch\
																																				<<",  threads_per_DoF="				<<threads_per_DoF\
																																				<<",  DoF_per_workgroup="			<<DoF_per_workgroup\
																																				<<",  device_work_size_multiple="	<<device_work_size_multiple\
																																				<<flush;
																																		}
	_clSetKernelArg( kernel,  0, sizeof( uint),								&cols_blocks,				fname);							//__private	uint		cols,					//0
	_clSetKernelArg( kernel,  1, sizeof( uint),								&rows_blocks,				fname);							//__private	uint		rows,					//1
	_clSetKernelArg( kernel,  2, sizeof( uint),								&row_offset,				fname);							//__private	uint		row_offset,				//2
	_clSetKernelArg( kernel,  3, sizeof( uint),								&thread_offset,				fname);							//__private	uint		thread_offset,			//3
	_clSetKernelArg( kernel,  4, sizeof( uint),								&mm_cols,					fname);							//__private	uint		mm_cols,				//4
	//global
	_clSetKernelArg( kernel,  5, sizeof( cl_mem),							&SE3_rho_map_mem,			fname);							//__global	float2*		Rho_,					//5		// { sum rho^2 ,  count of valid pixels used } Writen to dense patches.
	_clSetKernelArg( kernel,  6, sizeof( cl_mem),							&SE3_incr_map_mem,			fname);							//__global	float2*		SE3_incr_map_,			//6
	//local
	_clSetKernelArg( kernel,  7, ( local_work_size/2 )*sizeof( cl_float2),	NULL, 						fname);							//__local	float2*		local_Rho_,				//7		// used for sum-reduce. Need to be [groupsize/2], set in host fn.
	_clSetKernelArg( kernel,  8, ( local_work_size/2 )*sizeof( cl_float2),	NULL, 						fname);							//__local	float2*		local_SE3_incr_map_,	//8

	cl_command_queue 	queue_to_call		= m_queue;
	cl_int				res, status;
	cl_event			ev;
																									auto step_0 = high_resolution_clock::now();
	res		= clEnqueueNDRangeKernel(queue_to_call, kernel, 1, 0, &threads_to_launch, &device_work_size_multiple, 0, NULL, &ev); 	// run mipmap_float4_kernel, NB wait for own previous iteration.
																									if (res    != CL_SUCCESS)	{ cout << "\nres = " << checkerror(res) <<"\n"<<flush; exit_(res);}
	status 	= clFlush(queue_to_call);																if (status != CL_SUCCESS)	{ cout << "\nRunCL::reduce_patch_Rho( ..) call_kernel( cl_kernel "<<kernel<<",  clFlush(queue_to_call) status  = "<<status<<" "<< checkerror(status) <<"\n"<<flush; exit_(status);}
																									auto step_1 = high_resolution_clock::now();
	status 	= clWaitForEvents(1, &ev);																if (status != CL_SUCCESS)	{ cout << "\nRunCL::reduce_patch_Rho( ..) call_kernel( cl_kernel "<<kernel<<") final,  clWaitForEventsh(1, &ev) ="<<status<<" "<<checkerror(status)  <<"\n"<<flush; exit_(status);}
																									auto step_2 = high_resolution_clock::now();
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::reduce_patch_Rho( ..)_chk_2 . "<<\
																																				"Execution time = "<<  duration_cast<microseconds>(step_1 - step_0).count() \
																																				<<" , "<<duration_cast<microseconds>(step_2 - step_1).count() <<flush;

																																				float old_result_arry[6];
																																				ReadOutput( (uchar*)old_result_arry, old_results_buf, sizeof(float), 0);	//ReadOutput(uchar* outmat, cl_mem buf_mem, size_t data_size, size_t offset/*=0*/)
																																				cout<<"\n old_result_arry = " <<old_result_arry[0]<<flush;

																																				stringstream ss;
																																				ss << "_ds-framenum"<<dataset_frame_num<<"_img_layer"<<layer<<"_iter"<<iter<<"_out_bock_size"<<out_block_size<<"_rho_sq()";
																																				stringstream  ss_path;
																																				bool show				= false;
																																				float max_range			= -1;
																																				uint vol_layers			= 1;
																																				bool exception_tiff 	= false;
																																				bool display			= false;	//cout << "\nRunCL::rho_sq( ..)_chk_7   display="<< display<< endl << flush;
																																				bool old_tiff			= tiff;
																																				tiff					= true;
																																				DownloadAndSave_2Channel_volume(  SE3_rho_map_mem,		ss.str( ), paths.at( "SE3_rho_map_mem"),	2*mm_size_bytes_C1,   mm_Image_size,	CV_32FC2, show, max_range,	1);
																																				DownloadAndSave_2Channel_volume(  SE3_weight_map_mem,	ss.str( ), paths.at( "SE3_weight_map_mem"),	2*mm_size_bytes_C1,   mm_Image_size,	CV_32FC2, show, max_range,	vol_layers);
																																				DownloadAndSave_2Channel_volume(  SE3_incr_map_mem,		ss.str( ), paths.at( "SE3_incr_map_mem"),	2*mm_size_bytes_C1,   mm_Image_size,	CV_32FC2, show, max_range,	vol_layers);
																																				tiff = old_tiff;

																																				cout<<"\nRunCL::reduce_patch_Rho( ..)_finished _____________________"<<flush;
																																			}
}


void RunCL::update_k2k_cpu( uint layer, float delta_theta, float delta, Matx44f GT_pose ){
	string		fname						= "RunCL::update_k2k_cpu(..)";
	int			local_verbosity_threshold	= V_RUNCL_UPDATE_K2K;
																																	if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::update_k2k_cpu( ..)_chk_0 . ################################"<< flush;}
																																					cout << "\nlayer = "	<< layer 	<<endl<<flush;
/*
	uint		J_offset	=	layer	* 48;																										cout << "\nJ_offset = "	<< J_offset	<<endl<<flush;

//	vector<Matx16f>	J_vec	=			ReadOutput_16f_vec(					SE3_hessian_pinv_map_mem,		4*J_offset		* sizeof(float) );		PRINT_MATX16F( J_vec[0],);	PRINT_MATX16F( J_vec[1],);	PRINT_MATX16F( J_vec[2],)	 PRINT_MATX16F( J_vec[3],);
//	Matx16f		J			=			J_vec[0];																									PRINT_MATX16F( J,		);									// TODO this is a cl_float4  1x6 matrix

	//Matx66f		H_pinv	=			ReadOutput_66f(						SE3_hessian_pinv_map_mem,		(J_offset + 6)	* sizeof(float)	);		PRINT_MATX66F( H_pinv,	);									// TODO this is a cl_float4  6x6 matrix

//	vector<Matx66f>	H_vec	=			ReadOutput_66f_vec(					SE3_hessian_pinv_map_mem,		4*(J_offset + 6)* sizeof(float)	);		PRINT_MATX66F( H_vec[0],);	PRINT_MATX66F( H_vec[1],);	PRINT_MATX66F( H_vec[2],);	PRINT_MATX66F( H_vec[3],);
//	Matx66f		H			=			H_vec[0];																									PRINT_MATX66F( H,	);										// TODO this is a cl_float4  6x6 matrix
*/
	cl_float2	Rho 		=	{{0}};	ReadOutput( 		(uchar*)&Rho,				SE3_rho_map_mem, 	sizeof(cl_float2),		32*sizeof(cl_float2)	);		cout<<"\nRho	= "<< Rho.x <<", "<< Rho.y 	<<endl<<flush;
	float		rho			=	sqrtf(Rho.x) / Rho.y;																												cout<<"\nrho	= "<< rho 					<<endl<<flush;

	Matx16f		J			=			current_frames[ current_frames_idx[0] ].Jacobian[layer];													PRINT_MATX16F( J, );
	Matx66f		H			=			current_frames[ current_frames_idx[0] ].invHessian[layer];													PRINT_MATX66F( H, );	PRINT_MATX66F( H.inv(), );

	float		SE3_incr_arry[6*2];		ReadOutput(			(uchar*)SE3_incr_arry,		SE3_incr_map_mem,	6*sizeof(cl_float2),	32*sizeof(cl_float2)	);
																																						cout<<"\nSE3_incr_arry[]= (";
																																						for(int i=0; i<6*2; i++) cout << ", "<< SE3_incr_arry[i];
																																						cout<<" ) "<<endl<<flush;
	Matx16f		SE3_incr;				for (int i=0;	i<6; i++){	SE3_incr.operator()(i)	=	SE3_incr_arry[i*2];  }								PRINT_MATX16F( SE3_incr, );
//				SE3_incr	/=			SE3_incr_arry[1];																							PRINT_MATX16F( SE3_incr, );

	Matx44f		pose		=			ReadOutput_44f( 					pose_buf );																PRINT_MATX44F( pose, 	);
	Matx44f		invK		=			ReadOutput_44f(						inv_K_buf);																PRINT_MATX44F( invK,	);
	Matx44f		K			=			ReadOutput_44f(						K_buf	 );																PRINT_MATX44F( K,		);

																																	if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::update_k2k_cpu( ..)_chk_1 . ################################"<< flush;
																																			Matx44f pose_error						= pose	*	GT_pose.inv();	// correct, i.e. reproduces the artif error:  pose = poseStep * pose
																																			Matx16f pose_error_algebra				= PToLie(pose_error);
																																			PRINT_MATX16F( pose_error_algebra, );

																																			PRINT_MATX44F( GT_pose, );
																																			PRINT_MATX44F( pose, );
																																			PRINT_MATX44F( pose_error, );
																																	}
/*
	//for (uint i=0; i<num_SE3_DoF; i++) {	J[i] 	= Hessian_map[i 	+ layer*8*6];	}

	//
	pose_update_J[lid]			=	SE3_incr_map_[lid].s0	-	(  rho  *  J[lid].x );

	//
	uint 	elem				=	lid/num_SE3_DoF;
	float	pose_update_		=	pose_update_J[	elem];
	float	H_elem				=	Hessian_map[	lid + 6	+ layer*8*6].x;
	pose_update_H[lid]			=	H_elem * pose_update_;

	//
	for(uint i=0; i<num_SE3_DoF; i++) pose_update += pose_update_H[lid*6 +i];
*/
	Matx16f pose_update_cpu		= SE3_incr * H.inv();   /* - rho * J */ 	/*NB should compute H.inv() once and store */							PRINT_MATX16F( pose_update_cpu, );		// TODO order of matrix multiplication & transpose 1x6  vs 6x1 ?
	Matx61f	pose_update_cpu_1	= H.inv() * SE3_incr.t(); 	/*pose_update_cpu *  H.inv();*/															PRINT_MATX61F( pose_update_cpu_1, );
//	Matx61f	pose_update_cpu_2	= H/*.inv()*/ * pose_update_cpu.t();																				PRINT_MATX61F( pose_update_cpu_2, );

	Matx44f newPose				= LieToP_Matx( pose_update_cpu )  *  pose;																			PRINT_MATX44F( newPose,			);		// TODO order of matrix multiplication ?
	Matx44f newK2K				= invK  * newPose  * K ;																							PRINT_MATX44F( newK2K,			);
///////////
																																	if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::update_k2k_cpu( ..)_chk_2 . ################################"<< flush;
																																			Matx44f pose_error						= pose	*	GT_pose.inv();	// correct, i.e. reproduces the artif error:  pose = poseStep * pose
																																			Matx16f pose_error_algebra				= PToLie(pose_error);
																																			PRINT_MATX16F( pose_error_algebra, );

																																			PRINT_MATX44F( GT_pose, );
																																			PRINT_MATX44F( pose, );
																																			PRINT_MATX44F( pose_error, );
																																											  cout<<"\n\nRunCL::update_k2k_cpu( ..)_finished ###############################"<< flush;}
}


void RunCL::update_k2k( uint layer, float delta_theta, float delta, Matx44f GT_pose )
{
	string		fname						= "RunCL::update_k2k(..)";
	int			local_verbosity_threshold	= V_RUNCL_UPDATE_K2K;
	cl_kernel	kernel 						= update_k2k_kernel;

	float		img_var						= img_stats[ layer*2*4 + IMG_VAR*4 ] + img_stats[ layer*2*4 + IMG_VAR*4 +1 ] + img_stats[ layer*2*4 + IMG_VAR*4  +2 ];	//5		sum image variance over 3channels, for this layer.
	cl_float2	delta_SE3					= {{delta_theta, delta}};

																																	if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::update_k2k( ..)_chk_1 . "<< flush;}
	// private input
	_clSetKernelArg( kernel,  0, sizeof( cl_float2),					&delta_SE3,					fname);							//__private	float2		delta_SE3,				//0
	_clSetKernelArg( kernel,  1, sizeof( uint),							&layer,						fname);							//__private	float2		delta_SE3,				//0
	// global inputs
	_clSetKernelArg( kernel,  2, sizeof( cl_mem),						&SE3_hessian_pinv_map_mem,	fname);							// __private	uint	Hessian_map,			//1
	_clSetKernelArg( kernel,  3, sizeof( cl_mem),						&SE3_rho_map_mem,			fname);							//__global	float2*		Rho_,					//2		// { sum rho^2 ,  count of valid pixels used } Writen to dense patches.
	_clSetKernelArg( kernel,  4, sizeof( cl_mem),						&SE3_incr_map_mem,			fname);							//__global	float2*		SE3_incr_map_,			//4

	// global input/outputs
	_clSetKernelArg( kernel,  5, sizeof( cl_mem), 						&old_results_buf,			fname);							//__global	float*		old_result				//6
	_clSetKernelArg( kernel,  6, sizeof( cl_mem), 						&pose_buf,					fname);							//__global	float*		Pose,					//7

	_clSetKernelArg( kernel,  7, sizeof( cl_mem), 						&K_buf,						fname);							//__global	float*		K,						//8
	_clSetKernelArg( kernel,  8, sizeof( cl_mem), 						&inv_K_buf,					fname);							//__global	float*		inv_K,					//9
	_clSetKernelArg( kernel,  9, sizeof( cl_mem), 						&k2kbuf,					fname);							//__global	float*		k2k						//10

	cl_command_queue 	queue_to_call		= m_queue;
	cl_int				res, status;
	cl_event			ev;
																																		if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::update_k2k( ..)_chk_1.5 #########################"<<flush;
																																			float pose_ary[16];
																																			ReadOutput( (uchar*)pose_ary, pose_buf, sizeof(float)*16, 0);		// ReadOutput(uchar* outmat, cl_mem buf_mem, size_t data_size, size_t offset/*=0*/)
																																			Matx44f pose;
																																			for(int i=0; i<4; i++){ for(int j=0; j<4; j++){ pose.operator()(i,j) = pose_ary[i*4 +j];  } }

																																			Matx44f pose_error						= pose	*	GT_pose.inv();	// correct, i.e. reproduces the artif error:  pose = poseStep * pose
																																			Matx16f pose_error_algebra				= PToLie(pose_error);
																																			PRINT_MATX16F( pose_error_algebra, );

																																			PRINT_MATX44F( GT_pose, );
																																			PRINT_MATX44F( pose, );
																																			PRINT_MATX44F( pose_error, );

																																			float k2k_ary[16];
																																			ReadOutput( (uchar*)k2k_ary, k2kbuf, sizeof(float)*16, 0);		// ReadOutput(uchar* outmat, cl_mem buf_mem, size_t data_size, size_t offset/*=0*/)
																																			PRINT_FLOAT_16( k2k_ary, before );
																																		}
																									auto step_0 = high_resolution_clock::now();
	res		= clEnqueueNDRangeKernel(queue_to_call, kernel, 1, 0, &device_work_size_multiple, &device_work_size_multiple, 0, NULL, &ev); 	// launch just one minimal workgroup. Only 16 threads req.
																									if (res    != CL_SUCCESS)	{ cout << "\nres = " << checkerror(res) <<"\n"<<flush; exit_(res);}
	status 	= clFlush(queue_to_call);																if (status != CL_SUCCESS)	{ cout << "\nRunCL::update_k2k( ..) call_kernel( cl_kernel "<<kernel<<",  clFlush(queue_to_call) status  = "<<status<<" "<< checkerror(status) <<"\n"<<flush; exit_(status);}
																									auto step_1 = high_resolution_clock::now();
	status 	= clWaitForEvents(1, &ev);																if (status != CL_SUCCESS)	{ cout << "\nRunCL::update_k2k( ..) call_kernel( cl_kernel "<<kernel<<") final,  clWaitForEventsh(1, &ev) ="<<status<<" "<<checkerror(status)  <<"\n"<<flush; exit_(status);}
																									auto step_2 = high_resolution_clock::now();

																																		if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::update_k2k( ..)_chk_2 ##############################"<<\
																																				"Execution time = "<<  duration_cast<microseconds>(step_1 - step_0).count() \
																																				<<" , "<<duration_cast<microseconds>(step_2 - step_1).count() <<flush;

																																			float k2k_ary[16];
																																			ReadOutput( (uchar*)k2k_ary, k2kbuf, sizeof(float)*16, 0);		// ReadOutput(uchar* outmat, cl_mem buf_mem, size_t data_size, size_t offset/*=0*/)
																																			PRINT_FLOAT_16( k2k_ary, after );

																																			float pose_update_ary[6];
																																			ReadOutput( (uchar*)pose_update_ary, old_results_buf, sizeof(float)*6, 0);	//ReadOutput(uchar* outmat, cl_mem buf_mem, size_t data_size, size_t offset/*=0*/)
																																			cout<<"\n pose_update_ary = { 	"; for (int i=0; i<6; i++){ cout<<pose_update_ary[i]<<", 	"; }cout<<"}"<<flush;

																																			// compute pose error
																																			float pose_ary[16];
																																			ReadOutput( (uchar*)pose_ary, pose_buf, sizeof(float)*16, 0);		// ReadOutput(uchar* outmat, cl_mem buf_mem, size_t data_size, size_t offset/*=0*/)
																																			Matx44f pose;
																																			for(int i=0; i<4; i++){ for(int j=0; j<4; j++){ pose.operator()(i,j) = pose_ary[i*4 +j];  } }

																																			Matx44f pose_error						= pose	*	GT_pose.inv();	// correct, i.e. reproduces the artif error:  pose = poseStep * pose
																																			Matx16f pose_error_algebra				= PToLie(pose_error);
																																			PRINT_MATX16F( pose_error_algebra, );

																																			PRINT_MATX44F( GT_pose, );
																																			PRINT_MATX44F( pose, );
																																			PRINT_MATX44F( pose_error, );

																																			cout<<"\nRunCL::update_k2k( ..)_finished _____________________"<<flush;
																																		}
}

void RunCL::se3_rho_sq( const uint local_num_samples,  const uint start_sample_idx,  float Rho_sq_results[tracking_tot_samples][max_mipmap_layers][tracking_num_colour_channels], const float count[4], uint start, uint stop,  float k2k_3_16_[tracking_tot_samples][16]   ){
	string fname = "RunCL::se3_rho_sq( ..)";
	int local_verbosity_threshold = V_RUNCL_SE3_RHO_SQ;
	const int num_samples  = 1; //tracking_num_samples;

	const cl_mem imgmem_   = current_frames[ current_frames_idx[0] ].img_buf;
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::se3_rho_sq( ..)_chk0 .##################################################################"<<flush;
																																				cout << "\nRunCL::se3_rho_sq( ..)__chk_0.3: K2K= ";
																																				for ( int i=0; i<16; i++){ cout << ",  "<< fp32_k2keyframe[i];  }	cout << flush;

																																				for ( int sample=start_sample_idx; sample<( start_sample_idx + local_num_samples); sample++){
																																					cout << "\n\nk2k_3_16_["<<sample<<"][16]= ";
																																					print_float_16( k2k_3_16_[sample] );
																																					cout << "\n\n" <<flush;
																																				}
																																				cout<<"\n\nRunCL::se3_rho_sq( ..)_chk0.4 ,  dataset_frame_num="<<dataset_frame_num<<",   count="<<count[0]<<flush;
																																			}
																													if ( local_num_samples > tracking_num_samples) {
																														cerr  << "\n\n void RunCL::se3_rho_sq( ..) : Error! ( local_num_samples > tracking_num_samples), nee dto allocate a larger buffer." << flush;
																														exit_( 1);
																													}
	_clEnqueueWriteBuffer( uload_queue, k2kbuf,	CL_FALSE, 0, local_num_samples*16*sizeof( float), k2k_3_16_[start_sample_idx],  	fname);
	float zero  = 0;
	_clEnqueueFillBuffer( uload_queue, SE3_rho_map_mem, 	&zero, sizeof( float), 0, num_samples*2*mm_size_bytes_C4, 	fname);
	_clEnqueueFillBuffer( uload_queue, se3_sum_rho_sq_mem, 	&zero, sizeof( float), 0, num_samples*pix_sum_size_bytes, 	fname);
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::se3_rho_sq( ..)_chk0.7 "<<flush;}
																																			// NB GT_depth loaded to depth_mem by void RunCL::loadFrameData( ..)
	const uint wg_divisor =2;  // 1,2,4,8  reduction in workgroup size for this kernel.
	//input integers
	//      __private	 uint layer, set in mipmap_call_kernel( ..) below                                                                         __private	    uint	    layer,		                    //0
	_clSetKernelArg( se3_rho_sq_kernel, 1, sizeof( uint),   &local_num_samples, 	fname);														//__private		uint		local_num_samples				//1
	_clSetKernelArg( se3_rho_sq_kernel, 2, sizeof( uint),   &se3_sum_size, 			fname);														//__private		uint		pix_sum_size_bytes,				//2
	//input buffers
    _clSetKernelArg( se3_rho_sq_kernel, 3, sizeof( cl_mem), &mipmap_buf, 			fname);														//__constant    uint*	    mipmap_params,	                //3
	_clSetKernelArg( se3_rho_sq_kernel, 4, sizeof( cl_mem), &uint_param_buf, 		fname);														//__constant	uint*		uint_params,					//4
	_clSetKernelArg( se3_rho_sq_kernel, 5, sizeof( cl_mem), &fp32_param_buf, 		fname);														//__constant	float*		fp32_params						//5
	_clSetKernelArg( se3_rho_sq_kernel, 6, sizeof( cl_mem), &k2kbuf, 				fname);														//__global		float* 		k2k,							//6		// TODO keyframe2K
	_clSetKernelArg( se3_rho_sq_kernel, 7, sizeof( cl_mem), &keyframe_imgmem, 		fname);														//__global 		float4*		keyframe_imgmem,				//7		// TODO need keyframe mipmap   keyframe_imgmem , keyframe_depth_mem
	_clSetKernelArg( se3_rho_sq_kernel, 8, sizeof( cl_mem), &imgmem_, 				fname);														//__global 		float4*		imgmem,							//8
	_clSetKernelArg( se3_rho_sq_kernel, 9, sizeof( cl_mem), &keyframe_depth_mem, 	fname);														//__global 	 	float*		keyframe_depth_mem				//9		// NB GT_depth, now stoed as inv_depth	// TODO need keyframe mipmap
	_clSetKernelArg( se3_rho_sq_kernel,10, sizeof( cl_mem), &keyframe_g1mem, 		fname);														//__global 	 	float8*		g1p								//10

	//output
	_clSetKernelArg( se3_rho_sq_kernel,11, sizeof( cl_mem), 													&SE3_rho_map_mem, 		fname);	//__global	    float4*     Rho_					        //11
	_clSetKernelArg( se3_rho_sq_kernel,12, ( local_num_samples*local_work_size*4/wg_divisor)*sizeof( float),	 NULL, 					fname);	//__local		float4*		local_sum_rho_sq				//12	// 1 DoF, float4 channels
	_clSetKernelArg( se3_rho_sq_kernel,13, sizeof( cl_mem), 													&se3_sum_rho_sq_mem, 	fname);	//__global 		float4*		global_sum_rho_sq,				//13

																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::se3_rho_sq( ..)_chk1 .  start="<<start<<",  stop="<<stop<<flush;}
	mipmap_call_kernel( se3_rho_sq_kernel, m_queue, start, stop, false, local_work_size/wg_divisor);
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::se3_rho_sq( ..)_chk3 ."<<flush;
																																				stringstream ss;	ss << dataset_frame_num <<"_iter_"<<count[0]<<"_layer"<<count[1]<<"_n_samples"<<local_num_samples<<"_factor"<<count[2]<<"_se3_rho_sq_";
																																				stringstream ss_path;

																																				bool show 				= false;
																																				float max_range 		= -1; 					// i.e. gray = zero.
																																				uint vol_layers 		= local_num_samples;	// i.e. 3 samples.
																																				bool exception_tiff 	= false;
																																				bool display 			= false; 				//obj["sample_se3_incr"].asBool( );
																																				cout << "\nRunCL::se3_rho_sq( ..)_chk3.5   display="<< display<< endl << flush;

																																				DownloadAndSave_3Channel_volume(  SE3_rho_map_mem,  ss.str( ), paths.at( "SE3_rho_map_mem"),  mm_size_bytes_C4, mm_Image_size, CV_32FC4, show, max_range, vol_layers, exception_tiff, count[0], display );
																																			}
																																			// void RunCL::DownloadAndSave_3Channel_volume( cl_mem buffer,   std::string count,   std::filesystem::path folder,   size_t image_size_bytes,   cv::Size size_mat,   int type_mat,   bool show,   float max_range,   uint vol_layers,    bool exception_tiff /*=false*/,   float iter,   bool display)

	for ( int sample = 0; sample<local_num_samples; sample++){
		read_Rho_sq(  Rho_sq_results[sample + start_sample_idx], sample  );
	}
}

void RunCL::estimateSE3_LK( float local_k2k[16], float SE3_results[max_mipmap_layers][num_SE3_DoF][tracking_num_colour_channels], float SE3_weights_results[max_mipmap_layers][num_SE3_DoF][tracking_num_colour_channels], float Rho_sq_results[max_mipmap_layers][4], int count, uint start, uint stop){ //estimateSE3_LK( ); 	( uint start=0, uint stop=8)			// TODO replace arbitrary fixed constant with a const uint variable in the header...
	string fname = "RunCL::estimateSE3_LK( ..)";
	int local_verbosity_threshold = V_RUNCL_ESTIMATESE3_LK;

	const cl_mem imgmem_   = current_frames[ current_frames_idx[0] ].img_buf;
																																			if( verbosity>local_verbosity_threshold) {cout << "\nRunCL::estimateSE3_LK( ..)_chk_0: K2K= ";
																																				PRINT_FLOAT_16( local_k2k,);
																																				cout<<"\n dataset_frame_num="	<<dataset_frame_num
																																				<<",      count="				<<count				<<flush;
																																				stringstream ss;	ss << fname << "_" << dataset_frame_num << "_iter_"<< count << "_estimateSE3_LK_";
																																				DownloadAndSave( 	keyframe_depth_mem,   	ss.str( ), 	paths.at( "keyframe_depth_mem"), mm_size_bytes_C1,   mm_Image_size,   CV_32FC1, 	false , fp32_params[MAX_INV_DEPTH]);
																																			}
	_clEnqueueWriteBuffer( uload_queue, k2kbuf,	CL_FALSE, 0,  16 * sizeof( float), local_k2k, 				fname);
	float zero  = 0;
	_clEnqueueFillBuffer( uload_queue, SE3_rho_map_mem, 	&zero, sizeof( float), 0, 2*mm_size_bytes_C4,	fname);
	_clEnqueueFillBuffer( uload_queue, se3_sum_rho_sq_mem, 	&zero, sizeof( float), 0, pix_sum_size_bytes, 	fname);
	_clEnqueueFillBuffer( uload_queue, se3_weight_sum_mem, 	&zero, sizeof( float), 0, se3_sum_size_bytes, 	fname);
	_clEnqueueFillBuffer( uload_queue, se3_sum_mem, 		&zero, sizeof( float), 0, se3_sum_size_bytes, 	fname);
	_clEnqueueFillBuffer( uload_queue, SE3_incr_map_mem, 	&zero, sizeof( float), 0, mm_size_bytes_C1*24, 	fname);
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::estimateSE3_LK( ..)_chk_1 "<<flush;}
																																			// NB GT_depth loaded to depth_mem by void RunCL::loadFrameData( ..)
	const 	uint wg_divisor =2;  // 1,2,4,8  reduction in workgroup size for this kernel.
	// inputs
	// __private	 uint layer, set in mipmap_call_kernel( ..) below																			  __private		uint		layer,							//0
	_clSetKernelArg( se3_lk_grad_kernel, 1, sizeof( cl_mem), &mipmap_buf, 					fname);												//__constant	uint*		mipmap_params,					//1
	_clSetKernelArg( se3_lk_grad_kernel, 2, sizeof( cl_mem), &uint_param_buf, 				fname);												//__constant	uint*		uint_params,					//2
	_clSetKernelArg( se3_lk_grad_kernel, 3, sizeof( cl_mem), &fp32_param_buf, 				fname);												//__constant	float*		fp32_params						//3
	_clSetKernelArg( se3_lk_grad_kernel, 4, sizeof( cl_mem), &k2kbuf, 						fname);												//__global		float* 		k2k,							//4		// TODO keyframe2K
	_clSetKernelArg( se3_lk_grad_kernel, 5, sizeof( cl_mem), &keyframe_imgmem, 				fname);												//__global 		float4*		keyframe_imgmem,				//5		// TODO need keyframe mipmap   keyframe_imgmem , keyframe_depth_mem
	_clSetKernelArg( se3_lk_grad_kernel, 6, sizeof( cl_mem), &imgmem_, 						fname);												//__global 		float4*		imgmem,							//6
	_clSetKernelArg( se3_lk_grad_kernel, 7, sizeof( cl_mem), &keyframe_SE3_grad_map_mem, 	fname);												//__global 	 	float4*		keyframe_SE3_grad_map_mem		//7
	_clSetKernelArg( se3_lk_grad_kernel, 8, sizeof( cl_mem), &SE3_grad_map_mem, 			fname);												//__global 	 	float4*		SE3_grad_map					//8
	_clSetKernelArg( se3_lk_grad_kernel, 9, sizeof( cl_mem), &keyframe_depth_mem, 			fname);												//__global 	 	float*		keyframe_depth_mem				//9		// NB GT_depth, now stoed as inv_depth	// TODO need keyframe mipmap
	//outputs
	_clSetKernelArg( se3_lk_grad_kernel,10, sizeof( cl_mem), 									&SE3_rho_map_mem, 		fname);					//__global	    float4*     Rho_							//10

	_clSetKernelArg( se3_lk_grad_kernel,11, ( local_work_size*1*4/wg_divisor)*sizeof( float),	NULL, 					fname);					//__local		float4*		local_sum_rho_sq				//11	1 DoF, float4 channels
	_clSetKernelArg( se3_lk_grad_kernel,12, sizeof( cl_mem),									&se3_sum_rho_sq_mem, 	fname);		 			//__global 		float4*		global_sum_rho_sq,				//12
	_clSetKernelArg( se3_lk_grad_kernel,13, sizeof( cl_mem), 									&SE3_weight_map_mem, 	fname);					//__global		float4* 	weights_map,					//13

	_clSetKernelArg( se3_lk_grad_kernel,14, ( local_work_size*6*4/wg_divisor)*sizeof( float),	NULL, 					fname);					//__local		float4*		local_sum_weight,				//14	6 DoF, float4 channels
	_clSetKernelArg( se3_lk_grad_kernel,15, sizeof( cl_mem), 									&se3_weight_sum_mem, 	fname);		 			//__global 		float4*		global_sum_weight,				//15
	_clSetKernelArg( se3_lk_grad_kernel,16, sizeof( cl_mem), 									&SE3_incr_map_mem, 		fname);					//__global 	 	float4*		SE3_incr_map_					//16

	_clSetKernelArg( se3_lk_grad_kernel,17, ( local_work_size*6*4/wg_divisor)*sizeof( float),	NULL, 					fname);					//__local		float4*		local_sum_grads					//17	6 DoF, float4 channels
	_clSetKernelArg( se3_lk_grad_kernel,18, sizeof( cl_mem), 									&se3_sum_mem, 			fname);		 			//__global 		float4*		global_sum_grads,				//18

	_clSetKernelArg( se3_lk_grad_kernel,19, sizeof( cl_mem), 									&keyframe_g1mem, 		fname);					//__global 	 	float8*		g1p								//19

																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::estimateSE3_LK( ..)_chk_3 ."<<flush;}
																									auto step_0 = high_resolution_clock::now();
	mipmap_call_kernel( se3_lk_grad_kernel, m_queue, start, stop, true, local_work_size/wg_divisor); 										// false // reduced worksize to allow for local memory limit 4kb on rtx 3030
																									auto step_1 = high_resolution_clock::now();
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::estimateSE3_LK( ..)_chk_3.5 . "<<\
																																				"Execution time = "<<  duration_cast<microseconds>(step_1 - step_0).count() <<flush;
																																			}
																																			if( verbosity>local_verbosity_threshold-2) {cout<<"\n\nRunCL::estimateSE3_LK( ..)_chk_4 ."<<flush;
																																				stringstream ss;	ss << dataset_frame_num << "_iter_"<< count << "_estimateSE3_LK_";
																																				stringstream ss_path;
																																				bool show 				= false;
																																				bool display 			= false;
																																				bool exception_tiff 	= false;
																																				uint vol_layers 		= 1;
																																				float max_range 		= -1; 		// i.e. gray = zero.

																																				DownloadAndSave_3Channel_volume(  SE3_rho_map_mem,		ss.str( ), paths.at( "SE3_rho_map_mem"),  	mm_size_bytes_C4, mm_Image_size, CV_32FC4, show, max_range, vol_layers, exception_tiff, count, display );
																																				DownloadAndSave_3Channel_volume(  SE3_weight_map_mem,	ss.str( ), paths.at( "SE3_weight_map_mem"), mm_size_bytes_C4, mm_Image_size, CV_32FC4, show, max_range, 6, 			exception_tiff, count, display );
																																				DownloadAndSave_3Channel_volume(  SE3_incr_map_mem,		ss.str( ), paths.at( "SE3_incr_map_mem"), 	mm_size_bytes_C4, mm_Image_size, CV_32FC4, show, max_range, 6, 			exception_tiff, count, display );
																																			}
																																			if( obj["sample_se3_incr"].asBool( )==true) {
																																				PrepareResults_3Channel_volume(  SE3_rho_map_mem,  	mm_size_bytes_C4, mm_Image_size, CV_32FC4, -1, 1,  count );
																																				PrepareResults_3Channel_volume(  SE3_incr_map_mem, 	mm_size_bytes_C4, mm_Image_size, CV_32FC4, -1, 6,  count );
																																			}
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::estimateSE3_LK( ..)_chk_5 ."<<flush;}
	read_Rho_sq( Rho_sq_results);
	read_se3_weights( SE3_weights_results);
	read_se3_incr( SE3_results);																											if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::estimateSE3_LK( ..)_finished ."<<flush;}
}

void RunCL::read_Rho_sq( float Rho_sq_results[max_mipmap_layers][4],  int offset/*=0*/ ){
	string fname = "RunCL::read_Rho_sq( ..)";
	int local_verbosity_threshold = V_RUNCL_READ_RHO_SQ;
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::read_Rho_sq( ..)_chk1,  offset="<<offset
																																				<<",   offset*pix_sum_size_bytes="<<offset*pix_sum_size_bytes<<flush;}
	cv::Mat rho_sq_sum_mat = cv::Mat::zeros ( se3_sum_size, 4, CV_32FC1); // cv::Mat::zeros ( int rows, int cols, int type)					// NB the data returned is one float4 per group, holding HSV, plus entry[3]=pixel count.
	ReadOutput( rho_sq_sum_mat.data, se3_sum_rho_sq_mem, pix_sum_size_bytes, offset*pix_sum_size_bytes );																//float Rho_sq_reults[8][4] = {{0}};
	/*
	 * void RunCL::ReadOutput( uchar* outmat,   cl_mem buf_mem,   size_t data_size,   size_t offset/ *=0* / ) {...
	 * 		status = clEnqueueReadBuffer( dload_queue,		// command_queue
											buf_mem,		// buffer
											CL_FALSE,		// blocking_read
											offset,			// offset
											data_size,		// size
											outmat,			// pointer
											0,				// num_events_in_wait_list
											NULL,			// event_waitlist				needs to know about preceeding events:
											&readEvt);		// event
	 * ...}
	 */

																																			if( verbosity>local_verbosity_threshold+1) {cout<<"\nRunCL::read_Rho_sq( ..)_chk2,  offset="<<offset<<flush;}

																																			if( verbosity>local_verbosity_threshold+2) {
																																				cout << "\n\nRunCL::read_Rho_sq( ..)_chk3,  offset="<<offset<<flush;
																																				cout << "\nrho_sq_sum_mat.size( )="<<rho_sq_sum_mat.size( )<<flush;
																																				cout << "\nse3_sum_size="<<se3_sum_size<<flush;
																																				cout << "\n mm_num_reductions = " << mm_num_reductions << endl << flush;
																																				cout << "\n\nrho_sq_sum_mat.at<float> ( i*num_DoFs + j,  k) ,  i=group,  j=SO3 DoF,  i=delta4 ( H,S,V, ( valid pixels/group_size) )";
																																				for ( int i=0; i< se3_sum_size ; i++){//&& i<30
																																					cout << "\ngroup ="<<i<<":   ";
																																					cout << ",     \t( ";
																																					for ( int k=0; k<4; k++){	cout << ", \t" << rho_sq_sum_mat.at<float>( i, k); }
																																					cout << ")";
																																				}cout << endl << endl;
																																			}

	for ( int layer=mm_start; layer<=mm_stop; layer++){
		uint groups_to_sum 			= rho_sq_sum_mat.at<float>( layer, 0);
		uint start_group 			= rho_sq_sum_mat.at<float>( layer, 2);  //global_sum_offset;
		uint stop_group 			= start_group + groups_to_sum ;   																		// -1
																																			if( verbosity>local_verbosity_threshold+2) {
																																				cout << "\nRunCL::read_Rho_sq( ..)_chk4,  offset="<<offset<<",  layer = "<<layer<<
																																				", groups_to_sum = "<<groups_to_sum<<
																																				", start_group = "<<start_group<<
																																				", stop_group = "<<stop_group<< flush;
																																			}
		for ( int group=start_group; group< stop_group; group++){	for ( int chan=0; chan<4; chan++){ 		Rho_sq_results[layer][chan] 	+= rho_sq_sum_mat.at<float>( group, chan);		};
		}																									// sum j groups for this layer of the MipMap.
	}
																																			if( verbosity>local_verbosity_threshold+1) {
																																				cout << "\n\nRunCL::read_Rho_sq( ..)_chk5,  offset="<<offset<<flush;
																																				for ( int layer=0; layer<=mm_num_reductions+1; layer++){ 														// results / ( num_valid_px * img_variance)
																																					cout << "\nLayer "<<layer<<" mm_num_reductions = "<< mm_num_reductions <<",  Rho_sq_results/num_groups = ( ";
																																					if ( Rho_sq_results[layer][3] > 0){
																																						for ( int chan=0; chan<3; chan++){	cout << ",   \t" << Rho_sq_results[layer][chan] / ( Rho_sq_results[layer][3]  *  img_stats[layer*8 + IMG_VAR*4 +chan]  );
																																						}
																																						cout << ", \t" << Rho_sq_results[layer][3] << ")";
																																					}
																																					else{	for ( int chan=0; chan<3; chan++){	cout << ", \t" << 0.0f;		}
																																						cout << ", \t" << Rho_sq_results[layer][3] << ")";
																																					}
																																				}cout << "\nRunCL::read_Rho_sq( ..)_finish . ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"<<flush;
																																			}
}

void RunCL::read_se3_weights( float SE3_weights_results[max_mipmap_layers][num_SE3_DoF][tracking_num_colour_channels]){
	string fname = "RunCL::read_se3_weights( ..)";
	int local_verbosity_threshold = V_RUNCL_READ_SE3_WEIGHTS;
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::read_se3_weights( ..)_chk1 ."<<flush;}
                                                                                                                                            // directly read higher layers
	uint num_DoFs = 6;
    cv::Mat se3_sum_mat = cv::Mat::zeros ( se3_sum_size, num_DoFs*4, CV_32FC1); // cv::Mat::zeros ( int rows, int cols, int type)				// NB the data returned is one float8 per group, holding one float per 6DoF of SE3, plus entry[7]=pixel count.
	ReadOutput( se3_sum_mat.data, se3_weight_sum_mem, se3_sum_size_bytes );                                                                        // se3_sum_size_bytes
																																			if( verbosity>local_verbosity_threshold+2) {cout << "\n\nRunCL::read_se3_weights( ..)_chk2 ."<<flush;
																																				cout << "\nse3_sum_mat.size( )="<<se3_sum_mat.size( )<<flush;
																																				cout << "\nse3_sum_size="<<se3_sum_size<<flush;
																																				cout << "\n mm_num_reductions = " << mm_num_reductions << endl << flush;
																																				cout << "\n\nse3_sum_mat.at<float> ( i*num_DoFs + j,  k) ,  i=group,  j=SO3 DoF,  i=delta4 ( H,S,V, ( valid pixels/group_size) )";
																																				for ( int i=0; i< se3_sum_size ; i++){//&& i<30
																																					cout << "\ngroup ="<<i<<":   ";
																																					for ( int j=0; j<num_DoFs; j++){
																																						cout << ",     \t( ";	for ( int k=0; k<4; k++){	cout << ", \t" << se3_sum_mat.at<float>( i, j*4 + k); }	cout << ")";
																																					}cout << flush;
																																				}cout << endl << endl;
																																			}
	for ( int layer=mm_start; layer<=mm_stop; layer++){
        /*
		uint read_offset_ 		= MipMap[i*8 + MiM_READ_OFFSET];																			// mipmap_params_[MiM_READ_OFFSET];
        uint global_sum_offset 	= read_offset_ / local_work_size ;
        uint groups_to_sum 		= se3_sum_mat.at<float>( i, 0);
        uint start_group 		= global_sum_offset + 1;
        uint stop_group 		= start_group + groups_to_sum ;   // -1																		// skip the last group due to odd 7th value.
        */
        uint groups_to_sum 			= se3_sum_mat.at<float>( layer, 0);
		uint start_group 			= se3_sum_mat.at<float>( layer, 2);
		uint stop_group 			= start_group + groups_to_sum;
																																			if( verbosity>local_verbosity_threshold+1) {
																																				cout << "\ni="<<layer<<
																																				",  groups_to_sum="<<groups_to_sum<<
																																				",  start_group="<<start_group<<
																																				",  stop_group="<<stop_group;
																																			}
		for ( int group=start_group; group< stop_group  ; group++){	for ( int dof=0; dof<num_DoFs; dof++){ 	for ( int chan=0; chan<4; chan++){	SE3_weights_results[layer][dof][chan] += se3_sum_mat.at<float>( group, dof*4 + chan);	} }	}	//l =4 =num channels	// sum j groups for this layer of the MipMap. // se3_sum_mat.at<float>( j, k);
    }
																																			if( verbosity>local_verbosity_threshold+1) {
																																				cout << endl << " SE3_weights_results/num_groups = ( H, S, V, alpha=num_groups) ";
																																				for ( int layer=mm_start; layer<=mm_stop; layer++){ 																// results / ( num_valid_px * img_variance)
																																					cout << "\nLayer "<<layer<<" SE3_weights_results = ( ";												// raw results
																																					for ( int dof=0; dof<num_DoFs; dof++){
																																						cout << "\nse3 dof="<<dof<<" : ( ";  for ( int chan=0; chan<4; chan++){	cout << ",   \t" << SE3_weights_results[layer][dof][chan] ;	}cout << ")";
																																					}cout << ")";
																																					///
																																					/*
																																					cout << "\nLayer "<<i<<" SE3_weights_results/num_groups = ( ";
																																					for ( int k=0; k<num_DoFs; k++){
																																						cout << "\nDoF="<<k<<" ( "; for ( int l=0; l<3; l++){	cout << ", \t" << SE3_weights_results[i][k][l] / ( SE3_weights_results[i][k][3]  *  img_stats[IMG_VAR+l]  ); } cout << ", " << SE3_weights_results[i][k][3] << ")";
																																					}cout << ")";																					// << "{"<< img_stats[i*4 +IMG_VAR+l] <<"}"
																																					*/

																																				}cout << "\nRunCL::read_se3_weights( ..)_finish . ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"<<flush;
																																			}
	}

void RunCL::read_se3_incr( float SE3_results[max_mipmap_layers][num_SE3_DoF][tracking_num_colour_channels]){
	string fname = "RunCL::read_se3_incr( ..)";
	int local_verbosity_threshold = V_RUNCL_READ_SE3_INCR;
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::read_se3_incr( ..)_chk1 ."<<flush;}
                                                                                                                                            // directly read higher layers
	uint num_DoFs = 6;
    cv::Mat se3_sum_mat = cv::Mat::zeros ( se3_sum_size, num_DoFs*4, CV_32FC1); // cv::Mat::zeros ( int rows, int cols, int type)				// NB the data returned is one float8 per group, holding one float per 6DoF of SE3, plus entry[7]=pixel count.
	ReadOutput( se3_sum_mat.data, se3_sum_mem, se3_sum_size_bytes );                                                                        // se3_sum_size_bytes
																																			if( verbosity>local_verbosity_threshold+2) {cout << "\nRunCL::read_se3_incr( ..)_chk2 ."<<flush;
																																				cout << "\nse3_sum_mat.size( )="<<se3_sum_mat.size( )<<flush;
																																				cout << "\nse3_sum_size="<<se3_sum_size<<flush;
																																				cout << "\n mm_num_reductions = " << mm_num_reductions << endl << flush;
																																				cout << "\n\nse3_sum_mat.at<float> ( i*num_DoFs + j,  k) ,  i=group,  j=SO3 DoF,  i=delta4 ( H,S,V, ( valid pixels/group_size) )";
																																				for ( int i=0; i< se3_sum_size ; i++){//&& i<30
																																					cout << "\ngroup ="<<i<<":   ";
																																					for ( int j=0; j<num_DoFs; j++){
																																						cout << ",     \t( ";	for ( int k=0; k<4; k++){	cout << ", \t" << se3_sum_mat.at<float>( i, j*4 + k); }	cout << ")";
																																					}cout << flush;
																																				}cout << endl << endl;
																																			}
	for ( int layer=mm_start; layer<=mm_stop; layer++){
		uint groups_to_sum 			= se3_sum_mat.at<float>( layer, 0);
		uint start_group 			= se3_sum_mat.at<float>( layer, 2);
        uint stop_group 			= start_group + groups_to_sum;
																																			if( verbosity>local_verbosity_threshold+1) {
																																				cout << "\ni="<<layer<<
																																				",  groups_to_sum="<<groups_to_sum<<
																																				",  start_group="<<start_group<<
																																				",  stop_group="<<stop_group;
																																			}
		for ( int group=start_group; group< stop_group  ; group++){	for ( int dof=0; dof<num_DoFs; dof++){ 	for ( int chan=0; chan<4; chan++){	SE3_results[layer][dof][chan] += se3_sum_mat.at<float>( group, dof*4 + chan);	} }	}	//l =4 =num channels	// sum j groups for this layer of the MipMap. // se3_sum_mat.at<float>( j, k);
    }
																																			if( verbosity>local_verbosity_threshold+1) {
																																				cout << endl << " SE3_results/num_groups = ( H, S, V, alpha=num_groups) ";
																																				for ( int layer=mm_start; layer<=mm_stop; layer++){ 																// results / ( num_valid_px * img_variance)
																																					cout << "\nLayer "<<layer<<" SE3_results = ( ";														// raw results
																																					for ( int dof=0; dof<num_DoFs; dof++){
																																						cout << "\nse3 dof="<<dof<<" : ( ";  for ( int chan=0; chan<4; chan++){	cout << ", \t" << SE3_results[layer][dof][chan] ;	}cout << ")";
																																					}cout << ")";
																																					///
																																					/*
																																					cout << "\nLayer "<<i<<" SE3_results/num_groups = ( ";
																																					for ( int k=0; k<num_DoFs; k++){
																																						cout << "\nDoF="<<k<<" ( "; for ( int l=0; l<3; l++){	cout << ", \t" << SE3_results[i][k][l] / ( SE3_results[i][k][3]  *  img_stats[IMG_VAR+l]  ); } cout << ", " << SE3_results[i][k][3] << ")";
																																					}cout << ")";																					// << "{"<< img_stats[i*4 +IMG_VAR+l] <<"}"
																																					*/
																																				}cout << "\nRunCL::read_se3_incr( ..)_finish . ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"<<flush;
																																			}
}

void RunCL::tracking_result( string result){
	string fname = "RunCL::tracking_result( ..)";
	if( verbosity>V_RUNCL_TRACKING_RESULT) {
		cout<<"\n\nRunCL::tracking_result( ..)_chk0"<<flush;
		stringstream ss;			ss << dataset_frame_num <<  "_img_grad_" << result;				// "_iter_"<< count <<
		stringstream ss_path_rho;	ss_path_rho << "SE3_rho_map_mem";
		cout << " , " << ss_path_rho.str( ) << " , " <<  paths.at( ss_path_rho.str( )) << " , " << ss.str( )  <<flush;
		DownloadAndSave_3Channel_volume(  SE3_rho_map_mem,  ss.str( ), paths.at( ss_path_rho.str( )), mm_size_bytes_C4, mm_Image_size, CV_32FC4, false, -1, /*display*/true );
	}
}

/*
* 	atomic_test1_buf	= clCreateBuffer( m_context, CL_MEM_READ_WRITE 						, 4*local_work_size*sizeof( int),	0, &res);	if( res!=CL_SUCCESS){cout<<"\nres 42= "<<checkerror( res)<<"\n"<<flush;exit_( res);}
*/

/*
// void RunCL::atomic_test1( ){
// 	string fname = "RunCL::atomic_test1( )";
// 	int local_verbosity_threshold = V_RUNCL_ATOMIC_TEST1;
//
// 	const int data_size = 4*local_work_size;
// 	const size_t num_threads = 2 * local_work_size;
// 	int num_threads_int = 1.5*local_work_size;
//
// 	int zero  = 0;
// 	_clEnqueueFillBuffer( uload_queue, atomic_test1_buf, 	&zero, sizeof( int), 0, data_size*sizeof( int), 	fname);
//
// 	_clSetKernelArg( atomic_test1_kernel, 0, sizeof( int), 		&num_threads_int, 	fname);
// 	_clSetKernelArg( atomic_test1_kernel, 1, sizeof( cl_mem), 	&atomic_test1_buf, 	fname);
//
// 	_clEnqueueNDRangeKernel( m_queue, atomic_test1_kernel, 1, 0, &num_threads, &local_work_size,fname);
//
// 	int atomic_test_output[ data_size ];
// 	for ( int i = 0; i<data_size; i++) atomic_test_output[i] = -1;
// 	cl_event readEvt;
// 	cl_int status;
// 	status = clEnqueueReadBuffer( 	dload_queue,			// command_queue
// 									atomic_test1_buf,		// buffer
// 									CL_FALSE,				// blocking_read
// 									0,						// offset
// 									data_size*sizeof( int),	// size
// 									atomic_test_output,		// pointer
// 									0,						// num_events_in_wait_list
// 									NULL,					// event_waitlist				needs to know about preceeding events:
// 									&readEvt				// event
// 								 );																												if ( status != CL_SUCCESS) { cout << "\nclEnqueueReadBuffer( ..) status=" << checkerror( status) <<"\n"<<flush; exit_( status);}
// 	_cl_flush_finish( dload_queue, fname);
//
// 	cout << "\n\n void RunCL::atomic_test1( ): \t  local_work_size = "<<local_work_size<<", \t ( buffer size) data_size = "<<data_size<<", \t ( num threards launched)  num_threads="<<num_threads<<", \t ( num threards run) num_threads_int = "<<num_threads_int<<", \t  ( ";
// 	int i=0;
// 	for ( ; i< data_size; i++) cout << ", " << atomic_test_output[i];
// 	cout << ") \t i="<< i<< "\n\n" << flush;
// }
//
//
// void RunCL::atomic_test2( ){
// 	string fname = "RunCL::atomic_test2( )";
// 	int local_verbosity_threshold = V_RUNCL_ATOMIC_TEST2;
//
// 	cl_int res, status;
// 	cl_event ev, writeEvt;
// 	const 	int 	data_size 		= 4   *local_work_size;
// 	const 	size_t 	num_threads 	= 2   *local_work_size;
// 			int 	num_threads_int = 1.5 *local_work_size;
//
//
//
//
// 	float number  = 0.674;
// 	status = clEnqueueFillBuffer( uload_queue, atomic_test2_buf, &number, sizeof( int), 0, data_size*sizeof( float), 	0, NULL, &writeEvt);	if ( status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror( status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.3\n" << endl;exit_( status);}
//
// 	clFlush( uload_queue); status = clFinish( uload_queue); 																					if ( status != CL_SUCCESS)	{ cout << "\nclFinish( uload_queue)=" << status << checkerror( status) <<"\n"  << flush; exit_( status);}
//
// 	res = clSetKernelArg( atomic_test2_kernel, 0, sizeof( int), 		&num_threads_int);														if ( res    !=CL_SUCCESS)	{ cout <<"\nres = "<<checkerror( res)<<"\n"<<flush;exit_( res);}
// 	res = clSetKernelArg( atomic_test2_kernel, 1, sizeof( cl_mem), 	&atomic_test2_buf);															if ( res    !=CL_SUCCESS)	{ cout <<"\nres = "<<checkerror( res)<<"\n"<<flush;exit_( res);}
// 	/ *
// 	cl_int clEnqueueNDRangeKernel(
// 									cl_command_queue 		command_queue,
// 									cl_kernel 				kernel,
// 									cl_uint 				work_dim,
// 									const size_t* 			global_work_offset,
// 									const size_t* 			global_work_size,
// 									const size_t* 			local_work_size,
// 									cl_uint 				num_events_in_wait_list,
// 									const cl_event* 		event_wait_list,
// 									cl_event* 				event
// 									);
// 	* /
// 	res 	= clEnqueueNDRangeKernel( m_queue, atomic_test2_kernel, 1, 0, &num_threads, &local_work_size, 0, NULL, &ev);						if ( res    != CL_SUCCESS)	{ cout << "\nres = " << checkerror( res) <<"\n"<<flush; exit_( res);}
// 	status 	= clFlush( m_queue);																												if ( status != CL_SUCCESS)	{ cout << "\nRunCL::atomic_test1( ),  clFlush( queue_to_call) status  = "<<status<<" "<< checkerror( status) <<"\n"<<flush; exit_( status);}
//
// 	float atomic_test_output[ data_size ];
// 	for ( int i = 0; i<data_size; i++) atomic_test_output[i] = -1;
// 	cl_event readEvt;
// 	status = clEnqueueReadBuffer( 	dload_queue,			// command_queue
// 									atomic_test2_buf,		// buffer
// 									CL_FALSE,				// blocking_read
// 									0,						// offset
// 									data_size*sizeof( float),	// size
// 									atomic_test_output,		// pointer
// 									0,						// num_events_in_wait_list
// 									NULL,					// event_waitlist				needs to know about preceeding events:
// 									&readEvt				// event
// 								 );
// 														if ( status != CL_SUCCESS) { cout << "\nclEnqueueReadBuffer( ..) status=" 	<< checkerror( status) <<"\n"<<flush; exit_( status);}
// 	status = clFlush( dload_queue);						if ( status != CL_SUCCESS) { cout << "\nclFlush( m_queue) status = " 		<< checkerror( status) <<"\n"<<flush; exit_( status);}
// 	status = clWaitForEvents( 1, &readEvt); 			if ( status != CL_SUCCESS) { cout << "\nclWaitForEvents status="			<< checkerror( status) <<"\n"<<flush; exit_( status);}
//
// 	cout << "\n\n void RunCL::atomic_test2( ): \t  local_work_size = "<<local_work_size<<", \t ( buffer size) data_size = "<<data_size<<", \t ( num threards launched)  num_threads="<<num_threads<<", \t ( num threards run) num_threads_int = "<<num_threads_int<<", \t  ( ";
// 	int i=0;
// 	for ( ; i< data_size; i++) cout << ", " << atomic_test_output[i];
// 	cout << ") \t i="<< i<< "\n\n" << flush;
//
//
//
// }
*/
