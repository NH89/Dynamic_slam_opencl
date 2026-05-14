#include "RunCL.hpp"
// rho_sq(..) -> rho, rho sq, params_incr for use with paras_inv_H in LK optimizarion.		Uses: SE3_tracking, camera_matrix, lens_distortion.  Possibly ST3 patches for depth, rel_vel, rel_accel.

// patch kernels, but not using look up table /////////////////////////////////////////////////////////////////////////////////////////
void RunCL::rho_sq_set_params( uint out_block_size ){
																	// Needs 16 elements of local mem per 32x32 patch, to pass data between threads in recursive square reduction.
																	// Needs 32 elem array of private mem per thread.
																	// Writes answer to SE3_rho_map_mem, BUT as float2
	string fname					= "RunCL::rho_sqset_params( ..)";
	int local_verbosity_threshold	= V_RUNCL_RHO_SQ;
	cl_kernel	kernel 				= rho_sq_kernel;
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::rho_sqset_params( ..)_chk0 .##################################################################"<<flush;
																																				cout<<endl<<endl <<",  out_block_size="	<<out_block_size << flush;
																																			}
																														if (fmod(device_work_size_multiple, patch_size)!=0)   {
																															cerr <<"\nRunCL::rho_sqset_params( ..)  Error: fmod(device_work_size_multiple, patch_size) != 0 \n"<<flush;
																															exit_(0);
																														}
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::rho_sqset_params( ..)_chk_1 "<<flush;}

	cl_int k_wg_info =  clGetKernelWorkGroupInfo(
							kernel,											//cl_kernel kernel,
							deviceId,										//cl_device_id device,
							CL_KERNEL_WORK_GROUP_SIZE,						//cl_kernel_work_group_info param_name,
							sizeof(rho_sq_params.kernel_workgroup_size),	//size_t param_value_size,
							&rho_sq_params.kernel_workgroup_size,			//void* param_value,
							NULL											//size_t* param_value_size_ret
						);																								if( k_wg_info!=CL_SUCCESS){ 	cerr<<"\nRunCL::rho_sqset_params() ( k_wg_info!=CL_SUCCESS)"		<<k_wg_info<<" = "<<checkerror(k_wg_info)			<<endl<<flush;	exit_(1); }

	cl_int device_info_1 = clGetDeviceInfo(
							deviceId,										//cl_device_id device,
							CL_DEVICE_MAX_WORK_ITEM_SIZES,					//cl_device_info param_name,
							sizeof(rho_sq_params.device_max_workitem_sizes),//size_t param_value_size,
							rho_sq_params.device_max_workitem_sizes,		//void* param_value,
							NULL											//size_t* param_value_size_ret
	);																													if( device_info_1!=CL_SUCCESS){ cerr<<"\nRunCL::rho_sqset_params() ( device_info_1!=CL_SUCCESS)"	<<device_info_1<<" = "<<checkerror(device_info_1)	<<endl<<flush;	exit_(1); }

	cl_int device_info_2 = clGetDeviceInfo(
							deviceId,										//cl_device_id device,
							CL_DEVICE_MAX_COMPUTE_UNITS,					//cl_device_info param_name,
							sizeof(rho_sq_params.device_max_compute_units),	//size_t param_value_size,
							&rho_sq_params.device_max_compute_units,		//void* param_value,
							NULL											//size_t* param_value_size_ret
	);																													if( device_info_2!=CL_SUCCESS){ cerr<<"\nRunCL::rho_sqset_params() ( device_info_2!=CL_SUCCESS)"	<<device_info_2<<" = "<<checkerror(device_info_2)	<<endl<<flush;	exit_(1); }
	rho_sq_params.max_workgroup_size	= min(rho_sq_params.kernel_workgroup_size, rho_sq_params.device_max_workitem_sizes[0] );

																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::rho_sqset_params( ..)_chk_2 "<<flush;
																																				cout << "\n"
																																					<<",  device_max_compute_units="				<<device_max_compute_units
																																					//<<",  max_workgroup_size="					<<max_workgroup_size
																																					//<<",  kernel_workgroup_size="					<<kernel_workgroup_size
																																					<<",  device_max_workitem_sizes[0,1,2]={"		<<device_max_workitem_sizes[0]<<",  "
																																																	<<device_max_workitem_sizes[1]<<",  "
																																																	<<device_max_workitem_sizes[2]<<"}"
																																					<< flush;
																																			}
}

void RunCL::rho_sq( uint out_block_size, uint iter, uint frame_idx, uint layer, cl_mem k2k_buf, uint num_DoF, string calling_fn ){	// To be launched with 1 thread per col for 32x32 patches, and an integer multiple of 32 threads.
																	// Needs 16 elements of local mem per 32x32 patch, to pass data between threads in recursive square reduction.
																	// Needs 32 elem array of private mem per thread.
																	// Writes answer to SE3_rho_map_mem, BUT as float2
	string fname					= "RunCL::rho_sq( ..)";
	int local_verbosity_threshold	= V_RUNCL_RHO_SQ;
	cl_kernel	kernel 				= rho_sq_kernel;
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::rho_sq( ..)_chk0 .##################################################################"<<flush;
																																				cout<<endl<<endl
																																					<<"   dataset_frame_num="	<<dataset_frame_num
																																					<<",  out_block_size="		<<out_block_size
																																					<<",  iter="				<<iter
																																					<<",  layer="				<<layer
																																					<<",  frame_idx="			<<frame_idx
																																					<< flush;
																																				float pose_ary[16];
																																				ReadOutput( (uchar*)pose_ary, pose_buf, sizeof(float)*16, 0);		// ReadOutput(uchar* outmat, cl_mem buf_mem, size_t data_size, size_t offset/*=0*/)
																																				PRINT_FLOAT_16(pose_ary, gpu buf);

																																				float k2kbuf_ary[16];
																																				ReadOutput( (uchar*)k2kbuf_ary, k2kbuf, sizeof(float)*16, 0);		// ReadOutput(uchar* outmat, cl_mem buf_mem, size_t data_size, size_t offset/*=0*/)
																																				PRINT_FLOAT_16(k2kbuf_ary, gpu buf);

																																				for (uint i=0; i<5 ;i++){
																																					cout<<"\n\n## current_frames[ current_frames_idx["<<i<<"] ].frame_num = "<< current_frames[ current_frames_idx[i] ].dataset_frame_num << flush;
																																					PRINT_FLOAT_16( current_frames[ current_frames_idx[i] ].pose,		);
																																					PRINT_MATX44F(	current_frames[ current_frames_idx[i] ].pose_gt,	);
																																					PRINT_FLOAT_16( current_frames[ current_frames_idx[i] ].k2k_0to1_est,	);
																																				}
																																			}
 	_clEnqueueFillBuffer( uload_queue, SE3_rho_map_mem, 	&zero_flt, sizeof( float), 0, 			  2*mm_size_bytes_C1, 	fname);				//_clEnqueueWriteBuffer( uload_queue, k2kbuf, CL_FALSE, 0, local_num_samples*16*sizeof( float), k2k_3_16_[start_sample_idx], fname);
// //	_clEnqueueFillBuffer( uload_queue, SE3_weight_map_mem, 	&zero, sizeof( float), 0, num_SE3_DoF*2*mm_size_bytes_C1, 	fname);
 	_clEnqueueFillBuffer( uload_queue, SE3_incr_map_mem, 	&zero_flt, sizeof( float), 0, 			  2*mm_size_bytes_C1, 	fname);

	uint				read_rows					= MipMap[layer * 8 + MiM_READ_ROWS] ;
	uint				read_cols					= MipMap[layer * 8 + MiM_READ_COLS] ;
	uint				rows_blocks					= ceil( (float)  read_rows / patch_size );
	uint				cols_blocks					= ceil( (float)  read_cols / patch_size );
	uint				cols_per_row				= cols_blocks  * patch_size;
	uint				patches_required			= cols_blocks  * rows_blocks;

	uint				patches_per_compute_uint	= ceil( (float)patches_required / device_max_compute_units );
	uint				blocks_per_k_wg_size		= rho_sq_params.max_workgroup_size			/ device_work_size_multiple;
						patches_per_compute_uint	= min( patches_per_compute_uint,  blocks_per_k_wg_size );

	uint				blocks_required				= ceil( (float)patches_required / patches_per_compute_uint );
	size_t				local_work_size_[1]			= { patches_per_compute_uint	* patch_size };
	size_t				threads_to_launch			= blocks_required 				* local_work_size_[0];									// TO DO precompute an array for this function. ? where to store
																																			// ? Have a subclass and object for each kernel ?
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::rho_sq( ..)_chk_3 "<<flush;
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
	_clSetKernelArg( kernel, 0, sizeof( uint),									&frame_idx,	 											fname);		//__private		uint 		layer,					//0
	_clSetKernelArg( kernel, 1, sizeof( uint),									&layer,	 												fname);		//__private		uint 		layer,					//0
	_clSetKernelArg( kernel, 2, sizeof( uint),									&cols_per_row,											fname);		//__private		uint 		cols_per_row,			//1
	_clSetKernelArg( kernel, 3, sizeof( uint),									&out_block_size,										fname);		//__private		uint 		out_block_size,			//2
	_clSetKernelArg( kernel, 4, sizeof( uint),									&num_DoF,												fname);		//__private		float2		delta_SE3,				//3

	_clSetKernelArg( kernel, 5, sizeof( cl_mem), 								&mipmap_buf,											fname);		//__constant	uint8*		mipmap_params,			//3
	_clSetKernelArg( kernel, 6, sizeof( cl_mem), 								&uint_param_buf,										fname);		//__constant	uint*		uint_params,			//4
	_clSetKernelArg( kernel, 7, sizeof( cl_mem), 								&fp32_param_buf,										fname);		//__constant	float*		fp32_params,			//5
	_clSetKernelArg( kernel, 8, sizeof( cl_mem), 								&k2k_buf,												fname);		//__constant	float16*	inv_k2k,				//6		// transforms for 4 past frames

	_clSetKernelArg( kernel, 9, sizeof( cl_mem),								&current_frames[current_frames_idx[0]].img_buf,			fname);		//__global		float4*		img_cur,				//7		// multiple past frames. NB retain frames at powers of 2, and vary starting power plus num franes.
	_clSetKernelArg( kernel,10, sizeof( cl_mem), 								&current_frames[current_frames_idx[1]].img_buf,			fname);		//__global		float4*		img_past_0,				//8
	_clSetKernelArg( kernel,11, sizeof( cl_mem), 								&current_frames[current_frames_idx[2]].img_buf,			fname);		//__global		float4*		img_past_1,				//9
	_clSetKernelArg( kernel,12, sizeof( cl_mem), 								&current_frames[current_frames_idx[3]].img_buf,			fname);		//__global		float4*		img_past_2,				//10
	_clSetKernelArg( kernel,13, sizeof( cl_mem), 								&current_frames[current_frames_idx[4]].img_buf,			fname);		//__global		float4*		img_past_3,				//11
																																																				// NB GT_depth loaded to depth_mem by void RunCL::loadFrameData( ..)
	_clSetKernelArg( kernel,14, sizeof( cl_mem), 								&depth_mem,												fname);		//__global		float2* 	depth_map,				//12	// current frame depth, now stored as inv_depth
	_clSetKernelArg( kernel,15, sizeof( cl_mem), 								&g1mem,													fname);		//__global		float8* 	g1p,					//13	// current frame g1mem
	_clSetKernelArg( kernel,16, sizeof( cl_mem), 								&SE3_grad_map_mem,										fname);		//__global 		float8*		SE3_grad_map_cur_frame,	//14

	_clSetKernelArg( kernel,17, sizeof( cl_mem), 								&current_frames[current_frames_idx[0]].r_vel_buf,		fname);		//__global		float4*		img_cur,				//15	// multiple past frames.
	_clSetKernelArg( kernel,18, sizeof( cl_mem), 								&current_frames[current_frames_idx[1]].r_vel_buf,		fname);		//__global		float4*		img_past_0,				//16
	_clSetKernelArg( kernel,19, sizeof( cl_mem), 								&current_frames[current_frames_idx[2]].r_vel_buf,		fname);		//__global		float4*		img_past_1,				//17
	_clSetKernelArg( kernel,20, sizeof( cl_mem), 								&current_frames[current_frames_idx[3]].r_vel_buf,		fname);		//__global		float4*		img_past_2,				//18
	_clSetKernelArg( kernel,21, sizeof( cl_mem), 								&current_frames[current_frames_idx[4]].r_vel_buf,		fname);		//__global		float4*		img_past_3,				//19
	//output
	_clSetKernelArg( kernel,22, sizeof( cl_mem), 								&SE3_rho_map_mem, 										fname);		//__global		float2* 	Rho_,					//20	// { sum rho^2 ,  count of valid pixels used } Writen to dense patches.
	_clSetKernelArg( kernel,23, sizeof( cl_float2)*local_work_size,				NULL, 													fname);		//__local		float2*		local_rho				//21	// float2 local_rho[ local_work_size/2 ]  hence sizeof( float)*local_work_size.

	_clSetKernelArg( kernel,24, sizeof( cl_mem), 								&SE3_incr_map_mem,										fname);		//__global 		float2*		SE3_incr_map_,			//22
	_clSetKernelArg( kernel,25, sizeof( cl_float2)*local_work_size*num_SE3_DoF,	NULL,													fname);		//__local 		float2*		local_SE3_incr			//23
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
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::rho_sq( ..)_chk_6 ."<<flush;
																																				stringstream ss;
																																				ss << "_ds-framenum"<<dataset_frame_num<<"_img_layer"<<layer<<"_iter"<<iter<<"_out_bock_size"<<out_block_size<<"_rho_sq()"<<calling_fn;
																																				stringstream ss_path;
																																				bool show				= false;
																																				float max_range			= -1;
																																				uint vol_layers			= 1;
																																				bool old_tiff			= tiff;
																																				tiff					= true;
																																				DownloadAndSave_2Channel_volume(  SE3_rho_map_mem,		ss.str( ), paths.at( "SE3_rho_map_mem"),	2*mm_size_bytes_C1,   mm_Image_size,	CV_32FC2, show, max_range,	1);
																																				float max_range_ = 0;
																																				uint offset_depth_bytes	=0;
																																				DownloadAndSave_2Channel( depth_mem, ss.str(),  paths.at("depth_mem"),  2*mm_size_bytes_C1, mm_Image_size, CV_32FC2, 	show , max_range_, offset_depth_bytes);	cout << "\nDownloadAndSave_2Channel(.. depth_mem ..)\n"<<flush;
																																				for(int i=0; i<5; i++){
																																					stringstream ss_; ss_<< ss.str()<<"_current_frames_idx"<<i;
																																					DownloadAndSave_3Channel( current_frames[current_frames_idx[i]].img_buf,  ss_.str(),   paths.at("imgmem"),   	mm_size_bytes_C4,   mm_Image_size,   CV_32FC4, 	show , max_range_);
																																				}
																																				//DownloadAndSave_2Channel_volume(  SE3_weight_map_mem,	ss.str( ), paths.at( "SE3_weight_map_mem"),	2*mm_size_bytes_C1,   mm_Image_size,	CV_32FC2, show, max_range,	vol_layers);
																																				DownloadAndSave_2Channel_volume(  SE3_incr_map_mem,		ss.str( ), paths.at( "SE3_incr_map_mem"),	2*mm_size_bytes_C1,   mm_Image_size,	CV_32FC2, show, max_range,	vol_layers);
																																				tiff = old_tiff;

																																				cout<<"\n\nRunCL::rho_sq( ..) finished ########################################################################"<<endl<< flush;
																																			}
}

void RunCL::reduce_patch_Rho ( uint out_block_size, uint iter, uint layer, uint num_DoF )									// NB good for images upto 640x480 layer zero, above that need a patch kernel approach to ensure each DoF fits in 1 workgroup. see device_work_size_multiple
{
	string fname = "RunCL::reduce_patch_Rho( ..)";
	int local_verbosity_threshold = V_RUNCL_REDUCE_PATCH_RHO;
																																		if( verbosity>local_verbosity_threshold) {cout<<"\nRunCL::reduce_patch_Rho( ..)_chk_1 _____________________"<<flush;
																																		cout<<" layer = "<<layer<<flush;
																																			// if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::update_SE3( ..)_chk0 .##################################################################"<<flush;
																																			// 	float pose_update_ary[6];
																																			// 	ReadOutput( (uchar*)pose_update_ary, pose_update_buf, sizeof(float)*6, 0);	//ReadOutput(uchar* outmat, cl_mem buf_mem, size_t data_size, size_t offset/*=0*/)
																																			// 	cout<<"\n pose_update_ary = {"; for (int i=0; i<6; i++){ cout<<pose_update_ary[i]<<", "; }cout<<"}"<<flush;
																																		}
	cl_kernel		kernel 				= reduce_patch_Rho_kernel;																			//NB call just one workgroup to sum the whole image maps from the patch kernel.
	uint			read_rows			= MipMap[layer * 8 + MiM_READ_ROWS] ;
	uint			read_cols			= MipMap[layer * 8 + MiM_READ_COLS] ;																// NB the largest (layer 0) read_cols, is the unreduced size of the input image. (here 640x480)
	uint			rows_blocks			= read_rows/patch_size ; //ceil( (float) read_rows/patch_size );								//0 // num rows in the fully reduced map  640/32=20 => 32 cols_blocs.  32*6=192 which would fit IFF groupsize >=256.
	uint			cols_blocks			= ceil( (float) read_cols/patch_size );															//1 // num cols in the fully reduced map. 1920x1080 1920/32=60 => 64 cols_blocs
																											// threads_per_DoF must be the first 2^n > cols per SE3 patch.
	uint			threads_per_DoF		= powf(2,ceil( log2((float)cols_blocks) )); 						// 10 layer 1 =>  pown(2,ciel(log2(10.0f) ))=16; 6*16=96.      // * rows_blocks  ;//	8x10=80 layer1 => 96 threads to launch?		// num pixels in fully reduced map. Req per SE3 DoF.
																											//cout <<" cols_blocks = "<<cols_blocks

	float 			DoF_per_workgroup	= device_work_size_multiple / threads_per_DoF;
																																		if( verbosity>local_verbosity_threshold) {cout<<"\nRunCL::reduce_patch_Rho( ..)_chk_1.1"
																																			<<"   log2((float)cols_blocks) = "		<<log2((float)cols_blocks)
																																			<<"   threads_per_DoF = "				<<threads_per_DoF
																																			<<"   _device_work_size_multiple = "	<<device_work_size_multiple
																																			<<"   * SE3_DoF = "						<<num_DoF
																																			<<") / DoF_per_workgroup = "			<<DoF_per_workgroup <<flush;
																																		}
	size_t			threads_required	= ceil((device_work_size_multiple * num_DoF ) / DoF_per_workgroup);	// NB device_work_size_multiple is usually a poer of 2, DoF_per_workgroup will also be a power of 2.
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

	cl_command_queue	queue_to_call		= m_queue;
	cl_int				res, status;
	cl_event			ev;
																									//auto step_0 = high_resolution_clock::now();
	res		= clEnqueueNDRangeKernel(queue_to_call, kernel, 1, 0, &threads_to_launch, &device_work_size_multiple, 0, NULL, &ev); 	// run mipmap_float4_kernel, NB wait for own previous iteration.
																									if (res    != CL_SUCCESS)	{ cout << "\nres = " << checkerror(res) <<"\n"<<flush; exit_(res);}
	status 	= clFlush(queue_to_call);																if (status != CL_SUCCESS)	{ cout << "\nRunCL::reduce_patch_Rho( ..) call_kernel( cl_kernel "<<kernel<<",  clFlush(queue_to_call) status  = "<<status<<" "<< checkerror(status) <<"\n"<<flush; exit_(status);}
																									//auto step_1 = high_resolution_clock::now();
	status 	= clWaitForEvents(1, &ev);																if (status != CL_SUCCESS)	{ cout << "\nRunCL::reduce_patch_Rho( ..) call_kernel( cl_kernel "<<kernel<<") final,  clWaitForEventsh(1, &ev) ="<<status<<" "<<checkerror(status)  <<"\n"<<flush; exit_(status);}
																									//auto step_2 = high_resolution_clock::now();
	clReleaseEvent(ev);
																																			// /*
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::reduce_patch_Rho( ..)_chk_2 . "<<flush;//<<\
																																				// "Execution time = "<<  duration_cast<microseconds>(step_1 - step_0).count() \
																																				// <<" , "<<duration_cast<microseconds>(step_2 - step_1).count() <<flush;

																																				// float old_result_arry[6];
																																				// ReadOutput( (uchar*)old_result_arry, old_results_buf, sizeof(float), 0);	//ReadOutput(uchar* outmat, cl_mem buf_mem, size_t data_size, size_t offset)  //  size_t offset=0
																																				// cout<<"\n old_result_arry = " <<old_result_arry[0]<<flush;

																																				stringstream ss;
																																				ss << "_ds-framenum"<<dataset_frame_num<<"_img_layer"<<layer<<"_iter"<<iter<<"_out_bock_size"<<out_block_size<<"_reduce_rho_sq()";
																																				stringstream  ss_path;
																																				bool show				= false;
																																				float max_range			= -1;
																																				uint vol_layers			= 1;
																																				bool exception_tiff 	= false;
																																				bool display			= false;	//cout << "\nRunCL::rho_sq( ..)_chk_7   display="<< display<< endl << flush;
																																				bool old_tiff			= tiff;
																																				tiff					= true;
																																				DownloadAndSave_2Channel_volume(  SE3_rho_map_mem,		ss.str( ), paths.at( "SE3_rho_map_mem"),	2*mm_size_bytes_C1,   mm_Image_size,	CV_32FC2, show, max_range,	1);
																																				//DownloadAndSave_2Channel_volume(  SE3_weight_map_mem,	ss.str( ), paths.at( "SE3_weight_map_mem"),	2*mm_size_bytes_C1,   mm_Image_size,	CV_32FC2, show, max_range,	vol_layers);
																																				DownloadAndSave_2Channel_volume(  SE3_incr_map_mem,		ss.str( ), paths.at( "SE3_incr_map_mem"),	2*mm_size_bytes_C1,   mm_Image_size,	CV_32FC2, show, max_range,	vol_layers);
																																				tiff = old_tiff;

																																				cout<<"\nRunCL::reduce_patch_Rho( ..)_finished _____________________"<<flush;
																																			}
																																			// */
}


void RunCL::update_k2k_cpu( uint layer ){
	constexpr int		local_verbosity_threshold	= V_RUNCL_UPDATE_K2K;
																																	if( verbosity>local_verbosity_threshold-3) { cout<<"\n\nRunCL::update_k2k_cpu( ..)_chk_0 . ################################"<< flush;
																																		cout << "\nlayer = "	<< layer 	<<endl<<flush;
																																		Matx44f	current_frame_pose_gt	=	current_frames[ current_frames_idx[0] ].pose_gt;			//PRINT_MATX44F( current_frame_pose_gt, );
																																		Matx44f	previous_frame_pose_gt	=	current_frames[ current_frames_idx[1] ].pose_gt;			//PRINT_MATX44F( previous_frame_pose_gt, );

																																		Matx44f pose_gt					=	previous_frame_pose_gt	*	current_frame_pose_gt.inv();	//PRINT_MATX44F( pose_gt, );
																																		//Matx16f pose_gt_algebra			=	PToLie( pose_gt );											//PRINT_MATX16F( pose_gt_algebra, );

																																		Matx44f	pose					=	ReadOutput_44f( 					pose_buf );				//PRINT_MATX44F( pose,	from pose_buf );	PRINT_MATX16F( PToLie(pose),);
																																		Matx44f pose_update_gt			=	pose.inv() * pose_gt;										/* PRINT_MATX44F( pose_update_gt, );*/		PRINT_MATX16F( PToLie(pose_update_gt),);
																																	}
	ReadOutput( (uchar*)&se3_rho_result.Rho,				SE3_rho_map_mem, 	sizeof(cl_float2),		32*sizeof(cl_float2)	);
	ReadOutput(	(uchar*)se3_rho_result.param_incr_arry,		SE3_incr_map_mem,	6*sizeof(cl_float2),	32*sizeof(cl_float2)	);
																																	if( verbosity>local_verbosity_threshold) { cout<<"\n\nRunCL::update_k2k_cpu( ..)_finished ###############################"<< flush;
																																		cout<<"\nse3_rho_result.param_incr_arry=("<<flush;
																																		cout	<<se3_rho_result.param_incr_arry[0]<<","<<se3_rho_result.param_incr_arry[1]<<"),("<<flush;
																																		cout	<<se3_rho_result.param_incr_arry[2]<<","<<se3_rho_result.param_incr_arry[3]<<"),("<<flush;
																																		cout	<<se3_rho_result.param_incr_arry[4]<<","<<se3_rho_result.param_incr_arry[5]<<"),("<<flush;
																																		cout	<<se3_rho_result.param_incr_arry[6]<<","<<se3_rho_result.param_incr_arry[7]<<"),("<<flush;
																																		cout	<<se3_rho_result.param_incr_arry[8]<<","<<se3_rho_result.param_incr_arry[9]<<"),("<<flush;
																																		cout	<<se3_rho_result.param_incr_arry[10]<<","<<se3_rho_result.param_incr_arry[11]<<")"<<flush;
																																	}
}

void RunCL::get_rho_result ( Rho_result &rho_result, uint layer, uint num_DoF){	// NB must pass struct by reference to send data to the calling fn.
	constexpr int		local_verbosity_threshold	= V_RUNCL_UPDATE_K2K;
																																	if( verbosity>local_verbosity_threshold-3) { cout<<"\n\nRunCL::get_rho_result( ..)_chk_0 . ################################"<< flush;
																																		cout << "\nlayer = "	<< layer 	<<endl<<flush;
																																		Matx44f	current_frame_pose_gt	=	current_frames[ current_frames_idx[0] ].pose_gt;			//PRINT_MATX44F( current_frame_pose_gt, );
																																		Matx44f	previous_frame_pose_gt	=	current_frames[ current_frames_idx[1] ].pose_gt;			//PRINT_MATX44F( previous_frame_pose_gt, );

																																		Matx44f pose_gt					=	previous_frame_pose_gt	*	current_frame_pose_gt.inv();	//PRINT_MATX44F( pose_gt, );
																																		//Matx16f pose_gt_algebra			=	PToLie( pose_gt );											//PRINT_MATX16F( pose_gt_algebra, );

																																		Matx44f	pose					=	ReadOutput_44f( 					pose_buf );				//PRINT_MATX44F( pose,	from pose_buf );	PRINT_MATX16F( PToLie(pose),);
																																		Matx44f pose_update_gt			=	pose.inv() * pose_gt;										/* PRINT_MATX44F( pose_update_gt, );*/		PRINT_MATX16F( PToLie(pose_update_gt),);
																																	}
	ReadOutput( (uchar*)&rho_result.Rho,				SE3_rho_map_mem, 	sizeof(cl_float2),			sizeof(cl_float2)	);
	ReadOutput(	(uchar*)rho_result.param_incr_arry,		SE3_incr_map_mem,	num_DoF*sizeof(cl_float2),	32*sizeof(cl_float2)	);
																																	if( verbosity>local_verbosity_threshold) { cout<<"\n\nRunCL::get_rho_result( ..)_finished ###############################"<< flush;
																																		cout<<"\nrho_result.Rho="<<rho_result.Rho.x<<","<<rho_result.Rho.y<<flush;
																																		cout<<"\nrho_result.param_incr_arry=("<<flush;
																																		cout	<<rho_result.param_incr_arry[0]<<","<<rho_result.param_incr_arry[1]<<"),("<<flush;
																																		cout	<<rho_result.param_incr_arry[2]<<","<<rho_result.param_incr_arry[3]<<"),("<<flush;
																																		cout	<<rho_result.param_incr_arry[4]<<","<<rho_result.param_incr_arry[5]<<"),("<<flush;
																																		cout	<<rho_result.param_incr_arry[6]<<","<<rho_result.param_incr_arry[7]<<"),("<<flush;
																																		cout	<<rho_result.param_incr_arry[8]<<","<<rho_result.param_incr_arry[9]<<"),("<<flush;
																																		cout	<<rho_result.param_incr_arry[10]<<","<<rho_result.param_incr_arry[11]<<")"<<flush;
																																	}
}












void RunCL::update_k2k( uint layer, float delta_theta, float delta, Matx44f GT_pose )		// To be GPU kernel to iteratively update model..... to be written.
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
	_clSetKernelArg( kernel,  2, sizeof( cl_mem),						&SE3_hessian_map_mem,	fname);							// __private	uint	Hessian_map,			//1
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


// end of patch kernels //
////////////////////////////////////////////////////////////////////////////////////
