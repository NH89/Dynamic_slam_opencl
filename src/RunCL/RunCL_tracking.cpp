#include "RunCL.hpp"

void RunCL::precom_param_maps( float SE3_k2k[6*16]){ //  Compute maps of pixel motion for each SE3 DoF, and camera params // Derived from RunCL::mipmap
	string fname = "RunCL::precom_param_maps( ..)";
	int local_verbosity_threshold = V_RUNCL_PRECOM_PARAM_MAPS;
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::precom_param_maps( float SE3_k2k[6*16])_chk_0 "<<flush;}
	cv::Mat depth		= cv::Mat::ones ( mm_height, mm_width, CV_32FC1);																	// NB must recompute translation maps at run time. NB parallax motion is proportional to inv depth.
	float mid_depth 	= ( fp32_params[MAX_INV_DEPTH] + fp32_params[MIN_INV_DEPTH])/2.0;                                                   // TODO fix : depthmap not used as a kernel arg. NB want to match scale of depth range, but ? parallax may vary.
	depth 				*= mid_depth;

	_clEnqueueWriteBuffer( uload_queue, SE3_k2kbuf,	CL_FALSE, 0, 6*16*sizeof( float), SE3_k2k,		fname);
	_clEnqueueWriteBuffer( uload_queue, depth_mem,	CL_FALSE, 0, mm_size_bytes_C1,	 depth.data,	fname);

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
																																														DownloadAndSave_2Channel_volume( SE3_map_mem, ss.str( ), paths.at( "SE3_map_mem"), mm_size_bytes_C1*2, mm_Image_size, CV_32FC2, false, 1.0, 6 /*SE3, 6DoF */);
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

/*
// void RunCL::initialize_tracking_depthmap( float initial_depth){
// 	string fname = "RunCL::initialize_tracking_depthmap( float initial_depth)";
// 	int local_verbosity_threshold = V_RUNCL_INITIALIZE_TRACKING_DEPTHMAP;
//
// 	_clEnqueueFillBuffer( uload_queue, depth_mem, 	&initial_depth, sizeof( float), 0, mm_size_bytes_C1, 	fname);
// 																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::initialize_tracking_depthmap( ..)_finished ."<<flush;}
// }
*/

void RunCL::se3_rho_sq( const uint local_num_samples,  const uint start_sample_idx,  float Rho_sq_results[tracking_tot_samples][max_mipmap_layers][tracking_num_colour_channels], const float count[4], uint start, uint stop,  float k2k_3_16_[tracking_tot_samples][16]   ){
	string fname = "RunCL::se3_rho_sq( ..)";
	int local_verbosity_threshold = V_RUNCL_SE3_RHO_SQ;
	const int num_samples  = 1; //tracking_num_samples;
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
	_clEnqueueFillBuffer( uload_queue, SE3_rho_map_mem, 		&zero, sizeof( float), 0, num_samples*2*mm_size_bytes_C4, 	fname);
	_clEnqueueFillBuffer( uload_queue, se3_sum_rho_sq_mem, 	&zero, sizeof( float), 0, num_samples*pix_sum_size_bytes, 	fname);
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::se3_rho_sq( ..)_chk0.7 "<<flush;}
																																			// NB GT_depth loaded to depth_mem by void RunCL::loadFrameData( ..)
	const uint wg_divisor =2;  // 1,2,4,8  reduction in workgroup size for this kernel.
	//input integers
	//      __private	 uint layer, set in mipmap_call_kernel( ..) below                                                                         __private	    uint	    layer,		                    //0
	_clSetKernelArg( se3_rho_sq_kernel, 1, sizeof( size_t), &local_num_samples, 	fname);														//__private		uint		local_num_samples				//1
	_clSetKernelArg( se3_rho_sq_kernel, 2, sizeof( size_t), &se3_sum_size, 			fname);														//__private		uint		pix_sum_size_bytes,				//2
	//input buffers
    _clSetKernelArg( se3_rho_sq_kernel, 3, sizeof( cl_mem), &mipmap_buf, 			fname);														//__constant    uint*	    mipmap_params,	                //3
	_clSetKernelArg( se3_rho_sq_kernel, 4, sizeof( cl_mem), &uint_param_buf, 		fname);														//__constant	uint*		uint_params,					//4
	_clSetKernelArg( se3_rho_sq_kernel, 5, sizeof( cl_mem), &fp32_param_buf, 		fname);														//__constant	float*		fp32_params						//5
	_clSetKernelArg( se3_rho_sq_kernel, 6, sizeof( cl_mem), &k2kbuf, 				fname);														//__global		float* 		k2k,							//6		// TODO keyframe2K
	_clSetKernelArg( se3_rho_sq_kernel, 7, sizeof( cl_mem), &keyframe_imgmem, 		fname);														//__global 		float4*		keyframe_imgmem,				//7		// TODO need keyframe mipmap   keyframe_imgmem , keyframe_depth_mem
	_clSetKernelArg( se3_rho_sq_kernel, 8, sizeof( cl_mem), &imgmem, 				fname);														//__global 		float4*		imgmem,							//8
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
																																			// void RunCL::DownloadAndSave_3Channel_volume( cl_mem buffer,   std::string count,   boost::filesystem::path folder,   size_t image_size_bytes,   cv::Size size_mat,   int type_mat,   bool show,   float max_range,   uint vol_layers,    bool exception_tiff /*=false*/,   float iter,   bool display)

	for ( int sample = 0; sample<local_num_samples; sample++){
		read_Rho_sq(  Rho_sq_results[sample + start_sample_idx], sample  );
	}
}


void RunCL::estimateSE3_LK( float local_k2k[16], float SE3_results[max_mipmap_layers][num_SE3_DoF][tracking_num_colour_channels], float SE3_weights_results[max_mipmap_layers][num_SE3_DoF][tracking_num_colour_channels], float Rho_sq_results[max_mipmap_layers][4], int count, uint start, uint stop){ //estimateSE3_LK( ); 	( uint start=0, uint stop=8)			// TODO replace arbitrary fixed constant with a const uint variable in the header...
	string fname = "RunCL::estimateSE3_LK( ..)";
	int local_verbosity_threshold = V_RUNCL_ESTIMATESE3_LK;
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
	_clSetKernelArg( se3_lk_grad_kernel, 6, sizeof( cl_mem), &imgmem, 						fname);												//__global 		float4*		imgmem,							//6
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
	mipmap_call_kernel( se3_lk_grad_kernel, m_queue, start, stop, false, local_work_size/wg_divisor); 										// reduced worksize to allow for local memory limit 4kb on rtx 3030
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::estimateSE3_LK( ..)_chk_4 ."<<flush;
																																				stringstream ss;	ss << dataset_frame_num << "_iter_"<< count << "_estimateSE3_LK_";
                                                                                                                                                stringstream ss_path;
																																				bool show 				= false;
																																				bool display 			= false;
																																				bool exception_tiff 	= false;
																																				uint vol_layers 		= 1;
																																				float max_range 		= -1; 		// i.e. gray = zero.

																																				DownloadAndSave_3Channel_volume(  SE3_rho_map_mem,  	ss.str( ), paths.at( "SE3_rho_map_mem"),  	mm_size_bytes_C4, mm_Image_size, CV_32FC4, show, max_range, vol_layers, exception_tiff, count, display );
																																				DownloadAndSave_3Channel_volume(  SE3_weight_map_mem, 	ss.str( ), paths.at( "SE3_weight_map_mem"), 	mm_size_bytes_C4, mm_Image_size, CV_32FC4, show, max_range, 6, 			exception_tiff, count, display );
																																				DownloadAndSave_3Channel_volume(  SE3_incr_map_mem, 	ss.str( ), paths.at( "SE3_incr_map_mem"), 	mm_size_bytes_C4, mm_Image_size, CV_32FC4, show, max_range, 6, 			exception_tiff, count, display );
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
																																						for ( int chan=0; chan<3; chan++){	cout << ",   \t" << Rho_sq_results[layer][chan] / ( Rho_sq_results[layer][3]  *  img_stats[IMG_VAR+chan]  );
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

void RunCL::atomic_test1( ){
	string fname = "RunCL::atomic_test1( )";
	int local_verbosity_threshold = V_RUNCL_ATOMIC_TEST1;

	const int data_size = 4*local_work_size;
	const size_t num_threads = 2 * local_work_size;
	int num_threads_int = 1.5*local_work_size;

	int zero  = 0;
	_clEnqueueFillBuffer( uload_queue, atomic_test1_buf, 	&zero, sizeof( int), 0, data_size*sizeof( int), 	fname);

	_clSetKernelArg( atomic_test1_kernel, 0, sizeof( int), 		&num_threads_int, 	fname);
	_clSetKernelArg( atomic_test1_kernel, 1, sizeof( cl_mem), 	&atomic_test1_buf, 	fname);

	_clEnqueueNDRangeKernel( m_queue, atomic_test1_kernel, 1, 0, &num_threads, &local_work_size,fname);

	int atomic_test_output[ data_size ];
	for ( int i = 0; i<data_size; i++) atomic_test_output[i] = -1;
	cl_event readEvt;
	cl_int status;
	status = clEnqueueReadBuffer( 	dload_queue,			// command_queue
									atomic_test1_buf,		// buffer
									CL_FALSE,				// blocking_read
									0,						// offset
									data_size*sizeof( int),	// size
									atomic_test_output,		// pointer
									0,						// num_events_in_wait_list
									NULL,					// event_waitlist				needs to know about preceeding events:
									&readEvt				// event
								 );																												if ( status != CL_SUCCESS) { cout << "\nclEnqueueReadBuffer( ..) status=" << checkerror( status) <<"\n"<<flush; exit_( status);}
	_cl_flush_finish( dload_queue, fname);

	cout << "\n\n void RunCL::atomic_test1( ): \t  local_work_size = "<<local_work_size<<", \t ( buffer size) data_size = "<<data_size<<", \t ( num threards launched)  num_threads="<<num_threads<<", \t ( num threards run) num_threads_int = "<<num_threads_int<<", \t  ( ";
	int i=0;
	for ( ; i< data_size; i++) cout << ", " << atomic_test_output[i];
	cout << ") \t i="<< i<< "\n\n" << flush;
}


void RunCL::atomic_test2( ){
	string fname = "RunCL::atomic_test2( )";
	int local_verbosity_threshold = V_RUNCL_ATOMIC_TEST2;

	cl_int res, status;
	cl_event ev, writeEvt;
	const 	int 	data_size 		= 4   *local_work_size;
	const 	size_t 	num_threads 	= 2   *local_work_size;
			int 	num_threads_int = 1.5 *local_work_size;




	float number  = 0.674;
	status = clEnqueueFillBuffer( uload_queue, atomic_test2_buf, &number, sizeof( int), 0, data_size*sizeof( float), 	0, NULL, &writeEvt);	if ( status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror( status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.3\n" << endl;exit_( status);}

	clFlush( uload_queue); status = clFinish( uload_queue); 																					if ( status != CL_SUCCESS)	{ cout << "\nclFinish( uload_queue)=" << status << checkerror( status) <<"\n"  << flush; exit_( status);}

	res = clSetKernelArg( atomic_test2_kernel, 0, sizeof( int), 		&num_threads_int);														if ( res    !=CL_SUCCESS)	{ cout <<"\nres = "<<checkerror( res)<<"\n"<<flush;exit_( res);}
	res = clSetKernelArg( atomic_test2_kernel, 1, sizeof( cl_mem), 	&atomic_test2_buf);															if ( res    !=CL_SUCCESS)	{ cout <<"\nres = "<<checkerror( res)<<"\n"<<flush;exit_( res);}
	/*
	cl_int clEnqueueNDRangeKernel(
									cl_command_queue 		command_queue,
									cl_kernel 				kernel,
									cl_uint 				work_dim,
									const size_t* 			global_work_offset,
									const size_t* 			global_work_size,
									const size_t* 			local_work_size,
									cl_uint 				num_events_in_wait_list,
									const cl_event* 		event_wait_list,
									cl_event* 				event
									);
	*/
	res 	= clEnqueueNDRangeKernel( m_queue, atomic_test2_kernel, 1, 0, &num_threads, &local_work_size, 0, NULL, &ev);						if ( res    != CL_SUCCESS)	{ cout << "\nres = " << checkerror( res) <<"\n"<<flush; exit_( res);}
	status 	= clFlush( m_queue);																												if ( status != CL_SUCCESS)	{ cout << "\nRunCL::atomic_test1( ),  clFlush( queue_to_call) status  = "<<status<<" "<< checkerror( status) <<"\n"<<flush; exit_( status);}

	float atomic_test_output[ data_size ];
	for ( int i = 0; i<data_size; i++) atomic_test_output[i] = -1;
	cl_event readEvt;
	status = clEnqueueReadBuffer( 	dload_queue,			// command_queue
									atomic_test2_buf,		// buffer
									CL_FALSE,				// blocking_read
									0,						// offset
									data_size*sizeof( float),	// size
									atomic_test_output,		// pointer
									0,						// num_events_in_wait_list
									NULL,					// event_waitlist				needs to know about preceeding events:
									&readEvt				// event
								 );
														if ( status != CL_SUCCESS) { cout << "\nclEnqueueReadBuffer( ..) status=" 	<< checkerror( status) <<"\n"<<flush; exit_( status);}
	status = clFlush( dload_queue);						if ( status != CL_SUCCESS) { cout << "\nclFlush( m_queue) status = " 		<< checkerror( status) <<"\n"<<flush; exit_( status);}
	status = clWaitForEvents( 1, &readEvt); 			if ( status != CL_SUCCESS) { cout << "\nclWaitForEvents status="			<< checkerror( status) <<"\n"<<flush; exit_( status);}

	cout << "\n\n void RunCL::atomic_test2( ): \t  local_work_size = "<<local_work_size<<", \t ( buffer size) data_size = "<<data_size<<", \t ( num threards launched)  num_threads="<<num_threads<<", \t ( num threards run) num_threads_int = "<<num_threads_int<<", \t  ( ";
	int i=0;
	for ( ; i< data_size; i++) cout << ", " << atomic_test_output[i];
	cout << ") \t i="<< i<< "\n\n" << flush;



}
