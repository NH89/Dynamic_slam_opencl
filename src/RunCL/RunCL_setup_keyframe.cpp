#include "RunCL.hpp"

void RunCL::estimateCalibration( ){ //estimateCalibration( ); 		// own thread, one iter.
	string fname = "RunCL::estimateCalibration( )";
	int local_verbosity_threshold = V_RUNCL_ESTIMATECALIBRATION;//verbosity_mp["RunCL::estimateCalibration"];

}

void RunCL::transform_depthmap( /*cv::Matx44f K2K_*/ float K2K_arry[16] , cl_mem depthmap_ ){																		// NB must be used _before_ initializing the new cost_volume, because it uses keyframe_imgmem.
	string fname = "RunCL::transform_depthmap( )";
	int local_verbosity_threshold = V_RUNCL_TRANSFORM_DEPTHMAP;
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::transform_depthmap( ..)_chk0 .   runcl.dataset_frame_num="<< dataset_frame_num<<flush;}
	cl_int status;
	//float K2K_arry[16];Matx44f_To_float16arry( K2K_, K2K_arry );
																																			if( verbosity>local_verbosity_threshold) { PRINT_FLOAT_16( K2K_arry, RunCL::transform_depthmap( ..) ); }
	const float zero  = 0;
	_clEnqueueFillBuffer ( uload_queue, 	depth_mem_temp, &zero,    sizeof( float),  0,     mm_size_bytes_C1, 	fname);
	_clEnqueueWriteBuffer( uload_queue, 	k2kbuf,	   CL_FALSE, 0, 16 * sizeof( float), K2K_arry, 			fname);

	stringstream ss1;
	ss1 << "_transform_depthmap_1_";
	ss1 << save_index;
	DownloadAndSave( 	keyframe_depth_mem,   	ss1.str( ), 	paths.at( "keyframe_depth_mem"), mm_size_bytes_C1,   mm_Image_size,   CV_32FC1, 	false , fp32_params[MAX_INV_DEPTH]); 	cout<<"\n\nRunCL::transform_depthmap( ..)_chk0.1 ."	<<flush;
	DownloadAndSave( 	depth_mem_temp,   		ss1.str( ), 	paths.at( "depth_mem_temp"),	 mm_size_bytes_C1,   mm_Image_size,   CV_32FC1, 	false , fp32_params[MAX_INV_DEPTH]); 	cout<<"\n\nRunCL::transform_depthmap( ..)_chk4 ."	<<flush;   // ### corrupted !

	// inputs
	// __private	 uint layer, set in mipmap_call_kernel( ..) below																		//__private	    uint	    layer,							//0
	_clSetKernelArg( transform_depthmap_kernel,  1, sizeof( cl_mem), &mipmap_buf, 		fname);												//__constant    uint*	    mipmap_params,					//1
	_clSetKernelArg( transform_depthmap_kernel,  2, sizeof( cl_mem), &uint_param_buf, 	fname);												//__constant	uint*		uint_params,					//2
	_clSetKernelArg( transform_depthmap_kernel,  3, sizeof( cl_mem), &k2kbuf, 			fname);												//__global		float* 		k2k,							//3
	_clSetKernelArg( transform_depthmap_kernel,  4, sizeof( cl_mem), &keyframe_imgmem, 	fname);												//__global		float4* 	keyframe_imgmem,				//4		// uses alpha channel to check bounds
	_clSetKernelArg( transform_depthmap_kernel,  5, sizeof( cl_mem), &depthmap_, 			fname);												//__global		float* 		keyframe_depth_mem,				//5
	// output
	_clSetKernelArg( transform_depthmap_kernel,  6, sizeof( cl_mem), &depth_mem_temp, 			fname);												//__global		float* 		depth_mem,						//6
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::transform_depthmap( ..)_chk1 ."<<flush;}
	mipmap_call_kernel( transform_depthmap_kernel, m_queue );
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::transform_depthmap( ..)_chk3 ."<<flush;}
	stringstream ss;
	ss << "_transform_depthmap_";
	ss << save_index;
	DownloadAndSave( 	keyframe_depth_mem,   	ss.str( ), 	paths.at( "keyframe_depth_mem"), mm_size_bytes_C1,   mm_Image_size,   CV_32FC1, 	false , fp32_params[MAX_INV_DEPTH]); 	cout<<"\n\nRunCL::transform_depthmap( ..)_chk0.1 ."	<<flush;
	DownloadAndSave( 	depth_mem_temp,   		ss.str( ), 	paths.at( "depth_mem_temp"),   	 mm_size_bytes_C1,   mm_Image_size,   CV_32FC1, 	false , fp32_params[MAX_INV_DEPTH]); 	cout<<"\n\nRunCL::transform_depthmap( ..)_chk4 ."	<<flush;   // ### corrupted !

	cl_mem_swap_ptr( 	keyframe_depth_mem, 	depth_mem_temp	);
	clFlush( m_queue); status = clFinish( m_queue);																							if( status!= CL_SUCCESS){cout << " status = " << checkerror( status) <<", Error: RunCL::transform_depthmap( ..)_clfinish_clEnqueueCopyBuffer\n" << flush;exit_( status);}
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::transform_depthmap( ..)_finished ."<<flush;}
}


void RunCL::swap_costvol_pointers(){
	string fname = "RunCL::swap_costvol_pointers( )";
	int local_verbosity_threshold = V_RUNCL_SWAP_COSTVOL_POINTERS;
																																			if( verbosity>local_verbosity_threshold){ cout << "\n\nRunCL::swap_costvol_pointers( )_chk 0,  "<< flush; }
	cl_mem temp_mem;

	temp_mem 		= cdatabuf;
	cdatabuf		= temp_cdatabuf;
	temp_cdatabuf	= temp_mem;

	temp_mem 		= hdatabuf;
	hdatabuf		= temp_hdatabuf;
	temp_hdatabuf	= temp_mem;
}


void RunCL::transform_costvolume( float K2K_arry[16]){																						// NB must be used _after_ initializing the new cost_volume.
	string fname = "RunCL::transform_costvolume( )";
	int local_verbosity_threshold = V_RUNCL_TRANSFORM_COSTVOLUME;
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::transform_costvolume( ..)_chk0 .   runcl.dataset_frame_num="<< dataset_frame_num<<flush;
																																				PRINT_FLOAT_16( K2K_arry, RunCL::transform_costvolume( ..) );
																																			}
	_clEnqueueWriteBuffer( uload_queue, invk2kbuf,	CL_FALSE, 0, 16 * sizeof( float), K2K_arry, 	fname);
	// inputs
	//     __private	 uint layer, set in mipmap_call_kernel( ..) below																	//__private	    uint	    layer,				//0
	_clSetKernelArg( transform_costvolume_kernel,  1, sizeof( cl_mem), &mipmap_buf, 	fname);												//__constant    uint*	    mipmap_params,		//1
	_clSetKernelArg( transform_costvolume_kernel,  2, sizeof( cl_mem), &uint_param_buf, fname);												//__constant	uint*		uint_params,		//2
	_clSetKernelArg( transform_costvolume_kernel,  3, sizeof( cl_mem), &fp32_param_buf, fname);												//__constant	uint*		fp32_params,		//3
	_clSetKernelArg( transform_costvolume_kernel,  4, sizeof( cl_mem), &invk2kbuf, 		fname);												//__global		float16* 	k2k,				//4
	_clSetKernelArg( transform_costvolume_kernel,  5, sizeof( cl_mem), &temp_cdatabuf, 	fname);												//__global		float*		old_cdata,			//5		photometric cost volume
	_clSetKernelArg( transform_costvolume_kernel,  6, sizeof( cl_mem), &temp_hdatabuf, 	fname);												//__global		float*		old_hdata,			//7		hit count volume
	// outputs
	_clSetKernelArg( transform_costvolume_kernel,  7, sizeof( cl_mem), &cdatabuf, 		fname);												//__global		float*		new_cdata,			//6
	_clSetKernelArg( transform_costvolume_kernel,  8, sizeof( cl_mem), &hdatabuf, 		fname);												//__global		float*		new_hdata,			//8
	_clSetKernelArg( transform_costvolume_kernel,  9, sizeof( cl_mem), &lomem, 			fname);												//__global		float*		lo_,				//9		lo, hi, and mean of this ray of the cost volume.
	_clSetKernelArg( transform_costvolume_kernel, 10, sizeof( cl_mem), &himem, 			fname);												//__global		float*		hi_,				//10
	_clSetKernelArg( transform_costvolume_kernel, 11, sizeof( cl_mem), &mean_mem, 		fname);												//__global		float*		mean_				//11
	
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::transform_costvolume( ..)_chk1 ."<<flush;}
/*
																																			if( verbosity>local_verbosity_threshold ) {
																																				stringstream ss;
																																				ss << "tempbufs before_transform_costvolume" << save_index;													// Save buffers to file ###########
																																				bool show = false;
																																				bool exception_tiff = false;
																																				if ( verbosity_mp["RunCL::updateDepthCostVol::cdatabuf"])																// divisor max_range = 0 -> val/max_val.   max_range <0 -> grey=0
																																					DownloadAndSaveVolume( 		temp_cdatabuf, 			ss.str( ), paths.at( "cdatabuf"), mm_size_bytes_C1,	mm_Image_size,   CV_32FC1,  show , 0 , exception_tiff );
																																				if ( verbosity_mp["RunCL::updateDepthCostVol::hdatabuf"])
																																					DownloadAndSaveVolume( 		temp_hdatabuf, 			ss.str( ), paths.at( "hdatabuf"), mm_size_bytes_C1,	mm_Image_size,   CV_32FC1,  show , 0 , exception_tiff );
																																			}
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::transform_costvolume( ..)_chk2 ."<<flush;}
*/
	mipmap_call_kernel( transform_costvolume_kernel, m_queue );
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::transform_costvolume( ..)_chk3 ."<<flush;
																																				stringstream ss;
																																				ss << "transform_costvolume" << save_index;													// Save buffers to file ###########
																																				bool show = false;
																																				bool exception_tiff = false;
																																				DownloadAndSave( 	lomem,  ss.str( ), paths.at( "lomem"),  mm_size_bytes_C1,   mm_Image_size,   CV_32FC1, 	show , 8);	// a little more than the num images in costvol.
																																				DownloadAndSave( 	himem,  ss.str( ), paths.at( "himem"),  mm_size_bytes_C1,   mm_Image_size,   CV_32FC1, 	show , 8);	//params[COSTVOL_LAYERS]

																																				if ( V_RUNCL_UPDATEDEPTHCOSTVOL_CDATABUF)																// divisor max_range = 0 -> val/max_val.   max_range <0 -> grey=0
																																					DownloadAndSaveVolume( 		cdatabuf, 			ss.str( ), paths.at( "cdatabuf"), mm_size_bytes_C1,	mm_Image_size,   CV_32FC1,  show , 0 /*float max_range*/ , exception_tiff );
																																				if ( V_RUNCL_UPDATEDEPTHCOSTVOL_HDATABUF)
																																					DownloadAndSaveVolume( 		hdatabuf, 			ss.str( ), paths.at( "hdatabuf"), mm_size_bytes_C1,	mm_Image_size,   CV_32FC1,  show , 0 /*float max_range*/ , exception_tiff );
																																			}
	
}


void RunCL::initializeDepthCostVol( cl_mem key_frame_depth_map_src){			 															// Uses the current frame as the keyframe for a new depth cost volume.
	string fname = "RunCL::initializeDepthCostVol( )";																						// Dynamic_slam::initialize_from_GT( ), Dynamic_slam::initialize_new_keyframe( );
	int local_verbosity_threshold = V_RUNCL_INITIALIZEDEPTHCOSTVOL;
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::initializeDepthCostVol( ..)_chk0 ."<<flush;}
	costvol_frame_num = 0;
	_clEnqueueCopyBuffer( m_queue, imgmem, 			keyframe_imgmem, 			0, 0, mm_size_bytes_C4, 	fname);							// Load keyframe
	_clEnqueueCopyBuffer( m_queue, HSV_grad_mem, 	keyframe_imgmem_HSV_grad, 	0, 0, mm_size_bytes_C8, 	fname);

	/*if( vtp==true)*/ Store_keyframe( );

	stringstream ss;
	ss << "__buildDepthCostVol";
	save_index = keyFrameCount*1000 + costvol_frame_num;
	ss << save_index;

	DownloadAndSave( 		 	key_frame_depth_map_src,   	ss.str( ),   paths.at( "key_frame_depth_map_src"),   	mm_size_bytes_C1,   mm_Image_size,   CV_32FC1, 	false , fp32_params[MAX_INV_DEPTH]);
																																			if( verbosity>local_verbosity_threshold)	cout << "\nDownloadAndSave ( .. key_frame_depth_map_src ..) finished\n"<<flush;
	_clEnqueueCopyBuffer( m_queue, key_frame_depth_map_src, keyframe_depth_mem,			0, 0, mm_size_bytes_C1, 		fname);
	_clEnqueueCopyBuffer( m_queue, depth_mem_GT, 			keyframe_depth_mem_GT,		0, 0, mm_size_bytes_C1, 		fname);
	_clEnqueueCopyBuffer( m_queue, SE3_grad_map_mem, 		keyframe_SE3_grad_map_mem, 	0, 0, mm_size_bytes_C1*6*8, 	fname);
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::initializeDepthCostVol( ..)_chk1 ."<<flush;}
	float zero  = 0;
	_clEnqueueFillBuffer( uload_queue, dbg_databuf, 	&zero, sizeof( float),   0, mm_vol_size_bytes, 	fname);
	_clEnqueueFillBuffer( uload_queue, cdatabuf, 		&zero, sizeof( float),   0, mm_vol_size_bytes, 	fname);
	_clEnqueueFillBuffer( uload_queue, hdatabuf, 		&zero, sizeof( float),   0, mm_vol_size_bytes, 	fname);
	_clEnqueueFillBuffer( uload_queue, img_sum_buf, 	&zero, sizeof( float),   0, mm_vol_size_bytes, 	fname);

	_clEnqueueFillBuffer( uload_queue, dmem, 			&zero, sizeof( float),   0, mm_size_bytes_C1, 	fname);
	_clEnqueueFillBuffer( uload_queue, amem, 			&zero, sizeof( float),   0, mm_size_bytes_C1, 	fname);
	_clEnqueueFillBuffer( uload_queue, qmem, 			&zero, sizeof( float),   0, mm_size_bytes_C1, 	fname);
	_clEnqueueFillBuffer( uload_queue, qmem2, 			&zero, sizeof( float),   0, mm_size_bytes_C1, 	fname);

	_clEnqueueFillBuffer( uload_queue, lomem, 			&zero, sizeof( float),   0, mm_size_bytes_C1, 	fname);
	_clEnqueueFillBuffer( uload_queue, himem, 			&zero, sizeof( float),   0, mm_size_bytes_C1, 	fname);
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::initializeDepthCostVol( ..)_chk1.5 ."<<flush;
																																				ss << "initializeDepthCostVol";
																																				ss << save_index;													// Save buffers to file ###########
																																				cout<<"\n\nRunCL::initializeDepthCostVol( ..)_chk1.5.1 ."<<flush;

																																				DownloadAndSave_3Channel( 	keyframe_imgmem, 			ss.str( ), paths.at( "keyframe_imgmem"),  mm_size_bytes_C4, mm_Image_size,  CV_32FC4, 	false );
																																				DownloadAndSave_HSV_grad(	keyframe_imgmem_HSV_grad, 	ss.str( ), paths.at( "keyframe_imgmem_HSV_grad"),2*mm_size_bytes_C4, mm_Image_size, CV_32FC( 8),	false, -1, 0 );
																																				cout<<"\n\nRunCL::initializeDepthCostVol( ..)_chk1.6 ."<<flush;

																																				DownloadAndSave( 		 	keyframe_depth_mem,   		ss.str( ), paths.at( "keyframe_depth_mem"),   		mm_size_bytes_C1,   mm_Image_size,   CV_32FC1, 	false , fp32_params[MAX_INV_DEPTH]);
																																				cout<<"\n\nRunCL::initializeDepthCostVol( ..)_chk1.7 ."<<flush;

																																				DownloadAndSave_6Channel_volume(  keyframe_SE3_grad_map_mem, ss.str( ), paths.at( "keyframe_SE3_grad_map_mem"), mm_size_bytes_C4, mm_Image_size, CV_32FC4, false, -1, 6 );
																																			}
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::initializeDepthCostVol( ..)_Finished ########################################## ."<<flush;}
}


void RunCL::initializeFirstDepthCostVol( float default_depth ){			 																	// Uses the current frame as the keyframe for a new depth cost volume.
	string fname = "RunCL::initializeFirstDepthDepthCostVol( )";																			// Dynamic_slam::initialize_from_GT( ), Dynamic_slam::initialize_new_keyframe( );
	int local_verbosity_threshold = V_RUNCL_INITIALIZEDEPTHCOSTVOL;
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::initializeFirstDepthDepthCostVol( ..)_chk0, default_depth="<<default_depth<<flush;}
	_clEnqueueFillBuffer( uload_queue,	depth_mem_temp,			&default_depth, 			sizeof( float), 0,		mm_size_bytes_C1, 		fname);			// TODO one of depth_mem, or keyframe_depth_mem is redundant.
	_clEnqueueFillBuffer( uload_queue,	keyframe_depth_mem, 	&default_depth, 			sizeof( float), 0,		mm_size_bytes_C1, 		fname);

	_clEnqueueCopyBuffer( m_queue,		depth_mem_GT, 			keyframe_depth_mem_GT,		0, 				0,		mm_size_bytes_C1, 		fname);
	_clEnqueueCopyBuffer( m_queue,		SE3_grad_map_mem, 		keyframe_SE3_grad_map_mem, 	0, 				0,		mm_size_bytes_C1*6*8, 	fname);
	_clEnqueueCopyBuffer( m_queue,		imgmem,					keyframe_imgmem, 			0, 				0,		mm_size_bytes_C4, 		fname);			// Load keyframe
	_clEnqueueCopyBuffer( m_queue,		HSV_grad_mem,			keyframe_imgmem_HSV_grad, 	0, 				0,		mm_size_bytes_C8, 		fname);

	Store_keyframe( );																														// if( vtp==true)

	stringstream ss;
	ss << "__buildDepthCostVol";
	costvol_frame_num = 0;
	save_index = keyFrameCount*1000 + costvol_frame_num;
	ss << save_index;
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::initializeFirstDepthDepthCostVol( ..)_chk1 ."<<flush;}
	float zero  = 0;
	_clEnqueueFillBuffer( uload_queue, dbg_databuf, 			&zero, 						sizeof( float),   0, 	mm_vol_size_bytes, 		fname);
	_clEnqueueFillBuffer( uload_queue, cdatabuf, 				&zero, 						sizeof( float),   0, 	mm_vol_size_bytes, 		fname);
	_clEnqueueFillBuffer( uload_queue, hdatabuf, 				&zero, 						sizeof( float),   0, 	mm_vol_size_bytes, 		fname);
	_clEnqueueFillBuffer( uload_queue, temp_cdatabuf,			&zero, 						sizeof( float),   0, 	mm_vol_size_bytes, 		fname);
	_clEnqueueFillBuffer( uload_queue, temp_hdatabuf,			&zero, 						sizeof( float),   0, 	mm_vol_size_bytes, 		fname);

	_clEnqueueFillBuffer( uload_queue, img_sum_buf, 			&zero, 						sizeof( float),   0, 	mm_vol_size_bytes, 		fname);

	_clEnqueueFillBuffer( uload_queue, dmem, 					&default_depth, 			sizeof( float),   0, 	mm_size_bytes_C1, 		fname);
	_clEnqueueFillBuffer( uload_queue, amem, 					&default_depth, 			sizeof( float),   0, 	mm_size_bytes_C1, 		fname);
	_clEnqueueFillBuffer( uload_queue, qmem, 					&zero, 						sizeof( float),   0, 	mm_size_bytes_C1, 		fname);
	_clEnqueueFillBuffer( uload_queue, qmem2, 					&zero, 						sizeof( float),   0, 	mm_size_bytes_C1, 		fname);

	_clEnqueueFillBuffer( uload_queue, lomem, 					&zero, 						sizeof( float),   0, 	mm_size_bytes_C1, 		fname);
	_clEnqueueFillBuffer( uload_queue, himem, 					&zero, 						sizeof( float),   0, 	mm_size_bytes_C1, 		fname);
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::initializeFirstDepthDepthCostVol( ..)_chk1.5 ."<<flush;
																																				ss << "initializeDepthCostVol";
																																				ss << save_index;													// Save buffers to file ###########
																																				cout<<"\n\nRunCL::initializeDepthCostVol( ..)_chk1.5.1 ."<<flush;

																																				DownloadAndSave_3Channel( 	keyframe_imgmem, 			ss.str( ), paths.at( "keyframe_imgmem"),  mm_size_bytes_C4, mm_Image_size,  CV_32FC4, 	false );
																																				DownloadAndSave_HSV_grad(  keyframe_imgmem_HSV_grad, 	ss.str( ), paths.at( "keyframe_imgmem_HSV_grad"),2*mm_size_bytes_C4, mm_Image_size, CV_32FC( 8),	false, -1, 0 );
																																				cout<<"\n\nRunCL::initializeDepthCostVol( ..)_chk1.6 ."<<flush;

																																				DownloadAndSave( 		 	keyframe_depth_mem,   		ss.str( ), paths.at( "keyframe_depth_mem"),   		mm_size_bytes_C1,   mm_Image_size,   CV_32FC1, 	false , fp32_params[MAX_INV_DEPTH]);
																																				cout<<"\n\nRunCL::initializeDepthCostVol( ..)_chk1.7 ."<<flush;

																																				DownloadAndSave_6Channel_volume(  keyframe_SE3_grad_map_mem, ss.str( ), paths.at( "keyframe_SE3_grad_map_mem"), mm_size_bytes_C4, mm_Image_size, CV_32FC4, false, -1, 6 );
																																			}
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::initializeFirstDepthDepthCostVol( ..)_Finished ########################################## ."<<flush;}
}
