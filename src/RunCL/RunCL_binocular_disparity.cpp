#include "RunCL.hpp"

void RunCL::compute_lookup_table( uint start, uint stop){
    string fname = "RunCL::compute_lookup_table( )";
	int local_verbosity_threshold = V_RUNCL_COMPUTE_LOOKUP_TABLE;
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::compute_lookup_table( ..)_chk0 #############################################################"<<flush;
																																				cout << "\n local_work_size = " << local_work_size
																																				<< ",  start = " << start << ",  stop = " << stop
																																				<< flush;
																																			}
	// inputs
	// __private	(compute_lookup_table_kernel,  0, sizeof( int),    &reduction);	 set in for(uint reduction..) loop below			//__private	    uint		layer,				//0
	// __constant
	_clSetKernelArg( compute_lookup_table_kernel,  2, sizeof( cl_mem), &mipmap_buf,				fname);									//__constant	uint*		mipmap_params,		//1
	_clSetKernelArg( compute_lookup_table_kernel,  3, sizeof( cl_mem), &uint_param_buf,			fname);									//__constant 	uint*		uint_params,		//2
	_clSetKernelArg( compute_lookup_table_kernel,  4, sizeof( cl_mem), &fp32_param_buf,			fname);									//__constant	float*		fp32_params,		//3

	// output
	_clSetKernelArg( compute_lookup_table_kernel,  5, sizeof( cl_mem), &lookup_table_buf,		fname);									//__global 		uint4*		lookup_table		//4
																																			if(verbosity>local_verbosity_threshold) {
																																				cout<<"\nRunCL::compute_lookup_table( ..)_chk1,  cl_kernel: compute_lookup_table_kernel,  cl_command_queue: m_queue,   start="
																																				<<start<<",   stop="<<stop<<"local_work_size="<<local_work_size<<" _chk0"<<flush;
																																			}
	cl_event		ev;
	cl_int			res, status;
	lookup_table_offset[0]	= 0; 																										// NB 'start' may not be set to zero

	for(uint reduction = 0; reduction <= stop; reduction++) {																			// NB processes largest layer first.
																																			if(verbosity>local_verbosity_threshold) { cout<<"\nRunCL::mipmap_call_kernel(..)_chk1,  reduction="\
																																				<<reduction<<",  num_threads[reduction]="<<num_threads[reduction]<<"  local_work_size="<<local_work_size<<flush; }
		uint 	lookup_table_offset_uint = lookup_table_offset[reduction];
		res 	= clSetKernelArg(compute_lookup_table_kernel, 0, sizeof(int), &reduction); 						if (res    !=CL_SUCCESS)	{ cout <<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	;
		res 	= clSetKernelArg(compute_lookup_table_kernel, 1, sizeof(int), &lookup_table_offset_uint ); 		if (res    !=CL_SUCCESS)	{ cout <<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	;

		res 	= clEnqueueNDRangeKernel(m_queue, compute_lookup_table_kernel, 1, 0, &num_threads[reduction], &local_work_size, 0, NULL, &ev); // run mipmap_float4_kernel, NB wait for own previous iteration.
																		if (res    != CL_SUCCESS)	{ cout << "\nres = " << checkerror(res) <<"\n"<<flush; exit_(res);}
		status 	= clFlush(m_queue);										if (status != CL_SUCCESS)	{ cout << "\nRunCL::mipmap_call_kernel( cl_kernel: compute_lookup_table_kernel,  clFlush(queue_to_call) status  = "		<<status<<" "<< checkerror(status) <<"\n"<<flush; exit_(status);}
		status 	= clWaitForEvents (1, &ev);								if (status != CL_SUCCESS)	{ cout << "\nRunCL::mipmap_call_kernel( cl_kernel: compute_lookup_table_kernel) for loop,  clWaitForEventsh(1, &ev) ="	<<status<<" "<<checkerror(status)  <<"\n"<<flush; exit_(status);}

		lookup_table_offset[reduction +1]=  lookup_table_offset[reduction] + num_threads[reduction];										// NB this pads the lookup table, so that local work groups will not be shared betwen layers.
																																			// It also  means that this offset should be used to launch layers from the lookup table.
	}
	for(uint reduction = 0; reduction <= stop; reduction++) {
			cout << "\nnum_threads["<<reduction<<"] = "<<num_threads[reduction]<<" MipMap[reduction*8 +MiM_PIXELS] = "<<MipMap[reduction*8 +MiM_PIXELS]<<flush;
	}
	cout <<"\nNB there will only be a gap in the lookuptable when:  num_threads[reduction] > MipMap[reduction*8 +MiM_PIXELS]  , which depends on the image dimensions and local_work_size"<<flush;

																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::compute_lookup_table( ..)_chk2 ."<<flush;	// Save buffers to file ###########
																																				stringstream ss;
																																				ss << "compute_lookup_table_" << save_index ;
																																				bool show 		= false;
																																				bool old_tiff 	= tiff;
																																				tiff 			= true;
																																				_cl_flush_finish(m_queue, fname);
																																				DownloadAndSave_3Channel( 	lookup_table_buf,	ss.str( ), paths.at( "lookup_table_buf"),  		mm_size_bytes_C4,   mm_Image_size,   CV_32FC4, 	show);
																																				// NB the tiff file holda the int32 values as float32. This is okay because they fit in the mantissa.
																																				// BGRA format, B=u, G=v, R=read_index, A=alpha.
																																				tiff 			= old_tiff;
																																			}
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::compute_lookup_table( ..)_finished #############################################################"<<flush;
																																			}
}


void RunCL::disparity_load_frame(cl_mem input_img, cl_mem output_img, std::string folder ){													// Loads an image into layer zero of a padded image pyramid, e.g. from basemem to img_mem or keyframe_img_mem.
    string fname = "RunCL::disparity_load_frame( )";
	int local_verbosity_threshold = V_RUNCL_WARP_IMAGE;
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::disparity_load_frame(..)_chk0 #############################################################"<<flush;
																																				cout << "\n local_work_size = "<<local_work_size<<",  num_threads[0] = "<<num_threads[0]<< flush;
																																			}
	uint baseImagePixels =   baseImage_size.height * baseImage_size.width;																	cout << "\nbaseImagePixels = "<<baseImagePixels<<",  baseImage_size.height = "<<baseImage_size.height<<",  baseImage_size.width = "<<baseImage_size.width <<flush;
																																			cout << "\nimage_size_bytes = "<<image_size_bytes<<flush;
	// __private
	//_clSetKernelArg( warp_image_kernel,			0, sizeof( uint), 	&read_offset,			fname);										// __private	uint	read_offset,			//0
	_clSetKernelArg( disparity_load_frame_kernel,	1, sizeof( uint), 	&baseImagePixels,		fname);										// __private	uint	baseImage_size,			//1
	// __global
	_clSetKernelArg( disparity_load_frame_kernel,  	2, sizeof( cl_mem), &lookup_table_buf,		fname);										// __global		float4*	lookup_table,			//2
	_clSetKernelArg( disparity_load_frame_kernel,  	3, sizeof( cl_mem), &input_img,				fname);										// __global		float* 	depth_map,				//3
	// output
	_clSetKernelArg( disparity_load_frame_kernel,  	4, sizeof( cl_mem), &output_img,			fname);										// __global		float2*	warp,					//4

																																				if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::disparity_load_frame( ..)_chk1 ."<<flush;}
	int layer = 0;																															// NB this kernel is called for layer 0 only.
	layer_call_kernel( disparity_load_frame_kernel, m_queue, layer, local_work_size);
																																			if( verbosity>local_verbosity_threshold /*&& layer==1*/ ) {cout<<"\n\nRunCL::disparity_load_frame( ..)_chk2 ."<<flush;								// Save buffers to file ###########
																																				stringstream ss;
																																				ss << "_binoc__disparity_load_frame_" << save_index <<"_layer_"<<layer ;
																																				bool show 		= false;
																																				bool old_tiff 	= tiff;
																																				tiff 			= true;
																																				float max_range = 1; 		// -1 -> gray = zero.
																																				_cl_flush_finish(m_queue, fname);
																																				DownloadAndSave_3Channel(  output_img,  ss.str( ),  paths.at(folder),  mm_size_bytes_C4,  mm_Image_size,  CV_32FC4, show,  max_range);
																																				tiff 			= old_tiff;
																																			}
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::disparity_load_frame( ..)_finished ."<<flush;}
}


void RunCL::zero_warp_buffer(){
	cl_int 		status;
	cl_event 	writeEvt;
	float 		zero  		= 0;
	status = clEnqueueFillBuffer(uload_queue, warp_buf,	&zero, sizeof(float), 0, mm_size_bytes_C1 * 2,	0, NULL, &writeEvt);	if(status != CL_SUCCESS){ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.8\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	clFlush(uload_queue);
	status = clFinish(uload_queue);


}


void RunCL::set_warp_new_image(uint layer, float reduction){					// Computed once each iteration of warping, for each layer of image pyramid
    string fname = "RunCL::set_warp_new_image( )";
	int local_verbosity_threshold = V_RUNCL_WARP_IMAGE;
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::set_warp_new_image(..)_chk0 #############################################################"<<flush;
																																				cout << "\n local_work_size = " << local_work_size
																																				<< ",  layer = " << layer
																																				<< flush;
																																			}

	// inputs									// Warp describes where to sample the new image to match the reference image.
	// __private
	//_clSetKernelArg( warp_image_kernel,			0, sizeof( uint), 	&read_offset,			fname);										// __private	uint	read_offset,			//0
	_clSetKernelArg( set_warp_new_image_kernel,  	1, sizeof( uint), 	&layer,					fname);										// __private	uint	layer,					//1
	_clSetKernelArg( set_warp_new_image_kernel,  	2, sizeof( float), 	&reduction,				fname);										// __private	float	reduction,				//2
	_clSetKernelArg( set_warp_new_image_kernel,  	3, sizeof( uint), 	&mm_width,				fname);										// 	__private	uint	mm_cols,				//3
	// __constant
	_clSetKernelArg( set_warp_new_image_kernel,  	4, sizeof( cl_mem), &uint_param_buf,		fname);										// __constant	uint*	uint_params,			//4
	_clSetKernelArg( set_warp_new_image_kernel,  	5, sizeof( cl_mem), &fp32_param_buf,		fname);										// __constant	float*  fp32_params,			//5
	// __global
	_clSetKernelArg( set_warp_new_image_kernel,  	6, sizeof( cl_mem), &k2kbuf,				fname);										// __global		float16*k2k,					//6		// keyframe2K[3]
	_clSetKernelArg( set_warp_new_image_kernel,  	7, sizeof( cl_mem), &lookup_table_buf,		fname);										// __global		float4*	lookup_table,			//7
	_clSetKernelArg( set_warp_new_image_kernel,  	8, sizeof( cl_mem), &keyframe_depth_mem,	fname);										// __global		float* 	depth_map,				//8
	// output
	_clSetKernelArg( set_warp_new_image_kernel,  	9, sizeof( cl_mem), &warp_buf,				fname);										// __global		float2*	warp,					//9

																																				if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::set_warp_new_image( ..)_chk1 ."<<flush;}
	layer_call_kernel( set_warp_new_image_kernel, m_queue, layer, local_work_size);
																																			if( verbosity>local_verbosity_threshold /*&& layer==1*/ ) {cout<<"\n\nRunCL::set_warp_new_image( ..)_chk2 ."<<flush;								// Save buffers to file ###########
																																				stringstream ss;
																																				ss << "_binoc__set_warp_new_image_" << save_index <<"_layer_"<<layer ;
																																				bool show 		= false;
																																				bool old_tiff 	= tiff;
																																				tiff 			= true;
																																				float max_range = -1.0f; 		// -ve or 0 -> gray = zero.  0 -> use max range from image.
																																				uint vol_layers	= 1;
																																				_cl_flush_finish(m_queue, fname);
																																				DownloadAndSave_2Channel_volume( warp_buf,	ss.str( ), paths.at( "warp_buf"),  			2*mm_size_bytes_C1,   mm_Image_size,   CV_32FC2, show,  /* max_range*/-10.0f,   vol_layers );  // 1, 2D warp, 1DoF
																																				DownloadAndSave( 	keyframe_depth_mem,   	ss.str( ), paths.at( "keyframe_depth_mem"),   mm_size_bytes_C1,   mm_Image_size,   CV_32FC1, false , fp32_params[MAX_INV_DEPTH]);

																																				tiff 			= old_tiff;
																																			}
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::set_warp_new_image( ..)_finished ."<<flush;}
}

void RunCL::warp_image( uint layer, int iter ){																					// computed once each iteration of warping, for each layer of image pyramid
    string fname = "RunCL::warp_image( )";
	int local_verbosity_threshold = V_RUNCL_WARP_IMAGE;
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::warp_image( ..)_chk0 #############################################################"<<flush;
																																				cout << "\n local_work_size = " << local_work_size
																																				<< ",  layer = " << layer
																																				<< flush;
																																			}
	// inputs
	// __private
	//_clSetKernelArg( warp_image_kernel,			0, sizeof( uint), 	&read_offset,			fname);										// 	__private	uint	read_offset,			//0
	_clSetKernelArg( warp_image_kernel,  			1, sizeof( uint), 	&mm_width,				fname);										// 	__private	uint	mm_cols,				//1
	_clSetKernelArg( warp_image_kernel,  			2, sizeof( uint), 	&layer,					fname);										// 	__private	uint	layer,					//2
	// __constant
	_clSetKernelArg( warp_image_kernel,  			3, sizeof( cl_mem), &mipmap_buf,			fname);										// 	__constant 	uint*	mipmap_params,			//3
	// __global
	_clSetKernelArg( warp_image_kernel,  			4, sizeof( cl_mem), &warp_buf,				fname);										// 	__global 	float2*	warp,					//4
	_clSetKernelArg( warp_image_kernel,  			5, sizeof( cl_mem), &lookup_table_buf,		fname);										// 	__global 	uint4*	lookup_table,			//5
	_clSetKernelArg( warp_image_kernel,  			6, sizeof( cl_mem), &new_img_buf,			fname);										// 	__global 	float4*	new_img,				//6

	// outputs
	_clSetKernelArg( warp_image_kernel,  			7, sizeof( cl_mem), &warped_img_buf,		fname);										// 	__global 	float4*	new_img_warped			//7
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::warp_image( ..)_chk1 ."<<flush;}
	layer_call_kernel( warp_image_kernel, m_queue, layer, local_work_size);
																																			if( verbosity>local_verbosity_threshold /*&& layer==1*/ ) {cout<<"\n\nRunCL::warp_image( ..)_chk2 ."<<flush;								// Save buffers to file ###########
																																				stringstream ss;
																																				ss << "_binoc__warp_image" << save_index <<"_layer_"<<layer<<"_iter_"<<iter ;
																																				bool show 		= false;
																																				bool old_tiff 	= tiff;
																																				tiff 			= true;
																																				float max_range = 1.0f; 		// -1 -> gray = zero.
																																				_cl_flush_finish(m_queue, fname);
																																				DownloadAndSave_2Channel_volume( warp_buf,			ss.str( ), paths.at( "warp_buf"),  			2*mm_size_bytes_C1,	mm_Image_size,   CV_32FC2, show, /*max_range*/-10.0f,	 2 );  // 1, 2D warp, 1DoF
																																				DownloadAndSave_3Channel( 		 warped_img_buf,	ss.str( ), paths.at( "warped_img_buf"),  	mm_size_bytes_C4,	mm_Image_size,   CV_32FC4, show, max_range);
																																				tiff 			= old_tiff;
																																			}
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::warp_image( ..)_finished ."<<flush;}
}


void RunCL::correlation_one_step ( uint layer, uint iter ){
    string fname = "RunCL::correlation_one_step( )";
	int local_verbosity_threshold = V_RUNCL_IMG_VARIANCE;
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::correlation_one_step( ..)_chk0 #############################################################"<<flush;
																																				cout << "\t local_work_size = " << local_work_size
																																				<< ",  layer = " << layer
																																				<< flush;
																																			}
	// inputs
	// __private
	//_clSetKernelArg( warp_image_kernel,  				 0, sizeof( uint), 							&read_offset,			fname);			// 	__private	uint	read_offset,		//0
	_clSetKernelArg( correlation_one_step_kernel,  		 1, sizeof( uint), 							&mm_width,				fname);			// 	__private	uint	mm_cols,			//1
	_clSetKernelArg( correlation_one_step_kernel,  		 2, sizeof( uint), 							&mm_layerstep,			fname);			//	 __private	uint 	mm_size,			//2
	// __local
	_clSetKernelArg( correlation_one_step_kernel, 		 3, (local_work_size+2)*3*sizeof(cl_float4), NULL,					fname);			// __local		float4*	local_ref_img,		//3		// 3*(group_size + 2) * sizeof(float4)
	_clSetKernelArg( correlation_one_step_kernel, 		 4, (local_work_size+4)*5*sizeof(cl_float4), NULL,					fname);			// __local		float*	local_confidence	//4		// 5*(group_size + 4) * sizeof(float4)
	// __global
	_clSetKernelArg( correlation_one_step_kernel,  		 5, sizeof( cl_mem), 						&lookup_table_buf,		fname);			//__global 	float4*	lookup_table,			//5
	_clSetKernelArg( correlation_one_step_kernel,  		 6, sizeof( cl_mem), 						&ref_img_buf,			fname);			//__global 	float4*	ref_img,				//6
	_clSetKernelArg( correlation_one_step_kernel,  		 7, sizeof( cl_mem), 						&warped_img_buf,		fname);			//__global 	float4*	warped_img,				//7
	// output
	_clSetKernelArg( correlation_one_step_kernel,  		 8, sizeof( cl_mem), 						&covariance_buf,		fname);			//__global 	float4*	covariance				//8
	_clSetKernelArg( correlation_one_step_kernel,  		 9, sizeof( cl_mem), 						&correlation_buf,		fname);			//__global 	float4*	correlation				//9
	// _clSetKernelArg( correlation_one_step_kernel,  		10, sizeof( cl_mem), &warp_buf,				fname);									// __global 	float2*	warp				//10
	// _clSetKernelArg( correlation_one_step_kernel,  		11, sizeof( cl_mem), &confidence_buf,		fname);									// __global 	float*	confidence			//11

																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::correlation_one_step( ..)_chk1 ."<<flush;}
	layer_call_kernel( correlation_one_step_kernel, m_queue, layer, local_work_size);
																																			if( verbosity>local_verbosity_threshold /*&& layer==1*/ ) {cout<<"\n\nRunCL::correlation( ..)_chk2 ."<<flush;				// Save buffers to file ###########
																																				stringstream ss;
																																				ss << "_binoc_" <<fname<< save_index <<"_layer_"<<layer<<"_iter_"<<iter ;
																																				bool show 		= false;
																																				bool display 	= false;
																																				bool old_tiff 	= tiff;
																																				tiff 			= true;
																																				uint vol_layers = 5;			// NB 5 samples of possible warp.
																																				float max_range = -1.0f; 		// -1 -> gray = zero.
																																				_cl_flush_finish(m_queue, fname);
																																				DownloadAndSave_3Channel_volume( 	covariance_buf,		ss.str( ), paths.at( "covariance_buf" ),  	mm_size_bytes_C4,   mm_Image_size,   CV_32FC4, 	show, 1.0f, 	vol_layers, tiff, iter, display);
																																				DownloadAndSave_3Channel_volume( 	correlation_buf,	ss.str( ), paths.at( "correlation_buf" ),  	mm_size_bytes_C4,   mm_Image_size,   CV_32FC4, 	show, 1.0f, 	vol_layers, tiff, iter, display); // just the denominator
																																				// DownloadAndSave_2Channel_volume( 	warp_buf,			ss.str( ), paths.at( "warp_buf"),  		  2*mm_size_bytes_C1,   mm_Image_size,   CV_32FC2, 	show, /*max_range*/-10.0f, 		1 );
																																				// DownloadAndSave( 			  		confidence_buf,		ss.str( ), paths.at( "confidence_buf"),		mm_size_bytes_C1,   mm_Image_size,   CV_32FC1,  show, max_range);
																																				tiff 			= old_tiff;
																																			}
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::correlation_one_step( ..)_finished ."<<flush;}
}



void RunCL::blur_volume(cl_mem in_buff, cl_mem blurred_buf, std::string folder, uint vol_layers, uint mipmap_layer, uint iter  ){
	string fname = "RunCL::blur_volume()";
	int local_verbosity_threshold = V_RUNCL_IMG_VARIANCE;//verbosity_mp["RunCL::blur_volume"];// -1;

	//inputs
	//__private
	//_clSetKernelArg( warp_image_kernel,  				0, sizeof( uint), 							&read_offset,			fname);			//__private	uint	read_offset,		//0
	_clSetKernelArg( blur_volume_kernel,  				1, sizeof( uint), 							&mm_width,				fname);			//__private	uint	mm_cols,			//1
	_clSetKernelArg( blur_volume_kernel,  				2, sizeof( uint), 							&mm_layerstep,			fname);			//__private	uint 	mm_size,			//2
	_clSetKernelArg( blur_volume_kernel,  				3, sizeof( uint), 							&vol_layers,			fname);			//__private	uint 	vol_layers,			//3
	//__local
	_clSetKernelArg( blur_volume_kernel, 				4, 5*(local_work_size+4)*sizeof(cl_float4), NULL,					fname);			//__local	float4*	local_img_patch,	//4		// 5*(local_work_size+4)*sizeof(cl_float4)
	//__global
	_clSetKernelArg( blur_volume_kernel,  				5, sizeof( cl_mem), 						&lookup_table_buf,		fname);			//__global	float4*	lookup_table,		//5
	_clSetKernelArg( blur_volume_kernel, 				6, sizeof(cl_mem), 							&in_buff,				fname );		//__global	float4*	img,				//6
	// output
	_clSetKernelArg( blur_volume_kernel, 				7, sizeof(cl_mem), 							&blurred_buf,			fname );		//__global	float4*	img,				//7

																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::blur_volume_step( ..)_chk1 ."<<flush;}
	layer_call_kernel( blur_volume_kernel, m_queue, mipmap_layer, local_work_size);
																																			if( verbosity>local_verbosity_threshold /*&& layer==1*/ ) {cout<<"\n\nRunCL::correlation( ..)_chk2 ."<<flush;				// Save buffers to file ###########
																																				stringstream ss;
																																				ss << "_binoc_" <<fname<< save_index <<"_layer_"<<mipmap_layer<<"_iter_"<<iter ;
																																				bool show 		= false;
																																				bool display 	= false;
																																				bool old_tiff 	= tiff;
																																				tiff 			= true;
																																				uint vol_layers = 5;			// NB 5 samples of possible warp.
																																				float max_range = 1.0f; 		// -1 -> gray = zero.
																																				_cl_flush_finish(m_queue, fname);
																																				DownloadAndSave_3Channel_volume( 	blurred_buf,	ss.str( ), paths.at( folder ),	mm_size_bytes_C4,   mm_Image_size,   CV_32FC4, 	show, max_range, 	vol_layers, tiff, iter, display);
																																				tiff 			= old_tiff;
																																			}
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::blur_volume_step( ..)_finished ."<<flush;}
}



void RunCL::compute_warp(uint mipmap_layer, uint iter  ){
	string fname = "RunCL::compute_warp()";
	int local_verbosity_threshold = V_RUNCL_IMG_VARIANCE;//verbosity_mp["RunCL::compute_warp"];// -1;

	//inputs
	//__private
	//_clSetKernelArg( warp_image_kernel,  			0, sizeof( uint), 				&read_offset,							fname);			//__private		uint	read_offset,		//0
	_clSetKernelArg( compute_warp_kernel,  			1, sizeof( uint), 				&mm_layerstep,							fname);			//__private		uint 	mm_size,			//1
	_clSetKernelArg( compute_warp_kernel,  			2, sizeof( uint), 				&mipmap_layer,							fname);			//__private		uint 	mm_size,			//2
	//__constant
	_clSetKernelArg( compute_warp_kernel,  			3, sizeof( cl_mem), 			&mipmap_buf,							fname);			//__constant	uint8*	mipmap_params,		//3
	//__global
	_clSetKernelArg( compute_warp_kernel,  			4, sizeof( cl_mem), 			&lookup_table_buf,						fname);			//__global		float4*	lookup_table,		//4
	_clSetKernelArg( compute_warp_kernel, 			5, sizeof(cl_mem), 				&correlation_blurred_buf,				fname );		//__global		float4*	correlation_blurred,//5
	// output
	_clSetKernelArg( compute_warp_kernel,	  		6, sizeof( cl_mem), 			&warp_buf,								fname);			//__global		float2*	warp				//6
	_clSetKernelArg( compute_warp_kernel,	  		7, sizeof( cl_mem), 			&confidence_buf,						fname);			//__global		float2*	confidence			//7

																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::compute_warp( ..)_chk1 ."<<flush;}
	layer_call_kernel( compute_warp_kernel, m_queue, mipmap_layer, local_work_size);
																																			if( verbosity>local_verbosity_threshold /*&& layer==1*/ ) {cout<<"\n\nRunCL::compute_warp( ..)_chk2 ."<<flush;				// Save buffers to file ###########
																																				stringstream ss;
																																				ss << "_binoc_" <<fname<< save_index <<"_layer_"<<mipmap_layer<<"_iter_"<<iter ;
																																				bool show 		= false;
																																				bool old_tiff 	= tiff;
																																				tiff 			= true;
																																				uint vol_layers = 1;			// NB 5 samples of possible warp.
																																				float max_range = 1.0f; 		// -1 -> gray = zero.
																																				_cl_flush_finish(m_queue, fname);
																																				DownloadAndSave_2Channel_volume( 		  warp_buf,	ss.str( ), paths.at( "warp_buf"),  	 	2*mm_size_bytes_C1,   mm_Image_size,   CV_32FC2, show, /*max_range*/-10.0f,		 vol_layers);
																																				DownloadAndSave( 			  		confidence_buf,	ss.str( ), paths.at( "confidence_buf"),   mm_size_bytes_C1,   mm_Image_size,   CV_32FC1, show, max_range);
																																				tiff 			= old_tiff;
																																			}
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::compute_warp( ..)_finished ."<<flush;}
}


void RunCL::regularize_warp( uint layer, uint iter ){
	    string fname = "RunCL::regularize_warp( )";
	int local_verbosity_threshold = V_RUNCL_IMG_VARIANCE;
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::regularize_warp( )( ..)_chk0 #############################################################"<<flush;
																																				cout << "\t local_work_size = " << local_work_size
																																				<< ",  layer = " << layer
																																				<< flush;
																																			}
	// inputs
	// __private
	//_clSetKernelArg( warp_image_kernel,  			0, sizeof( uint), 						&read_offset,					fname);		// __private	uint	read_offset,			//0
	_clSetKernelArg( regularize_warp_kernel,  		1, sizeof( uint), 						&mm_width,						fname);		// __private	uint	mm_cols,				//1
	// __local
	_clSetKernelArg( regularize_warp_kernel, 		2, (local_work_size+2)*6*sizeof(float), NULL,							fname);		// __local		float2*	local_warp				//2
	_clSetKernelArg( regularize_warp_kernel, 		3, (local_work_size+2)*3*sizeof(float), NULL,							fname);		// __local		float*	local_confidence		//3
	// __global
	_clSetKernelArg( regularize_warp_kernel,  		4, sizeof( cl_mem), 					&lookup_table_buf,				fname);		// __global 	float4*	lookup_table,			//4
	_clSetKernelArg( regularize_warp_kernel,  		5, sizeof( cl_mem), 					&warp_buf,						fname);		// __global 	float2*	warp					//5
	_clSetKernelArg( regularize_warp_kernel,  		6, sizeof( cl_mem), 					&confidence_buf,				fname);		// __global 	float2*	confidence				//6

	// outputs
	_clSetKernelArg( regularize_warp_kernel,  		7, sizeof( cl_mem), 					&warp_buf_regularized,			fname);		// __global 	float2*	new_warp				//7
	_clSetKernelArg( regularize_warp_kernel,  		8, sizeof( cl_mem), 					&confidence_buf_regularized,	fname);		// __global 	float*	new_confidence			//8
																																		if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::regularize_warp( ..)_chk1 ."<<flush;}
	layer_call_kernel( regularize_warp_kernel, m_queue, layer, local_work_size);
																																			if( verbosity>local_verbosity_threshold /*&& layer==1*/ ) {cout<<"\n\nRunCL::regularize_warp( ..)_chk2 ."<<flush;				// Save buffers to file ###########
																																				stringstream ss;
																																				ss << "_binoc__regularize_warp_" << save_index <<"_layer_"<<layer<<"_iter_"<<iter ;
																																				bool show 		= false;
																																				bool old_tiff 	= tiff;
																																				tiff 			= true;
																																				float max_range = 1.0f; 		// -1 -> gray = zero.
																																				uint vol_layers	= 1;
																																				_cl_flush_finish(m_queue, fname);
																																				DownloadAndSave_2Channel_volume( 	warp_buf_regularized,	ss.str( ), paths.at( "warp_buf_regularized"),  	 	2*mm_size_bytes_C1,   mm_Image_size,   CV_32FC2, show, /*max_range*/-10.0f,		 vol_layers);
																																				DownloadAndSave( 			  confidence_buf_regularized,	ss.str( ), paths.at( "confidence_buf_regularized"),   mm_size_bytes_C1,   mm_Image_size,   CV_32FC1, show, max_range);
																																				tiff 			= old_tiff;
																																			}

	_clSetKernelArg( regularize_warp_kernel,  		5, sizeof( cl_mem), 					&warp_buf_regularized,			fname);		// __global 	float2*	warp					//5
	_clSetKernelArg( regularize_warp_kernel,  		6, sizeof( cl_mem), 					&confidence_buf_regularized,	fname);		// __global 	float2*	confidence				//6

	// outputs
	_clSetKernelArg( regularize_warp_kernel,  		7, sizeof( cl_mem), 					&warp_buf,						fname);		// __global 	float2*	new_warp				//7
	_clSetKernelArg( regularize_warp_kernel,  		8, sizeof( cl_mem), 					&confidence_buf,				fname);		// __global 	float*	new_confidence			//8

	layer_call_kernel( regularize_warp_kernel, m_queue, layer, local_work_size);
																																			if( verbosity>local_verbosity_threshold /*&& layer==1*/ ) {cout<<"\n\nRunCL::regularize_warp( ..)_chk2 ."<<flush;				// Save buffers to file ###########
																																				stringstream ss;
																																				ss << "_binoc__regularize_warp_" << save_index <<"_layer_"<<layer<<"_iter_"<<iter ;
																																				bool show 		= false;
																																				bool old_tiff 	= tiff;
																																				tiff 			= true;
																																				float max_range = 1.0f; 		// -1 -> gray = zero.
																																				uint vol_layers	= 1;
																																				_cl_flush_finish(m_queue, fname);
																																				DownloadAndSave_2Channel_volume( 	warp_buf,	ss.str( ), paths.at( "warp_buf_regularized"),  	 	2*mm_size_bytes_C1,   mm_Image_size,   CV_32FC2, show, /*max_range*/-10.0f,		 vol_layers);
																																				DownloadAndSave( 			  confidence_buf,	ss.str( ), paths.at( "confidence_buf_regularized"),   mm_size_bytes_C1,   mm_Image_size,   CV_32FC1, show, max_range);
																																				tiff 			= old_tiff;
																																			}
	swap(warp_buf,	warp_buf_regularized);
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::regularize_warp( ..)_finished ."<<flush;}
}


void RunCL::propagate_warp( uint layer ){
	if (layer <= mm_start) return;
	string fname = "RunCL::propagate_warp( )";
	int local_verbosity_threshold = V_RUNCL_PROPAGATE_WARP;
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::propagate_warp( ..)_chk0 #############################################################"<<flush;
																																				cout << "\t local_work_size = " << local_work_size
																																				<< ",  layer = " << layer
																																				<< flush;
																																			}
	uint 	reduction 		= layer-1;
	uint 	rows_in 		= MipMap[layer*8 + MiM_READ_ROWS];
	uint 	cols_in 		= MipMap[layer*8 + MiM_READ_COLS];
	uint 	write_offset	= lookup_table_offset[reduction];
cout << "\nRunCL::propagate_warp : reduction="<<reduction<<",  rows_in="<<rows_in<<",  cols_in="<<cols_in<<",  write_offset="<<write_offset<<",  mm_width="<<mm_width<< flush;
	// inputs
	// __private
	//_clSetKernelArg( warp_image_kernel,  			0, sizeof( uint), 						&read_offset,			fname);				//	__private	uint	read_offset,			//0
	_clSetKernelArg( propagate_warp_kernel,  		1, sizeof( uint), 						&rows_in,				fname);				//	__private	uint 	rows_in,				//1
	_clSetKernelArg( propagate_warp_kernel,  		2, sizeof( uint), 						&cols_in,				fname);				//	__private	uint	cols_in,				//2
	_clSetKernelArg( propagate_warp_kernel,  		3, sizeof( uint),						&write_offset,			fname);				//	__private	uint	write_offset,			//3
	_clSetKernelArg( propagate_warp_kernel,  		4, sizeof( uint),						&mm_width,				fname);				//	__private	uint	mm_cols,				//4
	// __local
	_clSetKernelArg( propagate_warp_kernel, 		5, (local_work_size+1)*4*sizeof(float), NULL,					fname);				// __local		float2*	local_warp				//5
	// __global
	_clSetKernelArg( propagate_warp_kernel,  		6, sizeof( cl_mem), 					&lookup_table_buf,		fname);				//	__global 	float4*	lookup_table,			//6

	// input_output
	_clSetKernelArg( propagate_warp_kernel,  		7, sizeof( cl_mem), 					&warp_buf,				fname);				//	__global 	float2*	warp					//7

																																		if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::propagate_warp( ..)_chk1 ."<<flush;}
	layer_call_kernel( propagate_warp_kernel, m_queue, layer, local_work_size);
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::propagate_warp( ..)_chk2 ."<<flush;								// Save buffers to file ###########
																																				stringstream ss;
																																				ss << "_binoc__propagate_warp_" << save_index <<"_layer_"<<layer<<"_" ;
																																				bool show 				= false;
																																				bool old_tiff 			= tiff;
																																				tiff 					= true;
																																				float max_range 		= -1.0f; 		// -1 -> gray = zero.
																																				uint vol_layers			= 1;
																																				_cl_flush_finish(m_queue, fname);

																																				cout<<"\n\nRunCL::compute_warp( ..)_chk2.3 ."<<flush;
																																				DownloadAndSave_2Channel_volume( 	warp_buf,		ss.str( ), paths.at( "warp_buf"),  		2*mm_size_bytes_C1,   mm_Image_size,   CV_32FC2, 	show , /*max_range*/-10.0f, vol_layers /*1*/ /*2D warp, 1DoF */);
																																				tiff 			= old_tiff;
																																			}
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::propagate_warp( ..)_finished ."<<flush;}
}



////////////////////////////////////////////////////////////

/*
void RunCL::compute_warp( uint layer, uint iter ){
    string fname = "RunCL::compute_warp( )";
	int local_verbosity_threshold = V_RUNCL_COMPUTE_WARP;
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::compute_warp( ..)_chk0 #############################################################"<<flush;
																																				cout << "\t local_work_size = " << local_work_size
																																				<< ",  layer = " << layer
																																				<< flush;
																																			}
	// inputs
	// __private
	//_clSetKernelArg( warp_image_kernel,  			0, sizeof( uint), 	&read_offset,			fname);										// 	__private	uint	read_offset,			//0
	_clSetKernelArg( compute_warp_kernel,  			1, sizeof( uint), 	&mm_layerstep,			fname);										// 	__private	uint 	mm_size,				//1
	_clSetKernelArg( compute_warp_kernel,  			2, sizeof( uint), 	&mm_width,				fname);										// 	__private	uint	mm_cols,				//2
	// __global
	_clSetKernelArg( compute_warp_kernel,  			3, sizeof( cl_mem), &lookup_table_buf,		fname);										// 	__global 	uint4*	lookup_table,			//3
	_clSetKernelArg( compute_warp_kernel,  			4, sizeof( cl_mem), &curr_img_buf,			fname);										// 	__global 	float4*	curr_img,				//4
	_clSetKernelArg( compute_warp_kernel,  			5, sizeof( cl_mem), &new_img_buf,			fname);										// 	__global 	float4*	new_img,				//5			// NB warped version of the new image.
	_clSetKernelArg( compute_warp_kernel,  			6, sizeof( cl_mem), &curr_img_var_buf,		fname);										// 	__global 	float4*	curr_img_var,			//6
	_clSetKernelArg( compute_warp_kernel,  			7, sizeof( cl_mem), &new_img_var_buf,		fname);										// 	__global 	float4*	new_img_var,			//7

	// output
	_clSetKernelArg( compute_warp_kernel,  			8, sizeof( cl_mem), &img_covar_buf,			fname);										// 	__global 	float4*	img_covar,				//8		// 5*float4*mm_size
	_clSetKernelArg( compute_warp_kernel,  			9, sizeof( cl_mem), &img_corr_buf,			fname);										// 	__global 	float4*	img_corr,				//9		// 5*float4*mm_size
	_clSetKernelArg( compute_warp_kernel,  		   10, sizeof( cl_mem), &warp_buf,				fname);										// 	__global 	float2*	warp					//10


																																		if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::compute_warp( ..)_chk1 ."<<flush;}
	layer_call_kernel( compute_warp_kernel, m_queue, layer, local_work_size);
																																			if( verbosity>local_verbosity_threshold  ) {cout<<"\n\nRunCL::compute_warp( ..)_chk2 ."<<flush;								// Save buffers to file ###########
																																				stringstream ss;
																																				ss << "_binoc_" << save_index <<"_layer_"<<layer<<"_iter_"<<iter<<"_" ;
																																				bool show 				= false;
																																				bool display 			= false;
																																				bool old_tiff 			= tiff;
																																				tiff 					= true;
																																				uint vol_layers 		= 5;
																																				float max_range 		= 1; 		// -1 -> gray = zero.

																																				_cl_flush_finish(m_queue, fname);
							   //DownloadAndSave_3Channel(cl_mem buffer, std::string count, std::filesystem::path folder_tiff, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, cv::Mat *bufImg, float max_range / *=1* /, 					uint offset / *=0* /, bool exception_tiff / *=false* /){
						//DownloadAndSave_3Channel_volume(cl_mem buffer, std::string count, std::filesystem::path folder,      size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, 				 float max_range, 		 uint vol_layers,  						bool exception_tiff / *=false* /,  float iter,  bool display){

																																			  //DownloadAndSave_3Channel_volume(  SE3_rho_map_mem,  ss.str( ), paths.at( "SE3_rho_map_mem"),  	mm_size_bytes_C4, mm_Image_size,   CV_32FC4,	show, max_range, vol_layers, exception_tiff, count, display );
																																				cout<<"\n\nRunCL::compute_warp( ..)_chk2.1 ."<<flush;
																																				DownloadAndSave_3Channel_volume( 	img_covar_buf,	ss.str( ), paths.at( "img_covar_buf"),  mm_size_bytes_C4,   mm_Image_size,   CV_32FC4, 	show, max_range, vol_layers, tiff, iter, display);

																																				cout<<"\n\nRunCL::compute_warp( ..)_chk2.2 ."<<flush;
																																				DownloadAndSave_3Channel_volume( 	img_corr_buf,	ss.str( ), paths.at( "img_corr_buf"),  	mm_size_bytes_C4,   mm_Image_size,   CV_32FC4, 	show, max_range, vol_layers, tiff, iter, display);

																																				cout<<"\n\nRunCL::compute_warp( ..)_chk2.3 ."<<flush;
																																			  //DownloadAndSave_2Channel_volume( SE3_map_mem, 		ss.str( ), paths.at( "SE3_map_mem"),    mm_size_bytes_C1*2,   mm_Image_size,   CV_32FC2,	false, 1.0, 6 ); / *SE3, 6DoF * /
																																				DownloadAndSave_2Channel_volume( 	warp_buf,		ss.str( ), paths.at( "warp_buf"),  		2*mm_size_bytes_C1,   mm_Image_size,   CV_32FC2, 	show , -1,  2 ); / *1* / / *2D warp, 1DoF * /

																																				tiff 			= old_tiff;
																																			}
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::compute_warp( ..)_finished ."<<flush;}
}
*/


void RunCL::correlation_2nd_step( uint layer, uint iter ){
	string fname = "RunCL::correlation_2nd_step( )";
	int local_verbosity_threshold = V_RUNCL_IMG_VARIANCE;
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::correlation_2nd_step( ..)_chk0 #############################################################"<<flush;
																																				cout << "\t local_work_size = " << local_work_size
																																				<< ",  layer = " << layer
																																				<< flush;
																																			}
	// inputs
	// __private
	//_clSetKernelArg( warp_image_kernel,  				0, sizeof( uint), 	&read_offset,			fname);									// __private	uint	read_offset,		//0
	_clSetKernelArg( correlation_2nd_step_kernel,  		1, sizeof( uint), 	&mm_width,				fname);									// __private	uint	mm_cols,			//1
	_clSetKernelArg( correlation_2nd_step_kernel,  		2, sizeof( uint), 	&mm_layerstep,			fname);									// __private	uint 	mm_size,			//2
	// __global
	_clSetKernelArg( correlation_2nd_step_kernel,  		3, sizeof( cl_mem), &lookup_table_buf,		fname);									// __global 	float4*	lookup_table,		//3
	_clSetKernelArg( correlation_2nd_step_kernel,  		4, sizeof( cl_mem), &covariance_buf,		fname);									// __global 	float4*	covariance			//4
	// output
	_clSetKernelArg( correlation_2nd_step_kernel,  		5, sizeof( cl_mem), &correlation_buf,		fname);									// __global 	float4*	correlation			//5
	_clSetKernelArg( correlation_2nd_step_kernel,  		6, sizeof( cl_mem), &warp_buf,				fname);									// __global 	float2*	warp				//6
	_clSetKernelArg( correlation_2nd_step_kernel,  		7, sizeof( cl_mem), &confidence_buf,		fname);									// __global 	float*	confidence			//7
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::correlation_2nd_step( ..)_chk1 ."<<flush;}
	layer_call_kernel( correlation_2nd_step_kernel, m_queue, layer, local_work_size);
																																			if( verbosity>local_verbosity_threshold /*&& layer==1*/ ) {cout<<"\n\nRunCL::correlation( ..)_chk2 ."<<flush;				// Save buffers to file ###########
																																				stringstream ss;
																																				ss << "_binoc_"<<fname<< save_index <<"_layer_"<<layer<<"_iter_"<<iter ;
																																				bool show 		= false;
																																				bool display 	= false;
																																				bool old_tiff 	= tiff;
																																				tiff 			= true;
																																				uint vol_layers = 5;			// NB 5 samples of possible warp.
																																				float max_range = -1.0f; 		// -1 -> gray = zero.
																																				_cl_flush_finish(m_queue, fname);
																																				DownloadAndSave_3Channel_volume( 	covariance_buf,		ss.str( ), paths.at( "covariance_buf" ),  	mm_size_bytes_C4,   mm_Image_size,   CV_32FC4, 	show, 1.0f, 	vol_layers, tiff, iter, display);
																																				DownloadAndSave_3Channel_volume( 	correlation_buf,	ss.str( ), paths.at( "correlation_buf" ),  	mm_size_bytes_C4,   mm_Image_size,   CV_32FC4, 	show, 1.0f, 	vol_layers, tiff, iter, display); // just the denominator
																																				//DownloadAndSave_2Channel_volume( 	warp_buf,			ss.str( ), paths.at( "warp_buf"),  		  2*mm_size_bytes_C1,   mm_Image_size,   CV_32FC2, 	show, -1.0f, 		1 );
																																				//DownloadAndSave( 			  		confidence_buf,		ss.str( ), paths.at( "confidence_buf"),		mm_size_bytes_C1,   mm_Image_size,   CV_32FC1,  show, max_range);
																																				tiff 			= old_tiff;
																																			}
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::correlation_2nd_step( ..)_finished ."<<flush;}
}


void RunCL::mean_3rows ( uint layer, uint iter, std::string folder_mean_rows, cl_mem img_buf, cl_mem mean_rows_buf ){
    string fname = "RunCL::mean_sq_3rows( )";
	int local_verbosity_threshold = V_RUNCL_IMG_VARIANCE;
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::mean_sq_3rows( ..)_chk0 #############################################################"<<flush;
																																				cout << "\t local_work_size = " << local_work_size
																																				<< ",  layer = " << layer
																																				<< flush;
																																			}
	// inputs
	//_clSetKernelArg( warp_image_kernel,  			0, sizeof( uint), 	&read_offset,			fname);										// 	__private	uint	read_offset,			//0

	_clSetKernelArg( mean_sq_3rows_kernel,  		1, sizeof( cl_mem), &lookup_table_buf,		fname);										// 	__global 	float4*	lookup_table,			//1
	_clSetKernelArg( mean_sq_3rows_kernel,  		2, sizeof( cl_mem), &img_buf,				fname);										// 	__global 	float4*	img,					//2
	// output
	_clSetKernelArg( mean_sq_3rows_kernel,  		3, sizeof( cl_mem), &mean_rows_buf,			fname);										// 	__global 	float4*	sq_mean_rows			//3

																																		if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::mean_sq_3rows( ..)_chk1 ."<<flush;}
	layer_call_kernel( mean_sq_3rows_kernel, m_queue, layer, local_work_size);
																																			if( verbosity>local_verbosity_threshold /*&& layer==1*/ ) {cout<<"\n\nRunCL::mean_sq_3rows( ..)_chk2 ."<<flush;								// Save buffers to file ###########
																																				stringstream ss;
																																				ss << "_binoc__mean_sq_3rows_" << save_index <<"_layer_"<<layer<<"_iter_"<<iter ;
																																				bool show 		= false;
																																				bool old_tiff 	= tiff;
																																				tiff 			= true;
																																				float max_range = 1.0f; 		// -1 -> gray = zero.
																																				_cl_flush_finish(m_queue, fname);
																																				DownloadAndSave_3Channel( 	mean_rows_buf,	ss.str( ), paths.at( folder_mean_rows ),	mm_size_bytes_C4,   mm_Image_size,   CV_32FC4, 	show , max_range);
																																				tiff 			= old_tiff;
																																			}
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::mean_sq_3rows( ..)_finished ."<<flush;}
}


void RunCL::mean_cols ( uint layer, uint iter, std::string folder_mean, std::string folder_sd, cl_mem img_buf, cl_mem mean_rows_buf, cl_mem mean_buf, cl_mem diff_buf ){
    string fname = "RunCL::mean_sq_cols( )";
	int local_verbosity_threshold = V_RUNCL_IMG_VARIANCE;
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::mean_sq_cols( ..)_chk0 #############################################################"<<flush;
																																				cout << "\t local_work_size = " << local_work_size
																																				<< ",  layer = " << layer
																																				<< flush;
																																			}
	// inputs
	//_clSetKernelArg( warp_image_kernel,  			0, sizeof( uint), 	&read_offset,			fname);										// 	__private	uint	read_offset,			//0
	_clSetKernelArg( mean_sq_cols_kernel,  			1, sizeof( uint), 	&mm_width,				fname);										//	__private	uint	mm_cols,				//1

	_clSetKernelArg( mean_sq_cols_kernel,  			2, sizeof( cl_mem), &lookup_table_buf,		fname);										// 	__global 	float4*	lookup_table,			//2
	_clSetKernelArg( mean_sq_cols_kernel,  			3, sizeof( cl_mem), &mean_rows_buf,			fname);										// 	__global 	float4*	sq_mean_rows,			//3
	_clSetKernelArg( mean_sq_cols_kernel,  			4, sizeof( cl_mem), &img_buf,				fname);										// 	__global 	float4*	img,					//4
	// output
	_clSetKernelArg( mean_sq_cols_kernel,  			5, sizeof( cl_mem), &mean_buf,				fname);										// 	__global 	float4*	sq_mean					//5
	_clSetKernelArg( mean_sq_cols_kernel,  			6, sizeof( cl_mem), &diff_buf,				fname);										// 	__global 	float4*	sd						//6
																																		if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::mean_sq_cols( ..)_chk1 ."<<flush;}
	layer_call_kernel( mean_sq_cols_kernel, m_queue, layer, local_work_size);
																																			if( verbosity>local_verbosity_threshold /*&& layer==1*/ ) {cout<<"\n\nRunCL::mean_sq_cols( ..)_chk2 ."<<flush;					// Save buffers to file ###########
																																				stringstream ss;
																																				ss << "_binoc__mean_sq_cols_" << save_index <<"_layer_"<<layer<<"_iter_"<<iter ;
																																				bool show 		= false;
																																				bool old_tiff 	= tiff;
																																				tiff 			= true;
																																				float max_range = 1.0f; 		// -1 -> gray = zero.
																																				_cl_flush_finish( m_queue, fname);
																																				DownloadAndSave_3Channel( 	mean_buf,	ss.str( ), paths.at( folder_mean ),		mm_size_bytes_C4,   mm_Image_size,   CV_32FC4, 	show , max_range);
																																				DownloadAndSave_3Channel( 	diff_buf,	ss.str( ), paths.at( folder_sd ),		mm_size_bytes_C4,   mm_Image_size,   CV_32FC4, 	show , -1);
																																				tiff 			= old_tiff;
																																			}
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::mean_sq_cols( ..)_finished ."<<flush;}
}

////

void RunCL::sigma_3rows( uint layer, uint iter, std::string folder_sigma_3rows, cl_mem sd_buf, cl_mem sigma_rows_buf){
    string fname = "RunCL::sigma_3rows( )";
	int local_verbosity_threshold = V_RUNCL_IMG_VARIANCE;
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::sigma_3rows( ..)_chk0 #############################################################"<<flush;
																																				cout << "\t local_work_size = " << local_work_size
																																				<< ",  layer = " << layer
																																				<< flush;
																																			}
	// inputs
	// __private
	//_clSetKernelArg( warp_image_kernel,  		0, sizeof( uint), 	&read_offset,			fname);										// 	__private	uint	read_offset,			//0
	// __global
	_clSetKernelArg( sigma_3rows_kernel,  		1, sizeof( cl_mem), &lookup_table_buf,		fname);										// 	__global 	float4*	lookup_table,			//1
	_clSetKernelArg( sigma_3rows_kernel,  		2, sizeof( cl_mem), &sd_buf,				fname);										// 	__global 	float4*	ref_img,				//2
	// output
	_clSetKernelArg( sigma_3rows_kernel,  		3, sizeof( cl_mem), &sigma_rows_buf,		fname);										// 	__global 	float4*	co_mean_rows			//3
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::sigma_3rows( ..)_chk1 ."<<flush;}
	layer_call_kernel( sigma_3rows_kernel, m_queue, layer, local_work_size);
																																			if( verbosity>local_verbosity_threshold /*&& layer==1*/ ) {cout<<"\n\nRunCL::sigma_3rows( ..)_chk2 ."<<flush;					// Save buffers to file ###########
																																				stringstream ss;
																																				ss << "_binoc__sigma__mean_rows_" << save_index <<"_layer_"<<layer<<"_iter_"<<iter ;
																																				bool show 		= false;
																																				bool old_tiff 	= tiff;
																																				tiff 			= true;
																																				float max_range = -1.0f; 		// -1 -> gray = zero.
																																				_cl_flush_finish(m_queue, fname);
																																				DownloadAndSave_3Channel( 	sigma_rows_buf,	ss.str( ), paths.at( folder_sigma_3rows ),  mm_size_bytes_C4,   mm_Image_size,   CV_32FC4, 	show, max_range);
																																				tiff 			= old_tiff;
																																			}
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::sigma_3rows( ..)_finished ."<<flush;}
}

//	ref_img_mean_rows_buf, warped_img_mean_rows_buf, warped_img_mean_rows_buf;
//	ref_img_mean_buf,      warped_img_mean_buf,      co_mean_rows_buf, correlation_buf;


void RunCL::sigma_3cols( uint layer, uint iter, std::string folder_sigma_3cols, cl_mem sigma_rows_buf, cl_mem mean_sigma_buf ){
    string fname = "RunCL::sigma_3cols( )";
	int local_verbosity_threshold = V_RUNCL_IMG_VARIANCE;
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::sigma_3cols( ..)_chk0 #############################################################"<<flush;
																																				cout << "\t local_work_size = " << local_work_size
																																				<< ",  layer = " << layer
																																				<< flush;
																																			}
	// inputs
	// __private
	//_clSetKernelArg( warp_image_kernel,  			0, sizeof( uint), 	&read_offset,				fname);								// __private	uint	read_offset,			//0
	_clSetKernelArg( sigma_3cols_kernel,  			1, sizeof( uint), 	&mm_width,					fname);								// __private	uint	mm_cols,				//1
	// __global
	_clSetKernelArg( sigma_3cols_kernel,  			2, sizeof( cl_mem), &lookup_table_buf,			fname);								// __global 	float4*	lookup_table,			//2
	_clSetKernelArg( sigma_3cols_kernel,  			3, sizeof( cl_mem), &sigma_rows_buf,			fname);								// __global 	float4*	co_mean_rows,			//3
	// output
	_clSetKernelArg( sigma_3cols_kernel,  			4, sizeof( cl_mem), &mean_sigma_buf,			fname);								// __global 	float4*	correlation				//4
																																		if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::sigma_3cols( ..)_chk1 ."<<flush;}
	layer_call_kernel( sigma_3cols_kernel, m_queue, layer, local_work_size);
																																			if( verbosity>local_verbosity_threshold /*&& layer==1*/ ) {cout<<"\n\nRunCL::sigma_3cols( ..)_chk2 ."<<flush;				// Save buffers to file ###########
																																				stringstream ss;
																																				ss << "_binoc__covariance_cols_" << save_index <<"_layer_"<<layer<<"_iter_"<<iter ;
																																				bool show 		= false;
																																				bool old_tiff 	= tiff;
																																				tiff 			= true;
																																				float max_range = -1.0f; 		// -1 -> gray = zero.
																																				_cl_flush_finish(m_queue, fname);
																																				DownloadAndSave_3Channel( 	mean_sigma_buf,	ss.str( ), paths.at( folder_sigma_3cols ),  mm_size_bytes_C4,   mm_Image_size,   CV_32FC4, 	show, max_range);
																																				tiff 			= old_tiff;
																																			}
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::sigma_3cols( ..)_finished ."<<flush;}
}

////

void RunCL::covariance_3rows( uint layer, uint iter){
    string fname = "covariance_3rows( )";
	int local_verbosity_threshold = V_RUNCL_IMG_VARIANCE;
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::covariance_3rows( ..)_chk0 #############################################################"<<flush;
																																				cout << "\t local_work_size = " << local_work_size
																																				<< ",  layer = " << layer
																																				<< flush;
																																			}
	// inputs
	// __private
	//_clSetKernelArg( warp_image_kernel,  		0, sizeof( uint), 	&read_offset,			fname);										// 	__private	uint	read_offset,			//0
	_clSetKernelArg( covariance_3rows_kernel,	1, sizeof( uint), 	&mm_width,				fname);										// 	__private	uint	mm_cols,				//1
	_clSetKernelArg( covariance_3rows_kernel,  	2, sizeof( uint), 	&mm_layerstep,			fname);										//	__private	uint	mm_cols,				//2
	// __global
	_clSetKernelArg( covariance_3rows_kernel,	3, sizeof( cl_mem), &lookup_table_buf,		fname);										// 	__global 	float4*	lookup_table,			//3
	_clSetKernelArg( covariance_3rows_kernel,	4, sizeof( cl_mem), &ref_img_diff_buf,		fname);										// 	__global 	float4*	ref_img,				//4
	_clSetKernelArg( covariance_3rows_kernel,	5, sizeof( cl_mem), &warped_img_diff_buf,	fname);										// 	__global 	float4*	warped_img,				//5
	// output
	_clSetKernelArg( covariance_3rows_kernel,	6, sizeof( cl_mem), &covariance_rows_buf,	fname);										// 	__global 	float4*	covariance_rows			//6
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::covariance_3rows( ..)_chk1 ."<<flush;}
	layer_call_kernel( covariance_3rows_kernel, m_queue, layer, local_work_size);
																																			if( verbosity>local_verbosity_threshold /*&& layer==1*/ ) {cout<<"\n\nRunCL::covariance_3rows( ..)_chk2 ."<<flush;					// Save buffers to file ###########
																																				stringstream ss;
																																				ss << "_binoc__co_mean_rows_" << save_index <<"_layer_"<<layer<<"_iter_"<<iter ;
																																				bool show 		= false;
																																				bool display 	= false;
																																				bool old_tiff 	= tiff;
																																				tiff 			= true;
																																				uint vol_layers = 5;			// NB 5 samples of possible warp.
																																				float max_range = -1.0f; 		// -1 -> gray = zero.
																																				_cl_flush_finish(m_queue, fname);
																																				//DownloadAndSave_3Channel( 		co_mean_rows_buf,	ss.str( ), paths.at( "co_mean_rows_buf" ),	mm_size_bytes_C4,   mm_Image_size,   CV_32FC4, 	show, max_range);
																																				DownloadAndSave_3Channel_volume( 	covariance_rows_buf,	ss.str( ), paths.at( "covariance_rows_buf" ),  mm_size_bytes_C4,   mm_Image_size,   CV_32FC4, 	show, max_range, vol_layers, tiff, iter, display);
																																				tiff 			= old_tiff;
																																			}
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::covariance_3rows( ..)_finished ."<<flush;}
}


void RunCL::covariance_cols( uint layer, uint iter){//, std::string folder_covariance_cols, cl_mem sigma_rows_buf, cl_mem mean_sigma_buf ){
    string fname = "RunCL::covariance_cols( )";
	int local_verbosity_threshold = V_RUNCL_IMG_VARIANCE;
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::covariance_cols( ..)_chk0 #############################################################"<<flush;
																																				cout << "\t local_work_size = " << local_work_size
																																				<< ",  layer = " << layer
																																				<< flush;
																																			}
	// inputs
	// __private
	//_clSetKernelArg( warp_image_kernel,			0, sizeof( uint), 	&read_offset,				fname);								// __private	uint	read_offset,			//0
	_clSetKernelArg( covariance_cols_kernel,		1, sizeof( uint), 	&mm_width,					fname);								// __private	uint	mm_cols,				//1
	_clSetKernelArg( covariance_cols_kernel,		2, sizeof( uint), 	&mm_layerstep,				fname);								//	__private	uint	mm_cols,				//2
	// __global
	_clSetKernelArg( covariance_cols_kernel,		3, sizeof( cl_mem), &lookup_table_buf,			fname);								// __global 	float4*	lookup_table,			//3
	_clSetKernelArg( covariance_cols_kernel,		4, sizeof( cl_mem), &covariance_rows_buf,		fname);								// __global 	float4*	co_mean_rows,			//4
	// output
	_clSetKernelArg( covariance_cols_kernel,		5, sizeof( cl_mem), &covariance_buf,			fname);								// __global 	float4*	correlation				//5
																																		if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::covariance_cols( ..)_chk1 ."<<flush;}
	layer_call_kernel( covariance_cols_kernel, m_queue, layer, local_work_size);
																																			if( verbosity>local_verbosity_threshold /*&& layer==1*/ ) {cout<<"\n\nRunCL::covariance_cols( ..)_chk2 ."<<flush;				// Save buffers to file ###########
																																				stringstream ss;
																																				ss << "_binoc__covariance_cols_" << save_index <<"_layer_"<<layer<<"_iter_"<<iter ;
																																				bool show 		= false;
																																				bool display 	= false;
																																				bool old_tiff 	= tiff;
																																				tiff 			= true;
																																				uint vol_layers = 5;			// NB 5 samples of possible warp.
																																				float max_range = -1.0f; 		// -1 -> gray = zero.
																																				_cl_flush_finish(m_queue, fname);
																																				DownloadAndSave_3Channel_volume( 	covariance_buf,	ss.str( ), paths.at( "covariance_buf" ),  mm_size_bytes_C4,   mm_Image_size,   CV_32FC4, 	show, max_range, vol_layers, tiff, iter, display);
																																				tiff 			= old_tiff;
																																			}
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::covariance_cols( ..)_finished ."<<flush;}
}


void RunCL::correlation( uint layer, uint iter ){
    string fname = "RunCL::covariance_cols( )";
	int local_verbosity_threshold = V_RUNCL_IMG_VARIANCE;
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::correlation( ..)_chk0 #############################################################"<<flush;
																																				cout << "\t local_work_size = " << local_work_size
																																				<< ",  layer = " << layer
																																				<< flush;
																																			}
	// inputs
	// __private
	//_clSetKernelArg( warp_image_kernel,  			0, sizeof( uint), 	&read_offset,				fname);								// __private	uint	read_offset,			//0
	_clSetKernelArg( correlation_kernel,  			1, sizeof( uint), 	&mm_width,					fname);								// __private	uint	mm_cols,				//1
	_clSetKernelArg( correlation_kernel,  			2, sizeof( uint), 	&mm_layerstep,				fname);								// __private	uint	mm_size,				//2
	// __global
	_clSetKernelArg( correlation_kernel,  			3, sizeof( cl_mem), &lookup_table_buf,			fname);								// __global 	float4*	lookup_table,			//3
	_clSetKernelArg( correlation_kernel,  			4, sizeof( cl_mem), &ref_img_mean_sigma_buf,	fname);								// __global 	float4*	sd_ref_img,				//4
	_clSetKernelArg( correlation_kernel,  			5, sizeof( cl_mem), &warped_img_mean_sigma_buf,	fname);								// __global 	float4*	sd_warped_img,			//5
	_clSetKernelArg( correlation_kernel,  			6, sizeof( cl_mem), &covariance_rows_buf,		fname);								// __global 	float4*	covariance,				//6		// sizeof(float4) * 5 * mm_size

	// output
	_clSetKernelArg( correlation_kernel,  			7, sizeof( cl_mem), &correlation_buf,			fname);								// __global 	float4*	correlation				//7		// sizeof(float4) * 5 * mm_size
	_clSetKernelArg( correlation_kernel,  			8, sizeof( cl_mem), &warp_buf,					fname);								// __global 	float2*	warp					//8
	_clSetKernelArg( correlation_kernel,  			9, sizeof( cl_mem), &confidence_buf,			fname);								// __global 	float*	confidence				//9
																																		if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::correlation( ..)_chk1 ."<<flush;}
	layer_call_kernel( correlation_kernel, m_queue, layer, local_work_size);
																																			if( verbosity>local_verbosity_threshold /*&& layer==1*/ ) {cout<<"\n\nRunCL::correlation( ..)_chk2 ."<<flush;				// Save buffers to file ###########
																																				stringstream ss;
																																				ss << "_binoc__correlation_" << save_index <<"_layer_"<<layer<<"_iter_"<<iter ;
																																				bool show 		= false;
																																				bool display 	= false;
																																				bool old_tiff 	= tiff;
																																				tiff 			= true;
																																				uint vol_layers = 5;			// NB 5 samples of possible warp.
																																				float max_range = -1.0f; 		// -1 -> gray = zero.
																																				_cl_flush_finish(m_queue, fname);
																																				DownloadAndSave_3Channel_volume( 	correlation_buf,	ss.str( ), paths.at( "correlation_buf" ),  	mm_size_bytes_C4,   mm_Image_size,   CV_32FC4, 	show, -1.0f, 	vol_layers, tiff, iter, display);
																																				DownloadAndSave_2Channel_volume( 	warp_buf,			ss.str( ), paths.at( "warp_buf"),  		  2*mm_size_bytes_C1,   mm_Image_size,   CV_32FC2, 	show, -1.0f, 		1 );
																																				DownloadAndSave( 			  		confidence_buf,		ss.str( ), paths.at( "confidence_buf"),		mm_size_bytes_C1,   mm_Image_size,   CV_32FC1,  show, max_range);
																																				tiff 			= old_tiff;
																																			}
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::correlation( ..)_finished ."<<flush;}
}
