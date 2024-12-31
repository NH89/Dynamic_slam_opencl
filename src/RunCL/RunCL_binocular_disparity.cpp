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

// void RunCL::binocular_disparity_copy_buffers(  ){
// 	string fname = "RunCL::binocular_disparity_copy_buffers( )";
// 	int local_verbosity_threshold = V_RUNCL_WARP_IMAGE;
// 																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::binocular_disparity_copy_buffers( ..)_chk0 #############################################################"<<flush;}
// 	_clEnqueueCopyBuffer( m_queue, keyframe_imgmem, 	curr_img_buf, 			0, 0, mm_size_bytes_C4, 	fname);
// 	_clEnqueueCopyBuffer( m_queue, imgmem, 				new_img_buf, 			0, 0, mm_size_bytes_C4, 	fname);
// 																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::binocular_disparity_copy_buffers( ..)_chk_finished"<<flush;
// }


void RunCL::warp_image( uint layer, uint iter ){																					// computed once each iteration of warping, for each layer of image pyramid
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
	// __constant
	_clSetKernelArg( warp_image_kernel,  			2, sizeof( cl_mem), &uint_param_buf,		fname);										// 	__constant 	uint*	uint_params,			//2
	// __global
	_clSetKernelArg( warp_image_kernel,  			3, sizeof( cl_mem), &warp_buf,				fname);										// 	__global 	float2*	warp,					//3
	_clSetKernelArg( warp_image_kernel,  			4, sizeof( cl_mem), &lookup_table_buf,		fname);										// 	__global 	uint4*	lookup_table,			//4
	_clSetKernelArg( warp_image_kernel,  			5, sizeof( cl_mem), &new_img_buf,			fname);										// 	__global 	float4*	new_img,				//5

	// outputs
	_clSetKernelArg( warp_image_kernel,  			6, sizeof( cl_mem), &new_img_warped_buf,	fname);										// 	__global 	float4*	new_img_warped			//6
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::warp_image( ..)_chk1 ."<<flush;}
	layer_call_kernel( warp_image_kernel, m_queue, layer, local_work_size);
																																			if( verbosity>local_verbosity_threshold && layer==0 ) {cout<<"\n\nRunCL::warp_image( ..)_chk2 ."<<flush;								// Save buffers to file ###########
																																				stringstream ss;
																																				ss << "_binoc_" << save_index <<"_layer_"<<layer<<"_iter_"<<iter ;
																																				bool show 		= false;
																																				bool old_tiff 	= tiff;
																																				tiff 			= true;
																																				float max_range = -1; 		// -1 -> gray = zero.
																																				_cl_flush_finish(m_queue, fname);
																																				DownloadAndSave_3Channel( 	new_img_warped_buf,	ss.str( ), paths.at( "new_img_warped_buf"),  		mm_size_bytes_C4,   mm_Image_size,   CV_32FC4, 	show , max_range);
																																				tiff 			= old_tiff;
																																			}
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::warp_image( ..)_finished ."<<flush;}
}


void RunCL::img_sq( uint layer, uint iter, cl_mem img_buf, cl_mem img_sq_buf, std::string folder){
    string fname = "RunCL::img_sq( )";
	int local_verbosity_threshold = V_RUNCL_IMG_SQ;
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::img_sq( ..)_chk0 #############################################################"<<flush;
																																				cout << "\t local_work_size = " << local_work_size
																																				<< ",  layer = " << layer
																																				<< flush;
																																			}
	// inputs
	// __private
	//_clSetKernelArg( warp_image_kernel,  			0, sizeof( uint), &read_offset,				fname);										// 	__private	uint	read_offset,			//0
	// __global
	_clSetKernelArg( img_sq_kernel,  				1, sizeof( cl_mem), &lookup_table_buf,		fname);										// 	__global 	uint4*	lookup_table,			//0
	_clSetKernelArg( img_sq_kernel,		  			2, sizeof( cl_mem), &img_buf,				fname);										// 	__global 	float4*	img,					//1
	// output
	_clSetKernelArg( img_sq_kernel,		  			3, sizeof( cl_mem), &img_sq_buf,			fname);										// 	__global 	float4*	img_sq					//2
																																		if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::img_sq( ..)_chk1 ."<<flush;}
	layer_call_kernel( img_sq_kernel, m_queue, layer, local_work_size);
																																			if( verbosity>local_verbosity_threshold && layer==0 ) {cout<<"\n\nRunCL::img_sq( ..)_chk2 ."<<flush;								// Save buffers to file ###########
																																				stringstream ss;
																																				ss << "_binoc_" << save_index <<"_layer_"<<layer<<"_iter_"<<iter ;
																																				bool show 		= false;
																																				bool old_tiff 	= tiff;
																																				tiff 			= true;
																																				float max_range = 1; 		// -1 -> gray = zero.
																																				_cl_flush_finish(m_queue, fname);
																																				DownloadAndSave_3Channel( 	img_sq_buf,	ss.str( ), paths.at( folder ),  mm_size_bytes_C4,   mm_Image_size,   CV_32FC4, 	show , max_range);
																																				tiff 			= old_tiff;
																																			}
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::img_sq( ..)_finished ."<<flush;}
}


void RunCL::img_variance( uint layer, uint iter, cl_mem img_sq_buf, cl_mem img_var_buf, std::string folder){
    string fname = "RunCL::img_variance( )";
	int local_verbosity_threshold = V_RUNCL_IMG_VARIANCE;
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::img_variance( ..)_chk0 #############################################################"<<flush;
																																				cout << "\t local_work_size = " << local_work_size
																																				<< ",  layer = " << layer
																																				<< flush;
																																			}
	// inputs
	// __private
	//_clSetKernelArg( warp_image_kernel,  			0, sizeof( uint), 	&read_offset,			fname);										// 	__private	uint	read_offset,			//0
	_clSetKernelArg( img_variance_kernel,  			1, sizeof( uint), 	&mm_width,				fname);										// 	__private	uint	mm_cols,				//1
	// __global
	_clSetKernelArg( img_variance_kernel,  			2, sizeof( cl_mem), &lookup_table_buf,		fname);										// 	__global 	uint4*	lookup_table,			//2
	_clSetKernelArg( img_variance_kernel,  			3, sizeof( cl_mem), &img_sq_buf,			fname);										// 	__global 	float4*	img_sq					//3

	// output
	_clSetKernelArg( img_variance_kernel,  			4, sizeof( cl_mem), &img_var_buf,			fname);										// 	__global 	float4*	img_var					//4
																																		if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::img_variance( ..)_chk1 ."<<flush;}
	layer_call_kernel( img_variance_kernel, m_queue, layer, local_work_size);
																																			if( verbosity>local_verbosity_threshold && layer==0 ) {cout<<"\n\nRunCL::img_variance( ..)_chk2 ."<<flush;								// Save buffers to file ###########
																																				stringstream ss;
																																				ss << "_binoc_" << save_index <<"_layer_"<<layer<<"_iter_"<<iter ;
																																				bool show 		= false;
																																				bool old_tiff 	= tiff;
																																				tiff 			= true;
																																				float max_range = 1; 		// -1 -> gray = zero.
																																				_cl_flush_finish(m_queue, fname);
																																				DownloadAndSave_3Channel( 	img_var_buf,	ss.str( ), paths.at( folder ),	mm_size_bytes_C4,   mm_Image_size,   CV_32FC4, 	show , max_range);
																																				tiff 			= old_tiff;
																																			}
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::img_variance( ..)_finished ."<<flush;}
}


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
																																			if( verbosity>local_verbosity_threshold && layer==0  ) {cout<<"\n\nRunCL::compute_warp( ..)_chk2 ."<<flush;								// Save buffers to file ###########
																																				stringstream ss;
																																				ss << "_binoc_" << save_index <<"_layer_"<<layer<<"_iter_"<<iter<<"_" ;
																																				bool show 				= false;
																																				bool display 			= false;
																																				bool old_tiff 			= tiff;
																																				tiff 					= true;
																																				uint vol_layers 		= 5;
																																				float max_range 		= 1; 		// -1 -> gray = zero.

																																				_cl_flush_finish(m_queue, fname);
							   //DownloadAndSave_3Channel(cl_mem buffer, std::string count, std::filesystem::path folder_tiff, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, cv::Mat *bufImg, float max_range /*=1*/, 					uint offset /*=0*/, bool exception_tiff /*=false*/){
						//DownloadAndSave_3Channel_volume(cl_mem buffer, std::string count, std::filesystem::path folder,      size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, 				 float max_range, 		 uint vol_layers,  						bool exception_tiff /*=false*/,  float iter,  bool display){

																																			  //DownloadAndSave_3Channel_volume(  SE3_rho_map_mem,  ss.str( ), paths.at( "SE3_rho_map_mem"),  	mm_size_bytes_C4, mm_Image_size,   CV_32FC4,	show, max_range, vol_layers, exception_tiff, count, display );
																																				cout<<"\n\nRunCL::compute_warp( ..)_chk2.1 ."<<flush;
																																				DownloadAndSave_3Channel_volume( 	img_covar_buf,	ss.str( ), paths.at( "img_covar_buf"),  mm_size_bytes_C4,   mm_Image_size,   CV_32FC4, 	show, max_range, vol_layers, tiff, iter, display);

																																				cout<<"\n\nRunCL::compute_warp( ..)_chk2.2 ."<<flush;
																																				DownloadAndSave_3Channel_volume( 	img_corr_buf,	ss.str( ), paths.at( "img_corr_buf"),  	mm_size_bytes_C4,   mm_Image_size,   CV_32FC4, 	show, max_range, vol_layers, tiff, iter, display);

																																				cout<<"\n\nRunCL::compute_warp( ..)_chk2.3 ."<<flush;
																																			  //DownloadAndSave_2Channel_volume( SE3_map_mem, 		ss.str( ), paths.at( "SE3_map_mem"),    mm_size_bytes_C1*2,   mm_Image_size,   CV_32FC2,	false, 1.0, 6 /*SE3, 6DoF */);
																																				DownloadAndSave_2Channel_volume( 	warp_buf,		ss.str( ), paths.at( "warp_buf"),  		2*mm_size_bytes_C1,   mm_Image_size,   CV_32FC2, 	show , -1, 2 /*1*/ /*2D warp, 1DoF */);

																																			  //DownloadAndSave_3Channel_volume( 	warp_buf,		ss.str( ), paths.at( "warp_buf"),  		mm_size_bytes_C4/*2*mm_size_bytes_C1*/,   mm_Image_size,   CV_32FC4/*C2*/, 	show , -1, 2 /*1*/ /*2D warp, 1DoF */);
																																				tiff 			= old_tiff;
																																			}
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::compute_warp( ..)_finished ."<<flush;}
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
	uint 	rows_in 		= MipMap[reduction*8 + MiM_READ_ROWS];
	uint 	cols_in 		= MipMap[reduction*8 + MiM_READ_COLS];
	uint 	write_offset	= lookup_table_offset[reduction];
cout << "\nRunCL::propagate_warp : reduction="<<reduction<<",  rows_in="<<rows_in<<",  cols_in="<<cols_in<<",  write_offset="<<write_offset<<",  mm_width="<<mm_width<< flush;
	// inputs
	// __private
	//_clSetKernelArg( warp_image_kernel,  			0, sizeof( uint), 	&read_offset,			fname);										//	__private	uint	read_offset,			//0
	_clSetKernelArg( propagate_warp_kernel,  			1, sizeof( uint), 	&rows_in,				fname);										//	__private	uint 	rows_in,				//1
	_clSetKernelArg( propagate_warp_kernel,  			2, sizeof( uint), 	&cols_in,				fname);										//	__private	uint	cols_in,				//2
	_clSetKernelArg( propagate_warp_kernel,  			3, sizeof( uint),	&write_offset,			fname);										//	__private	uint	write_offset,			//3
	_clSetKernelArg( propagate_warp_kernel,  			4, sizeof( uint),	&mm_width,				fname);										//	__private	uint	mm_cols,				//4
	// __global
	_clSetKernelArg( propagate_warp_kernel,  			5, sizeof( cl_mem), &lookup_table_buf,		fname);										//	__global 	float4*	lookup_table,			//5

	// input_output
	_clSetKernelArg( propagate_warp_kernel,  			6, sizeof( cl_mem), &warp_buf,				fname);										//	__global 	float2*	warp					//6

																																		if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::propagate_warp( ..)_chk1 ."<<flush;}
	layer_call_kernel( propagate_warp_kernel, m_queue, layer, local_work_size);
																																			if( verbosity>local_verbosity_threshold && layer==0  ) {cout<<"\n\nRunCL::propagate_warp( ..)_chk2 ."<<flush;								// Save buffers to file ###########
																																				stringstream ss;
																																				ss << "_binoc_propagate_warp_" << save_index <<"_layer_"<<layer<<"_" ;
																																				bool show 				= false;
																																				bool old_tiff 			= tiff;
																																				tiff 					= true;
																																				float max_range 		= -1; 		// -1 -> gray = zero.

																																				_cl_flush_finish(m_queue, fname);

																																				cout<<"\n\nRunCL::compute_warp( ..)_chk2.3 ."<<flush;
																																				DownloadAndSave_2Channel_volume( 	warp_buf,		ss.str( ), paths.at( "warp_buf"),  		2*mm_size_bytes_C1,   mm_Image_size,   CV_32FC2, 	show , max_range, 2 /*1*/ /*2D warp, 1DoF */);
																																				tiff 			= old_tiff;
																																			}
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::propagate_warp( ..)_finished ."<<flush;}
}
