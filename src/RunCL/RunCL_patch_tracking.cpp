#include "RunCL.hpp"

// old functions

void RunCL::precomp_param_maps ( float SE3_k2k[  max_mipmap_layers*num_SE3_DoF*16  ]){ //  Compute maps of pixel motion for each SE3 DoF, and camera params // Derived from RunCL::mipmap
	string fname = "RunCL::precom_param_maps( ..)";
	int local_verbosity_threshold = V_RUNCL_PRECOM_PARAM_MAPS;
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::precom_param_maps( float SE3_k2k[6*16])_chk_0 "<<flush;}
	// cv::Mat depth		= cv::Mat::ones ( mm_height, mm_width, CV_32FC1);																	// NB must recompute translation maps at run time. NB parallax motion is proportional to inv depth.
	// float mid_depth 	= ( fp32_params[MAX_INV_DEPTH] + fp32_params[MIN_INV_DEPTH])/2.0;                                                   // TO DO fix : depthmap not used as a kernel arg. NB want to match scale of depth range, but ? parallax may vary.
	// depth 				*= mid_depth;
	//_clEnqueueWriteBuffer( uload_queue, depth_mem_temp,	CL_FALSE, 0, mm_size_bytes_C1,	 								depth.data,		fname);

	// float fx			= current_frames[ current_frames_idx[0] ].K(0,0);
	// float fy			= current_frames[ current_frames_idx[0] ].K(1,1);
	float inv_depth		= 1; //(fx + fy)/2.0f;																									// sets inverse relative depth = focal_length
																																				// Makes 1 pixel ST3 and SE3 easy to set.
																																				// NB must convert GT depth <=> relative depth.
	_clEnqueueWriteBuffer( uload_queue, SE3_k2kbuf,		CL_FALSE, 0, max_mipmap_layers*num_SE3_DoF*16*sizeof( float), 	SE3_k2k,		fname);

	//      __private	 uint layer, set in mipmap_call_kernel( ..) below                                                                      __private	 uint	 layer,			//0
	_clSetKernelArg( comp_param_maps_kernel, 1, sizeof( float),		&inv_depth,	 fname);														//__private	float 	inv_depth,		//1
	_clSetKernelArg( comp_param_maps_kernel, 2, sizeof( cl_mem),	&mipmap_buf, fname);														//__constant uint*	mipmap_params,	//2
	_clSetKernelArg( comp_param_maps_kernel, 3, sizeof( cl_mem), 	&uint_param_buf, fname);													//__global 	uint*	uint_params		//3
	_clSetKernelArg( comp_param_maps_kernel, 4, sizeof( cl_mem), 	&SE3_k2kbuf, fname);														//__global 	float* 	k2k,			//4
	_clSetKernelArg( comp_param_maps_kernel, 5, sizeof( cl_mem), 	&SE3_map_mem, fname);														//__global 	float* 	SE3_map,		//5
																																			if( verbosity>local_verbosity_threshold) {cout<<"\nRunCL::precom_param_maps( float SO3_k2k[6*16])_chk_1 "<<flush;}
	// SE3_map_mem, k_map_mem, dist_map_mem;
	mipmap_call_kernel( comp_param_maps_kernel, m_queue );
																																			if( verbosity>local_verbosity_threshold) {
																																				cout<<"\n\nRunCL::precom_param_maps( float SO3_k2k[6*16])_output "<<flush;
																																				stringstream ss;	ss << dataset_frame_num << "_SE3_map";
																																				float max_range = 0.0f;		// i.e. find max value, and map 0.0->0.5.
																																				DownloadAndSave_2Channel_volume( SE3_map_mem, ss.str( ), paths.at( "SE3_map_mem"), mm_size_bytes_C1*2, mm_Image_size, CV_32FC2, false, max_range, num_SE3_DoF );

																																				cout<<"\nRunCL::precom_param_maps( float SE3_k2k[6*16])_chk.. Finished "<<flush;
																																			}
}

void RunCL::update_tracking_depthmap(cl_mem depthmap_){
	string fname = "RunCL::update_tracking_depthmap(cl_mem depthmap_)";

	_clEnqueueCopyBuffer( m_queue, depthmap_, depth_mem, 0, 0, mm_size_bytes_C1, fname);
}


void RunCL::update_k2k_buf( float k2k_array[16],		float pose_arry[16] ) {
	string fname = "RunCL::update_k2k_buf( ..)";
	int local_verbosity_threshold = V_RUNCL_UPDATE_K2K_BUF;
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::update_k2k_buf( ..)_chk0 .##################################################################"<<flush;
																																				PRINT_FLOAT_16( k2k_array, );
																																				PRINT_FLOAT_16( pose_arry, );
																																			}
	_clEnqueueWriteBuffer( uload_queue, 	k2kbuf,		CL_FALSE, 0, 16*sizeof( float), k2k_array,  	fname);
	_clEnqueueWriteBuffer( uload_queue, 	pose_buf,	CL_FALSE, 0, 16*sizeof( float), pose_arry, 		fname);

	for (int i=0; i<16; i++){	current_frames[	current_frames_idx[0]	].k2k_0to1_est[i]	=	k2k_array[i];	}
	for (int i=0; i<16; i++){	current_frames[	current_frames_idx[0]	].pose[i]			=	pose_arry[i];	}
}

void RunCL::update_k2k_buf( 	Matx44f k2k, 	Matx44f pose ){
	float 	k2k_array[16], 	pose_array[16];
	Matx44f_To_float16arry(	k2k,	k2k_array );
	Matx44f_To_float16arry( pose,	pose_array );

	update_k2k_buf( k2k_array, pose_array);
}


// new functions ////////////////////////////////////////////////

// Image pyramid

void RunCL::build_img_pyramid( std::string folder ){
	string fname = "RunCL::build_img_pyramid()";
	int local_verbosity_threshold = V_RUNCL_REDUCE_IMG;
	int layer		= 0;																										if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::build_img_pyramid(..)_chk0"<<flush;
																																	cout << "\nlayer = "<<layer<<flush;
																																	cout << "\nmm_stop = "<<mm_stop<<flush;
																																}
	reduce_img( layer, folder);
	for (layer=1; layer<mm_stop ; layer++){
																																if(verbosity>local_verbosity_threshold) {	cout<<"\n\nRunCL::build_img_pyramid(..) pyramid layer = "<<layer<<flush;
																																	cout << "\nlayer = "<<layer<<flush;
																																}
		blur_image_layer( layer);
		reduce_img( layer, folder);
	}
	size_t			local_size		= local_work_size;
	const cl_mem 	imgmem_			= current_frames[	current_frames_idx[0] ].img_buf;
																																if(verbosity>local_verbosity_threshold) {
																																	cout << "\nlast layer = "<<layer<<flush;
																																	cout<<"\n\nRunCL::build_img_pyramid(..)_chk3 Finished all loops."<<flush;
																																	stringstream ss;	ss << dataset_frame_num << "build_img_pyramid";
																																	cv::Size new_Image_size = cv::Size(mm_width, mm_height);
																																	size_t   new_size_bytes = mm_width * mm_height * 4*4;
																																	ss << "_raw_";
																																	DownloadAndSave_3Channel( imgmem_, ss.str(), paths.at(folder), new_size_bytes, new_Image_size, CV_32FC4, false, 1, 0, true );
																																	cout << "\n  (local_size+4) *5*4* sizeof(float) = "<<  (local_size+4) *5*4* sizeof(float) << " ,   (local_size+4) = " <<  (local_size+4) << endl << flush;
																																}
																																if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::build_img_pyramid(..)_chk4 Finished"<<flush;}
}

void RunCL::blur_image_layer( uint layer ){
		pad_image_top_bottom2(	layer );
		vertcal_blur5(			layer );
		pad_image_left_right2(	layer );
		horiz_blur5(			layer );
}

void RunCL::pad_image_top_bottom2( uint layer ){
	string fname = "RunCL::pad_image_top_bottom2()";
	int local_verbosity_threshold 	= V_RUNCL_REDUCE_IMG;																		if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::pad_image_top_bottom2(..)_chk0"<<flush;}
	cl_kernel 		kernel 			= pad_image_top_bottom2_kernel;
	const cl_mem 	imgmem_			= current_frames[	current_frames_idx[0] ].img_buf;

	size_t	local_work_size_		= patch_img_gradients_workgroup_size[	layer];							// From void RunCL::patch_img_gradients( uint layer )
	size_t	threads_to_launch		= patch_num_threads[					layer];
	uint	offset1					= MipMap[layer * 8 + MiM_READ_OFFSET];									//0	top left corner
	uint	buf_width				= uint_params[MM_COLS];													//2 mm_cols, i.e. width of the buffer holding the image pyramid
	uint	offset2					= offset1 + buf_width * (MipMap[layer * 8 + MiM_READ_ROWS] - 1);		//1 bottom left corner
	uint	img_cols				= MipMap[layer * 8 + MiM_READ_COLS];									//3 num cols of this level of the image pyramid

																																if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::pad_image_top_bottom2(..)_chk1"<<flush;
																																	cout \
																																	<<"\n layer = "				<< layer
																																	<<"\n local_work_size_ = "	<< local_work_size_ 	<< " = patch_img_gradients_workgroup_size[	layer];"
																																	<<"\n threads_to_launch = "	<< threads_to_launch	<< " = patch_num_threads[					layer];"
																																	<<"\n offset1 = "			<< offset1
																																	<<"\n buf_width = "			<< buf_width
																																	<<"\n offset2 = "			<< offset2
																																	<<"\n img_cols = "			<< img_cols
																																	<<flush;
																																}

	_clSetKernelArg( kernel,	0, sizeof(int), 		&offset1,						fname );								//	__private	uint	offset1,			//0	top left corner
	_clSetKernelArg( kernel,	1, sizeof(int), 		&offset2,						fname );								//	__private	uint	offset2,			//1 bottom left corner
	_clSetKernelArg( kernel,	2, sizeof(int), 		&buf_width,						fname );								//	__private	uint	buf_width,			//2 mm_cols, i.e. width of the buffer holding the image pyramid
	_clSetKernelArg( kernel,	3, sizeof(int), 		&img_cols,						fname );								//	__private	uint	img_cols,			//3 num rows of this level of the image pyramid

	_clSetKernelArg( kernel,	4, sizeof( cl_mem),		&imgmem_,						fname );								//	__global 	float4*	img					//5
	cl_event	ev;
	cl_int		res, status;
	res 	= clEnqueueNDRangeKernel(m_queue,		kernel, 1, 0, &threads_to_launch, &local_work_size_, 0, NULL, &ev);
																if (res    != CL_SUCCESS)	{ cout << "\nres = " << checkerror(res) <<"\n"<<flush; exit_(res);}
	status	= clFlush(m_queue);									if (status != CL_SUCCESS)	{ cout << "\nRunCL::pad_image_top_bottom2( ),  clFlush(m_queue) status  = "<<status<<" "<< checkerror(status) <<"\n"<<flush; exit_(status);}
	status	= clWaitForEvents (1, &ev);							if (status != CL_SUCCESS)	{ cout << "\nRunCL::pad_image_top_bottom2( ),  clWaitForEventsh(1, &ev) ="	<<status<<" "<<checkerror(status)  <<"\n"<<flush; exit_(status);}
																																			if(verbosity>local_verbosity_threshold) {
																																				cout<<"\n\nRunCL::pad_image_top_bottom2(..)_chk3 Finished all loops."<<flush;
																																				stringstream ss;	ss << dataset_frame_num << "pad_image_top_bottom2";
																																				cv::Size new_Image_size = cv::Size(mm_width, mm_height);
																																				size_t   new_size_bytes = mm_width * mm_height * 4*4;
																																				ss << "_raw_";
																																				DownloadAndSave_3Channel( imgmem_, ss.str(), paths.at("imgmem"), new_size_bytes, new_Image_size, CV_32FC4, false, 1, 0, true );
																																			}
}

void RunCL::vertcal_blur5( uint layer ){
	string fname = "RunCL::vertcal_blur5()";
	int local_verbosity_threshold = V_RUNCL_REDUCE_IMG;																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::vertcal_blur5(..)_chk0"<<flush;}
	cl_kernel 		kernel 			= vertcal_blur5_kernel;
	const cl_mem 	imgmem_			= current_frames[ 	current_frames_idx[0] ].img_buf;
	const cl_mem 	tmp_img			= imgmem_blurred;

	size_t	local_work_size_		= patch_img_gradients_workgroup_size[	layer];							// From void RunCL::patch_img_gradients( uint layer )
	size_t	threads_to_launch		= patch_num_threads[					layer];
	uint	offset1					= MipMap[layer * 8 + MiM_READ_OFFSET];									//0	top left corner
	uint	buf_width				= uint_params[MM_COLS];													//1 mm_cols, i.e. width of the buffer holding the image pyramid
	uint	img_pixels				= MipMap[layer * 8 + MiM_PIXELS];										//2 num rows of this level of the image pyramid
	uint	img_cols				= MipMap[layer * 8 + MiM_READ_COLS];									//3
	uint	img_rows				= MipMap[layer * 8 + MiM_READ_ROWS];									//
	uint	stop_offset				= offset1 + (img_rows-1) * buf_width + img_cols;						//4
																																if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::vertcal_blur5(..)_chk1"<<flush;}
	_clSetKernelArg( kernel,	0, sizeof(int), 		&offset1,						fname );								//	__private	uint	offset1,			//0	top left corner
	_clSetKernelArg( kernel,	1, sizeof(int), 		&buf_width,						fname );								//	__private	uint	buf_width,			//1 mm_cols, i.e. width of the buffer holding the image pyramid
	_clSetKernelArg( kernel,	2, sizeof(int), 		&img_pixels,					fname );								//	__private	uint	img_pixels,			//2 num rows of this level of the image pyramid
	_clSetKernelArg( kernel,	3, sizeof(int), 		&img_cols,						fname );								//	__private	uint	img_cols,			//3
	_clSetKernelArg( kernel,	4, sizeof(int), 		&stop_offset,					fname );								//	__private	uint	img_pixels,			//3 num rows of this level of the image pyramid

	_clSetKernelArg( kernel,	5, sizeof( cl_mem),		&imgmem_,						fname );								//	__global 	float4*	img,				//4
	_clSetKernelArg( kernel,	6, sizeof( cl_mem),		&tmp_img,						fname );								//	__global 	float4*	tmp_img				//5
																																if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::vertcal_blur5(..)_chk2"<<flush;}
	cl_event	ev;
	cl_int		res, status;
	res 	= clEnqueueNDRangeKernel(m_queue,		kernel, 1, 0, &threads_to_launch, &local_work_size_, 0, NULL, &ev);
																if (res    != CL_SUCCESS)	{ cout << "\nres = " << checkerror(res) <<"\n"<<flush; exit_(res);}
	status	= clFlush(m_queue);									if (status != CL_SUCCESS)	{ cout << "\nRunCL::vertcal_blur5( ),  clFlush(m_queue) status  = "<<status<<" "<< checkerror(status) <<"\n"<<flush; exit_(status);}
	status	= clWaitForEvents (1, &ev);							if (status != CL_SUCCESS)	{ cout << "\nRunCL::vertcal_blur5( ),  clWaitForEventsh(1, &ev) ="	<<status<<" "<<checkerror(status)  <<"\n"<<flush; exit_(status);}
																																			if(verbosity>local_verbosity_threshold) {
																																				cout<<"\n\nRunCL::vertcal_blur5(..)_chk3 Finished all loops."<<flush;
																																				stringstream ss;	ss << dataset_frame_num << "vertcal_blur5";
																																				cv::Size new_Image_size = cv::Size(mm_width, mm_height);
																																				size_t   new_size_bytes = mm_width * mm_height * 4*4;
																																				ss << "_raw_";
																																				DownloadAndSave_3Channel( tmp_img, ss.str(), paths.at("imgmem"), new_size_bytes, new_Image_size, CV_32FC4, false, 1, 0, true );
																																			}
}

void RunCL::pad_image_left_right2( uint layer ){
	string fname = "RunCL::pad_image_left_right2()";
	int local_verbosity_threshold = V_RUNCL_REDUCE_IMG;																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::pad_image_left_right2(..)_chk0"<<flush;}
	cl_kernel 		kernel 			= pad_image_left_right2_kernel;
	const cl_mem 	tmp_img			= imgmem_blurred;

	size_t	local_work_size_		= patch_img_gradients_workgroup_size[	layer];	// From void RunCL::patch_img_gradients( uint layer )
	size_t	threads_to_launch		= patch_num_threads[					layer];
	uint	offset1					= MipMap[layer * 8 + MiM_READ_OFFSET];				//0 top left corner
	uint	img_cols				= MipMap[layer * 8 + MiM_READ_COLS];
	uint	offset2					= offset1 + img_cols;								//1 top right corner
	uint	buf_width				= uint_params[MM_COLS];								//2 mm_cols, i.e. width of the buffer holding the image pyramid
	uint	img_rows				= MipMap[layer * 8 + MiM_READ_ROWS];				//3 num rows of this level of the image pyramid

	_clSetKernelArg( kernel,	0, sizeof(int), 		&offset1,						fname );								//	__private	uint	offset1,			//0 top left corner
	_clSetKernelArg( kernel,	1, sizeof(int), 		&offset2,						fname );								//	__private	uint	offset2,			//1 top right corner
	_clSetKernelArg( kernel,	2, sizeof(int), 		&buf_width,						fname );								//	__private	uint	buf_width,			//2 mm_cols, i.e. width of the buffer holding the image pyramid
	_clSetKernelArg( kernel,	3, sizeof(int), 		&img_rows,						fname );								//	__private	uint	img_pixels,			//3 num rows of this level of the image pyramid

	_clSetKernelArg( kernel,	4, sizeof( cl_mem),		&tmp_img,						fname );								//	__global 	float4*	img					//4
	cl_event	ev;
	cl_int		res, status;
	res 	= clEnqueueNDRangeKernel(m_queue,		kernel, 1, 0, &threads_to_launch, &local_work_size_, 0, NULL, &ev);
																if (res    != CL_SUCCESS)	{ cout << "\nres = " << checkerror(res) <<"\n"<<flush; exit_(res);}
	status	= clFlush(m_queue);									if (status != CL_SUCCESS)	{ cout << "\nRunCL::pad_image_left_right2( ),  clFlush(m_queue) status  = "<<status<<" "<< checkerror(status) <<"\n"<<flush; exit_(status);}
	status	= clWaitForEvents (1, &ev);							if (status != CL_SUCCESS)	{ cout << "\nRunCL::pad_image_left_right2( ),  clWaitForEventsh(1, &ev) ="	<<status<<" "<<checkerror(status)  <<"\n"<<flush; exit_(status);}
																																			if(verbosity>local_verbosity_threshold) {
																																				cout<<"\n\nRunCL::pad_image_left_right2(..)_chk3 Finished all loops."<<flush;
																																				stringstream ss;	ss << dataset_frame_num << "pad_image_left_right2";
																																				cv::Size new_Image_size = cv::Size(mm_width, mm_height);
																																				size_t   new_size_bytes = mm_width * mm_height * 4*4;
																																				ss << "_raw_";
																																				DownloadAndSave_3Channel( tmp_img, ss.str(), paths.at("imgmem"), new_size_bytes, new_Image_size, CV_32FC4, false, 1, 0, true );
																																			}
}

void RunCL::horiz_blur5( uint layer ){
	string fname = "RunCL::horiz_blur5()";
	int local_verbosity_threshold = V_RUNCL_REDUCE_IMG;																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::horiz_blur5(..)_chk0"<<flush;}
	cl_kernel 		kernel 			= horiz_blur5_kernel;
	const cl_mem 	imgmem_			= current_frames[ 	current_frames_idx[0] ].img_buf;
	const cl_mem 	tmp_img			= imgmem_blurred;

	size_t	local_work_size_		= patch_img_gradients_workgroup_size[	layer];	// From void RunCL::patch_img_gradients( uint layer )
	size_t	threads_to_launch		= patch_num_threads[					layer];
	uint	offset1					=  MipMap[layer * 8 + MiM_READ_OFFSET];				//0	top left corner
	uint	buf_width				=  uint_params[MM_COLS];							//1 mm_cols, i.e. width of the buffer holding the image pyramid
	uint	img_pixels				=  MipMap[layer * 8 + MiM_PIXELS];					//2 num rows of this level of the image pyramid
	uint	img_cols				=  MipMap[layer * 8 + MiM_READ_COLS];				//3
	uint	img_rows				= MipMap[layer * 8 + MiM_READ_ROWS];				//
	uint	stop_offset				= offset1 + (img_rows-1) * buf_width + img_cols;	//4


	_clSetKernelArg( kernel,	0, sizeof(int), 		&offset1,						fname );								//	__private	uint	offset1,			//0	top left corner
	_clSetKernelArg( kernel,	1, sizeof(int), 		&buf_width,						fname );								//	__private	uint	buf_width,			//1 mm_cols, i.e. width of the buffer holding the image pyramid
	_clSetKernelArg( kernel,	2, sizeof(int), 		&img_pixels,					fname );								//	__private	uint	img_pixels,			//2 num rows of this level of the image pyramid
	_clSetKernelArg( kernel,	3, sizeof(int), 		&img_cols,						fname );								//	__private	uint	img_cols,			//3
	_clSetKernelArg( kernel,	4, sizeof(int), 		&stop_offset,					fname );								//	__private	uint	img_pixels,			//3 num rows of this level of the image pyramid

	_clSetKernelArg( kernel,	5, sizeof( cl_mem),		&tmp_img,						fname );								//	__global 	float4*	img,				//4
	_clSetKernelArg( kernel,	6, sizeof( cl_mem),		&imgmem_,						fname );								//	__global 	float4*	tmp_img				//5
	cl_event	ev;
	cl_int		res, status;
	res 	= clEnqueueNDRangeKernel(m_queue,		kernel, 1, 0, &threads_to_launch, &local_work_size_, 0, NULL, &ev);
																if (res    != CL_SUCCESS)	{ cout << "\nres = " << checkerror(res) <<"\n"<<flush; exit_(res);}
	status	= clFlush(m_queue);									if (status != CL_SUCCESS)	{ cout << "\nRunCL::horiz_blur5( ),  clFlush(m_queue) status  = "<<status<<" "<< checkerror(status) <<"\n"<<flush; exit_(status);}
	status	= clWaitForEvents (1, &ev);							if (status != CL_SUCCESS)	{ cout << "\nRunCL::horiz_blur5( ),  clWaitForEventsh(1, &ev) ="	<<status<<" "<<checkerror(status)  <<"\n"<<flush; exit_(status);}
																																			if(verbosity>local_verbosity_threshold) {
																																				cout<<"\n\nRunCL::horiz_blur5(..)_chk3 Finished all loops."<<flush;
																																				stringstream ss;	ss << dataset_frame_num << "horiz_blur5";
																																				cv::Size new_Image_size = cv::Size(mm_width, mm_height);
																																				size_t   new_size_bytes = mm_width * mm_height * 4*4;
																																				ss << "_raw_";
																																				DownloadAndSave_3Channel( imgmem_, ss.str(), paths.at("imgmem"), new_size_bytes, new_Image_size, CV_32FC4, false, 1, 0, true );
																																			}
}


void RunCL::reduce_img(uint layer, std::string folder){
	string fname = "RunCL::reduce_img()";
	int local_verbosity_threshold = V_RUNCL_REDUCE_IMG;																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::reduce_img(..)_chk0,  layer = "<<layer<< "#############################################################"<<flush;}
	cl_kernel 		kernel 						= reduce_img_kernel;
	const cl_mem 	imgmem_						= current_frames[ 						current_frames_idx[0] ].img_buf;

		size_t	local_work_size_	= patch_img_gradients_workgroup_size[	layer];	// From void RunCL::patch_img_gradients( uint layer )
		size_t	threads_to_launch	= patch_num_threads[					layer+1];
		uint	offset1				= MipMap[layer * 8 + MiM_READ_OFFSET];			//0	top left corner source image
		uint	offset2				= MipMap[layer * 8 + MiM_WRITE_OFFSET];			//1	top left corner dest image

		uint	buf_width			= uint_params[MM_COLS]; 						//2 mm_cols, i.e. width of the buffer holding the image pyramid
		uint	img_pixels			= MipMap[layer * 8 + MiM_PIXELS];				//3 num rows of this level of the image pyramid
		uint	img_cols			= MipMap[layer * 8 + MiM_WRITE_COLS];			//4

		uint	img_rows			= MipMap[layer * 8 + MiM_WRITE_ROWS];			//5
		uint	stop_offset			= offset2 + (img_rows-1) * buf_width + img_cols;	//7 bottom right corner of dest image
																																if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::reduce_img(..)_chk1 #############################################################"<<flush;
																																	cout << "\n local_work_size = " << local_work_size
																																			<<"\nthreads_to_launch = " 	<< threads_to_launch
																																			<<"\noffset1 = " 			<< offset1
																																			<<"\noffset2 = " 			<< offset2
																																			<<"\nrow = " 				<< offset2/buf_width
																																			<<"\ncol = "	 			<< offset2%buf_width

																																			<<"\nbuf_width = " 			<< buf_width
																																			<<"\nimg_pixels = " 		<< img_pixels
																																			<<"\nimg_cols = " 			<< img_cols
																																			<<"\nimg_rows = " 			<< img_rows
																																			<<"\nstop_offset = " 		<< stop_offset
																																	<< flush;
																																}
		_clSetKernelArg( kernel,	0, sizeof(int), 		&offset1,						fname );								//	__private	uint	offset1,			//0	top left corner source image
		_clSetKernelArg( kernel,	1, sizeof(int), 		&offset2,						fname );								//	__private	uint	offset2,			//1	top left corner dest image

		_clSetKernelArg( kernel,	2, sizeof(int), 		&buf_width,						fname );								//	__private	uint	buf_width,			//2 mm_cols, i.e. width of the buffer holding the image pyramid
		_clSetKernelArg( kernel,	3, sizeof(int), 		&img_pixels,					fname );								//	__private	uint	img_pixels,			//3 num rows of this level of the image pyramid
		_clSetKernelArg( kernel,	4, sizeof(int), 		&img_cols,						fname );								//	__private	uint	img_cols,			//4

		_clSetKernelArg( kernel,	5, sizeof(int), 		&img_rows,						fname );								//	__private	uint	img_rows,			//5
		_clSetKernelArg( kernel,	6, sizeof(int), 		&patch_size,					fname );								//	__private	uint	num_iter,			//6 patch height
		_clSetKernelArg( kernel,	7, sizeof(int), 		&stop_offset,					fname );								//	__private	uint	stop_offset,		//7 bottom right corner of dest image

		_clSetKernelArg( kernel,	8, sizeof( cl_mem),		&imgmem_,						fname );								//	__global 	float4*	img,				//8

		cl_event	ev;
		cl_int		res, status;

		res 	= clEnqueueNDRangeKernel(m_queue,		kernel, 1, 0, &threads_to_launch, &local_work_size_, 0, NULL, &ev);
																	if (res    != CL_SUCCESS)	{ cout << "\nres = " << checkerror(res) <<"\n"<<flush; exit_(res);}
		status	= clFlush(m_queue);									if (status != CL_SUCCESS)	{ cout << "\nRunCL::reduce_img( ),  clFlush(m_queue) status  = "<<status<<" "<< checkerror(status) <<"\n"<<flush; exit_(status);}
		status	= clWaitForEvents (1, &ev);							if (status != CL_SUCCESS)	{ cout << "\nRunCL::reduce_img( ),  clWaitForEventsh(1, &ev) ="	<<status<<" "<<checkerror(status)  <<"\n"<<flush; exit_(status);}
																																			if(verbosity>local_verbosity_threshold) {
																																				cout<<"\n\nRunCL::reduce_img(..)_chk3 Finished all loops."<<flush;
																																				stringstream ss;	ss << dataset_frame_num << "reduce_img";
																																				cv::Size new_Image_size = cv::Size(mm_width, mm_height);
																																				size_t   new_size_bytes = mm_width * mm_height * 4*4;
																																				ss << "_raw_";
																																				DownloadAndSave_3Channel( imgmem_, ss.str(), paths.at("imgmem"), new_size_bytes, new_Image_size, CV_32FC4, false, 1, 0, true );
																																			}
																																if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::reduce_img(..)_finished layer = "<<layer<< "#############################################################"<<flush; }
}





// Image gradietnt d(value)/d(u,v)



// d(u,v)/d(se3)



// Jacobian J = d(value)/d(se3)  = d(value)/d(u,v)  *  d(u,v)/d(se3)



// Pixels weights, W = Huber norm on Rho



// Rho photometric error



// Weighted Gauss-Newton aprox Hessian = sum(Jt W J)



// Pseudo inverse or inverse of 6x6 matrix.




// Compute se3 update = WGN_inv  *  sum(W J Rho)



// Scale update wrt delta_se3  [1,1,1,theta,theta,theta]



// Apply update, ie compute new SE3 and new K2K



// Test if (SSD > old_SSD || isnan(SSD) )
/*	change pyramid layer + reduction
 *  or use separate depth maps & k, inv_k for each scale.
 */



/* Repeat using more frames, for:
 *(i) Camera matrix k & lens distortion params
 *(ii) Depth pyramid
 *(iii) Rel_vel & Accel, & camera accel & jolt
 *(iv) Reflectance & illumination map
 * (ii-iv) using
 * (a) parsimony
 * (b) local smoothing
 * (c) edge preseeving - anisotropy wrt edges
 * (d) penalize mutually cancelling information wave artifacts
 *
 *
 * NB Foveation & ability to direct attention i.e. move the fovea.
 * Also for each patch, find pixels with max gradient in u, v, uv, -uv directions, i.e. octagon sample set.
 * Use semi-sparse to accelerate lower img pyr levels.
 */
