#include "RunCL.hpp"

// Image pyramid

void RunCL::build_img_pyramid( uint reductions, uint blur_layers, std::string folder ){
	string fname = "RunCL::build_img_pyramid()";
	int local_verbosity_threshold = V_RUNCL_REDUCE_IMG;																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::build_img_pyramid(..)_chk0"<<flush;}

	int stop 		= min(mm_num_reductions, max_mipmap_layers-1);	// TODO compute required reduction and blur depending on img size
	int layer 		= 0;
																																			cout << "\nlayer = "<<layer<<flush;
	reduce_img( layer, folder);
	for (layer=1; layer<=stop ; layer++){
																																			if(verbosity>local_verbosity_threshold) {	cout<<"\n\nRunCL::build_img_pyramid(..) pyramid layer = "<<layer<<flush;	}
																																			cout << "\nlayer = "<<layer<<flush;
		blur_image_layer( layer);
		reduce_img( layer, folder);
	}
																																			cout << "\nlayer = "<<layer<<",  old stop = "<< stop << flush;
	stop 			= min( (mm_num_reductions + mm_num_blur_layers),  max_mipmap_layers-1);
	layer--;
																																			cout << "\nnew stop = "<< stop << flush;
	for (; layer<stop  ; layer++){
																																			cout << "\nblur layer = "<<layer<<flush;
		copy_translate_img( layer );																										if(verbosity>local_verbosity_threshold) {	cout<<"\n\nRunCL::build_img_pyramid(..) blur layer = "<<layer<<flush;	}

		for (int iter = 0; iter<3; iter++){				// NB 3x (5x5) boxblur
			blur_image_layer( layer );					// NB will need new locations for non-reduced layers.
		}
	}
																																			cout << "\nlast layer = "<<layer<<flush;
	size_t 			local_size 		= local_work_size;
	const cl_mem 	imgmem_			= current_frames[	current_frames_idx[0] ].img_buf;
																																			if(verbosity>local_verbosity_threshold) {
																																				cout<<"\n\nRunCL::build_img_pyramid(..)_chk3 Finished all loops."<<flush;
																																				stringstream ss;	ss << dataset_frame_num << "build_img_pyramid";
																																				cv::Size new_Image_size = cv::Size(mm_width, mm_height);
																																				size_t   new_size_bytes = mm_width * mm_height * 4*4;
																																				ss << "_raw_";

																																				DownloadAndSave_3Channel( imgmem_, ss.str(), paths.at(folder), new_size_bytes, new_Image_size, CV_32FC4, false, 1, 0, true );
																																				cout << "\n  (local_size+4) *5*4* sizeof(float) = "<<  (local_size+4) *5*4* sizeof(float) << " ,   (local_size+4) = " <<  (local_size+4) << endl << flush;
																																			}
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::mipmap_linear(..)_chk4 Finished"<<flush;}
}

void RunCL::blur_image_layer( uint layer ){
		pad_image_top_bottom2( 		layer );
		vertcal_blur5( 			layer );
		pad_image_left_right2( 	layer );
		horiz_blur5( 			layer );
}

void RunCL::pad_image_top_bottom2( uint layer ){
	string fname = "RunCL::pad_image_top2()";
	int local_verbosity_threshold 	= V_RUNCL_REDUCE_IMG;																		if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::pad_image_top2(..)_chk0"<<flush;}
	cl_kernel 		kernel 			= pad_image_top_bottom2_kernel;
	const cl_mem 	imgmem_			= current_frames[	current_frames_idx[0] ].img_buf;

	size_t	local_work_size_		= patch_img_gradients_workgroup_size[	layer];							// From void RunCL::patch_img_gradients( uint layer )
	size_t	threads_to_launch		= patch_num_threads[					layer];
	uint	offset1					= MipMap[layer * 8 + MiM_READ_OFFSET];									//0	top left corner
	uint	buf_width				= uint_params[MM_COLS];													//2 mm_cols, i.e. width of the buffer holding the image pyramid
	uint	offset2					= offset1 + buf_width * (MipMap[layer * 8 + MiM_READ_ROWS] - 1);		//1 bottom left corner
	uint	img_pixels				= MipMap[layer * 8 + MiM_PIXELS];										//3 num rows of this level of the image pyramid

	_clSetKernelArg( kernel,	0, sizeof(int), 		&offset1,						fname );								//	__private	uint	offset1,			//0	top left corner
	_clSetKernelArg( kernel,	1, sizeof(int), 		&offset2,						fname );								//	__private	uint	offset2,			//1 bottom left corner
	_clSetKernelArg( kernel,	2, sizeof(int), 		&buf_width,						fname );								//	__private	uint	buf_width,			//2 mm_cols, i.e. width of the buffer holding the image pyramid
	_clSetKernelArg( kernel,	3, sizeof(int), 		&img_pixels,					fname );								//	__private	uint	img_pixels,			//3 num rows of this level of the image pyramid

	_clSetKernelArg( kernel,	4, sizeof( cl_mem),		&imgmem_,						fname );								//	__global 	float4*	img					//5
	cl_event	ev;
	cl_int		res, status;
	res 	= clEnqueueNDRangeKernel(m_queue,		kernel, 1, 0, &threads_to_launch, &local_work_size_, 0, NULL, &ev);
																if (res    != CL_SUCCESS)	{ cout << "\nres = " << checkerror(res) <<"\n"<<flush; exit_(res);}
	status	= clFlush(m_queue);									if (status != CL_SUCCESS)	{ cout << "\nRunCL::reduce_img( ),  clFlush(m_queue) status  = "<<status<<" "<< checkerror(status) <<"\n"<<flush; exit_(status);}
	status	= clWaitForEvents (1, &ev);							if (status != CL_SUCCESS)	{ cout << "\nRunCL::reduce_img( ),  clWaitForEventsh(1, &ev) ="	<<status<<" "<<checkerror(status)  <<"\n"<<flush; exit_(status);}
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
	uint	stop_offset				= offset1 + img_rows * buf_width;										//4

	_clSetKernelArg( kernel,	0, sizeof(int), 		&offset1,						fname );								//	__private	uint	offset1,			//0	top left corner
	_clSetKernelArg( kernel,	1, sizeof(int), 		&buf_width,						fname );								//	__private	uint	buf_width,			//1 mm_cols, i.e. width of the buffer holding the image pyramid
	_clSetKernelArg( kernel,	2, sizeof(int), 		&img_pixels,					fname );								//	__private	uint	img_pixels,			//2 num rows of this level of the image pyramid
	_clSetKernelArg( kernel,	3, sizeof(int), 		&img_cols,						fname );								//	__private	uint	img_cols,			//3
	_clSetKernelArg( kernel,	4, sizeof(int), 		&stop_offset,					fname );								//	__private	uint	img_pixels,			//3 num rows of this level of the image pyramid

	_clSetKernelArg( kernel,	5, sizeof( cl_mem),		&imgmem_,						fname );								//	__global 	float4*	img,				//4
	_clSetKernelArg( kernel,	6, sizeof( cl_mem),		&tmp_img,						fname );								//	__global 	float4*	tmp_img				//5
	cl_event	ev;
	cl_int		res, status;
	res 	= clEnqueueNDRangeKernel(m_queue,		kernel, 1, 0, &threads_to_launch, &local_work_size_, 0, NULL, &ev);
																if (res    != CL_SUCCESS)	{ cout << "\nres = " << checkerror(res) <<"\n"<<flush; exit_(res);}
	status	= clFlush(m_queue);									if (status != CL_SUCCESS)	{ cout << "\nRunCL::reduce_img( ),  clFlush(m_queue) status  = "<<status<<" "<< checkerror(status) <<"\n"<<flush; exit_(status);}
	status	= clWaitForEvents (1, &ev);							if (status != CL_SUCCESS)	{ cout << "\nRunCL::reduce_img( ),  clWaitForEventsh(1, &ev) ="	<<status<<" "<<checkerror(status)  <<"\n"<<flush; exit_(status);}
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
	status	= clFlush(m_queue);									if (status != CL_SUCCESS)	{ cout << "\nRunCL::reduce_img( ),  clFlush(m_queue) status  = "<<status<<" "<< checkerror(status) <<"\n"<<flush; exit_(status);}
	status	= clWaitForEvents (1, &ev);							if (status != CL_SUCCESS)	{ cout << "\nRunCL::reduce_img( ),  clWaitForEventsh(1, &ev) ="	<<status<<" "<<checkerror(status)  <<"\n"<<flush; exit_(status);}
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
	uint	stop_offset				= offset1 + img_rows * buf_width;					//4


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
	status	= clFlush(m_queue);									if (status != CL_SUCCESS)	{ cout << "\nRunCL::reduce_img( ),  clFlush(m_queue) status  = "<<status<<" "<< checkerror(status) <<"\n"<<flush; exit_(status);}
	status	= clWaitForEvents (1, &ev);							if (status != CL_SUCCESS)	{ cout << "\nRunCL::reduce_img( ),  clWaitForEventsh(1, &ev) ="	<<status<<" "<<checkerror(status)  <<"\n"<<flush; exit_(status);}
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
		uint	stop_offset			= offset2 + img_rows * buf_width;				//7 bottom right corner of dest image
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

// Add blurred top layers to image pyramid
void RunCL::copy_translate_img( uint layer ){
	string fname = "RunCL::copy_translate_img()";
	int local_verbosity_threshold = V_RUNCL_REDUCE_IMG;																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::copy_translate_img(..)_chk0,  layer = "<<layer<< "#############################################################"<<flush;}
	//cl_kernel 		kernel 						= reduce_img_kernel;
	const cl_mem 	imgmem_						= current_frames[ 						current_frames_idx[0] ].img_buf;
	cl_event	ev;

	uint read_offset			= MipMap[layer * 8 + MiM_READ_OFFSET];
	uint read_row				= read_offset / mm_width;
	uint read_col				= read_offset % mm_width;
	const size_t src_origin[3] 	= { read_col * sizeof(cl_float4), read_row , 0 }; // {0,0,0}; //

	uint write_offset			= MipMap[layer * 8 + MiM_WRITE_OFFSET];
	uint write_row				= write_offset / mm_width;
	uint write_col				= write_offset % mm_width;
	const size_t dst_origin[3] 	= { write_col * sizeof(cl_float4)  , write_row  ,  0 }; // {0,0,0}; //

	uint read_rows				= MipMap[layer * 8 + MiM_READ_ROWS];
	uint read_cols				= MipMap[layer * 8 + MiM_READ_COLS];

	const size_t region[3]		= {read_rows*sizeof(cl_float4),read_cols,1 }; // { read_rows * sizeof(cl_float4), 	 read_rows * read_cols * sizeof(cl_float4),		read_rows * read_cols * sizeof(cl_float4) };  //  {1,1,1};

	size_t src_row_pitch				=	uint_params[MM_COLS] * sizeof(cl_float4); // 1;	//
	size_t src_slice_pitch				=	mm_size_bytes_C4;	//0;  // 1;	//

	size_t dst_row_pitch				=	uint_params[MM_COLS] * sizeof(cl_float4); // 1;	//
	size_t dst_slice_pitch				=	mm_size_bytes_C4;	//0;  // 1;	//

	cl_uint num_events_in_wait_list		=	0;
	const cl_event* event_wait_list		=	NULL;

																																if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::copy_translate_img(..)_chk1 "<<flush;
																																	cout <<
																																	"\n	mm_width = "		<< mm_width 						<<
																																	"\n	uint_params[MM_COLS] = " << uint_params[MM_COLS] 		<<

																																	"\n\n	src_origin  "		<<
																																	"\n	col = "				<< src_origin[0]/sizeof(cl_float4) 	<<
																																	"\n	row = "				<< src_origin[1]					<<

																																	"\n\n	dst_origin "		<<
																																	"\n	col = "				<< dst_origin[0]/sizeof(cl_float4)	<<
																																	"\n	row = "				<< dst_origin[1]					<<

																																	"\n\n	region = ( "		<< region[0]/sizeof(cl_float4)		<<", "<< region[1]/sizeof(cl_float4) <<", "<<  region[2]/sizeof(cl_float4) <<" ) "<<

																																	"\n\n	src_row_pitch = "	<< src_row_pitch/sizeof(cl_float4)	<<
																																	"\n	src_slice_pitch = "	<< src_slice_pitch/sizeof(cl_float4)<<

																																	"\n\n	dst_row_pitch = "	<< dst_row_pitch/sizeof(cl_float4) 	<<
																																	"\n	dst_slice_pitch = "	<< dst_slice_pitch/sizeof(cl_float4)<< flush;
																																}
	cl_int status = clEnqueueCopyBufferRect(
		m_queue, 					// cl_command_queue command_queue,
		imgmem_, 					// cl_mem src_buffer,     // qmem, //
		imgmem_,					// cl_mem dst_buffer,             // imgmem_blurred  // qmem2, //
		src_origin,					//const size_t* src_origin,
		dst_origin,					//const size_t* dst_origin,
		region,						//const size_t* region,
		src_row_pitch,				//size_t src_row_pitch,
		src_slice_pitch,			//size_t src_slice_pitch,
		dst_row_pitch,				//size_t dst_row_pitch,
		dst_slice_pitch,			//size_t dst_slice_pitch,
		0,							//cl_uint num_events_in_wait_list,
		NULL,						//const cl_event* event_wait_list,
		&ev							//cl_event* event
	);
																																if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::copy_translate_img(..)_chk2 "<<flush;}
	status	= clFlush(m_queue);									if (status != CL_SUCCESS)	{ cout << "\nRunCL::copy_translate_img( ),  clFlush(m_queue) status  = "<<status<<" "<< checkerror(status) <<"\n"<<flush; exit_(status);}
	status	= clWaitForEvents (1, &ev);							if (status != CL_SUCCESS)	{ cout << "\nRunCL::copy_translate_img( ),  clWaitForEventsh(1, &ev) ="	<<status<<" "<<checkerror(status)  <<"\n"<<flush; exit_(status);}

																																if(verbosity>local_verbosity_threshold) {
																																	cout<<"\n\nRunCL::build_img_pyramid(..)_chk3 copy_translate_img()."<<flush;
																																	stringstream ss;	ss << dataset_frame_num << "copy_translate_img";
																																	cv::Size new_Image_size = cv::Size(mm_width, mm_height);
																																	size_t   new_size_bytes = mm_width * mm_height * 4*4;
																																	ss << "_raw_";
																																	DownloadAndSave_3Channel( imgmem_, ss.str(), paths.at("imgmem"), new_size_bytes, new_Image_size, CV_32FC4, false, 1, 0, true );
																																}
																																if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::copy_translate_img(..)_finished "<<flush;}
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
