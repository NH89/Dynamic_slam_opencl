#include "RunCL.hpp"



void RunCL::precomp_cam_and_lens_maps ( uint layer, cl_float16 SE3_k2k[  num_camera_matrix_DoF +1  ],  cl_mem map_mem, uint num_vars, string calling_fn){ //  Compute maps of pixel motion for each DoF of camera intrinsic matrix, or lens distortion // Derived from RunCL::mipmap
	string fname = "RunCL::precomp_cam_and_lens_maps(..)";
	int local_verbosity_threshold = V_RUNCL_PRECOM_PARAM_MAPS;
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::precomp_cam_and_lens_maps(..)_chk_0 "
																																				<<"\nnum_vars = "<<num_vars
																																				<<"\ncalling_fn = "<<calling_fn
																																				<<flush;
																																				for(int mipmap_layer=0; mipmap_layer< max_mipmap_layers; mipmap_layer++){
																																					cout<<"\n########################################################################\n"<<flush;
																																					for(int param=0; param<=num_vars; param++){
																																						cout<<"\n######\n mipmap_layer="<<mipmap_layer<<"   se3="<<param<<"\n"<<flush;
																																						PRINT_CL_FLOAT16( SE3_k2k[  mipmap_layer*(num_camera_matrix_DoF +1)  + param  ], )
																																					}
																																				}
																																			}
	cl_kernel	kernel	= comp_cam_and_lens_maps_kernel;
	float inv_depth		= 1;
	_clEnqueueWriteBuffer( uload_queue, SE3_k2kbuf,		CL_FALSE, 0, (num_camera_matrix_DoF +1)*sizeof(cl_float16), 	SE3_k2k,		fname);

	//      __private	 uint layer, set in mipmap_call_kernel( ..) below                                                                      __private	 uint	 layer,		//0
	_clSetKernelArg( kernel, 1, sizeof( float),		&inv_depth,	 		fname);												//__private	float 	inv_depth,		//1
	_clSetKernelArg( kernel, 2, sizeof( uint),		&num_vars,	 		fname);												//__private	uint,	num_vars		//2

	_clSetKernelArg( kernel, 3, sizeof( cl_mem),	&mipmap_buf, 		fname);												//__constant uint*	mipmap_params,	//3
	_clSetKernelArg( kernel, 4, sizeof( cl_mem), 	&uint_param_buf,	fname);												//__global 	uint*	uint_params		//4
	_clSetKernelArg( kernel, 5, sizeof( cl_mem), 	&SE3_k2kbuf, 		fname);												//__global 	float* 	k2k,			//5
	_clSetKernelArg( kernel, 6, sizeof( cl_mem), 	&map_mem, 			fname);												//__global 	float* 	SE3_map,		//6
																																			if( verbosity>local_verbosity_threshold) {cout<<"\nRunCL::precomp_cam_and_lens_maps(..)_chk_1 "
																																				<<"\nnum_vars = "<<num_vars
																																				<<"\ncalling_fn = "<<calling_fn
																																				<<flush;}
	// SE3_map_mem, k_map_mem, dist_map_mem;
	bool layers_sequential=false;
	mipmap_call_kernel( kernel, m_queue, layer, layer, layers_sequential, local_work_size);

																																			if( verbosity>local_verbosity_threshold) {
																																				cout<<"\n\nRunCL::precomp_cam_and_lens_maps(..)_output "<<flush;
																																				stringstream ss;	ss << dataset_frame_num << "_param_map_"<<calling_fn;
																																				float max_range = -1.0f;		// i.e. find max value, and map 0.0->0.5.
																																				DownloadAndSave_2Channel_volume( map_mem, ss.str( ), paths.at( "SE3_map_mem"), mm_size_bytes_C1*2, mm_Image_size, CV_32FC2, false, max_range, num_vars );

																																				cout<<"\nRunCL::precomp_cam_and_lens_maps(..)_chk.. Finished "<<flush;
																																			}
}


void RunCL::patch_cam_and_lens_Hessian(  uint layer, cl_mem param_map_mem, cl_mem param_grad_map_mem, cl_mem param_hessian_map_mem ){														// called by Dynamic_slam::getFrame
	string fname = "RunCL::patch_cam_and_lens_Hessian()";
	int local_verbosity_threshold = V_RUNCL_PATCH_CAM_LENS_HESSIAN;																if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::patch_cam_and_lens_Hessian()_chk1 #############################################################"<<flush;}
	cl_kernel		kernel						= patch_cam_and_lens_Hessian_kernel;

	size_t			local_work_size_			= patch_img_gradients_workgroup_size[	layer];											// patch_local_work_size[	layer];						//patch_img_gradients_workgroup_size; // patch_img_gradients_local_work_size_;
	size_t			threads_to_launch			= patch_num_threads[					layer];
	size_t			local_Hessian_size			= sizeof(cl_float4)						*num_SE3_DoF *num_SE3_DoF	*local_work_size_;	//*patch_img_gradients_workgroup_size;
	uint			lookup_table_offset_uint 	= patch_lookup_table_offset[			layer];
																																if( verbosity>local_verbosity_threshold ) {
																																	for (uint layer_ =0; layer_<mm_stop; layer_++){
																																		cout<<"\npatch_ST3_hessian_start_idx[	layer="<<layer_<<"][row][col] = "<<patch_hessian_start_idx[layer_][0][0]<<flush;
																																	}
																																}
	uint			SE3_h_offset				= patch_hessian_start_idx[layer][0][0];
	uint			ST3_h_offset				= patch_ST3_hessian_start_idx[layer][0][0];
	cl_uint3		SE3_hessian_offset			= {{ SE3_h_offset,	(patch_hessian_start_idx[layer][0][1]     - SE3_h_offset) ,	(patch_hessian_start_idx[layer][1][0]     - SE3_h_offset)  }};
	cl_uint3		ST3_out_offset				= {{ ST3_h_offset,	(patch_ST3_hessian_start_idx[layer][0][1] - ST3_h_offset) ,	(patch_ST3_hessian_start_idx[layer][1][0] - ST3_h_offset)  }};
																																if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::patch_cam_and_lens_Hessian()_chk2 "
																																	<<" local_Hessian_size = "<<local_Hessian_size<<"  "<<flush;
																																	cout<<" out_block_size = "<<out_block_size
																																	<<"\n SE3_h_offset = "<<SE3_h_offset
																																	<<"\n ST3_h_offset = "<<ST3_h_offset
																																	<<"\n SE3_hessian_offset = ("<<SE3_hessian_offset.x <<", "<<SE3_hessian_offset.y <<", "<<SE3_hessian_offset.z <<")"
																																	<<"\n ST3_out_offset = ("<<ST3_out_offset.x <<", "<<ST3_out_offset.y <<", "<<ST3_out_offset.z <<")"
																																	<<flush;
																																}
	cl_event		ev;
	cl_int			res, status;
	//Inputs:
	//__private
	_clSetKernelArg( kernel,	0, sizeof(int),			&layer,							fname);									// __private	uint		layer,						//0
	_clSetKernelArg( kernel,	1, sizeof(int),			&lookup_table_offset_uint,		fname);									// __private	uint		lookup_table_offset_uint,	//1
	_clSetKernelArg( kernel,	2, sizeof(int), 		&out_block_size,				fname );								// __private	uint		out_block_size,				//2
	_clSetKernelArg( kernel,	3, sizeof(cl_uint3),	&SE3_hessian_offset,			fname);									// __private	uint		SE3_hessian_offset,			//2
	_clSetKernelArg( kernel,	4, sizeof(cl_uint3),	&ST3_out_offset,				fname);									// __private	uint		SE3_hessian_offset,			//3
	//__constant
	_clSetKernelArg( kernel,	5, sizeof( cl_mem), 	&mipmap_buf,					fname);									// __constant	uint8*		mipmap_params,			//5
	_clSetKernelArg( kernel,	6, sizeof( cl_mem), 	&uint_param_buf,				fname);									// __constant	uint*		uint_params,			//6
	//__global
	_clSetKernelArg( kernel,	7, sizeof( cl_mem), 	&param_map_mem,					fname);									// __constant 	float2*		SE3_map,				//7
	_clSetKernelArg( kernel,	8, sizeof( cl_mem), 	&patch_lookup_table_buf,		fname);									// __global 	float4*		lookup_table,			//8
	_clSetKernelArg( kernel,	9, sizeof( cl_mem), 	&img_grad_mem,					fname);									// __global 	float8*		img_grad_uv,			//11
	_clSetKernelArg( kernel,	10, sizeof( cl_mem), 	&param_grad_map_mem,			fname);									// __global 	float4*		SE3_grad_map,			//12	// We keep hsv sepate at this stage, so 6*4*2=24, but float16 is the largest type, so 6*float8.
	//Outputs:
	//__global
	_clSetKernelArg( kernel,	11, sizeof( cl_mem), 	&param_hessian_map_mem,			fname);									// __global 	float4*		SE3_Hessian_pinv_map,	//13	// HSV (6x6) matrix so 36*float8
	//__local
	_clSetKernelArg( kernel,	12,local_Hessian_size,	NULL,							fname);									// __local		float4*		local_Hessian,				//13	// local_Hessian[ sizeof(float4) *6*6 *local_size]
																																cout<<"\n\nRunCL::patch_cam_and_lens_Hessian()_chk3 "<<flush;
	res 	= clEnqueueNDRangeKernel(m_queue,		kernel, 1, 0, &threads_to_launch, &local_work_size_, 0, NULL, &ev);
																	if (res    != CL_SUCCESS)	{ cout << "\nres = " << checkerror(res) <<"\n"<<flush; exit_(res);}
	status	= clFlush(m_queue);										if (status != CL_SUCCESS)	{ cout << "\nRunCL::patch_cam_and_lens_Hessian( ),  clFlush(m_queue) status  = "<<status<<" "<<checkerror(status) <<"\n"<<flush; exit_(status);}
	status	= clWaitForEvents (1, &ev);								if (status != CL_SUCCESS)	{ cout << "\nRunCL::patch_cam_and_lens_Hessian( ),  clWaitForEventsh(1, &ev) =" <<status<<" "<<checkerror(status) <<"\n"<<flush; exit_(status);}

}


void  RunCL::patch_cam_and_lens__hessian_reduce (uint layer, cl_mem param_hessian_map_mem, Matx55d &inv_Hessian ){				// called by Dynamic_slam::getFrame
	string 		fname	= "RunCL::patch_cam_and_lens__hessian_reduce()";
	int local_verbosity_threshold = V_RUNCL_PATCH_CAM_LENS_HESSIAN_REDUCE;														if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::patch_cam_and_lens__hessian_reduce()_chk1 layer="<<layer<<" #############################################################"<<flush;}
	cl_kernel	kernel	= patch_hessian_reduce_kernel;

	uint	cols		=		patch_hessian_cols[ layer];
	uint	rows		=		patch_hessian_rows[ layer];
	uint	mm_cols		=		mm_width;

	_clSetKernelArg( kernel,	1, sizeof(int),			&layer,						fname);										// __private	uint	cols		//1
	_clSetKernelArg( kernel,	2, sizeof(int),			&cols,						fname);										// __private	uint	cols		//2
	_clSetKernelArg( kernel,	3, sizeof(int),			&rows,						fname);										// __private	uint	rows		//3
	_clSetKernelArg( kernel,	4, sizeof(int),			&mm_cols,					fname);										// __private	uint	mm_cols		//4

	_clSetKernelArg( kernel,	6, sizeof(int),			&mm_layerstep,				fname);										// __private	uint	mm_pixels	//6
	_clSetKernelArg( kernel,	7, sizeof(cl_mem),		&param_hessian_map_mem,		fname);										// __private	uint				//7

	cl_int		status	= CL_SUCCESS;
	cl_event	ev		= 0;
	cl_int		res		= 0;
	size_t		threads_to_launch	= block_size;	// NB could be a problem on AMD GPUs with minmum 64 threads, not 32.
	size_t		local_work_size_	= block_size;

	for( uint row = 0; row<num_camera_matrix_DoF; row++){
		for( uint col = 0; col<num_camera_matrix_DoF; col++){
			uint start_idx	=	patch_hessian_start_idx[layer][row][col];														if( verbosity>local_verbosity_threshold) {cout<<"\nRunCL::patch_cam_and_lens__hessian_reduce()_chk2  patch_hessian_start_idx["<<layer<<"]["<<row<<"]["<<col<<"] = "<<patch_hessian_start_idx[layer][row][col]<<flush; }
			uint elem 		=	row * 6 + col;																					// NB same spaceing as SE3 tracking.
			_clSetKernelArg( kernel,	0, sizeof(int),	&start_idx,					fname);										// __private	uint	start_idx	//0
			_clSetKernelArg( kernel,	5, sizeof(int),	&elem,						fname);										// __private	uint	elem		//5

			// launch workgroup for this elem.
			res = clEnqueueNDRangeKernel(m_queue,	kernel, 1, 0, &threads_to_launch, &local_work_size_, 0, NULL, &ev);		if (res    != CL_SUCCESS)	{ cout << "\nres = " << checkerror(res) <<"\n"<<flush; exit_(res);}
		}
	}
	status	= clFlush(m_queue);							if (status != CL_SUCCESS)	{ cout << "\nRunCL::patch_hessian_reduce( ),  clFlush(m_queue) status  = "<<status<<" "<<checkerror(status)  <<"\n"<<flush; exit_(status);}
	status	= clWaitForEvents (1, &ev);					if (status != CL_SUCCESS)	{ cout << "\nRunCL::patch_hessian_reduce( ),  clWaitForEventsh(1, &ev) = "<<status<<" "<<checkerror(status)  <<"\n"<<flush; exit_(status);}
	status	= clFinish(m_queue);						if (status != CL_SUCCESS)	{ cout << "\nRunCL::patch_hessian_reduce( ),  clFinish(m_queue) status = "<<status<<" "<<checkerror(status)  <<"\n"<<flush; exit_(status);}

	Matx55d		Hessian;
	Mat			hessian_Mat(	(num_camera_matrix_DoF+1),	num_SE3_DoF,	CV_32FC4);											// NB we use the same Patch Hessian reduce kernel as SE3 Hessian, hence same num_SE3_DoF data spacing.
	size_t		data_size	=	(num_camera_matrix_DoF+1) *	num_SE3_DoF *	sizeof(cl_float4);
	size_t		offset		=	layer*8*6 ;

	ReadOutput( hessian_Mat.data, param_hessian_map_mem, data_size, offset*sizeof(cl_float4) );
																																if( verbosity>local_verbosity_threshold) {cout<<"\nRunCL::patch_cam_and_lens__hessian_reduce()_chk2.1";
																																	cout<<"\noffset="<<offset<<flush;
																																	cout<<"\nhessian_Mat = \n"<<hessian_Mat<<flush;
																																}
	for(int row=0; row<num_camera_matrix_DoF; row++){																			// per_pixel division currently done in kernel, TO DO which is better ?
		for(int col=0; col<num_camera_matrix_DoF; col++){
			Hessian.operator()(row,col)							= hessian_Mat.at<cl_float4>( row+1,col ).x; 					// NB choose colour channel of Hessian
		}
	}

	//  Eigen pseudo-inverse
	Eigen::MatrixXd GN_H(num_camera_matrix_DoF,num_camera_matrix_DoF);															// TO DO replace Eigen with a kernel for 6x6 matrix pseudo-inverse or inverse.
	for (int i=0;i<num_camera_matrix_DoF;i++){																					// Hard code efficient computation of 6x6 inversion, & Det.
		for (int j=0;j<num_camera_matrix_DoF;j++){
			GN_H(i,j) 											= Hessian.operator()(i,j);	// GN_Hessian.operator()(i,j);
		}
	}
	Eigen::MatrixXd pinv 										= GN_H.completeOrthogonalDecomposition().pseudoInverse();
	for (int i=0;i<num_camera_matrix_DoF;i++){
		for (int j=0;j<num_camera_matrix_DoF;j++){
			inv_Hessian.operator()(i,j)							= pinv(i,j);
		}
	}
																																if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::patch_cam_and_lens__hessian_reduce()_chk3 ."<<flush;	// Save buffers to file ###########
																																	stringstream ss;
																																	ss << "patch_cam_and_lens__hessian_reduce__frame_num="<<current_frames[ current_frames_idx[0] ].dataset_frame_num<<"_layer="<<layer<<"_";
																																	bool show 		= false;
																																	bool old_tiff 	= tiff;
																																	tiff 			= true;
																																	float max_range	= 1;
																																	cv::Mat bufImg;
																																	_cl_flush_finish(m_queue, fname);
																																	DownloadAndSave_3Channel( 	param_hessian_map_mem,	ss.str( ), paths.at( "hessian"),  	mm_size_bytes_C4,   mm_Image_size,   CV_32FC4, 	show, &bufImg, max_range,  0,			false);
																																	DownloadAndSave_3Channel( 	param_hessian_map_mem,	ss.str( ), paths.at( "jacobian"),  	mm_size_bytes_C4,   mm_Image_size,   CV_32FC4, 	show, &bufImg, max_range,  mm_size_bytes_C4/*mm_layerstep*/, false);
																																	// NB the tiff file holda the int32 values as float32. This is okay because they fit in the mantissa.
																																	// BGRA format, B=u, G=v, R=read_index, A=alpha.
																																	////////////////

																																	if( layer==0){
																																		stringstream 	ss_path;
																																		ss_path.str(std::string()); // reset ss_path
																																		ss_path 		<< "SE3_grad_map_mem"<<flush;
																																		cout 			<< "\n" << ss_path.str() <<flush;
																																		cout 			<< "\n" << paths.at(ss_path.str()) <<flush;
																																		//DownloadAndSave_6Channel_volume(  SE3_grad_map_mem, ss.str(), paths.at(ss_path.str()), mm_size_bytes_C4, mm_Image_size, CV_32FC4, false, 1, 6 );
																																	}
																																	//////////
																																	tiff 			= old_tiff;

																																	PRINT_MATX55D( inv_Hessian, )
																																	cout<<"\n\nRunCL::patch_cam_and_lens__hessian_reduce()_Finished ."<<flush;
																																}
}
