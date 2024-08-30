#include "RunCL.hpp"


void RunCL::disparity(){
    string fname = "RunCL::disparity( )";
	int local_verbosity_threshold = V_RUNCL_DISPARITY;
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::disparity( ..)_chk0 #############################################################"<<flush;}
	// __private	 uint layer, set in mipmap_call_kernel( ..) below																		//__private	    uint		layer,				//0
	// constant
	_clSetKernelArg( disparity_kernel,  1, sizeof( cl_mem), &mipmap_buf,				fname);												//__constant	uint*		mipmap_params,		//1
	_clSetKernelArg( disparity_kernel,  2, sizeof( cl_mem), &uint_param_buf,			fname);												//__constant	uint*		uint_params,		//2
	_clSetKernelArg( disparity_kernel,  3, sizeof( cl_mem), &fp32_param_buf,			fname);												//__constant	uint*		fp32_params,		//3
    // inputs
	_clSetKernelArg( disparity_kernel,  4, sizeof( cl_mem), &keyframe_imgmem,			fname);												//__global		float4*		img_cur,			//4		// keyframe
	_clSetKernelArg( disparity_kernel,  5, sizeof( cl_mem), &imgmem,					fname);												//__global		float4*		img_new,			//5
	_clSetKernelArg( disparity_kernel,  6, sizeof( cl_mem), &keyframe_g1mem,			fname);												//__global		float8* 	g1p,				//6		// keyframe_g1mem
	// local
	_clSetKernelArg( disparity_kernel,  7, local_work_size*4*5*sizeof(float),	NULL, 	fname);												//__local	 	float4*		local_img_cur, 		//7
	_clSetKernelArg( disparity_kernel,  8, local_work_size*4*5*sizeof(float),	NULL, 	fname);												//__local	 	float4*		local_img_new, 		//8
	_clSetKernelArg( disparity_kernel,  9, local_work_size*4*5*sizeof(float),	NULL, 	fname);												//__local	 	float4*		local_img_cur_sq, 	//9
	_clSetKernelArg( disparity_kernel, 10, local_work_size*4*5*sizeof(float),	NULL, 	fname);												//__local	 	float4*		local_img_new_sq, 	//10
	// outputs
	_clSetKernelArg( disparity_kernel, 11, sizeof( cl_mem), &mean_mem,					fname);												//__global		float4*		Rho_,				//11
	_clSetKernelArg( disparity_kernel, 12, sizeof( cl_mem), &mean_mem,					fname);												//__global		float4*		disparity			//12

																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::disparity( ..)_chk1 ."<<flush;}
	mipmap_call_kernel( disparity_kernel, m_queue );
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::disparity( ..)_chk3 ."<<flush;
                                                                                                                                                stringstream ss;
																																				ss << "transform_costvolume" << save_index;													// Save buffers to file ###########
																																				bool show = false;
																																				DownloadAndSave( 	lomem,  ss.str( ), paths.at( "lomem"),  mm_size_bytes_C1,   mm_Image_size,   CV_32FC1, 	show , 8);	// a little more than the num images in costvol.
																																				DownloadAndSave( 	himem,  ss.str( ), paths.at( "himem"),  mm_size_bytes_C1,   mm_Image_size,   CV_32FC1, 	show , 8);	//params[COSTVOL_LAYERS]
                                                                                                                                            }
                                                                                                                                            if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::disparity( ..) Finished ###########################################################"<<flush;}
}
