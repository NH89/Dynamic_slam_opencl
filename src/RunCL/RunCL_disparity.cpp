#include "RunCL.hpp"

/*
// void RunCL::disparity(uint start, uint stop){
//     string fname = "RunCL::disparity( )";
// 	int local_verbosity_threshold = V_RUNCL_DISPARITY;
// 	const uint wg_divisor = 8; //4;	// 512 * 20 * 32 / 8 =  40,960 bytes # TODO automate computation of wg_divisor .
// 	//  Intel(R) Iris(R) Xe Graphics : Local memory size  65,536 (64KiB),  Max work item dimensions 3,   Max work item sizes 512x512x512,   Max work group size 512.
// / *
// 														// const uint wg_divisor = 4;	//  512 * 20 * 32 / 8 =  40,960 bytes
// 																						// 1024 * 20 * 32 /
//
// 														// Intel(R) Iris(R) Xe Graphics :
// 														// Local memory size  65,536 (64KiB),
// 														// Max work item dimensions 3,
// 														// Max work item sizes 512x512x512,
// 														// Max work group size 512.
//
// 														// NVIDIA GeForce GTX 970M
// 														// Local memory size               49,152 (48KiB)
// 														// Max work item dimensions        3
// 														// Max work item sizes             1024x1024x64
// 														// Max work group size             1024
// * /
// 																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::disparity( ..)_chk0 #############################################################"<<flush;
// 																																				cout << "\n local_work_size = " << local_work_size
// 																																				<< ",  start = " << start << ",  stop = " << stop
// 																																				<< ",  local_work_size*4*5/wg_divisor*sizeof(float) = " << local_work_size*4*5/wg_divisor*sizeof(float)
// 																																				<< ",  local_work_size*4*5/wg_divisor = " << local_work_size*4*5/wg_divisor
// 																																				<< flush;
// 																																			}
// 	// __private	 uint layer, set in mipmap_call_kernel( ..) below																		//__private	    uint		layer,				//0
// 	// constant
// 	_clSetKernelArg( disparity_kernel,  1, sizeof( cl_mem), &mipmap_buf,				fname);												//__constant	uint*		mipmap_params,		//1
// 	_clSetKernelArg( disparity_kernel,  2, sizeof( cl_mem), &uint_param_buf,			fname);												//__constant	uint*		uint_params,		//2
// 	_clSetKernelArg( disparity_kernel,  3, sizeof( cl_mem), &fp32_param_buf,			fname);												//__constant	uint*		fp32_params,		//3
//     // inputs
// 	_clSetKernelArg( disparity_kernel,  4, sizeof( cl_mem), &keyframe_imgmem,			fname);												//__global		float4*		img_cur,			//4		// keyframe
// 	_clSetKernelArg( disparity_kernel,  5, sizeof( cl_mem), &imgmem,					fname);												//__global		float4*		img_new,			//5
// 	_clSetKernelArg( disparity_kernel,  6, sizeof( cl_mem), &keyframe_g1mem,			fname);												//__global		float8* 	g1p,				//6		// keyframe_g1mem
// 	// local
// 	_clSetKernelArg( disparity_kernel,  7, local_work_size*4*5/wg_divisor*sizeof(float),	NULL, 	fname);									//__local	 	float4*		local_img_cur, 		//7
// 	_clSetKernelArg( disparity_kernel,  8, local_work_size*4*5/wg_divisor*sizeof(float),	NULL, 	fname);									//__local	 	float4*		local_img_new, 		//8
// 	_clSetKernelArg( disparity_kernel,  9, local_work_size*4*5/wg_divisor*sizeof(float),	NULL, 	fname);									//__local	 	float4*		local_img_cur_sq, 	//9
// 	_clSetKernelArg( disparity_kernel, 10, local_work_size*4*5/wg_divisor*sizeof(float),	NULL, 	fname);									//__local	 	float4*		local_img_new_sq, 	//10
// 	// outputs
// 	_clSetKernelArg( disparity_kernel, 11, sizeof( cl_mem), &binocular_rho,				fname);												//__global		float4*		Rho_,				//11
// 	_clSetKernelArg( disparity_kernel, 12, sizeof( cl_mem), &binocular_disparity,		fname);												//__global		float4*		disparity			//12
//
// 																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::disparity( ..)_chk1 .   local_work_size/wg_divisor = " << local_work_size/wg_divisor <<flush;}
// 	//uint start = 4, stop = 3;
// 	for (int iter = 0; iter<4; iter++){
// 		mipmap_call_kernel( disparity_kernel, m_queue, start, stop, false, local_work_size/(wg_divisor*2) );
// 																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::disparity( ..)_chk3 ."<<flush;								// Save buffers to file ###########
// 																																				stringstream ss;
// 																																				ss << "disparity_" << save_index << "_iter_" << iter;
// 																																				bool show 		= false;
// 																																				bool old_tiff 	= tiff;
// 																																				tiff 			= true;
// 																																				_cl_flush_finish(m_queue, fname);
// 																																				DownloadAndSave_3Channel( 	binocular_rho,  	  ss.str( ), paths.at( "binocular_rho"),  		mm_size_bytes_C4,   mm_Image_size,   CV_32FC4, 	show , -1);
// 																																				DownloadAndSave_3Channel( 	binocular_disparity,  ss.str( ), paths.at( "binocular_disparity"),  mm_size_bytes_C4,   mm_Image_size,   CV_32FC4, 	show , -1);
// 																																				tiff 			= old_tiff;
// 																																				if (iter==0){
// 																																					ss.str("");
// 																																					ss << "disparity_keyframe_imgmem" ;
// 																																					DownloadAndSave_3Channel( 	keyframe_imgmem,  ss.str( ), paths.at( "binocular_rho"),  		mm_size_bytes_C4,   mm_Image_size,   CV_32FC4, 	show , 1);
// 																																					ss.str("");
// 																																					ss << "disparity_imgmem" ;
// 																																					DownloadAndSave_3Channel( 	imgmem,			  ss.str( ), paths.at( "binocular_rho"),		mm_size_bytes_C4,   mm_Image_size,   CV_32FC4, 	show , 1);
// 																																				}
// 																																			}
// 	}
//                                                                                                                                             if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::disparity( ..) Finished ###########################################################"<<flush;}
// }
// // TODO Need sequential execution of mimpap layers AND propagation of warp to next layer, BUT also inherit detail from previous frame's warp.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void RunCL::compute_lookup_table(uint start, uint stop){
    string fname = "RunCL::disparity( )";
	int local_verbosity_threshold = V_RUNCL_DISPARITY;


}
//  __kernel void compute_lookup_table(					// computed once at start of program	// TODO when is it possible to roll the layers together ?  i.e. when local mem is not used.
// 	 // inputs
// 	__private	uint	layer,					//0
//
// 	__constant 	uint8*	mipmap_params,			//1
// 	__constant 	uint*	uint_params,			//2
// 	__constant  float*  fp32_params,			//3
//
// 	__global 	float4*	img_cur,				//4		// keyframe
//
// 	// output
// 	__global 	uint4*	lookup_table			//5
// )


void RunCL::warp_image(uint start, uint stop){
    string fname = "RunCL::disparity( )";
	int local_verbosity_threshold = V_RUNCL_DISPARITY;


}

//
//  __kernel void warp_image(								// computed once each iteration of warping, for each layer of image pyramid
// 	// inputs
// 	__private	uint	read_offset,			//0
// 	__private	uint	mm_cols,				//1
//
// 	__constant 	uint*	uint_params,			//2
//
// 	__global 	float2*	warp,					//3
// 	__global 	uint4*	lookup_table,			//4
// 	__global 	float4*	new_img,				//5
//
// 	// outputs
// 	__global 	float4*	new_img_warped			//6
// )
//
//


void RunCL::img_sq(uint start, uint stop){
    string fname = "RunCL::disparity( )";
	int local_verbosity_threshold = V_RUNCL_DISPARITY;


}
//  __kernel void img_sq(
// 	// inputs
// 	__global 	uint4*	lookup_table,			//0
// 	__global 	float4*	img,				//1
// 	// output
// 	__global 	float4*	img_sq					//2
// )
//


void RunCL::img_variance(uint start, uint stop){
    string fname = "RunCL::disparity( )";
	int local_verbosity_threshold = V_RUNCL_DISPARITY;


}
//  __kernel void img_variance(
// 	// inputs
// 	__private	uint	mm_cols,				//0
//
// 	__global 	uint4*	lookup_table,			//1
// 	__global 	float4*	img_sq,					//2
//
// 	// output
// 	__global 	float4*	img_var					//3
// )
//
//


void RunCL::compute_warp(uint start, uint stop){
    string fname = "RunCL::disparity( )";
	int local_verbosity_threshold = V_RUNCL_DISPARITY;


}
//  __kernel void compute_warp(
// 	// inputs
// 	__private	uint 	mm_size,				//0
// 	__private	uint	mm_cols,				//1
//
// 	__global 	uint4*	lookup_table,			//2
// 	__global 	float4*	curr_img,				//3
// 	__global 	float4*	new_img,				//4																		// NB warped version of the new image.
// 	__global 	float4*	curr_img_var,			//5
// 	__global 	float4*	new_img_var,			//6
//
// 	// output
// 	__global 	float4*	img_covar,				//7		// 5*float4*mm_size
// 	__global 	float4*	img_corr,				//8		// 5*float4*mm_size
// 	__global 	float2*	warp					//9
// )



