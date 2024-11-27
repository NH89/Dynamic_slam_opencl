#include "RunCL.hpp"

void RunCL::compute_lookup_table(uint start, uint stop){
    string fname = "RunCL::disparity( )";
	int local_verbosity_threshold = V_RUNCL_COMPUTE_LOOKUP_TABLE;
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::compute_lookup_table( ..)_chk0 #############################################################"<<flush;
																																				cout << "\n local_work_size = " << local_work_size
																																				<< ",  start = " << start << ",  stop = " << stop
																																				<< flush;
																																			}
	// inputs
	// __private	 uint layer, set in mipmap_call_kernel( ..) below																		//__private	    uint		layer,				//0
	// __constant
	_clSetKernelArg( compute_lookup_table_kernel,  1, sizeof( cl_mem), &mipmap_buf,				fname);									//__constant	uint*		mipmap_params,		//1
	_clSetKernelArg( compute_lookup_table_kernel,  2, sizeof( cl_mem), &uint_param_buf,			fname);									//__constant 	uint*		uint_params,		//2
	_clSetKernelArg( compute_lookup_table_kernel,  3, sizeof( cl_mem), &fp32_param_buf,			fname);									//__constant	float*		fp32_params,		//3

	// output
	_clSetKernelArg( compute_lookup_table_kernel,  4, sizeof( cl_mem), &lookup_table_buf,		fname);									//__global 		uint4*		lookup_table		//4
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::compute_lookup_table( ..)_chk1 ."<<flush;}
	//mipmap_call_kernel( disparity_kernel, m_queue, start, stop, false, local_work_size );

	//void RunCL::mipmap_call_kernel(cl_kernel kernel_to_call, cl_command_queue queue_to_call, uint start, uint stop, bool layers_sequential, const size_t local_work_size){
	//int local_verbosity_threshold = V_RUNCL_MIPMAP_CALL_KERNEL;//verbosity_mp["RunCL::mipmap_call_kernel"];// -2;
																																			if(verbosity>local_verbosity_threshold) {
																																				cout<<"\nRunCL::mipmap_call_kernel( cl_kernel: compute_lookup_table_kernel,  cl_command_queue: m_queue,   start="<<start<<",   stop="<<stop<<
																																				 "local_work_size="<<local_work_size<<" )_chk0"<<flush;
																																			}
	cl_event		ev;
	cl_int			res, status;

	for(uint reduction = start; reduction <= stop; reduction++) {																			// NB processes largest layer first.
																																			if(verbosity>local_verbosity_threshold) { cout<<"\nRunCL::mipmap_call_kernel(..)_chk1,  reduction="<<reduction<<",  num_threads[reduction]="<<num_threads[reduction]<<"  local_work_size="<<local_work_size<<flush; }
		//if (reduction>=start && reduction<stop){																							// compute num threads to launch & num_pixels in reduction
																																			//if(verbosity>local_verbosity_threshold) { cout<<"\nRunCL::mipmap_call_kernel(..)_chk2 :  num_threads[reduction]="<<num_threads[reduction]<<"  local_work_size="<<local_work_size<<flush; }
			res 	= clSetKernelArg(compute_lookup_table_kernel, 0, sizeof(int), &reduction);							if (res    !=CL_SUCCESS)	{ cout <<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	;
			res 	= clEnqueueNDRangeKernel(m_queue, compute_lookup_table_kernel, 1, &lookup_table_offset[reduction], &num_threads[reduction], &local_work_size, 0, NULL, &ev); // run mipmap_float4_kernel, NB wait for own previous iteration.
																											if (res    != CL_SUCCESS)	{ cout << "\nres = " << checkerror(res) <<"\n"<<flush; exit_(res);}
			status 	= clFlush(m_queue);																if (status != CL_SUCCESS)	{ cout << "\nRunCL::mipmap_call_kernel( cl_kernel: compute_lookup_table_kernel,  clFlush(queue_to_call) status  = "		<<status<<" "<< checkerror(status) <<"\n"<<flush; exit_(status);}
			status 	= clWaitForEvents (1, &ev);								if (status != CL_SUCCESS)	{ cout << "\nRunCL::mipmap_call_kernel( cl_kernel: compute_lookup_table_kernel) for loop,  clWaitForEventsh(1, &ev) ="	<<status<<" "<<checkerror(status)  <<"\n"<<flush; exit_(status);}

			lookup_table_offset[reduction +1]=  lookup_table_offset[reduction] + num_threads[reduction];									// NB this pads the lookup table, so that local work groups will not be shared betwen layers.
																																			// It also  means that this offset should be used to launch layers from the lookup table.
	}

																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::compute_lookup_table( ..)_chk2 ."<<flush;	// Save buffers to file ###########
																																				stringstream ss;
																																				ss << "compute_lookup_table_" << save_index ;
																																				bool show 		= false;
																																				bool old_tiff 	= tiff;
																																				tiff 			= true;
																																				_cl_flush_finish(m_queue, fname);
																																				DownloadAndSave_3Channel( 	lookup_table_buf,	ss.str( ), paths.at( "lookup_table"),  		mm_size_bytes_C4,   mm_Image_size,   CV_32FC4, 	show , -1);
																																				tiff 			= old_tiff;
																																			}
}


void RunCL::warp_image(uint start, uint stop, uint layer ){																					// computed once each iteration of warping, for each layer of image pyramid
    string fname = "RunCL::disparity( )";
	int local_verbosity_threshold = V_RUNCL_WARP_IMAGE;
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::warp_image( ..)_chk0 #############################################################"<<flush;
																																				cout << "\n local_work_size = " << local_work_size
																																				<< ",  start = " << start << ",  stop = " << stop
																																				<< flush;
																																			}
uint	read_offset =	0;
uint	mm_cols		=	0;

	// inputs
	// __private
	_clSetKernelArg( warp_image_kernel,  			0, sizeof( uint), &read_offset,				fname);										// 	__private	uint	read_offset,			//0
	_clSetKernelArg( warp_image_kernel,  			1, sizeof( uint), &mm_cols,					fname);										// 	__private	uint	mm_cols,				//1
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
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::( ..)_chk2 ."<<flush;								// Save buffers to file ###########
																																				stringstream ss;
																																				ss << "compute_lookup_table_" << save_index ;
																																				bool show 		= false;
																																				bool old_tiff 	= tiff;
																																				tiff 			= true;
																																				_cl_flush_finish(m_queue, fname);
																																				DownloadAndSave_3Channel( 	new_img_warped_buf,	ss.str( ), paths.at( "warp_image"),  		mm_size_bytes_C4,   mm_Image_size,   CV_32FC4, 	show , -1);
																																				tiff 			= old_tiff;
																																			}
}


void RunCL::img_sq(uint start, uint stop, cl_mem img_buf, cl_mem img_sq_buf){
    string fname = "RunCL::disparity( )";
	int local_verbosity_threshold = V_RUNCL_IMG_SQ;
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::img_sq( ..)_chk0 #############################################################"<<flush;
																																				cout << "\n local_work_size = " << local_work_size
																																				<< ",  start = " << start << ",  stop = " << stop
																																				<< flush;
																																			}
	// inputs
	// __global
	_clSetKernelArg( img_sq_kernel,  				0, sizeof( cl_mem), &lookup_table_buf,		fname);										// 	__global 	uint4*	lookup_table,			//0
	_clSetKernelArg( img_sq_kernel,		  			1, sizeof( cl_mem), &img_buf,				fname);										// 	__global 	float4*	img,					//1
	// output
	_clSetKernelArg( img_sq_kernel,		  			2, sizeof( cl_mem), &img_sq_buf,			fname);										// 	__global 	float4*	img_sq					//2



}


void RunCL::img_variance(uint start, uint stop, cl_mem img_sq_buf, cl_mem img_var_buf){
    string fname = "RunCL::disparity( )";
	int local_verbosity_threshold = V_RUNCL_IMG_VARIANCE;
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::img_variance( ..)_chk0 #############################################################"<<flush;
																																				cout << "\n local_work_size = " << local_work_size
																																				<< ",  start = " << start << ",  stop = " << stop
																																				<< flush;
																																			}
uint	mm_cols = 0;

	// inputs
	// __private
	_clSetKernelArg( img_variance_kernel,  			0, sizeof( uint), 	&mm_cols,				fname);										// 	__private	uint	mm_cols,				//0
	// __global
	_clSetKernelArg( img_variance_kernel,  			1, sizeof( cl_mem), &lookup_table_buf,		fname);										// 	__global 	uint4*	lookup_table,			//1
	_clSetKernelArg( img_variance_kernel,  			2, sizeof( cl_mem), &img_sq_buf,			fname);										// 	__global 	float4*	img_sq					//2

	// output
	_clSetKernelArg( img_variance_kernel,  			3, sizeof( cl_mem), &img_var_buf,			fname);										// 	__global 	float4*	img_var					//3



}


void RunCL::compute_warp(uint start, uint stop){
    string fname = "RunCL::disparity( )";
	int local_verbosity_threshold = V_RUNCL_COMPUTE_WARP;
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::compute_warp( ..)_chk0 #############################################################"<<flush;
																																				cout << "\n local_work_size = " << local_work_size
																																				<< ",  start = " << start << ",  stop = " << stop
																																				<< flush;
																																			}
uint 	mm_size = 0;
uint	mm_cols = 0;

	// inputs
	// __private
	_clSetKernelArg( compute_warp_kernel,  			0, sizeof( uint), &mm_size,					fname);										// 	__private	uint 	mm_size,				//0
	_clSetKernelArg( compute_warp_kernel,  			1, sizeof( uint), &mm_cols,					fname);										// 	__private	uint	mm_cols,				//1
	// __global
	_clSetKernelArg( compute_warp_kernel,  			2, sizeof( cl_mem), &lookup_table_buf,		fname);										// 	__global 	uint4*	lookup_table,			//2
	_clSetKernelArg( compute_warp_kernel,  			3, sizeof( cl_mem), &curr_img_buf,			fname);										// 	__global 	float4*	curr_img,				//3
	_clSetKernelArg( compute_warp_kernel,  			4, sizeof( cl_mem), &new_img_buf,			fname);										// 	__global 	float4*	new_img,				//4			// NB warped version of the new image.
	_clSetKernelArg( compute_warp_kernel,  			5, sizeof( cl_mem), &curr_img_var_buf,		fname);										// 	__global 	float4*	curr_img_var,			//5
	_clSetKernelArg( compute_warp_kernel,  			6, sizeof( cl_mem), &new_img_var_buf,		fname);										// 	__global 	float4*	new_img_var,			//6

	// output
	_clSetKernelArg( compute_warp_kernel,  			7, sizeof( cl_mem), &img_covar_buf,			fname);										// 	__global 	float4*	img_covar,				//7		// 5*float4*mm_size
	_clSetKernelArg( compute_warp_kernel,  			8, sizeof( cl_mem), &img_corr_buf,			fname);										// 	__global 	float4*	img_corr,				//8		// 5*float4*mm_size
	_clSetKernelArg( compute_warp_kernel,  			9, sizeof( cl_mem), &warp_buf,				fname);										// 	__global 	float2*	warp					//9


}
