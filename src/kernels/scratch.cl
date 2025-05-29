
#include "kernels_macros.h"
#include "kernels.h"

// RunCL::rho_sq( ..)_chk_4.6
// tracking_num_samples*2*mm_size_bytes_C4	=63523200
// 24 * mm_size_bytes_C1					=63523200

// cv::Mat temp(mm_height, mm_width, CV_32FC3);
//
// cv::Mat temp2(mm_height, mm_width, CV_32FC1);
//
// mm_size_bytes_C4	= temp.total() * 4 * sizeof(float);
//
// mm_size_bytes_C1	= temp2.total() * temp2.elemSize();


// const uint tracking_num_samples 		= TRACKING_NUM_SAMPLES +1;  // = 3
//
// SE3_weight_map_mem	= clCreateBuffer(m_context, CL_MEM_READ_WRITE 					,24 * mm_size_bytes_C1,			0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 9= "<<checkerror(res)<<"\n"<<flush;exit_(res);}
//
// SE3_incr_map_mem		= clCreateBuffer(m_context, CL_MEM_READ_WRITE 					,24 * mm_size_bytes_C1,			0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 9= "<<checkerror(res)<<"\n"<<flush;exit_(res);} // For debugging before summation.
//
// SE3_rho_map_mem		= clCreateBuffer(m_context, CL_MEM_READ_ONLY  , tracking_num_samples*2*mm_size_bytes_C4,  		0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 34= "<<checkerror(res)<<"\n"<<flush;exit_(res);}

// tracking_num_samples*2*mm_size_bytes_C4   = 3*2*4 * mm_size_bytes_C1 = 24 * mm_size_bytes_C1  ?

// 	const uint block_size		= 32;						// or send as __private arg ? BUT as hardcoded "const uint" it can be used to size arrays etc.
// 	const uint se3_dof			= 6;
//
// 	float2 rho[		block_size]					= {0.0f};																	// pvt variable for values in this column.
// 	float2 weights[	block_size * se3_dof]		= {0.0f};



__kernel void Rho_sq(

	__private	uint		layer,					//0
	__private	uint 		cols_per_row,			//1
	__private	uint 		out_block_size,			//2

	__constant	uint8*		mipmap_params,			//3
	__constant	uint*		uint_params,			//4
	__constant	float*		fp32_params,			//5

	//output
	__global	float2*		Rho_,					//20	// { sum rho^2 ,  count of valid pixels used } Writen to dense patches.
	__local		float2*		local_rho,				//21	// float2 local_rho[ local_work_size/2 ]  hence sizeof( float)*local_work_size.

	__global	float2*		weights_map,			//22
	__local		float2*		local_weights,			//23

	__global	float2*		SE3_incr_map_,			//24
	__local		float2*		local_SE3_incr			//25
	)
{
	const uint block_size		= 32;						// or send as __private arg ? BUT as hardcoded "const uint" it can be used to size arrays etc.
	const uint se3_dof			= 6;
	const uint num_past_frames	= 4;
	uint  global_id_u 			= get_global_id(0);
	float global_id_flt 		= global_id_u;
	uint  lid 					= get_local_id(0);

	const uint local_size 		= get_local_size(0);

	uint8 mipmap_params_ 		= mipmap_params[layer];
	uint layer_pixels			= mipmap_params_[MiM_PIXELS];
	uint mm_cols				= uint_params[MM_COLS];

	uint row_length				= cols_per_row; 												// blocks_cols * block_size;
	uint row_col				= fmod((float)global_id_u, row_length);
	uint block_row				= global_id_u / row_length;

	uint write_spacing			= block_size/out_block_size;
	uint write_index 			= row_col/out_block_size + block_row*write_spacing*mm_cols;

	float2 rho[block_size]				= {0.0f};																	// pvt variable for values in this column.
	float4 rho_pvt_flt4;
	float2 rho_pvt_flt2;

	float2 weights[block_size*se3_dof]	= {0.0f};																	// pvt variable for values in this column.
	float4 weights_pvt_flt4;
	float2 weights_pvt_flt2;

	float2 SE3_incr[block_size*se3_dof]	= {0.0f};																	// pvt variable for values in this column.
	float4 SE3_incr_pvt_flt4;
	float2 SE3_incr_pvt_flt2;

	float4 img_cur_pvt[block_size];																					// pvt variable for values in this column.
	float8 g1p_pvt[block_size];

	bool intersection;

	for (uint row_in_block=0; row_in_block<block_size; row_in_block +=2){						// step through pairs of rows of the patch, /////////////////////////////////////////////////////////////
		for (uint past_frame_idx=0; past_frame_idx</*num_past_frames*/1; past_frame_idx++){		// step though past frames
			if (intersection){
				for (uint se3_dim=0; se3_dim<se3_dof; se3_dim++) {
					int var =0; // place holder
				}
			}
			barrier(CLK_GLOBAL_MEM_FENCE );
			//...
			rho[row_in_block]		+= rho_pvt_flt2;

			if (intersection){
				//rho_pvt_flt4		= img_cur_pvt[row_in_block+1]	-	bilinear_flt4( img_past[past_frame_idx], u2_flt_2, v2_flt_2,  mm_cols, read_offset_ );									// find 2nd row pixel rho
				// TODO second row....
			}
			barrier(CLK_GLOBAL_MEM_FENCE );
			//...
			rho[row_in_block+1]		+= rho_pvt_flt2;
		}
	}

	uint step=2;
	for (; step<out_block_size; step *=2){

		for (uint block_row=0; block_row<block_size ; block_row += step){

			if (fmod((float)lid,step/2)==0) {																																					// selects threads separated by 1/2 step, i.e results of previous iteration of patch reduction.
																						rho[	  block_row ]							+= rho[		 block_row + step/2 ];						// sum pair of values in col,
				for (uint se3_dim=0; se3_dim<se3_dof; se3_dim++) {
																						SE3_incr[ block_row + se3_dim*block_size ]		+= SE3_incr[ block_row + step/2 + se3_dim*block_size ];
																						weights[  block_row + se3_dim*block_size ]		+= weights[  block_row + step/2 + se3_dim*block_size ];
				}
			}

			if( !(fmod((float)lid,step)==0) &&  (fmod((float)lid,step/2)==0)    ){																												// selects 2nd column, sends data
																						local_rho[		lid/step ] 						= rho[		 block_row];
				for (uint se3_dim=0; se3_dim<se3_dof; se3_dim++) {
																						local_SE3_incr[ lid/step + se3_dim*local_size ]	= SE3_incr[  block_row + se3_dim*block_size ];			// NB integer division. Hence both threads use the same index to local memory.
																						local_weights[  lid/step + se3_dim*local_size ]	= weights[   block_row + se3_dim*block_size ];
				}
			}
			barrier(CLK_LOCAL_MEM_FENCE );																																						// Using barrier as a semaphore, for local mem messages between threads.

			if( (fmod((float)lid,step)==0)  ){																																					// selects 1st column, adds data. Sum of patch now held in top left element of patch.
																						rho[block_row] 									+= local_rho[lid/step];
				for (uint se3_dim=0; se3_dim<se3_dof; se3_dim++) {
																						SE3_incr[  block_row + se3_dim*block_size ]		+= local_SE3_incr[ lid/step + se3_dim*local_size ];
																						weights[   block_row + se3_dim*block_size ]		+= local_weights[  lid/step + se3_dim*local_size ];
				}
			}
			barrier(CLK_LOCAL_MEM_FENCE );
		}
	}

	uint write_block_row=0;
	if( fmod((float)lid,out_block_size) ==0 ){																																					// selects columns i.e. threads within the workgroup

		for (uint block_row=0; block_row< block_size ; block_row += step, write_block_row++){
																						Rho_[			write_index + write_block_row*mm_cols]							= rho[		 block_row ];
			for (uint se3_dim=0; se3_dim</*se3_dof*/1; se3_dim++) {
																						SE3_incr_map_[	write_index + write_block_row*mm_cols + se3_dim*layer_pixels ] 	= SE3_incr[  block_row + se3_dim*block_size ];
																						weights[		write_index + write_block_row*mm_cols + se3_dim*layer_pixels ] 	= weights[   block_row + se3_dim*block_size ];
			}
		}
	}
	barrier(CLK_GLOBAL_MEM_FENCE );
}






















 __kernel void compute_warp(
	// inputs
	__private	uint	read_offset,			//0
	__private	uint 	mm_size,				//1
	__private	uint	mm_cols,				//2

	__global 	float4*	lookup_table,			//3
	__global 	float4*	curr_img,				//4
	__global 	float4*	new_img,				//5																		// NB warped version of the new image.
	__global 	float4*	curr_img_var,			//6
	__global 	float4*	new_img_var,			//7

	// output
	__global 	float4*	img_covar,				//8		// 5*float4*mm_size
	__global 	float4*	img_corr,				//9		// 5*float4*mm_size
	__global 	float2*	warp					//10	// 2*float4*mm_size // float2*
){
	uint read_index		= lookup_table[ get_global_id(0) + read_offset ].z;
	if (read_index ==0 ) {
		printf("\n_kernel compute_warp(..), mm_size=%u",mm_size);
		return;
	}
	float W[9] 			= { 1.0f/16, 2.0f/16, 1.0f/16, 2.0f/16, 4.0f/16, 2.0f/16, 1.0f/16, 2.0f/16, 1.0f/16 };			// 3x3 discrete Gaussian kernel
	float4 covar[5]		= {0};
	float4 corr[5]		= {0};

	uint 	read_index_3x3[9];
	read_index_3x3[1]	=	read_index 			-mm_cols;
	read_index_3x3[0]	=	read_index_3x3[1] 	-1;
	read_index_3x3[2]	=	read_index_3x3[1] 	+1;

	read_index_3x3[4]	=	read_index;
	read_index_3x3[3]	=	read_index 			-1;
	read_index_3x3[5]	=	read_index 			+1;

	read_index_3x3[7]	=	read_index 			-mm_cols;
	read_index_3x3[6]	=	read_index_3x3[7]	-1;
	read_index_3x3[8]	=	read_index_3x3[7] 	+1;

	int sample_idx[5];
	sample_idx[0] 		=	-mm_cols;
	sample_idx[1] 		=	-1;
	sample_idx[2] 		=	0;
	sample_idx[3] 		=	+1;
	sample_idx[4] 		=	+mm_cols;

	for (int j=0; j<5; j++){
		for (int i=0; i<9; i++ ){ covar[j]		+= curr_img[read_index_3x3[i] ] * new_img[read_index_3x3[i] + sample_idx[j] ] *  W[i]; }
		img_covar[read_index + j*mm_size]		= covar[j];

		float4 inv_denominator					= ( sqrt( curr_img_var[read_index] ) * sqrt( new_img_var[read_index + sample_idx[j] ]  ) );
		float4 denominator 						= isnormal(inv_denominator) ? 1/inv_denominator : 1;					// prevent div by zero

		corr[j] 								= covar[j] / denominator ;
		img_corr[read_index + j*mm_size]		= corr[j];
	}

 	float2 warp2								= warp[read_index];
 	float warp_u								= /*warp2.x +*/ compute_maximum( corr[1], corr[2], corr[3] );
 	float warp_v								= /*warp2.y +*/ compute_maximum( corr[0], corr[2], corr[4] );
 	warp_u										= clamp(warp_u, -1.0f, 1.0f);
 	warp_v										= clamp(warp_v, -1.0f, 1.0f);					// warp increment clamped to +/-1

	float2 warp2_new							= {warp_u, warp_v};
	warp[read_index]							= warp2_new;
}




//=======================

	__private	uint 		cols_per_row,			//1
	__private	uint 		out_block_size,			//2

const uint block_size	= 32;
uint mm_cols			= uint_params[MM_COLS];

uint8 mipmap_params_ 	= mipmap_params[layer];
uint read_rows_ 		= mipmap_params_[MiM_READ_ROWS];

uint row_length			= cols_per_row;
uint row_col			= fmod((float)global_id_u, row_length);
uint block_row			= global_id_u / row_length;

uint write_index_2 		= row_col/block_size + block_row*mm_cols; 		//fmod((float)global_id_u, blocks_cols * out_block_size)	+ (global_id_u / (uint)(blocks_cols * out_block_size)) * out_block_size;;


if( fmod((float)lid,block_size) ==0 ){		//out_block_size

	uint frame_offset_1 = write_index_2;

	step 				= block_size/2;

	for (uint block_row=0; block_row < block_size ; block_row += step, write_block_row++){

		uint offset_2 				= frame_offset_1 + write_block_row*mm_cols;

		Rho_[			offset_2  ]

		for (uint se3_dim=0; se3_dim<se3_dof; se3_dim++) {

			uint offset_3 			= offset_2 		+ se3_dim*( 4 + (read_rows_/block_size) )*mm_cols;

			SE3_incr_map_[	offset_3 ]



//=================
//=================
//=================

void RunCL::rho_sq(uint out_block_size, uint iter, uint layer  ){
	float zero  = 0;
	_clEnqueueFillBuffer( uload_queue, SE3_rho_map_mem, 	&zero, sizeof( float), 0, 		  2*mm_size_bytes_C1, 	fname);
	_clEnqueueFillBuffer( uload_queue, SE3_weight_map_mem, 	&zero, sizeof( float), 0, se3_dof*2*mm_size_bytes_C1, 	fname);
	_clEnqueueFillBuffer( uload_queue, SE3_incr_map_mem, 	&zero, sizeof( float), 0, se3_dof*2*mm_size_bytes_C1, 	fname);

	_clSetKernelArg( rho_sq_kernel, 2, sizeof( uint),   						&out_block_size,	 									fname);		//__private		uint 		out_block_size,			//1

	_clSetKernelArg( rho_sq_kernel, 3, sizeof( cl_mem), 						&mipmap_buf, 											fname);		//__constant	uint8*		mipmap_params,			//2
	_clSetKernelArg( rho_sq_kernel, 4, sizeof( cl_mem), 						&uint_param_buf, 										fname);		//__constant	uint*		uint_params,			//3
	_clSetKernelArg( rho_sq_kernel, 5, sizeof( cl_mem), 						&fp32_param_buf, 										fname);		//__constant	float*		fp32_params,			//4
	_clSetKernelArg( rho_sq_kernel, 6, sizeof( cl_mem), 						&k2kbuf, 												fname);		//__constant	float16*	inv_k2k,				//5		// transforms for 4 past frames

	_clSetKernelArg( rho_sq_kernel, 7, sizeof( cl_mem),							&current_frames[current_frames_idx[0]].img_buf,		 	fname);		//__global		float4*		img_cur,				//6		// multiple past frames. NB retain frames at powers of 2, and vary starting power plus num franes.
	_clSetKernelArg( rho_sq_kernel, 8, sizeof( cl_mem), 						&current_frames[current_frames_idx[1]].img_buf,			fname);		//__global		float4*		img_past_0,				//7
	_clSetKernelArg( rho_sq_kernel, 9, sizeof( cl_mem), 						&current_frames[current_frames_idx[2]].img_buf,			fname);		//__global		float4*		img_past_1,				//8
	_clSetKernelArg( rho_sq_kernel,10, sizeof( cl_mem), 						&current_frames[current_frames_idx[3]].img_buf,			fname);		//__global		float4*		img_past_2,				//9
	_clSetKernelArg( rho_sq_kernel,11, sizeof( cl_mem), 						&current_frames[current_frames_idx[4]].img_buf,			fname);		//__global		float4*		img_past_3,				//10

	_clSetKernelArg( rho_sq_kernel,12, sizeof( cl_mem), 						&depth_mem,												fname);		//__global		float* 		depth_map,				//11	// current frame depth, now stored as inv_depth
	_clSetKernelArg( rho_sq_kernel,13, sizeof( cl_mem), 						&g1mem,													fname);		//__global		float8* 	g1p,					//12	// current frame g1mem
	_clSetKernelArg( rho_sq_kernel,14, sizeof( cl_mem), 						&SE3_grad_map_mem,										fname);		//__global 		float8*		SE3_grad_map_cur_frame,	//14

	_clSetKernelArg( rho_sq_kernel,15, sizeof( cl_mem), 						&current_frames[current_frames_idx[0]].r_vel_buf,		fname);		//__global		float4*		img_cur,				//13	// multiple past frames.
	_clSetKernelArg( rho_sq_kernel,16, sizeof( cl_mem), 						&current_frames[current_frames_idx[1]].r_vel_buf,		fname);		//__global		float4*		img_past_0,				//14
	_clSetKernelArg( rho_sq_kernel,17, sizeof( cl_mem), 						&current_frames[current_frames_idx[2]].r_vel_buf,		fname);		//__global		float4*		img_past_1,				//15
	_clSetKernelArg( rho_sq_kernel,18, sizeof( cl_mem), 						&current_frames[current_frames_idx[3]].r_vel_buf,		fname);		//__global		float4*		img_past_2,				//16
	_clSetKernelArg( rho_sq_kernel,19, sizeof( cl_mem), 						&current_frames[current_frames_idx[4]].r_vel_buf,		fname);		//__global		float4*		img_past_3,				//17

	//output
	_clSetKernelArg( rho_sq_kernel,20, sizeof( cl_mem), 						&SE3_rho_map_mem, 										fname);		//__global		float2* 	Rho_,					//18	// { sum rho^2 ,  count of valid pixels used } Writen to dense patches.
	_clSetKernelArg( rho_sq_kernel,21, sizeof( float)*local_work_size,			NULL, 													fname);		//__local		float2*		local_rho				//19	// float2 local_rho[ local_work_size/2 ]  hence sizeof( float)*local_work_size.

	_clSetKernelArg( rho_sq_kernel,22, sizeof( cl_mem), 						&SE3_weight_map_mem,									fname);		//__global		float4* 	weights_map,			//22
	_clSetKernelArg( rho_sq_kernel,23, sizeof( float)*local_work_size*se3_dof,	NULL,													fname);		//__local		float4* 	local_weights,			//23	// float4 local_weights[ local_work_size/2 ]  hence sizeof( float)*local_work_size*4 NB only used as a message between threads.

	_clSetKernelArg( rho_sq_kernel,24, sizeof( cl_mem), 						&SE3_incr_map_mem,										fname);		//__global 		float4*		SE3_incr_map_,			//24
	_clSetKernelArg( rho_sq_kernel,25, sizeof( float)*local_work_size*se3_dof,	NULL,													fname);		//__local 		float4*		local_SE3_incr			//25

	const uint			patch_size 			= 32;																						//TODO set global patch size from device parameters		// generally:  device_work_size_multiple = patch_size * integer,   eg 32, 64, 128

	uint				reduction 			= layer;
	uint				read_rows			= MipMap[layer * 8 + MiM_READ_ROWS] ;
	uint				read_cols			= MipMap[layer * 8 + MiM_READ_COLS] ;
	uint				rows_blocks			= read_rows/patch_size  + (fmod(read_rows, patch_size) != 0) ;
	uint				cols_blocks			= read_cols/patch_size  + (fmod(read_cols, patch_size) != 0) ;
	uint				cols_per_row		= cols_blocks * patch_size;

	size_t				threads_required 	= rows_blocks * cols_blocks * patch_size ;
	size_t				threads_to_launch	= (threads_required/device_work_size_multiple  +  ( fmod( threads_required, device_work_size_multiple ) != 0 ) )  * device_work_size_multiple ; 	// smallest integer multiple of "device_work_size_multiple" >= threads_required;

	cl_kernel 			kernel_to_call		= rho_sq_kernel;
	cl_command_queue 	queue_to_call		= m_queue;

	_clSetKernelArg( rho_sq_kernel, 0, sizeof( uint),   				&layer,	 												fname);		//__private		uint 		layer,					//0
	_clSetKernelArg( rho_sq_kernel, 1, sizeof( uint),   				&cols_per_row,											fname);		//__private		uint 		cols_per_row,			//1

	res 	= clEnqueueNDRangeKernel(queue_to_call, kernel_to_call, 1, 0, &threads_to_launch, &device_work_size_multiple, 0, NULL, &ev); 	// run mipmap_float4_kernel, NB wait for own previous iteration.
}

__kernel void Rho_sq(
	)
{
	const uint block_size		= 32;						// or send as __private arg ? BUT as hardcoded "const uint" it can be used to size arrays etc.
	const uint se3_dof			= 6;
	const uint num_past_frames	= 4;						// 1,2,4,8,16,32,64 // variable select window of 4 frames.
	const float4 zero_f4		= {0.0f,0.0f,0.0f,0.0f};
	__global float4*	img_past[num_past_frames]		= { img_past_0, img_past_1, img_past_2, img_past_3 };
	__global float4*	vel_past[num_past_frames]		= { vel_past_0, vel_past_1, vel_past_2, vel_past_3 };

	uint  global_id_u 		= get_global_id(0);
	//float global_id_flt 	= global_id_u;
	uint  lid 				= get_local_id(0);
	uint  group_id			= get_group_id(0);
	const uint local_size 	= get_local_size(0);

	uint8 mipmap_params_ 	= mipmap_params[layer];
	uint read_offset_ 		= mipmap_params_[MiM_READ_OFFSET];
	uint read_cols_ 		= mipmap_params_[MiM_READ_COLS];
	uint read_rows_ 		= mipmap_params_[MiM_READ_ROWS];
	uint layer_pixels		= mipmap_params_[MiM_PIXELS];

	uint mm_cols			= uint_params[MM_COLS];
	uint mm_pixels			= uint_params[MM_PIXELS];

	float min_inv_depth		= fp32_params[MIN_INV_DEPTH];										//+ inv_d_step;
	float max_inv_depth		= fp32_params[MAX_INV_DEPTH];										//- inv_d_step;

	float reduction			= mm_cols/read_cols_;
	uint row_length			= cols_per_row; 													// blocks_cols * block_size;
	uint row_col			= fmod((float)global_id_u, row_length);
	uint block_row			= global_id_u / row_length;
	uint read_index			= read_offset_ + row_col + block_row*block_size*mm_cols;
	uint row_offset			= read_offset_/mm_cols;

	uint write_spacing		= block_size/out_block_size;
	uint write_index 		= row_col/out_block_size + block_row*write_spacing*mm_cols; 		//fmod((float)global_id_u, blocks_cols * out_block_size)	+ (global_id_u / (uint)(blocks_cols * out_block_size)) * out_block_size;;

	uint write_index_2 		= row_col/block_size + block_row*mm_cols; 		//fmod((float)global_id_u, blocks_cols * out_block_size)	+ (global_id_u / (uint)(blocks_cols * out_block_size)) * out_block_size;;


	float2 rho[block_size]				= {0.0f};												// pvt variable for values in this column.
	float4 rho_pvt_flt4;
	float2 rho_pvt_flt2;

	float2 weights[block_size*se3_dof]	= {0.0f};												// pvt variable for values in this column.
	float4 weights_pvt_flt4;
	float2 weights_pvt_flt2;

	float2 SE3_incr[block_size*se3_dof]	= {0.0f};												// pvt variable for values in this column.
	float4 SE3_incr_pvt_flt4;
	float2 SE3_incr_pvt_flt2;

	float4 img_cur_pvt[block_size];																// pvt variable for values in this column.
	float8 g1p_pvt[block_size];

	bool intersection;
	uint step;
	for ( step=2; step<block_size; step *=2){				// out_block_size																													// for each step size, (multiples of 2)
																																																// selects pairs of columns to sum  i.e. threads within the workgroup
		for (uint block_row=0; block_row<block_size ; block_row += step){																														// step through rows in column
			if (fmod((float)lid,step/2)==0) {																																					// selects threads separated by 1/2 step, i.e results of previous iteration of patch reduction.
																						rho[	  block_row ]							+= rho[		 block_row + step/2 ];						// sum pair of values in col,
				for (uint se3_dim=0; se3_dim<se3_dof; se3_dim++) {
																						SE3_incr[ block_row + se3_dim*block_size ]		+= SE3_incr[ block_row + step/2 + se3_dim*block_size ];
																						weights[  block_row + se3_dim*block_size ]		+= weights[  block_row + step/2 + se3_dim*block_size ];
				}
			}

			if( !(fmod((float)lid,step)==0) &&  (fmod((float)lid,step/2)==0)    ){																												// selects 2nd column, sends data
																						local_rho[		lid/step ] 						= rho[		 block_row];
				for (uint se3_dim=0; se3_dim<se3_dof; se3_dim++) {
																						local_SE3_incr[ lid/step + se3_dim*local_size ]	= SE3_incr[  block_row + se3_dim*block_size ];			// NB integer division. Hence both threads use the same index to local memory.
																						local_weights[  lid/step + se3_dim*local_size ]	= weights[   block_row + se3_dim*block_size ];
				}
			}
			barrier(CLK_LOCAL_MEM_FENCE );																																						// Using barrier as a semaphore, for local mem messages between threads.

			if( (fmod((float)lid,step)==0)  ){																																					// selects 1st column, adds data. Sum of patch now held in top left element of patch.
																						rho[block_row] 									+= local_rho[ lid/step ];
				for (uint se3_dim=0; se3_dim<se3_dof; se3_dim++) {
																						SE3_incr[  block_row + se3_dim*block_size ]		+= local_SE3_incr[ lid/step + se3_dim*local_size ];
																						weights[   block_row + se3_dim*block_size ]		+= local_weights[  lid/step + se3_dim*local_size ];
				}
			}
			barrier(CLK_LOCAL_MEM_FENCE );
		}
	}
	float2 temp2a											= { (float)/*block_row*/group_id, (float)/*block_col*/lid };
	if ( read_index < mm_pixels )	{	Rho_[read_index ] 	= temp2a;	}																														// Marks the area where the img buf is read, lines show top row of each patch.
																																																// Breaks show bondaries of patches.
	barrier(CLK_GLOBAL_MEM_FENCE );
	uint write_block_row=0;
	if( fmod((float)lid,block_size) ==0 ){		//out_block_size																																// selects columns i.e. threads within the workgroup
		uint frame_offset_1 = write_index_2;	// stacks frame SE3 results vertically.
		step = block_size/2;

		for (uint block_row=0; block_row < block_size ; block_row += step, write_block_row++){
																						uint offset_2 				= frame_offset_1 + write_block_row*mm_cols;
																						Rho_[			offset_2  ]	= rho[		 block_row ];
			for (uint se3_dim=0; se3_dim<se3_dof; se3_dim++) {																																	// All 6 DoF of SE3
																						uint offset_3 				= offset_2 		+ se3_dim*( 4 + (read_rows_/block_size) )*mm_cols;
																						uint offset_4				= block_row 	+ se3_dim*block_size;
																						SE3_incr_map_[	offset_3 ]	= SE3_incr[  offset_4 ];
																						weights_map[	offset_3 ]	= weights[   offset_4 ];
			}
		}
	}
	barrier(CLK_GLOBAL_MEM_FENCE );
}


void RunCL::update_SE3( uint layer, float delta_theta, float delta )									// NB good for images upto 640x480 layer zero, above that need a patch kernel approach to ensure each DoF fits in 1 workgroup. see device_work_size_multiple
{
	cl_kernel		kernel 				= update_SE3_kernel;																			//NB call just one workgroup to sum the whole image maps from the patch kernel.
	const uint		patch_size 			= 32;																							//TODO set global patch size from device parameters // generally:  device_work_size_multiple = patch_size * integer,   eg 32, 64, 128
	const uint		SE3_DoF				= 6;
	uint			read_rows			= MipMap[layer * 8 + MiM_READ_ROWS] ;
	uint			read_cols			= MipMap[layer * 8 + MiM_READ_COLS] ;
	uint			rows_blocks			= read_rows/patch_size  + (fmod(read_rows, patch_size) != 0) ;									// num rows in the fully reduced map
	uint			cols_blocks			= read_cols/patch_size  + (fmod(read_cols, patch_size) != 0) ;									// num cols in the fully reduced map

	uint			threads_per_DoF		= powf(2,ceil( log2((float)cols_blocks) )); // 10 layer 1 =>  pown(2,ciel(log2(10.0f) ))=16; 6*16=96.      // * rows_blocks  ;//	8x10=80 layer1 => 96 threads to launch?			// num pixels in fully reduced map. Req per SE3 DoF.
	uint 			DoF_per_workgroup	= device_work_size_multiple / threads_per_DoF;
	size_t			threads_required	= (device_work_size_multiple * SE3_DoF) / DoF_per_workgroup;		// NB device_work_size_multiple is usually a poer of 2, DoF_per_workgroup will also be a power of 2.
	uint 			workgroups_required	= ceil( (float)threads_required / device_work_size_multiple );
	size_t			threads_to_launch	= workgroups_required  *  device_work_size_multiple;

	uint 			row_offset			= rows_blocks + 4;																								//2
	uint			thread_offset		= threads_per_DoF;																								//3     2^n  > pixels in fully reduced patch
	uint			mm_cols				= uint_params[MM_COLS];																							//4
	float			img_var				= img_stats[ layer*4 + IMG_VAR ] + img_stats[ layer*4 + IMG_VAR +1 ] + img_stats[ layer*4 + IMG_VAR +2 ];		//5		sum image variance over 3channels, for this layer.
	cl_float2		delta_SE3			= {{delta_theta, delta}};																						//6

	//private
	_clSetKernelArg( kernel, 0, sizeof( uint),							&cols_blocks,				fname);								//__private	uint		cols,					//0
	_clSetKernelArg( kernel, 1, sizeof( uint),							&rows_blocks,				fname);								//__private	uint 		rows,					//1
	_clSetKernelArg( kernel, 2, sizeof( uint),							&row_offset,				fname);								//__private	uint 		row_offset,				//2
	_clSetKernelArg( kernel, 3, sizeof( uint),							&thread_offset,				fname);								//__private	uint		thread_offset,			//3
	_clSetKernelArg( kernel, 4, sizeof( uint),							&mm_cols,					fname);								//__private	uint		mm_cols,				//4
	_clSetKernelArg( kernel, 5, sizeof( float),							&img_var,		 			fname);								//__private	float		img_var,				//5
	_clSetKernelArg( kernel, 6, sizeof( cl_float2),						&delta_SE3,					fname);								//__private	float2		delta_SE3,				//6
	//global
	_clSetKernelArg( kernel, 7, sizeof( cl_mem),						&SE3_rho_map_mem,			fname);								//__global	float2*		Rho_,					//7		// { sum rho^2 ,  count of valid pixels used } Writen to dense patches.
	_clSetKernelArg( kernel, 8, sizeof( cl_mem),						&SE3_weight_map_mem,		fname);								//__global	float2*		weights_map,			//8
	_clSetKernelArg( kernel, 9, sizeof( cl_mem),						&SE3_incr_map_mem,			fname);								//__global	float2*		SE3_incr_map_,			//9
	//local
	_clSetKernelArg( kernel,10, ( local_work_size/2 )*sizeof( float),	 NULL, 						fname);								//__local	float2*		local_Rho_,				//10		// used for sum-reduce. Need to be [groupsize/2], set in host fn.
	_clSetKernelArg( kernel,11, ( local_work_size/2 )*sizeof( float),	 NULL, 						fname);								//__local	float2*		local_weights_map,		//11
	_clSetKernelArg( kernel,12, ( local_work_size/2 )*sizeof( float),	 NULL, 						fname);								//__local	float2*		local_SE3_incr_map_,	//12

	// input/output, global
	_clSetKernelArg( kernel, 13, sizeof( cl_mem), 						&pose_update_buf,			fname);								//__global	float*		pose_update,			//13	// 6_DoF
	_clSetKernelArg( kernel, 14, sizeof( cl_mem), 						&distorsion_update_buf, 	fname);								//__global	float*		distorsion_update,		//14
	_clSetKernelArg( kernel, 15, sizeof( cl_mem), 						&old_result_buf, 			fname);								//__global	float*		old_result				//15

res 	= clEnqueueNDRangeKernel(queue_to_call, kernel, 1, 0, &threads_to_launch, &device_work_size_multiple, 0, NULL, &ev); 	// run mipmap_float4_kernel, NB wait for own previous iteration.
}

__kernel void update_SE3(									// call just one workgroup to sum the whole image maps from the patch kernel.
){
	const uint	SE3_DoF		= 6;
	uint   global_id_u 		= get_global_id(0);
	float  global_id_f 		= global_id_u;
	uint   lid 				= get_local_id(0);
																								// read in global data : Rho, weights, SE3_incr
																								// NB 10x8 pactch for each SE3.
																								// Read & sum pixels in column, NB img overlap pixel count
	uint	SE3				= global_id_u / thread_offset;
	bool	in_range		= fmod( global_id_f, thread_offset) < cols  &&  (SE3 < SE3_DoF);	// NB integer division.

	if (in_range){
		row_offset 			*=SE3;

		for(uint idx = 0; idx<rows; idx ++){
			uint idx_2		= idx * mm_cols;
			pvt_rho			+= Rho_[idx_2];														// NB only one Rho[], but 6 DoF for weights_map[] & SE3_incr_map_[]
			idx_2 			+= row_offset;
			pvt_weights		+= weights_map[idx_2];
			pvt_incr		+= SE3_incr_map_[idx_2];											// pvt variable will hold sum for column in Rho_, weights_map, SE3_incr_map_  // TODO should these be combined BEFORE bering summed ?
		}
	}
}
