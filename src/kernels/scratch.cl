
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
