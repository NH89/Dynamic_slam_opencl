#include "kernels__macros.h"
#include "kernels.h"

__kernel void Rho_sq_depth(								// To be launched with 1 thread per col for 32x32 patches, and an integer multiple of 32 threads.
													// Needs 16 elements of local mem per 32x32 patch, to pass data between threads in recursive square reduction.
													// Needs 32 elem array of private mem per thread.
	__private	uint		cols_per_row,			//1

	///// from patch_img_grad(..)
	//Inputs:
	__private	uint		layer,					//0
	__private	uint		lookup_table_offset,	//1
	__private	uint		out_block_size,			//2

	__private	uint3		SE3_offset3,			//3
	__private	uint3		ST3_offset3,			//4

	__global 	float4*		lookup_table,			//8		// should ideally be a constant.
	///// end patch_img_grad(..)


	__constant	uint8*		mipmap_params,			//3
	__constant	uint*		uint_params,			//4
	__constant	float*		fp32_params,			//5
	__constant	float16*	inv_k2k,				//6		// transforms for 4 past frames,  k2k_buf

	__global	float4*		img_cur,				//7		// multiple past frames. NB retain frames at powers of 2, and vary starting power plus num franes.
	__global	float4*		img_past_0,				//8
	__global	float4*		img_past_1,				//9
	__global	float4*		img_past_2,				//10
	__global	float4*		img_past_3,				//11

	__global	float*		depth_map,				//12	// current frame depth, now stored as inv_depth
	__global	float8*		g1p,					//13	// current frame g1mem
	__global 	float4*		SE3_grad_map_cur_frame,	//14

	__global	float4*		vel_cur,				//15	// multiple past frames.
	__global	float4*		vel_past_0,				//16	// TO DO, relative velocity not used yet. Will use it to modify depth map with timestep for past frames.
	__global	float4*		vel_past_1,				//17
	__global	float4*		vel_past_2,				//18
	__global	float4*		vel_past_3,				//19

	//output
	__global	float2*		Rho_,					//20	// { sum rho^2 ,  count of valid pixels used } Writen to dense patches.
	__local		float2*		local_rho,				//21	// float2 local_rho[ local_work_size/2 ]  hence sizeof( float)*local_work_size.

	__global	float2*		SE3_incr_map_,			//22
	__local		float2*		local_SE3_incr			//23
	)
{
	////// from patch_img_grad(..)
	uint	global_id_uint								= get_global_id(0);
	uint	lid											= get_local_id(0);
	uint	group_id									= get_group_id(0);
	const	uint local_size								= get_local_size(0);

	float4	lookup_ref									= lookup_table[global_id_uint + lookup_table_offset];
	uint	read_index									= floor(lookup_ref.z);
	uint	u											= lookup_ref.x;														// read_column
	uint	v											= lookup_ref.y;														// read_row

	float4	lookup_ref_layer							= lookup_table[lookup_table_offset];	//0;
	uint	layer_offset								= floor(lookup_ref_layer.z);

	uint8	mipmap_params_ 								= mipmap_params[layer];
	uint	read_cols_									= mipmap_params_[MiM_READ_COLS];
	uint	read_rows_									= mipmap_params_[MiM_READ_ROWS];

	uint	mm_cols										= uint_params[MM_COLS];
	uint	mm_rows										= uint_params[MM_ROWS];
	uint	mm_pixels									= uint_params[MM_PIXELS];

	uint	stop_offset									= layer_offset + (read_rows_ -1) * mm_cols + read_cols_	;	// bottom right corner of source image layer

														//layer_offset/mm_cols;
	const uint ST3_offset		= ST3_offset3.s0;		//SE3_out_step_1 * (num_SE3_DoF + 1);// + layer_offset;
	const uint ST3_u_step		= ST3_offset3.s1;		// step between elements of the Hessian matrix
	const uint ST3_v_step		= ST3_offset3.s2;

	////// end patch_img_grad(..) ///////////////////
/*
	//const uint block_size							= 32;										// or send as __private arg ? BUT as hardcoded "const uint" it can be used to size arrays etc.
	//const uint num_SE3_DoF						= 6;
	//const uint num_past_frames					= 4;										// 1,2,4,8,16,32,64 // variable select window of 4 frames.
// 	const float4 zero_f4							= {0.0f,0.0f,0.0f,0.0f};
// 	const float2 zero_f2							= {0.0f,0.0f};

// 	uint  global_id_uint 							= get_global_id(0);
// 	uint  lid 										= get_local_id(0);
// 	uint  group_id									= get_group_id(0);
// 	const uint local_size 							= get_local_size(0);

// 	if (global_id_u < 1 / *num_past_frames* /){printf("\n__kernel void Rho_sq()  past_frame_num= %u,  invk2k buf = \n(%f,	%f,	%f,	%f),	\n(%f,	%f,	%f,	%f),	\n(%f,	%f,	%f,	%f),	\n(%f,	%f,	%f,	%f),	   ",\
// 		lid, inv_k2k[lid][0], inv_k2k[lid][1], inv_k2k[lid][2], inv_k2k[lid][3], 	inv_k2k[lid][4], inv_k2k[lid][5], inv_k2k[lid][6], inv_k2k[lid][7], 	inv_k2k[lid][8], inv_k2k[lid][9], inv_k2k[lid][10], inv_k2k[lid][11], 	inv_k2k[lid][12], inv_k2k[lid][13], inv_k2k[lid][14], inv_k2k[lid][15] );
// 	}

	//const uint8 mipmap_params_					= mipmap_params[layer];
	//uint read_cols_ 								= mipmap_params_[MiM_READ_COLS];
	//uint read_rows_ 								= mipmap_params_[MiM_READ_ROWS];
	//uint mm_cols									= uint_params[MM_COLS];
	//uint mm_pixels								= uint_params[MM_PIXELS];
	//uint read_index								= read_offset_ + row_col + block_row*block_size*mm_cols;
*/
	__global float4*	img_past[num_past_frames]	= { img_past_0, img_past_1, img_past_2, img_past_3 };
	__global float4*	vel_past[num_past_frames]	= { vel_past_0, vel_past_1, vel_past_2, vel_past_3 };

	uint read_offset_ 								= mipmap_params_[MiM_READ_OFFSET];
	uint layer_pixels								= mipmap_params_[MiM_PIXELS];
	uint base_cols									= uint_params[COLS];
	float min_inv_depth								= fp32_params[MIN_INV_DEPTH];
	float max_inv_depth								= fp32_params[MAX_INV_DEPTH];

	float reduction									= base_cols/read_cols_;
	uint row_length									= cols_per_row;								// blocks_cols * block_size;
	uint row_col									= fmod((float)global_id_uint, row_length);
	uint block_row									= global_id_uint / row_length;
	uint row_offset									= read_offset_/mm_cols;

	uint write_spacing								= block_size/out_block_size;
	uint write_index								= row_col/out_block_size 	+ block_row*write_spacing*mm_cols;
	uint write_index_2								= row_col/block_size 		+ block_row*mm_cols;

	float2 rho_pvt_arr[block_size]					= {zero_f2};								// pvt variable for values in this column.
	float4 rho_pvt_flt4								= zero_f4;
	float2 rho_pvt_flt2								= zero_f2;

	float2 SE3_incr_pvt_arr[block_size*num_ST3_DoF]	= {zero_f2};								// pvt variable for values in this column.
	float4 SE3_incr_pvt_flt4						= zero_f4;
	float2 SE3_incr_pvt_flt2						= zero_f2;

	float4 img_cur_pvt[block_size]					= {zero_f4};								// pvt variable for values in this column.
	float8 g1p_pvt[block_size]						= {zero_f8};
	float4 old_px									= zero_f4;;

	bool   intersection								= false;
	bool   print_ 									= false;
																																// 	if (global_id_u < 1 /*num_past_frames*/){printf("\n__kernel void Rho_sq()  layer = %d,  read_index=%d,  read_index/mm_cols=%f ",
																																// 																				layer,		read_index,  	(float)read_index/(float)mm_cols	);	}
	local_rho[lid]									= zero_f2;
	for (uint st3_dim=0; st3_dim<num_ST3_DoF; st3_dim++) {
		local_SE3_incr[lid + st3_dim*local_size]	= zero_f2;
	}
																								// PATCH KERNEL //  TO DO need to transfer computation of Huber Norm weighting, Jacobian and Hessian here,
																								// because Hessian must include weights and therefore be updated if weights change.
	////////////////////////////////////////////////////////////////////////////				// transfer data from global memory.
	for (uint past_frame_idx=0; past_frame_idx</*num_past_frames*/1; past_frame_idx++){			// step though past frames ///////////////////////////////////////////////////////////////////////////////

		////// from patch_img_grad(..)
		int		lfoff										= -(u >1);															//-(read_column != 0);
		int		rtoff										=  (u < read_cols_-2);												// (read_column < mm_cols-1);

		float4	Jacobian_pvt_arr[block_size][6]				= {{zero_f4}};
		float4	Hessian_pvt_arr[block_size][6][6]			= {{{zero_f4}}};													// pvt variable for values in this column.

		uint	write_index									= u/out_block_size			 + (v/out_block_size)*mm_cols	+ ST3_offset;
		uint 	offset_1_1_max								= mm_cols * read_rows_ / out_block_size 		 			+ ST3_offset;
		uint 	read_index_row								= read_index;

		for (uint row_in_block=0; (row_in_block<block_size)&&(read_index<=stop_offset&&read_index>0); row_in_block++, v++,  read_index_row +=mm_cols){	// stop offset prevents bottom row patches from overrunning the bottom of the image layer. // NB readindex may be 0 if not in range according to lookup table.
		////// end patch_img_grad(..)


		//for (uint row_in_block=0; row_in_block<block_size; row_in_block ++){					// step through rows of the patch, /////////////////////////////////////////////////////////////
			float						u2f, 	v2f;									// current frame

			img_cur_pvt[row_in_block]	= img_cur[		read_index_row];
			g1p_pvt[row_in_block]		= g1p[			read_index_row];
			float inv_depth 			= depth_map[	read_index_row];

			// Where to sample the past image frame //////
			px_k2k( inv_k2k[past_frame_idx],  reduction,  v,  u,  inv_depth, &u2f,  &v2f, print_ );		//

			const uint margin			= 4;
			intersection 				= 	(u>margin)			&& (u<=read_cols_-margin)			&& (v>margin)			&& (v<=read_rows_-margin)			&& \
											(u2f>margin)	&& (u2f<=read_cols_-margin)	&& (v2f>margin)	&& (v2f<=read_rows_-margin)	&& \
											(global_id_uint<=layer_pixels)		&&	(inv_depth>=min_inv_depth)	&& (inv_depth<=max_inv_depth);												// if images overlap
			rho_pvt_flt4				= zero_f4;
			//////////////////////////////////////////////////
			if (intersection){
				// Photometric error rho ///////
				old_px					= bilinear_flt4( img_past[past_frame_idx],  u2f,  v2f,  mm_cols,  read_offset_ )	;
				rho_pvt_flt4			= (img_cur_pvt[row_in_block] - old_px) ;
				rho_pvt_flt4.w			= 1.0f;																																					// rho.w holds pixel count.

				// Gradient of pixel value wrt SE3 rotation & translation, taking account of current depth map //////
				SE3_incr_pvt_flt2.y											=  1;
				for (uint st3_dim=0; st3_dim<num_SE3_DoF; st3_dim++) {
					SE3_incr_pvt_flt4										= rho_pvt_flt4 		* 	SE3_grad_map_cur_frame[ read_index_row + (st3_dim * mm_pixels) ] ;
					SE3_incr_pvt_flt2.x										= SE3_incr_pvt_flt4.x;
					SE3_incr_pvt_arr[ st3_dim*block_size + row_in_block ]	= SE3_incr_pvt_flt2 ;
				}





			}
			barrier( CLK_GLOBAL_MEM_FENCE );

			rho_pvt_flt2.x					=  rho_pvt_flt4.x;																																	// Sum Rho
			rho_pvt_flt2.y					=  rho_pvt_flt4.x * rho_pvt_flt4.x;																													// Sum Rho_squared
			rho_pvt_arr[row_in_block]		+= rho_pvt_flt2;																																	// save to pvt mem for this column
		}
	}

/*
				// Given st3 for this pair of frames, and rel_vel & rel_accel for this pixel,   find J(value / inv_depth) for this pixel.
				float t[num_ST3_DoF];
				float t[0]	= inv_k2k[past_frame_idx].s3;
				float t[1]	= inv_k2k[past_frame_idx].s7;
				float t[2]	= inv_k2k[past_frame_idx].s11;

				float J_inv_d = 0;

				for (uint st3_dim=0; st3_dim<num_SE3_DoF; st3_dim++) {
					J_inv_d = t[st3_dim]	*	SE3_incr_pvt_arr[ st3_dim*block_size + row_in_block ].x;
				}

*/




	// Sum-reduce image, /////////////  Save intermediate size ST3 patches for depth map updates, and maximally reduced SE3 patches for pose updates. Second reduce_patch_Rho(..) kernel required for SE3 from lareger image pyramid layers, before update_k2k(..) kernel.
	uint past_frame_idx =0; // TO DO remove and restore long outer loop.
	uint step;
	for ( step=1; step<block_size; step *=2){																																					// for each step size, (multiples of 2)
		for (uint block_row=0; block_row<block_size ; block_row += step){																														// step through rows in column
																						rho_pvt_arr[		block_row ]							+=rho_pvt_arr[		block_row + step ];			// sum pair of values in col,
			for (uint se3_dim=0; se3_dim<num_SE3_DoF; se3_dim++) {
																						SE3_incr_pvt_arr[	block_row + se3_dim*block_size ]	+=SE3_incr_pvt_arr[	block_row + step + se3_dim*block_size ];
			}

			if( !(fmod((float)lid,(step*2))==0) &&  (fmod((float)lid,step)==0)    ){																											// selects 2nd column, sends data
																						local_rho[			lid-step ]							= rho_pvt_arr[		block_row];
				for (uint se3_dim=0; se3_dim<num_SE3_DoF; se3_dim++) {																																// NB integer division. Hence both threads use the same index to local memory.
																						local_SE3_incr[		lid-step + se3_dim*local_size ]		= SE3_incr_pvt_arr[	block_row + se3_dim*block_size ];
				}
			}
			barrier(CLK_LOCAL_MEM_FENCE );																																						// Using barrier as a semaphore, for local mem messages between threads. This minimizes local_mem req, while allowing 2 patch sizes in output, full & ST3 map at out_block_size.

			if( (fmod((float)lid,(step*2))==0)  ){																																				// selects 1st column, adds data. Sum of patch now held in top left element of patch.
																						rho_pvt_arr[		block_row] 							+= local_rho[		lid ];
				for (uint se3_dim=0; se3_dim<num_SE3_DoF; se3_dim++) {
																						SE3_incr_pvt_arr[	block_row + se3_dim*block_size ]	+= local_SE3_incr[	lid		 + se3_dim*local_size ];
				}
			}
			barrier(CLK_LOCAL_MEM_FENCE );
		}
		// Save intermediate size ST3 patches for depth map updates, //////////
		if (step==out_block_size/2){																																							// save ST3 map at out_block_size, to use for updating depth_map and rel_vel_map
			uint frame_offset 		= write_index + past_frame_idx * 100 + 25 ;			// NB 100 works for current img size . // stacks frame ST3 maps in adjacent columns..
			uint write_block_row	= 0;
			if( fmod((float)lid,out_block_size) == 0 ){																																			// selects columns i.e. threads within the workgroup
				for (uint block_row=0; block_row < block_size ; block_row += step*2, write_block_row++){
																						uint offset_1 				= frame_offset		+ write_block_row*mm_cols;
																						Rho_[			offset_1]	= rho_pvt_arr[		block_row ];
/*
// 					if(block_row==10 && group_id==0 ){printf("\n__kernel void Rho_sq_2, global_id_u=%u,	block_row=%u,	group_id=%u,		rho_pvt_arr[ block_row ]=(%f, %f ) ", \
// 					global_id_u, block_row, group_id,	rho_pvt_arr[block_row].x, rho_pvt_arr[block_row].y ); }
*/
					for (uint se3_dim=3; se3_dim<num_SE3_DoF; se3_dim++) {																															// select only ST3
																						uint offset_2 				= offset_1			+ (se3_dim-3)*( 4+ (read_rows_/out_block_size) )*mm_cols;
																						uint offset_3 				= block_row			+ se3_dim*block_size;									// NB read_rows_/out_block_size = writre_rows
																						SE3_incr_map_[	offset_2 ]	= SE3_incr_pvt_arr[	offset_3 ];
					}
				}
			}
		}//////////////////////////////////////////////////////////////////////
	}

}
