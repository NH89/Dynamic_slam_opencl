#include "kernels_macros.h"
#include "kernels.h"

/* NB GPU limits
 * For Intel iRIS Xe
// Max number of constant args                     8
// Max constant buffer size                        4294959104 (4GiB)
NB shoud use these for things that never change during runtime, not for variables constant in a particular kernel but not another.
TODO Declare constants at top of the device prgram file.
*/

__kernel void compute_param_maps(
	__private	uint	layer,			//0
	__constant 	uint8*	mipmap_params,	//1
	__constant 	uint*	uint_params,	//2
	__constant 	float* 	SO3_k2k,		//3
	__global 	float2*	SE3_map			//4
		 )
{
	uint global_id_u 	= get_global_id(0);
	float global_id_flt = global_id_u;
	uint8 mipmap_params_= mipmap_params[layer];
	uint read_offset_ 	= mipmap_params_[MiM_READ_OFFSET];
	uint read_cols_ 	= mipmap_params_[MiM_READ_COLS];
	if (global_id_u >= mipmap_params_[MiM_PIXELS]) return;

	uint lid 			= get_local_id(0);
	uint group_size 	= get_local_size(0);

	uint margin 		= uint_params[MARGIN];
	uint mm_cols		= uint_params[MM_COLS];
	uint reduction		= mm_cols/read_cols_;
	uint v    			= global_id_u / read_cols_;													// read_row
	uint u 				= fmod(global_id_flt, read_cols_);											// read_column
	float u_flt			= u * reduction;															// NB this causes sparse sampling of the original space, to use the same k2k at every scale.
	float v_flt			= v * reduction;
	uint read_index 	= read_offset_  +  v  * mm_cols  + u ;

	for (uint i=0; i<6; i++) {																		// for each SE3 DoF
																									// Find new pixel position, h=homogeneous coords.
		int idx = i *16;
		float inv_depth = 1.0f;																		// mid point max-min inv depth
		float uh2 = SO3_k2k[idx+0]*u_flt + SO3_k2k[idx+1]*v_flt + SO3_k2k[idx+2]*1 + SO3_k2k[idx+3]*inv_depth;
		float vh2 = SO3_k2k[idx+4]*u_flt + SO3_k2k[idx+5]*v_flt + SO3_k2k[idx+6]*1 + SO3_k2k[idx+7]*inv_depth;
		float wh2 = SO3_k2k[idx+8]*u_flt + SO3_k2k[idx+9]*v_flt + SO3_k2k[idx+10]*1+ SO3_k2k[idx+11]*inv_depth;
		//float h/z  = SO3_k2k[12]*u_flt + SO3_k2k[13]*v + SO3_k2k[14]*1; 							// +SO3_k2k[15]/z

		float u2   = uh2/wh2;
		float v2   = vh2/wh2;
		float2 partial_gradient={u_flt-u2 , v_flt-v2}; 												// Find movement of pixel

		SE3_map[read_index + i* uint_params[MM_PIXELS]  ] = partial_gradient;
	}

	// TODO // Create a 'reproject' & 'img_grad_sum' kernels
}

__kernel void Rho_sq(								// To be launched with 1 thread per col for 32x32 patches, and an integer multiple of 32 threads.
													// Needs 16 elements of local mem per 32x32 patch, to pass data between threads in recursive square reduction.
													// Needs 32 elem array of private mem per thread.
	__private	uint		layer,					//0
	__private	uint 		cols_per_row,			//1
	__private	uint 		out_block_size,			//2

	__constant	uint8*		mipmap_params,			//3
	__constant	uint*		uint_params,			//4
	__constant	float*		fp32_params,			//5
	__constant	float16*	inv_k2k,				//6		// transforms for 4 past frames

	__global	float4*		img_cur,				//7		// multiple past frames. NB retain frames at powers of 2, and vary starting power plus num franes.
	__global	float4*		img_past_0,				//8
	__global	float4*		img_past_1,				//9
	__global	float4*		img_past_2,				//10
	__global	float4*		img_past_3,				//11

	__global	float* 		depth_map,				//12	// current frame depth, now stored as inv_depth
	__global	float8* 	g1p,					//13	// current frame g1mem
	__global 	float8*		SE3_grad_map_cur_frame,	//14

	__global	float4*		vel_cur,				//15	// multiple past frames.
	__global	float4*		vel_past_0,				//16	// TODO, relative velocity not used yet. Will use it to modify depth map with timestep for past frames.
	__global	float4*		vel_past_1,				//17
	__global	float4*		vel_past_2,				//18
	__global	float4*		vel_past_3,				//19

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
	//float2 temp2	= {1.0f,1.0f};
																								// PATCH KERNEL //
	////////////////////////////////////////////////////////////////////////////				// transfer data from global memory.
	for (uint past_frame_idx=0; past_frame_idx</*num_past_frames*/1; past_frame_idx++){		// step though past frames ///////////////////////////////////////////////////////////////////////////////

		for (uint row_in_block=0; row_in_block<block_size; row_in_block +=2){						// step through pairs of rows of the patch, /////////////////////////////////////////////////////////////
			// current frame
			uint read_index_row 		= read_index + row_in_block * mm_cols;
			img_cur_pvt[row_in_block]	= img_cur[read_index_row];
			img_cur_pvt[row_in_block+1]	= img_cur[read_index_row + mm_cols];											// sum two source pixels elem from column.

			g1p_pvt[row_in_block]		= g1p[read_index_row];
			g1p_pvt[row_in_block+1]		= g1p[read_index_row + mm_cols];

			float inv_depth_1 			= depth_map[read_index_row];
			float inv_depth_2 			= depth_map[read_index_row + mm_cols];

			float	u2_flt_1, 	v2_flt_1, 	u2_flt_2, 	v2_flt_2;
			uint index 					= read_index_row - read_offset_;
			uint v 						= index / mm_cols;
			uint u 						= fmod((float)index, mm_cols);
			float u_flt					= (float)u * reduction;																// NB this causes sparse sampling of the original space, to use the same k2k at every scale.
			float v_flt_1				= (float)v * reduction;
			float v_flt_2				= ((float)v+1.0f) * reduction;

			//if(global_id_u==0)printf("\n__kernel void Rho_sq() v=%u, v_flt_1=%f, v_flt_2=%f, read_index_row=%u, row_offset=%u,  read_rows_=%u, (read_index_row/mm_cols)-row_offset=%u ", v, v_flt_1, v_flt_2, read_index_row, row_offset, read_rows_, (read_index_row/mm_cols)-row_offset);

																													// TODO, relative velocity not used yet. Will use it to modify depth map with timestep for past frames.

			float uh2_1 			= inv_k2k[past_frame_idx][0]*u_flt 												+ inv_k2k[past_frame_idx][ 2]*1		+ inv_k2k[past_frame_idx][ 3]*inv_depth_1;		// + inv_k2k[past_frame_idx][1]*v_flt
			float vh2_1 			= inv_k2k[past_frame_idx][4]*u_flt 												+ inv_k2k[past_frame_idx][ 6]*1		+ inv_k2k[past_frame_idx][ 7]*inv_depth_1;		// + inv_k2k[past_frame_idx][5]*v_flt
			float wh2_1				= inv_k2k[past_frame_idx][8]*u_flt 	+ inv_k2k[past_frame_idx][9]*v_flt_1 		+ inv_k2k[past_frame_idx][10]*1		+ inv_k2k[past_frame_idx][11]*inv_depth_1;		//

			float uh2_2 			= inv_k2k[past_frame_idx][0]*u_flt 												+ inv_k2k[past_frame_idx][ 2]*1		+ inv_k2k[past_frame_idx][ 3]*inv_depth_2;		// + inv_k2k[past_frame_idx][1]*v_flt
			float vh2_2 			= inv_k2k[past_frame_idx][4]*u_flt 												+ inv_k2k[past_frame_idx][ 6]*1		+ inv_k2k[past_frame_idx][ 7]*inv_depth_2;		// + inv_k2k[past_frame_idx][5]*v_flt
			float wh2_2				= inv_k2k[past_frame_idx][8]*u_flt 	+ inv_k2k[past_frame_idx][9]*v_flt_2 		+ inv_k2k[past_frame_idx][10]*1		+ inv_k2k[past_frame_idx][11]*inv_depth_2;		//
			//float h/z  			= inv_k2k[past_frame_idx][12]*u_flt	+ inv_k2k[past_frame_idx][13]*v_flt 		+ inv_k2k[past_frame_idx][14]*1; 													// + inv_k2k[past_frame_idx][15]/z

			u2_flt_1				= (uh2_1 + inv_k2k[past_frame_idx][1]*v_flt_1 ) / ((wh2_1  )*reduction);
			v2_flt_1				= (vh2_1 + inv_k2k[past_frame_idx][5]*v_flt_1 ) / ((wh2_1  )*reduction);

			u2_flt_2				= (uh2_2 + inv_k2k[past_frame_idx][1]*v_flt_2 ) / ((wh2_2  )*reduction);
			v2_flt_2				= (vh2_2 + inv_k2k[past_frame_idx][5]*v_flt_2 ) / ((wh2_2  )*reduction);

			int  u2_1				= floor(u2_flt_1 + 0.5f) ;														// nearest neighbour interpolation, used by "bool intersection" to determine if images overlap at each pixel.
			int  v2_1				= floor(v2_flt_1 + 0.5f) ;														// NB this corrects the sparse sampling to the redued scales.
			int  u2_2				= floor(u2_flt_2 + 0.5f) ;
			int  v2_2				= floor(v2_flt_2 + 0.5f) ;
			//////////////////////////////////////////////////
			rho_pvt_flt4			= zero_f4;
			intersection 			= (u>2) && (u<=read_cols_-2) && (v>2) && (v<=read_rows_-2) && (u2_1>2) && (u2_1<=read_cols_-2)  \
									&& (v2_1>2) && (v2_1<=read_rows_-2)  &&  (global_id_u<=layer_pixels) && (inv_depth_1>=min_inv_depth) && (inv_depth_1<=max_inv_depth);						// if images overlap
			if (intersection){
				rho_pvt_flt4		= img_cur_pvt[row_in_block]		 -	bilinear_flt4( img_past[past_frame_idx],  u2_flt_1,  v2_flt_1,  mm_cols,  read_offset_ )	;							// find 1st row pixel rho	// img_past[past_frame_idx][read_index_row]
				//rho_sq_pvt_flt4		*= rho_pvt_flt4 * (1.0f - g1p[read_index].s3);																											// rho_sq weighted by img edges.

				float multiplier = fp32_params[MAX_INV_DEPTH]/((inv_depth_1 + 0.01) *5); 	// NB beware if inf depth, i.e. subnormal inv_depth, then div by zero error !						// de-weight foreground for rotation, & de-weight backgroud for translation.
				for (uint se3_dim=0; se3_dim<se3_dof; se3_dim++) {																																	// for each SE3 DoF
					float8 grad_v8 											= SE3_grad_map_cur_frame[ read_index_row + (se3_dim * mm_pixels) ] ;												// TODO change SE3_grad_map_cur_frame[] to float4 or less.
					weights_pvt_flt4										= grad_v8.hi + grad_v8.lo;																							// i.e. sum the u and v components of the gradient, for the four colour channels.
					SE3_incr_pvt_flt4										= weights_pvt_flt4 * rho_pvt_flt4;																					// TODO collapse to single float

					if (se3_dim>=3){multiplier 								= inv_depth_1/fp32_params[MAX_INV_DEPTH] ;}			// For ST3 only to emphasize foreground pixels for parallax motion: multiply pixel inv_depth by min depth in scene.  Office scene depth is in cm from approx 90 to 450cm.

					SE3_incr_pvt_flt2.x										= ( SE3_incr_pvt_flt4.x + SE3_incr_pvt_flt4.y + SE3_incr_pvt_flt4.z ) * multiplier   ;
					SE3_incr_pvt_flt2.y										= 1.0f;
					SE3_incr[ se3_dim*block_size + row_in_block ]			= SE3_incr_pvt_flt2;																								// pixelwise increment for this SE3 DoF

					weights_pvt_flt2.x 										= ( weights_pvt_flt4.x * weights_pvt_flt4.x + weights_pvt_flt4.y * weights_pvt_flt4.y + weights_pvt_flt4.z * weights_pvt_flt4.z );
					weights_pvt_flt2.y										= 1.0f;
					weights[ se3_dim*block_size + row_in_block ]			= weights_pvt_flt2;																									// save the SE3_grad^2, to use as divisor for this SE3 DoF, after summing.
				}
			}
			barrier(CLK_GLOBAL_MEM_FENCE );

			rho_pvt_flt2.x			= rho_pvt_flt4.x*rho_pvt_flt4.x  + rho_pvt_flt4.y*rho_pvt_flt4.y  +rho_pvt_flt4.z*rho_pvt_flt4.z;															// sum rho^2
			rho_pvt_flt2.x			*= (1.0f - g1p[read_index].s3);																																// Weight rho by edges. // TODO choose/ refine which edges to use.
			rho_pvt_flt2.y			= 1.0f;
			rho[row_in_block]		+= rho_pvt_flt2;																																			// save to pvt mem for this column

			//////////////////////////////////////////////////
			rho_pvt_flt4			= zero_f4;
			intersection 			= (u>2) && (u<=read_cols_-2) && (v>2) && (v<=read_rows_-2) && (u2_2>2) && (u2_2<=read_cols_-2) \
									&& (v2_2>2) && (v2_2<=read_rows_-2)  &&  (global_id_u<=layer_pixels) && (inv_depth_2>=min_inv_depth) && (inv_depth_2<=max_inv_depth);						// if images overlap
			if (intersection){
				rho_pvt_flt4		= img_cur_pvt[row_in_block+1]	-	bilinear_flt4( img_past[past_frame_idx], u2_flt_2, v2_flt_2,  mm_cols, read_offset_ );									// find 2nd row pixel rho

				float multiplier = fp32_params[MAX_INV_DEPTH]/((inv_depth_1 + 0.01) *5); 	// NB beware if inf depth, i.e. subnormal inv_depth, then div by zero error !						// de-weight foreground for rotation, & de-weight backgroud for translation.
				for (uint se3_dim=0; se3_dim<se3_dof; se3_dim++) {																																// for each SE3 DoF
					float8 grad_v8 											= SE3_grad_map_cur_frame[ read_index_row + mm_cols + (se3_dim * mm_pixels) ] ;										// TODO change SE3_grad_map_cur_frame[] to float4 or less.
					weights_pvt_flt4										= grad_v8.hi + grad_v8.lo;																							// i.e. sum the u and v components of the gradient, for the four colour channels.
					SE3_incr_pvt_flt4										= weights_pvt_flt4 * rho_pvt_flt4;																					// TODO collapse to single float

					if (se3_dim>=3){multiplier 								= inv_depth_1/fp32_params[MAX_INV_DEPTH] ;}			// For ST3 only to emphasize foreground pixels for parallax motion: multiply pixel inv_depth by min depth in scene.  Office scene depth is in cm from approx 90 to 450cm.

					SE3_incr_pvt_flt2.x										= ( SE3_incr_pvt_flt4.x + SE3_incr_pvt_flt4.y + SE3_incr_pvt_flt4.z ) * multiplier   ;
					SE3_incr_pvt_flt2.y										= 1.0f;
					SE3_incr[ se3_dim*block_size + row_in_block ]			= SE3_incr_pvt_flt2;																								// pixelwise increment for this SE3 DoF

					weights_pvt_flt2.x 										= ( weights_pvt_flt4.x * weights_pvt_flt4.x + weights_pvt_flt4.y * weights_pvt_flt4.y + weights_pvt_flt4.z * weights_pvt_flt4.z );
					weights_pvt_flt2.y										= 1.0f;
					weights[ se3_dim*block_size + row_in_block ]			= weights_pvt_flt2;																									// save the SE3_grad^2, to use as divisor for this SE3 DoF, after summing.
				}
			}
			barrier(CLK_GLOBAL_MEM_FENCE );
			rho_pvt_flt2.x			= rho_pvt_flt4.x*rho_pvt_flt4.x  + rho_pvt_flt4.y*rho_pvt_flt4.y  +rho_pvt_flt4.z*rho_pvt_flt4.z;															// sum rho^2
			rho_pvt_flt2.x			*= (1.0f - g1p[read_index+mm_cols].s3);
			rho_pvt_flt2.y			= 1.0f;
 			rho[row_in_block+1]		+= rho_pvt_flt2;																																			// accumulate rho for this patch
		}
	}
	uint past_frame_idx =0; // TODO remove and restore long outer loop.
	uint step;
	for ( step=2; step<block_size; step *=2){				// out_block_size																													// for each step size, (multiples of 2)
// 		if (global_id_u==0){printf("\n__kernel void Rho_sq() chk 2,  step=%u  ############",step);}//#############################################
																																																// selects pairs of columns to sum  i.e. threads within the workgroup
		for (uint block_row=0; block_row<block_size ; block_row += step){																														// step through rows in column
// 			if (global_id_u==0){printf("\n__kernel void Rho_sq() chk 3,  block_row=%u",block_row);}//#############################################

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
		////////////////////////////////
		if (step==out_block_size){																																								// save ST3 map at out_block_size, to use for updating depth_map and rel_vel_map
			uint frame_offset 		= write_index + past_frame_idx * 100 + 25 ; // NB 100 works for current img size . // stacks frame ST3 maps in adjacent collumns..
			uint write_block_row	= 0;
			if( fmod((float)lid,out_block_size) == 0 ){																																			// selects columns i.e. threads within the workgroup

				for (uint block_row=0; block_row < block_size ; block_row += step, write_block_row++){
																						uint offset_1 				= frame_offset  + write_block_row*mm_cols;
																						Rho_[			offset_1]	= rho[		 block_row ];
					for (uint se3_dim=3; se3_dim<se3_dof; se3_dim++) {																															// select only ST3
																						uint offset_2 				= offset_1 		+ (se3_dim-3)*( 4+ (read_rows_/out_block_size) )*mm_cols;
																						uint offset_3 				= block_row 	+ se3_dim*block_size;										// NB read_rows_/out_block_size = writre_rows
																						SE3_incr_map_[	offset_2 ] 	= SE3_incr[  offset_3 ];
																						weights_map[	offset_2 ] 	= weights[   offset_3 ];
					}
				}	// weights_map[	frame_offset  + write_block_row*mm_cols + (se3_dim-3)*((4*block_size+read_rows_)/out_block_size)*mm_cols ] 	=  weights[   block_row + se3_dim*block_size ];
			}
		}
	}

// 	if (global_id_u==0){printf("\n__kernel void Rho_sq() chk 4");}									//#############################################

	/// Write ouptut to global mem.																																								// Writes dense blocks. Reduces required transfer to host. // TODO need kernel update depth map
	float2 temp2a											= { (float)/*block_row*/group_id, (float)/*block_col*/lid };
	if ( read_index < mm_pixels )	{	Rho_[read_index ] 	= temp2a;	}																														// Marks the area where the img buf is read, lines show top row of each patch.
																																																// Breaks show bondaries of patches.
	barrier(CLK_GLOBAL_MEM_FENCE );
	uint write_block_row=0;
	if( fmod((float)lid,block_size) ==0 ){		//out_block_size																																// selects columns i.e. threads within the workgroup

		uint frame_offset_1 = write_index_2;	//past_frame_idx * 75 * mm_cols; 			// NB 75 works for current img size . 	// stacks frame SE3 results vertically.

		//for ( step=2; step<block_size; step *=2){}
		step = block_size/2;

		for (uint block_row=0; block_row < block_size ; block_row += step, write_block_row++){
																						printf("\n__kernel void Rho_sq()  group_id=%u,  lid=%u,  step=%u,  write_index=%u,  write_index_2=%u,  write_block_row=%u,  mm_cols=%u,  layer_pixels=%u,  mm_pixels=%u,  [write_index + write_block_row*mm_cols]=%u   block_row=%u,  se3_dim=0, block_size=%u,  SE3_incr[  block_row + se3_dim*block_size ]=%f, %f",\
																							group_id, lid, step, write_index, write_index_2, write_block_row, mm_cols, layer_pixels, mm_pixels, (write_index + write_block_row*mm_cols),   block_row,  block_size,  SE3_incr[block_row].x,  SE3_incr[block_row].y );

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

__kernel void update_SE3(									// call just one workgroup to sum the whole image maps from the patch kernel.

	__private	uint		cols,					//0
	__private	uint 		rows,					//1
	__private	uint 		row_offset,				//2		// index of 1st pixel of the 2nd patch, i.e. spacing between patches
	__private	uint		thread_offset,			//3		// smallest 2^n > rows * cols NB rows=3, cols=4, -> 12 ->16 for layer 1.  6*8=48 -> 64 for layer 0, where base image has 640*480 pixels. NB for larger images may need a patch approach to update_SE3, to kep each SE3 DoF within
	__private	uint		mm_cols,				//4
	__private	float		img_var,				//5
	__private	float2		delta_SE3,				//6

	__global	float2*		Rho_,					//7		// { sum rho^2 ,  count of valid pixels used } Writen to dense patches.
	__global	float2*		weights_map,			//8
	__global	float2*		SE3_incr_map_,			//9

	__local		float2*		local_Rho_,				//10	// used for sum-reduce. Need to be [groupsize/2], set in host fn.
	__local		float2*		local_weights_map,		//11
	__local		float2*		local_SE3_incr_map_,	//12

	// out
	__global	float*		pose_update,			//13	// 6_DoF
	__global	float*		distorsion_update,		//14
	__global	float*		old_result				//15	// 1 rho value
	)
{
	const uint	SE3_DoF		= 6;
	uint   global_id_u		= get_global_id(0);
	float  global_id_f		= global_id_u;
	uint   lid				= get_local_id(0);
	uint   local_group_size = get_local_size(0);
	uint   group_id			= get_group_id(0);
																								// read in global data : Rho, weights, SE3_incr
																								// NB 10x8 pactch for each SE3.
																								// Read & sum pixels in column, NB img overlap pixel count
	uint	SE3				= global_id_u / thread_offset;
	uint	read_col		= fmod(global_id_f, thread_offset);
	bool	in_range		= read_col < cols  &&  (SE3 < SE3_DoF);	// NB integer division.

	printf("\n__kernel void update_SE3()_0    global_id_u=,%u,   SE3=,%u,  read_col=,%u,  thread_offset=,%u,  lid=,%u,   local_group_size=,%u,   group_id=,%u,   in_range=,%u "\
											,   global_id_u,   SE3,  read_col,  thread_offset,  lid,   local_group_size,   group_id,   in_range  );

	float2 pvt_rho			= {0.0f,0.0f};
	float2 pvt_weights		= {0.0f,0.0f};
	float2 pvt_incr			= {0.0f,0.0f};

	if( lid < local_group_size/2){
		local_Rho_[lid]			= pvt_rho;
		local_weights_map[lid]	= pvt_weights;
		local_SE3_incr_map_[lid]= pvt_incr;
	}

	float  delta_SE3_[2]	= {delta_SE3.x, delta_SE3.y};

	if (in_range){																				//
		row_offset			*=(SE3 * mm_cols);

		for(uint idx = 0; idx<rows; idx ++){													// NB patch sum could be used for rows, BUT most iterations have too few rows to justify the ovehead.
			uint idx_2		= read_col + (idx * mm_cols);
			pvt_rho			+= Rho_[idx_2];														// NB only one Rho[], but 6 DoF for weights_map[] & SE3_incr_map_[]
			uint idx_3		= row_offset + idx_2;
			pvt_weights		+= weights_map[idx_3];
			pvt_incr		+= SE3_incr_map_[idx_3];											// pvt variable will hold sum for column in Rho_, weights_map, SE3_incr_map_  // TODO should these be combined BEFORE bering summed ?

//			printf("\n__kernel void update_SE3()_0  global_id_u=,%u,   SE3=,%u,   idx=,%u, idx_2=,%u,   Rho_[idx_2]={%f,%f},   weights_map[idx_2].x=,%f,   SE3_incr_map_[idx_2].x=,%f,   row_offset=,%u",global_id_u, SE3, idx, idx_2, Rho_[idx_2].x, Rho_[idx_2].y, weights_map[idx_2].x, SE3_incr_map_[idx_2].x, row_offset );
		}
	}
				if( in_range){ printf("\n__kernel void update_SE3()_0.2  global_id_u=,%u,   SE3=,%u,   pvt_rho={,%f,%f,},  lid=,%u,   local_group_size=,%u,   group_id=,%u,   rows=,%u   , read_col=,%u"\
																		,  global_id_u,  SE3,  pvt_rho.x,  pvt_rho.y,  lid,  local_group_size,  group_id,  rows,  read_col  ); }

	barrier(CLK_GLOBAL_MEM_FENCE);
																								// sum reduce  columns of each 10x8 patch (1ayer 1), 5x5 layer 2, 3x3 layer 3, 2x2 layer 4, 1x1 layer 5.
	uint max_iter								= ceil(log2((float)cols));
	uint step									= 2;
	uint old_step								= 1;
	float col									= global_id_u - SE3*thread_offset;

	bool mod_step, mod_step_1, mod_step_2;;
	for ( uint iter=0; iter<max_iter; iter++, step*=2 ){
		uint cols_2 = cols/step;
		mod_step								= fmod(col				, step)== 0;
		mod_step_1								= fmod((col-old_step)	, step)== 0;
		float upper_lid = ceil((float)lid/step);

		if( mod_step_1 && in_range  ){
			local_Rho_[				lid/step]	= pvt_rho;
			local_weights_map[		lid/step]	= pvt_weights;
			local_SE3_incr_map_[	lid/step]	= pvt_incr;
		}
		barrier(CLK_LOCAL_MEM_FENCE);

				if(/* mod_step_1 &&*/ in_range ){ printf("\n__kernel void update_SE3()_0.3  global_id_u=,%u,   SE3=,%u,   pvt_rho={,%f,%f,},  lid=,%u,  group_id=,%u,   read_col=,%u,   step=,%u,  old_step=,%u,  lid/step=,%u,  iter=,%u,  upper_lid=,%f,  cols=,%u,  cols_2=,%u,  col=,%f,  mod_step_1=,%u"\
																							,  global_id_u,    SE3,      pvt_rho.x, pvt_rho.y,  lid,    group_id,       read_col,       step,       old_step,     lid/step,      iter,      upper_lid,      cols,      cols_2,      col,      mod_step_1  ); }


		if( mod_step  /*&& in_range*/ && col+old_step<cols  ){															// && fmod( upper_lid, thread_offset)<(cols/2)
			pvt_rho								+= local_Rho_[				lid/step];
			pvt_weights							+= local_weights_map[		lid/step];
			pvt_incr							+= local_SE3_incr_map_[		lid/step];
		}
		barrier(CLK_LOCAL_MEM_FENCE);

				if( /*mod_step  &&*/ in_range /*&& col<cols*/ /*lid/step<cols_2*/){ printf("\n__kernel void update_SE3()_0.4  global_id_u=,%u,   SE3=,%u,   pvt_rho={,%f,%f,},  lid=,%u,  group_id=,%u,   read_col=,%u,   step=,%u,  lid/step=,%u,  iter=,%u,  upper_lid=,%f,  mod_step=,%u,  col<cols=%u,  col=%f"\
																													, global_id_u, SE3,  pvt_rho.x, pvt_rho.y,  lid,  group_id,  read_col,  step, lid/step,  iter,  upper_lid,  mod_step,  col<cols,  col  ); }

		old_step = step;
	}



	if (mod_step && in_range){
		pvt_rho.x								/= pvt_rho.y;																						// Divide by number of valid overlapping pixels in original image pair.
		pvt_weights.x							/= pvt_rho.y;
		pvt_incr.x								/= pvt_rho.y;																						// TODO reduce SE3_incr_map & weights_map to float1, to save data read/writes.
		local_Rho_[SE3]							= pvt_rho;																							// will hold sum_Rho for each [SE3]
	}
	barrier(CLK_LOCAL_MEM_FENCE);

				if (mod_step && in_range) { printf("\n__kernel void update_SE3()_0.5  global_id_u=,%u,   SE3=,%u,   pvt_rho={,%f,%f,}  local_Rho_[SE3]={%f,%f,%f,  %f,%f,%f}"\
																					,global_id_u, SE3,  pvt_rho.x, pvt_rho.y, local_Rho_[0].x,local_Rho_[1].x,local_Rho_[2].x,  local_Rho_[3].x,local_Rho_[4].x,local_Rho_[5].x ); }

	float old_pose_update	= -1;
	if (mod_step && in_range){
		old_pose_update							= pose_update[SE3];																					// will be zero if 1st iteration. }
	}
	barrier(CLK_GLOBAL_MEM_FENCE);

	if (mod_step && in_range){																														// compute updates. NB different for SO3 vs ST3, hence offset.
		uint offset 							= 3 * floor((float)SE3/3);																			// offset = 0 for SO3, 3 for ST3.
		float mag_S3							= fabs(local_Rho_[0+offset].x)  + fabs(local_Rho_[1+offset].x)  + fabs(local_Rho_[2+offset].x);		// float rho_ST3_mag  = fabs(local_Rho_[3].x) + fabs(local_Rho_[4].x) + fabs(local_Rho_[5].x);
																																					// result_[iter][SE3] = SE3_results[layer][SE3][channel]  / (SE3_weights[layer][SE3][channel] * runcl.img_stats[IMG_VAR+channel] )
		float result							= pvt_rho.x / ( pvt_weights.x  * img_var );
		float update;

		if ( old_pose_update==0.0f ){
			printf("\n__kernel void update_SE3()_1  global_id_u=,%u,  (old_pose_update==0)   old_pose_update=,%f,  SE3=,%u", global_id_u,  old_pose_update, SE3 );
			update								= result * delta_SE3_[ SE3/3 ] / mag_S3;															// NB integer division SE3/3 => 0=SO3, 1=ST3
		}else{
			printf("\n__kernel void update_SE3()_2  global_id_u=,%u,  (old_pose_update != 0.0f),   old_pose_update=,%f,  SE3=,%u", global_id_u,  old_pose_update, SE3 );
			float rho_S3_delta					=      local_Rho_[0+offset].x  -  old_result[0+offset] \
												   +   local_Rho_[1+offset].x  -  old_result[1+offset] \
												   +   local_Rho_[2+offset].x  -  old_result[2+offset] ;
			update								= result * rho_S3_delta / mag_S3;
		}
		old_result[SE3]							= local_Rho_[SE3].x;
		pose_update[SE3]						= clamp(    update, -delta_SE3_[ SE3/3 ], +delta_SE3_[ SE3/3 ] );
		printf("\n__kernel void update_SE3()_3  global_id_u=,%u,  pose_update[SE3]=,%f,  SE3=,%u, delta_SE3_[SE3/3]=,%f, cols=,%u, max_iter=,%u, pvt_rho={,%f,%f,}"\
												, global_id_u, pose_update[SE3], SE3, delta_SE3_[SE3/3], cols, max_iter, pvt_rho.x, pvt_rho.y );
		//distorsion_update[..]	=  ;
	}
	barrier(CLK_GLOBAL_MEM_FENCE);
}

__kernel void update_maps(  // ? integrate with patch kernel ?





	)
{
	// given global ST3 direction vector, fit depth map


	// given residual of local ST3 map, after depth update, fit rel_vel_map


}





__kernel void se3_Rho_sq(
	// inputs
	__private	uint	layer,					//0
	__private	uint	local_num_samples,		//1
	__private	uint	se3_sum_size,			//2

	__constant 	uint8*	mipmap_params,			//3
	__constant 	uint*	uint_params,			//4
	__constant  float*  fp32_params,			//5

	__global	float16*k2k,					//6		// keyframe2K[3]
	__global 	float4*	img_cur,				//7		// keyframe
	__global 	float4*	img_new,				//8
	__global	float* 	depth_map,				//9		// NB keyframe GT_depth, now stored as inv_depth
	__global	float8* g1p,					//10		// keyframe_g1mem

	// outputs
	__global	float4* Rho_,					//11
	__local		float4*	local_sum_rho_sq,		//12		// 1 DoF, float4 channels
	__global 	float4*	global_sum_rho_sq		//13
	)
 {																									// find gradient wrt SE3 find global sum for each of the 6 DoF
	uint  global_id_u 	= get_global_id(0);
	float global_id_flt = global_id_u;
	uint  lid 			= get_local_id(0);

	uint local_size 	= get_local_size(0);
	uint group_size 	= local_size;
	uint work_dim 		= get_work_dim();
	uint global_size	= get_global_size(0);


	uint8 mipmap_params_ = mipmap_params[layer];
	uint read_offset_ 	= mipmap_params_[MiM_READ_OFFSET];
	uint read_cols_ 	= mipmap_params_[MiM_READ_COLS];
	uint read_rows_ 	= mipmap_params_[MiM_READ_ROWS];
	uint layer_pixels	= mipmap_params_[MiM_PIXELS];

	uint base_cols		= uint_params[COLS];
	uint margin 		= uint_params[MARGIN];
	uint mm_cols		= uint_params[MM_COLS];
	uint mm_pixels		= uint_params[MM_PIXELS];

	//float SE3_LM_a		= fp32_params[SE3_LM_A];														// Optimisation parameters
	//float SE3_LM_b		= fp32_params[SE3_LM_B];
	float inv_d_step 	= fp32_params[INV_DEPTH_STEP];
	float min_inv_depth = fp32_params[MIN_INV_DEPTH] ; //+ inv_d_step;
	float max_inv_depth = fp32_params[MAX_INV_DEPTH] ; //- inv_d_step;


	uint reduction		= mm_cols/read_cols_;
	uint v    			= global_id_u / read_cols_;														// read_row
	uint u 				= fmod(global_id_flt, read_cols_);												// read_column
	float u_flt			= u * reduction;																// NB this causes sparse sampling of the original space, to use the same k2k at every scale.
	float v_flt			= v * reduction;
	uint read_index 	= read_offset_  +  v  * mm_cols  + u ;
	//float alpha			= img_cur[read_index].w;

	float inv_depth 	= depth_map[read_index ]; 				//1.0f;// mid point max-min inv depth	// Find new pixel position, h=homogeneous coords.//inv dept  //depth_index

	for (int sample=0; sample<local_num_samples; sample++){

		float16 k2k_pvt		= k2k[sample];															// NB we read  k2k[1] and  k2k[2]
		float uh2 			= k2k_pvt[0]*u_flt + k2k_pvt[1]*v_flt + k2k_pvt[2]*1 + k2k_pvt[3]*inv_depth;
		float vh2 			= k2k_pvt[4]*u_flt + k2k_pvt[5]*v_flt + k2k_pvt[6]*1 + k2k_pvt[7]*inv_depth;
		float wh2 			= k2k_pvt[8]*u_flt + k2k_pvt[9]*v_flt + k2k_pvt[10]*1+ k2k_pvt[11]*inv_depth;
		//float h/z  		= k2k_pvt[12]*u_flt + k2k_pvt[13]*v + k2k_pvt[14]*1; // +k2k_pvt[15]/z

		float u2_flt		= uh2/(wh2*reduction);
		float v2_flt		= vh2/(wh2*reduction);
		int  u2				= floor(u2_flt + 0.5f) ;													// nearest neighbour interpolation
		int  v2				= floor(v2_flt + 0.5f) ;													// NB this corrects the sparse sampling to the redued scales.
		//read_index_new[sample] = read_offset_ + v2 * mm_cols  + u2; // read_cols_
		{
																	VK_TRACKING(\
																	if(global_id_u == 1  ){\
																		printf("\n\n\n\n__kernel void se3_Rho_sq chk 0  (global_id_u == 1 ) : sample=%u,  local_num_samples=%u, local_size=%u,  reduction=%u,  layer=%u, read_offset_=%u, read_cols_=%u, read_rows_=%u, layer_pixels=%u, read_index=%u,  \nk2k_pvt=[\n%f,%f,%f,%f,   \n%f,%f,%f,%f,    \n%f,%f,%f,%f,   \n%f,%f,%f,%f   ]\n", \
																		sample, local_num_samples, local_size, reduction, layer, read_offset_, read_cols_, read_rows_, layer_pixels, read_index,   k2k_pvt[0],k2k_pvt[1],k2k_pvt[2],k2k_pvt[3],  k2k_pvt[4],k2k_pvt[5],k2k_pvt[6],k2k_pvt[7],    k2k_pvt[8],k2k_pvt[9],k2k_pvt[10],k2k_pvt[11],   k2k_pvt[12],k2k_pvt[13],k2k_pvt[14],k2k_pvt[15]  );\
																	}\
																	if((u==read_cols_-1) && (v== read_rows_-1 )){\
																		printf("\n\n__kernel void se3_Rho_sq chk 1  (u==read_cols_-1) && (v== read_rows_-1 ) :  local_size=%u,  reduction=%u,  layer=%u, read_offset_=%u, read_cols_=%u, read_rows_=%u, layer_pixels=%u, read_index=%u, ", \
																		local_size, reduction, layer, read_offset_, read_cols_, read_rows_, layer_pixels, read_index   );\
																	}\
																	)
		}
		uint num_DoFs 		= 6;
		float4 new_px;

		int sample_lid 				= lid + sample * local_size;
		local_sum_rho_sq[sample_lid] = 0;																// Essential to zero local mem.
	 /*{
			VK_TRACKING(\
			if ( u==5 && v==5  ){ \
				printf("\n__kernel se3_Rho_sq(..)_0: sample=%u,  layer=%i,  global_id_u=%i,  u=%i,  v=%i,   inv_depth=%f, u2=%f,  v2=%f,  u2_flt=%f,  v2_flt=%f,  u2=%i,  v2=%i,    k2k_pvt=(%f,%f,%f,%f    ,%f,%f,%f,%f    ,%f,%f,%f,%f    ,%f,%f,%f,%f)"\
				,sample,  layer, global_id_u, u, v, inv_depth, u_flt, v_flt, u2_flt, v2_flt, u2, v2,      k2k_pvt[0],k2k_pvt[1],k2k_pvt[2],k2k_pvt[3],    k2k_pvt[4],k2k_pvt[5],k2k_pvt[6],k2k_pvt[7],    k2k_pvt[8],k2k_pvt[9],k2k_pvt[10],k2k_pvt[11],    k2k_pvt[12],k2k_pvt[13],k2k_pvt[14],k2k_pvt[15]   )  ;\
			}\
			)
	}*/
		float4 rho 											= {0.0f,0.0f,0.0f,0.0f};
																										// Exclude all out-of-bounds threads:
		//bool intersection = (u>2) && (u<=read_cols_-2) && (v>2) && (v<=read_rows_-2) && (u2>2) && (u2<=read_cols_-2) && (v2>2) && (v2<=read_rows_-2)  &&  (global_id_u<=layer_pixels);
		bool intersection = (u>2) && (u<=read_cols_-2) && (v>2) && (v<=read_rows_-2) && (u2>2) && (u2<=read_cols_-2) && (v2>2) \
		&& (v2<=read_rows_-2)  &&  (global_id_u<=layer_pixels) && (inv_depth>=min_inv_depth) && (inv_depth<=max_inv_depth);

		{
			VK_TRACKING(\
			if( (u==10) && (v==10) ){\
				printf("\n\n__kernel void se3_Rho_sq chk 2   (u==10) && (v==10) :  intersection = %i,  (u>2) = %i,  (u<=read_cols_-2) = %i,  (v>2) = %i,  (v<=read_rows_-2) = %i,  (u2>2) = %i,  (u2<=read_cols_-2) = %i,  (v2>2) = %i,  (v2<=read_rows_-2) = %i,  (global_id_u<=layer_pixels) = %i,  (inv_depth>=min_inv_depth) = %i,  (inv_depth<=max_inv_depth) = %i", \
					intersection,  (u>2) , (u<=read_cols_-2) , (v>2) , (v<=read_rows_-2) , (u2>2) , (u2<=read_cols_-2) , (v2>2) , (v2<=read_rows_-2)  ,  (global_id_u<=layer_pixels) , (inv_depth>min_inv_depth) , (inv_depth<max_inv_depth)  );\
				\
				printf("\n\n__kernel void se3_Rho_sq chk 2.5   (u==10) && (v==10) :  intersection = %i,  u = %i,  read_cols_-2 = %i,  v = %i,  read_rows_-2 = %i,  u2 = %i,    v2 = %i,       global_id_u=%i,  layer_pixels = %i,  inv_depth=%f,  min_inv_depth = %f,  max_inv_depth = %f", \
					intersection,  u, read_cols_-2, v, read_rows_-2, u2,   v2,    global_id_u,  layer_pixels,  inv_depth,  min_inv_depth,  max_inv_depth  );\
			}\
			)
		}

	/*{
			VK_TRACKING(\
			if ( u==5 && v==5  ){ \
				printf("\n__kernel se3_Rho_sq(..)_1: sample=%u,  layer=%i,  global_id_u=%i,  u=%i,  v=%i,   se3_sum_size=%i,  intersection=%i, (u>2)=%i,  (u<=read_cols_-2)=%i,  (v>2)=%i,  (v<=read_rows_-2)=%i,  (u2>2)=%i,  (u2<=read_cols_-2)=%i,  (v2>2)=%i,  (v2<=read_rows_-2)=%i,  (global_id_u<=layer_pixels)=%i" \
				, sample,  layer,  global_id_u,  u,  v,  se3_sum_size,  intersection, (u>2),  (u<=read_cols_-2),  (v>2),  (v<=read_rows_-2),  (u2>2),  (u2<=read_cols_-2),  (v2>2),  (v2<=read_rows_-2),  (global_id_u<=layer_pixels) );\
			}\
			)
	}*/
		if (  intersection  ) {																			// if (not cleanly within new frame) skip  Problem u2&v2 are wrong.
			int idx 										= 0;										// float4 bilinear_flt4(__global float4* img, float u_flt, float v_flt, int cols, int read_offset_, uint reduction);
			new_px 											= bilinear_flt4(img_new, u2_flt, v2_flt, mm_cols, read_offset_);
			rho 											= (img_cur[read_index] - new_px)*(1.0f - g1p[read_index].s3);						// g1.s3 = Value channel. Weight rho by edges.
			rho.w 											= 1.0f; ///alpha;

			Rho_[read_index + sample*mm_pixels ] 			= rho;										// save pixelwise photometric error map to buffer. NB Outside if(){}, to zero non-overlapping pixels.
			float4 rho_sq 									= {rho.x*rho.x,  rho.y*rho.y,  rho.z*rho.z, rho.w};
			local_sum_rho_sq[sample_lid] 					= rho_sq;									// Also compute global Rho^2.
// TODO  [sample] index on local mem and final buffer.
			//if (layer==5) printf(",(%u,%f)", global_id_u ,inv_depth);									// debug chk on value of inv_depth
		}
	}
	////////////////////////////////////////////////////////////////////////////////////////			// Reduction ///////////////////////////////////////////////////////////////////////
	int max_iter 											= 9; 										//ceil(log2((float)(group_size)));
	group_size = local_size;
	for (uint iter=0; iter<=max_iter ; iter++) {		// for log2(local work group size)				// problem : how to produce one result for each mipmap layer ?
																										// NB kernels launched separately for each layer, but workgroup size varies between GPUs.
		barrier(CLK_LOCAL_MEM_FENCE);																	// No 'if->return' before fence between write & read local mem
		group_size   										/= 2;
		if (lid<group_size){
			for (int sample=0; sample<local_num_samples; sample++){
				int sample_lid 					= lid + sample * local_size;
				local_sum_rho_sq[ sample_lid ] += local_sum_rho_sq[ sample_lid + group_size ];			// Also compute global Rho^2.
			}
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	////////////////////////////////////////////////////////////////////////////////////////			// Export result ////////////////////////////////////////////////////////////////////
	if (lid==0) {
		uint group_id 										= get_group_id(0);
		uint rho_global_sum_offset 							= (read_offset_ / local_size);
		uint num_groups 									= get_num_groups(0);
		float4 layer_data 									= {num_groups, reduction, rho_global_sum_offset, 0.0f };		// Write layer data to first entry
		{
																										VK_TRACKING(\
																											printf("\n__kernel se3_Rho_sq(..) chk 3: layer=%i, u=%i, v=%i, group_id=%i,  rho_global_sum_offset=%i,  (float)read_offset_/local_size=%f,  local_size=%u, read_offset_=%u,  local_sum_rho_sq[lid]=(%f,%f,%f,%f), local_sum_rho_sq[lid][3]=%f ",\
																											layer, u, v, group_id, rho_global_sum_offset, ((float)read_offset_)/((float)local_size),  local_size, read_offset_, local_sum_rho_sq[lid].x, local_sum_rho_sq[lid].y, local_sum_rho_sq[lid].z, local_sum_rho_sq[lid].w, local_sum_rho_sq[lid][3] );\
																										)
		}
		for (int sample=0; sample<local_num_samples; sample++){
			uint sample_offset								= sample * se3_sum_size;
			uint rho_global_sum_offset_						= rho_global_sum_offset + (sample * se3_sum_size);
			int  sample_lid 								= lid + sample * local_size;

			if (global_id_u == 0) {
				global_sum_rho_sq[layer +  sample_offset] 	= layer_data;
				{
																										VK_TRACKING(\
 																										printf("\n__kernel se3_Rho_sq(..)  chk 4: sample=%i,  layer=%i,  [layer+sample_offset]=%i,  rho_global_sum_offset=%i,   group_id=%i,   global_id_u=%i,  layer_data=( %f,  %f,  %f,  %f )", \
 																										sample,  layer,  layer+sample_offset,  rho_global_sum_offset,   group_id,   global_id_u,  layer_data.x,  layer_data.y,  layer_data.z,  layer_data.w );\
																										)
				}
			}
			rho_global_sum_offset_ 							+= group_id;

			if (local_sum_rho_sq[lid][3] >0){															// Using last channel rho[3], to count valid pixels being summed.
				global_sum_rho_sq[rho_global_sum_offset_] 	= local_sum_rho_sq[sample_lid];
				{
																										VK_TRACKING(\
 																										printf("\n__kernel se3_Rho_sq(..)  chk 5: sample=%i,  layer=%i,  rho_global_sum_offset_=%i,  group_id=%i,   local_sum_rho_sq[lid]=( %f,  %f,  %f,  %f )", \
 																										sample,  layer, rho_global_sum_offset_, group_id,  local_sum_rho_sq[lid].x, local_sum_rho_sq[lid].y, local_sum_rho_sq[lid].z, local_sum_rho_sq[lid].w );\
																										)
				}

			}else {																						// If no matching pixels in this group, set values to zero.
				global_sum_rho_sq[rho_global_sum_offset_] 	= 0;
			}
		}
	}
}
///////////////////////////////////////////////////////


__kernel void se3_LK_grad(
	// inputs
	__private	uint	layer,					//0

	__constant 	uint8*	mipmap_params,			//1
	__constant 	uint*	uint_params,			//2
	__constant  float*  fp32_params,			//3

	__global	float16*k2k,					//4		// keyframe2K
	__global 	float4*	img_cur,				//5		// keyframe
	__global 	float4*	img_new,				//6
	__global 	float8*	SE3_grad_map_cur_frame,	//7		// keyframe
	__global 	float8*	SE3_grad_map_new_frame,	//8
	__global	float* 	depth_map,				//9		// NB keyframe GT_depth, now stored as inv_depth

	// outputs
	__global	float4* Rho_,					//10
	__local		float4*	local_sum_rho_sq,		//11	1 DoF, float 1 channels	NB local memory limited to 4kb on rtx3080 and local worksize is 1024
	__global 	float4*	global_sum_rho_sq,		//12

	__global	float4* weights_map,			//13
	__local		float4*	local_sum_weight,		//14	6 DoF, float 1 channels
	__global 	float4*	global_sum_weight,		//15

	__global 	float4*	SE3_incr_map_,			//16
	__local		float4*	local_sum_grads,		//17	6 DoF, float 1 channels
	__global	float4*	global_sum_grads,		//18

	__global	float8* g1p						//19	keyframe_g1mem
	)
 {																										// find gradient wrt SE3 find global sum for each of the 6 DoF
	uint  global_id_u 	= get_global_id(0);
	float global_id_flt = global_id_u;
	uint  lid 			= get_local_id(0);
																										//if(global_id_u == 1  ){ printf("\n__kernel void se3_LK_grad (global_id_u == 1 )  chk_1"   ); }
	uint local_size 	= get_local_size(0); // / wg_divisor;
	uint group_size 	= local_size;
	uint num_groups		= get_num_groups(0); //size_t get_num_groups (uint dimindx)
	uint work_dim 		= get_work_dim();
	uint global_size	= get_global_size(0);
	float16 k2k_pvt		= k2k[0];

	uint8 mipmap_params_ = mipmap_params[layer];
	uint read_offset_ 	= mipmap_params_[MiM_READ_OFFSET];
	uint read_cols_ 	= mipmap_params_[MiM_READ_COLS];
	uint read_rows_ 	= mipmap_params_[MiM_READ_ROWS];
	uint layer_pixels	= mipmap_params_[MiM_PIXELS];

	uint8 mipmap_params_0 = mipmap_params[0];
	uint read_offset_0 	= mipmap_params_0[MiM_READ_OFFSET];

	uint base_cols		= uint_params[COLS];
	uint margin 		= uint_params[MARGIN];
	uint mm_cols		= uint_params[MM_COLS];
	uint mm_pixels		= uint_params[MM_PIXELS];

	float inv_d_step 	= fp32_params[INV_DEPTH_STEP];
	float min_inv_depth = fp32_params[MIN_INV_DEPTH]; // + inv_d_step;
	float max_inv_depth = fp32_params[MAX_INV_DEPTH]; // - inv_d_step;

	uint reduction		= mm_cols/read_cols_;
	uint v 				= global_id_u / read_cols_;														// read_row
	uint u 				= fmod(global_id_flt, read_cols_);												// read_column
	float u_flt			= u * reduction;																// NB this causes sparse sampling of the original space, to use the same k2k at every scale.
	float v_flt			= v * reduction;
	uint read_index 	= read_offset_  +  v  * mm_cols  + u ;
	float alpha			= img_cur[read_index].w;

	float inv_depth 	= depth_map[read_index]; 														//1.0f;// mid point max-min inv depth	// Find new pixel position, h=homogeneous coords.//inv dept  //depth_index
	float uh2 			= k2k_pvt[0]*u_flt 	+ k2k_pvt[1]*v_flt 	+ k2k_pvt[2]*1 	+ k2k_pvt[3]*inv_depth;
	float vh2 			= k2k_pvt[4]*u_flt 	+ k2k_pvt[5]*v_flt 	+ k2k_pvt[6]*1 	+ k2k_pvt[7]*inv_depth;
	float wh2 			= k2k_pvt[8]*u_flt 	+ k2k_pvt[9]*v_flt 	+ k2k_pvt[10]*1	+ k2k_pvt[11]*inv_depth;
	//float h/z  		= k2k_pvt[12]*u_flt	+ k2k_pvt[13]*v_flt + k2k_pvt[14]*1; // +k2k_pvt[15]/z

	float u2_flt		= uh2/(wh2*reduction);
	float v2_flt		= vh2/(wh2*reduction);
	int  u2				= floor(u2_flt + 0.5f) ;														// nearest neighbour interpolation
	int  v2				= floor(v2_flt + 0.5f) ;														// NB this corrects the sparse sampling to the redued scales.
	uint num_DoFs 		= 6;
	float4 new_px;

	const float4 zero_v4f	= {0.0f,0.0f,0.0f,0.0f};
	float4 rho 				= zero_v4f;
	float weight;
	float8 se3_incr;
	{
																	VK_TRACKING(\
																	if(global_id_u == 1  ){\
																		printf("\n\n\n\n__kernel void se3_LK_grad chk 0  (global_id_u == 1 ) : local_size=%u,  reduction=%u,  layer=%u, read_offset_=%u, read_cols_=%u, read_rows_=%u, layer_pixels=%u, read_index=%u, alpha=%f, \nk2k_pvt=[\n%f,%f,%f,%f,   \n%f,%f,%f,%f,    \n%f,%f,%f,%f,   \n%f,%f,%f,%f   ]\n", \
																		local_size, reduction, layer, read_offset_, read_cols_, read_rows_, layer_pixels, read_index, alpha,  k2k_pvt[0],k2k_pvt[1],k2k_pvt[2],k2k_pvt[3],  k2k_pvt[4],k2k_pvt[5],k2k_pvt[6],k2k_pvt[7],    k2k_pvt[8],k2k_pvt[9],k2k_pvt[10],k2k_pvt[11],   k2k_pvt[12],k2k_pvt[13],k2k_pvt[14],k2k_pvt[15]  );\
																	}\
																	if((u==read_cols_-1) && (v== read_rows_-1 )){\
																		printf("\n\n__kernel void se3_LK_grad chk 1  (u==read_cols_-1) && (v== read_rows_-1 ) :  local_size=%u,  reduction=%u,  layer=%u, read_offset_=%u, read_cols_=%u, read_rows_=%u, layer_pixels=%u, read_index=%u, num_groups=%u, alpha=%f", \
																		local_size, reduction, layer, read_offset_, read_cols_, read_rows_, layer_pixels, read_index, num_groups, alpha   );\
																	}\
																	)
	}
	for (int i=0; i<6; i++) {																							// Essential to zero local mem.
		local_sum_rho_sq[i*local_size + lid] 	= zero_v4f;
		local_sum_weight[i*local_size + lid] 	= zero_v4f;
		local_sum_grads [i*local_size + lid] 	= zero_v4f;
	}

	////////////////////////////////////////////////////////////////////////////////////////							// Exclude all out-of-bounds threads:
	bool intersection = (u>2) && (u<=read_cols_-2) && (v>2) && (v<=read_rows_-2) && (u2>2) && (u2<=read_cols_-2) && (v2>2) && (v2<=read_rows_-2)  &&  (global_id_u<=layer_pixels) && (inv_depth>=min_inv_depth) && (inv_depth<=max_inv_depth);

	{
		VK_TRACKING(\
		if( (u==10) && (v==10) ){\
			printf("\n\n__kernel void se3_LK_grad chk 2   (u==10) && (v==10) :  intersection = %i,  (u>2) = %i,  (u<=read_cols_-2) = %i,  (v>2) = %i,  (v<=read_rows_-2) = %i,  (u2>2) = %i,  (u2<=read_cols_-2) = %i,  (v2>2) = %i,  (v2<=read_rows_-2) = %i,  (global_id_u<=layer_pixels) = %i,  (inv_depth>=min_inv_depth) = %i,  (inv_depth<=max_inv_depth) = %i", \
			intersection,  (u>2) , (u<=read_cols_-2) , (v>2) , (v<=read_rows_-2) , (u2>2) , (u2<=read_cols_-2) , (v2>2) , (v2<=read_rows_-2)  ,  (global_id_u<=layer_pixels) , (inv_depth>min_inv_depth) , (inv_depth<max_inv_depth)  );\
			\
			printf("\n\n__kernel void se3_LK_grad chk 2.5   (u==10) && (v==10) :  intersection = %i,  u = %i,  read_cols_-2 = %i,  v = %i,  read_rows_-2 = %i,  u2 = %i,    v2 = %i,       global_id_u=%i,  layer_pixels = %i,  inv_depth=%f,  min_inv_depth = %f,  max_inv_depth = %f", \
			intersection,  u, read_cols_-2, v, read_rows_-2, u2,   v2,    global_id_u,  layer_pixels,  inv_depth,  min_inv_depth,  max_inv_depth  );\
		}\
		)
	}

	if (  intersection  ) {																								// if (not cleanly within new frame) skip  Problem u2&v2 are wrong.
		int idx 					= 0;																				// float4 bilinear_flt4(__global float4* img, float u_flt, float v_flt, int cols, int read_offset_, uint reduction);
																														// if(global_id_u == 10000  ){ printf("\n__kernel void se3_LK_grad (global_id_u == 10000 )  chk_3  , read_offset_=%u,  inv_depth=%f, SO3_multiplier=%f, ST3_multiplier=%f ",  read_offset_, inv_depth,  fp32_params[MAX_INV_DEPTH]/((inv_depth + 0.01) *5),  inv_depth/fp32_params[MAX_INV_DEPTH] ); }
		new_px 						= bilinear_flt4(img_new, u2_flt, v2_flt,  mm_cols, read_offset_); //   /reduction
		rho 						= (img_cur[read_index] - new_px)*(1.0f - g1p[read_index].s3);						// g1.s3 = Value channel. Weight rho by edges.
		rho.w 						= 1.0f; ///alpha;
		//float rho_a4[4];
		//vstore4(rho ,0, rho_a4);

		Rho_[read_index] 			= rho;																				// save pixelwise photometric error map to buffer. NB Outside if(){}, to zero non-overlapping pixels.
		float4 rho_sq 				= {rho.x*rho.x,  rho.y*rho.y,  rho.z*rho.z, rho.w};
		local_sum_rho_sq[lid] 		= rho_sq;																			// Also compute global Rho^2.

		{
			VK_TRACKING(\
			if( (u==10) && (v==10) ){ printf("\n\n__kernel void se3_LK_grad chk 3  (u==10) && (v==10) : , img_cur[read_index] = new_px = [%f,%f,%f,%f],  g1p[read_index].s3 = %f,  new_px = [%f,%f,%f,%f],   rho = [%f,%f,%f,%f] ", img_cur[read_index].x,img_cur[read_index].y,img_cur[read_index].z,img_cur[read_index].w,   g1p[read_index].s3,    new_px.x,new_px.y,new_px.z,new_px.w,   rho.x,rho.y,rho.z,rho.w );  }
			)
		}
		/*
		float4 weights_v4[6] 		= {{0,0,0,0}}; // float4

		bilinear_SE3_grad_weight( weights_v4, SE3_grad_map_cur_frame, read_index,  SE3_grad_map_new_frame,  u2_flt,  v2_flt,  mm_cols,  read_offset_,  reduction,  mm_pixels,  alpha ); // , channel

		for (uint se3_dim=0; se3_dim<6; se3_dim++) 	{ local_sum_weight[ se3_dim*local_size + lid ]     		=  weights_v4[se3_dim]; }
		*/
		float multiplier = fp32_params[MAX_INV_DEPTH]/((inv_depth + 0.01) *5); 	// NB beware if inf depth, i.e. subnormal inv_depth, then div by zero error !					// de-weight foreground for rotation, & de-weight backgroud for translation.
		for (uint se3_dim=0; se3_dim<6; se3_dim++) {																																	// for each SE3 DoF
			float8 grad_v8 											= SE3_grad_map_cur_frame[ read_index + (se3_dim * mm_pixels) ] ;
			//grad_v8 												+= bilinear_SE3_grad (SE3_grad_map_new_frame, u2_flt, v2_flt, mm_cols, read_offset_0 + (se3_dim * mm_pixels)  );	// SE3_grad_map_new_frame[read_index_new + se3_dim * mm_pixels ] ;
			float4 grad_v4 											= grad_v8.hi + grad_v8.lo;							// i.e. sum the u and v components of the gradient, for the four colour channels.
			float4 incr_v4 											= grad_v4  * rho;

			if (se3_dim>=3){multiplier 								= inv_depth/fp32_params[MAX_INV_DEPTH] ;}			// For ST3 only to emphasize foreground pixels for parallax motion: multiply pixel inv_depth by min depth in scene.  Office scene depth is in cm from approx 90 to 450cm.
			incr_v4													*= multiplier;
																														if( fmod( (float)global_id_u, 999 ) &&   /*isinf(incr_v4.x) || isinf(incr_v4.y) ||*/ isinf(incr_v4.z) /*|| isinf(incr_v4.w)*/  ){ // !isnormal(incr_v4) ||
																															printf("\n_kernel void se3_LK_grad() se3_dim=%u,  global_id_u=%u,  inv_depth=%f,  fp32_params[MAX_INV_DEPTH]=%f,  multiplier=%f  ", \
																															se3_dim, global_id_u, inv_depth, fp32_params[MAX_INV_DEPTH], multiplier);
																														}
			incr_v4.w 												= 1.0f;
			local_sum_grads[se3_dim*local_size + lid] 				= incr_v4;											// pixelwise increment for this SE3 DoF

			grad_v4 												*= grad_v4;
			grad_v4.w												= 1.0f;
			local_sum_weight[se3_dim*local_size + lid] 				= grad_v4;											// save the SE3_grad^2, to use as divisor for this SE3 DoF, after summing.
			{
				VK_TRACKING(\
				if( (u==10) && (v==10) ){ printf("\n\n__kernel void se3_LK_grad chk 4  (u==10) && (v==10) : inv_depth=%f,  fp32_params[MAX_INV_DEPTH]=%f,  multiplier=%f,  incr_v4=%f,%f,%f,%f ",\
					inv_depth,   fp32_params[MAX_INV_DEPTH],  multiplier, incr_v4.x, incr_v4.y, incr_v4.z, incr_v4.w    ); }\
				)
			}
			/*
			float SE3_grad_cur_px[8];
			float SE3_grad_new_px[8];
			vstore8(SE3_grad_cur_px_v8, 0, SE3_grad_cur_px);
			vstore8(SE3_grad_new_px_v8, 0, SE3_grad_new_px);


			//float weights_a4[4];
			//vstore4(weights_v4[se3_dim], 0, weights_a4);

			float delta_a4[4];

			for (int chan=0; chan<3; chan++) {
				float SE3_grad 										= ( SE3_grad_cur_px[chan] + SE3_grad_cur_px[chan+4] + SE3_grad_new_px[chan] + SE3_grad_new_px[chan+4] ) / 4;
				delta_a4[chan]		 								= weights_a4[chan] * rho_a4[chan] / SE3_grad;
				if (!isnormal(delta_a4[chan])) 	delta_a4[chan] 		= 0;
			}


			if (se3_dim>=3){ 						multiplier 		= depth;}											// TODO investigate what multiplier should be used & when
			for (int chan=0; chan<3; chan++){		delta_a4[0] 	*= multiplier;   }									// Applies depth or inv depth multiplier.....

			float4 									delta_v4 		= {delta_a4[0]	,delta_a4[1], delta_a4[2], alpha  };

			local_sum_grads[se3_dim*local_size + lid] 				= delta_v4;
			*/
		}
/*
		for (uint se3_dim=0; se3_dim<3; se3_dim++) {	// translation, amplify nearby movement.						// NB SE3_incr_map_[ ].w = alpha for the image within the mipmap.
			local_sum_grads[se3_dim*local_size + lid] 				*= inv_depth;
			local_sum_grads[se3_dim*local_size + lid].w				= alpha;
			/ *
			local_sum_grads[se3_dim*local_size + lid].x 			*= inv_depth;// * 100;
			local_sum_grads[se3_dim*local_size + lid].y 			*= inv_depth;// * 100;
			local_sum_grads[se3_dim*local_size + lid].z 			*= inv_depth;// * 100;
			* /
		}
*/
/*
		for (uint se3_dim=3; se3_dim<6; se3_dim++) {	// rotation, amplify distant movement.							// NB SE3_incr_map_[ ].w = alpha for the image within the mipmap.
			local_sum_grads[se3_dim*local_size + lid].x 			/= (inv_depth * 50);
			local_sum_grads[se3_dim*local_size + lid].y 			/= (inv_depth * 50);
			local_sum_grads[se3_dim*local_size + lid].z 			/= (inv_depth * 50);
		}
*/
		float4 temp_v4f;
		for (uint se3_dim=0; se3_dim<6; se3_dim++) {
			weights_map  [read_index + se3_dim * mm_pixels ]		= local_sum_weight[ se3_dim*local_size + lid ];		// Save maps of SE3_grad^2 and pixelwise increment, for each SE3 DoF, for debugging.
			SE3_incr_map_[read_index + se3_dim * mm_pixels ]  		= local_sum_grads[se3_dim*local_size + lid];
		}
	}
/*
	// temp barrier for debugging, otherwise results are zero.
	barrier(CLK_LOCAL_MEM_FENCE);
	//if (true) {
		uint group_id 												= get_group_id(0);
		if (group_id == 25){
			float4 dof4 = local_sum_weight[ 4*local_size + lid ] ;
			float4 dof5 = local_sum_weight[ 5*local_size + lid ] ;

			float4 dof4_g = local_sum_grads[ 4*local_size + lid ] ;
			float4 dof5_g = local_sum_grads[ 5*local_size + lid ] ;

			printf ("\n__Kernel se3_LK_grad chk BEFORE parallel sum: layer=%u, \tlid=%u, \tdof4=%f, \tdof5=%f, \tdof4_g=%f, \tdof5_g=%f  ", layer, lid, dof4.w, dof5.w, dof4_g.w, dof5_g.w  );
		}
	//}
																		//if(global_id_u == 1  ){ printf("\n__kernel void se3_LK_grad (global_id_u == 1 )  chk_4,  alpha=%f", alpha   ); }
																		//if((u==read_cols_-1) && (v== read_rows_-1 )) { printf("\n__kernel void se3_LK_grad (global_id_u == 1 )  chk_4,  alpha=%f", alpha   ); }
*/
	////////////////////////////////////////////////////////////////////////////////////////		// Reduction
	int max_iter 												= 9;								//ceil(log2((float)(group_size)));
	barrier(CLK_LOCAL_MEM_FENCE);
	for (uint iter=0; iter<=max_iter ; iter++) {	// for log2(local work group size)				// problem : how to produce one result for each mipmap layer ?
		barrier(CLK_LOCAL_MEM_FENCE);																// NB kernels launched separately for each layer, but workgroup size varies between GPUs.
		group_size   											/= 2;
		for (int i=0; i<num_DoFs; i++){																// local_sum_grads
			if (lid<group_size){																	// No 'if->return' before fence between write & read local mem
				local_sum_weight[i*local_size + lid] 			+= local_sum_weight[i*local_size + lid + group_size];
				local_sum_grads [i*local_size + lid] 			+= local_sum_grads [i*local_size + lid + group_size];
			}
		}
	}
	group_size = local_size;
	for (uint iter=0; iter<=max_iter ; iter++) {	// for log2(local work group size)				// problem : how to produce one result for each mipmap layer ?
		barrier(CLK_LOCAL_MEM_FENCE);																// NB kernels launched separately for each layer, but workgroup size varies between GPUs.
		group_size   											/= 2;
		if (lid<group_size){
			local_sum_rho_sq[ lid] 			+= local_sum_rho_sq[ lid + group_size];
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);									//if(global_id_u == 1  ){ printf("\n__kernel void se3_LK_grad (global_id_u == 1 )  chk_5"   ); }
	/*
	for (uint se3_dim=0; se3_dim<6; se3_dim++) {
		weights_map  [read_index + se3_dim * mm_pixels ]		= local_sum_weight[ se3_dim*local_size + lid ];
	}

	//if (lid==0) {  // debug chk
		//uint group_id 											= get_group_id(0);
	*/
/*
		if (group_id == 25){
			float4 dof4 = local_sum_weight[ 4*local_size + lid ] ;
			float4 dof5 = local_sum_weight[ 5*local_size + lid ] ;

			float4 dof4_g = local_sum_grads[ 4*local_size + lid ] ;
			float4 dof5_g = local_sum_grads[ 5*local_size + lid ] ;

			float4 rho_chk = local_sum_rho_sq[ lid];

			printf ("\n__Kernel se3_LK_grad chk AFTER parallel sum: layer=%u, \tlid=%u, \tdof4=%f, \tdof5=%f, \tdof4_g=%f, \tdof5_g=%f, \trho_chk=%f  ", layer, lid, dof4.w, dof5.w, dof4_g.w, dof5_g.w, rho_chk.w  );
		}
	//}
*/
	////////////////////////////////////////////////////////////////////////////////////////		// Export result
	if (lid==0) {
		uint group_id 											= get_group_id(0);
		uint rho_global_sum_offset 								= read_offset_ / local_size ;					// Compute offset for this layer,   NB interger rounding !
		uint se3_global_sum_offset 								= rho_global_sum_offset *num_DoFs;				// 6 DoF of float4 channels, + 1 DoF to compute global Rho.
		rho_global_sum_offset 									+= group_id;
		se3_global_sum_offset 									+= group_id*num_DoFs;
																												//printf("\nkernel se3_grad_c(..): layer=%i,  group_id=%i,  read_offset_/local_size=%f,  local_size=%u, read_offset_=%u ", layer, group_id, ((float)read_offset_)/((float)local_size),  local_size, read_offset_);
		if (global_id_u == 0) {
			uint num_groups 									= get_num_groups(0);
			float4 layer_data 									= {num_groups, reduction, rho_global_sum_offset, se3_global_sum_offset };			// Write layer data to first entry
			global_sum_rho_sq [layer]							= layer_data;
			global_sum_weight [layer*num_DoFs] 					= layer_data;
			global_sum_grads  [layer*num_DoFs] 					= layer_data;
			{
																									VK_TRACKING(\
																									printf("\n\n__kernel se3_LK_grad chk_10   (global_id_u == 0) :  layer=%i,  rho_global_sum_offset=%i,   group_id=%i,   global_id_u=%i,  layer_data=( %f,  %f,  %f,  %f )", \
																										 layer,   rho_global_sum_offset,   group_id,   global_id_u,  layer_data.x,  layer_data.y,  layer_data.z,  layer_data.w );\
																									)
			}
/*
			//printf("\nkernel se3_grad_d(..)_2: layer=%i,  group_id=%i,  se3_global_sum_offset=%i,  layer_data=(%f,%f,%f,%f),    u=%i,  v=%i,   inv_depth=%f, u2=%f,  v2=%f,  u2_flt=%f,  v2_flt=%f,    k2k_pvt=(%f,%f,%f,%f    ,%f,%f,%f,%f    ,%f,%f,%f,%f    ,%f,%f,%f,%f),\t rho=(%f,%f,%f,%f), local_sum_grads=(%f,%f,%f,%f)"\
			//	,layer, group_id, se3_global_sum_offset,  layer_data.x, layer_data.y, layer_data.z, layer_data.w,  u, v, inv_depth, u_flt, v_flt, u2_flt, v2_flt,  k2k_pvt[0],k2k_pvt[1],k2k_pvt[2],k2k_pvt[3],   k2k_pvt[4],k2k_pvt[5],k2k_pvt[6],k2k_pvt[7],   k2k_pvt[8],k2k_pvt[9],k2k_pvt[10],k2k_pvt[11],   k2k_pvt[12],k2k_pvt[13],k2k_pvt[14],k2k_pvt[15], rho.x, rho.y, rho.z, rho.w, local_sum_grads[0][0],local_sum_grads[0][1],local_sum_grads[0][2],local_sum_grads[0][3]   )  ;
*/
		}
		if (local_sum_grads[0][3] >0){																// Using last channel local_sum_pix[0][7], to count valid pixels being summed.
			global_sum_rho_sq[rho_global_sum_offset]			= local_sum_rho_sq[lid];
			{
																									VK_TRACKING(\
																									printf("\n\n__kernel se3_LK_grad chk_11   (local_sum_grads[0][3] >0) :  layer=%i,  group_id=%i,   local_sum_rho_sq[lid]=( %f,  %f,  %f,  %f )", \
																										 layer, group_id,  local_sum_rho_sq[lid].x, local_sum_rho_sq[lid].y, local_sum_rho_sq[lid].z, local_sum_rho_sq[lid].w );\
																									)
			}
			for (int i=0; i<num_DoFs; i++){
				float4 temp_weights_float4 						= local_sum_weight[i*local_size + lid] / local_size;
				global_sum_weight[se3_global_sum_offset + i] 	= temp_weights_float4 ;

				float4 temp_float4 								= local_sum_grads[i*local_size + lid] / local_size; 	//   / local_sum_grads[i*local_size + lid].w ;
																														// Better to divide by local size, preserve information wrt number of valid pixels.
				global_sum_grads[se3_global_sum_offset + i] 	= temp_float4 ;						// local_sum_grads
				{
					VK_TRACKING(\
					printf("\n__kernel se3_LK_grad chk_12    layer=%i,  group_id=%i,   local_sum_grads[i*local_size + lid]=(%f,%f,%f,%f )", layer, group_id,   temp_float4.x,temp_float4.y,temp_float4.z,temp_float4.w );\
					)
				}
			}																						// Save to global_sum_grads // Count hits, and divide group by num hits, without using atomics!
		}else {																						// If no matching pixels in this group, set values to zero.
			global_sum_rho_sq[rho_global_sum_offset]			= 0;
			for (int i=0; i<num_DoFs; i++){
				global_sum_weight[se3_global_sum_offset + i] 	= 0;
				global_sum_grads[se3_global_sum_offset + i] 	= 0;	/*[rho_global_sum_offset]*/
			}
		}
	}
}



__kernel void reduce (																				// TODO use this for the second stage image summation tasks.
	__constant 	uint*		mipmap_params,		//0		// kernel not currently in use, needs to integrate with mipmap.
	__constant 	uint*		uint_params,		//1
	__global	float8*		se3_sum,			//2
	__local		float8*		local_sum_grads,	//3
	__global 	float8*		se3_sum2			//4
		 )
{
	uint global_id_u 	= get_global_id(0);
	float global_id_flt = global_id_u;
	if (global_id_u >= mipmap_params[MiM_PIXELS]) return;
	uint lid 			= get_local_id(0);
	uint local_size 	= get_local_size(0);
	uint group_size 	= local_size;
	uint read_offset_ 	= mipmap_params[MiM_READ_OFFSET];
	uint read_cols_ 	= mipmap_params[MiM_READ_COLS];
	uint mm_cols		= uint_params[MM_COLS];
	uint reduction		= mm_cols/read_cols_;

	uint global_sum_offset 	= read_offset_ / local_size ;											// Compute offset for this layer
	//
	local_sum_grads[lid] = se3_sum[global_sum_offset  + global_id_u];
	int max_iter = ilogb((float)(group_size));

	for (uint iter=0; iter<=max_iter ; iter++) {	// for log2(local work group size)				// problem : how to produce one result for each mipmap layer ?  NB kernels launched separately for each layer, but workgroup size varies between GPUs.
		group_size   /= 2;
		barrier(CLK_LOCAL_MEM_FENCE);																// No 'if->return' before fence between write & read local mem
		if (lid<group_size)  local_sum_grads[lid] += local_sum_grads[lid+group_size];				// local_sum_grads
	}

	if (lid==0) {
		uint group_id 	= get_group_id(0);
		uint global_sum_offset = read_offset_ / local_size ;										// Compute offset for this layer
		uint num_groups = get_num_groups(0);
		/*
		printf("\n\n reduction=%u,  global_sum_offset=%u,  num_groups=%u,  group_id=%u, \nlocal_sum_grads[lid]=( %f, %f, %f, %f,   %f, %f, %f, %f ),  \nse3_sum2[group_id]=( %f, %f, %f, %f,   %f, %f, %f, %f ) "\
		, reduction, global_sum_offset,  num_groups, group_id \
		, local_sum_grads[lid][0],local_sum_grads[lid][1],local_sum_grads[lid][2],local_sum_grads[lid][3], local_sum_grads[lid][4],local_sum_grads[lid][5],local_sum_grads[lid][6],local_sum_grads[lid][7] \
		, se3_sum2[group_id][0], se3_sum2[group_id][1], se3_sum2[group_id][2], se3_sum2[group_id][3],    se3_sum2[group_id][4], se3_sum2[group_id][5], se3_sum2[group_id][6], se3_sum2[group_id][7]\
		);
		*/
		float8 layer_data = {num_groups, reduction, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };			// Write layer data to first entry
		if (global_id_u == 0) {se3_sum2[global_sum_offset] = layer_data; }
		global_sum_offset += 1+ group_id;

		if (local_sum_grads[0][7] >0){
			se3_sum2[global_sum_offset] = local_sum_grads[0] / local_sum_grads[0][7];				// Save to se3_sum2 // Count hits, and divide group by num hits, without using atomics!
		}else se3_sum2[global_sum_offset] = 0;
	}
}


__kernel void atomic_test1(
	__private uint num_threads,
	volatile __global int *var_array
)
{
	uint gid = get_global_id(0);
	if (gid>=num_threads) return;
	int new_var = 1;
	int result = -1;
	result = atomic_add(&var_array[0],  new_var );			// int atomic_add(volatile __global int *p,  int val)
	{
		VK_TRACKING(\
		if (gid == 1) printf("\n__kernel void atomic_test1(..) result = %i",result );\
		)
	}
	if (gid==0)return;
	var_array[gid]=new_var;
}

/*
__kernel void atomic_test2(
	__private uint num_threads,
	volatile __global int *var_array
		)
{
	uint gid = get_global_id(0);
	if (gid>=num_threads) return;
	int new_var = 1;
	int result = -1;
//	atomic_fetch_add_explicit( &var_array[0], new_var, memory_order_relaxed ); //atomic_add(&var_array[0],  new_var );			// int atomic_add(volatile __global int *p,  int val)
	if (gid == 1) printf("\n__kernel void atomic_test1(..) result = %i",result );
	if (gid==0)return;
	var_array[gid]=new_var;
}

//atomic_fetch_add_explicit(&acnt, 1, memory_order_relaxed);
*/

/*
inline void atomic_maxf(															  				// from https://ingowald.blog/2018/06/24/float-atomics-in-opencl/
	volatile 	__global 	float 	*g_val,
							float 	myValue
){
	float cur = FLT_MIN;
	while (myValue > (cur = *g_val)) 		myValue 	= atomic_xchg( g_val,  fmax(cur,myValue) );
}*/


__kernel void atomic_test2(
				__private 	uint 	num_threads,
	volatile 	__global 	float 	*var_array
){
	uint gid 		= get_global_id(0);
	if (gid>=num_threads) return;
	float new_var 	=  gid + 0.36f;
	var_array[gid] = ((float)gid)/100.0f;
	//if (gid < 5) printf("\n__kernel void atomic_test2(..) before:   new_var = %f,   var_array[0] = %f,   &var_array[0]=%p  num_threads=%u",  new_var,  var_array[0],  &var_array[0], num_threads  );
	atomic_maxf(&var_array[0],  new_var );										// int atomic_add(volatile __global int *p,  int val)
	//if (gid < 5) printf("\n__kernel void atomic_test2(..) after: new_var = %f, var_array[0] = %f  ", new_var, var_array[0] );
	//if (gid == 0) return;
	//var_array[gid]	=	new_var;
}


/*
static float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(     address_as_i,     assumed,     __float_as_int(::fmaxf(val, __int_as_float(assumed) ) ) );

    } while (assumed != old);
    return __int_as_float(old);
}



// Cuda implementation of float atomicMax(..)  from  https://stackoverflow.com/questions/17399119/how-do-i-use-atomicmax-on-floating-point-values-in-cuda
__device__ static float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
         old = ::atomicCAS(     address_as_i,     assumed,     __float_as_int(::fmaxf(val, __int_as_float(assumed) ) ) );
    } while (assumed != old);
    return __int_as_float(old);
}*/

