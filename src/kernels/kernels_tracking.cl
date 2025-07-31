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
	__private	uint		cols_per_row,			//1
	__private	uint		out_block_size,			//2
	__private	float2		delta_SE3,				//3

	__constant	uint8*		mipmap_params,			//4
	__constant	uint*		uint_params,			//5
	__constant	float*		fp32_params,			//6
	__constant	float16*	inv_k2k,				//7		// transforms for 4 past frames,  k2k_buf

	__global	float4*		img_cur,				//8		// multiple past frames. NB retain frames at powers of 2, and vary starting power plus num franes.
	__global	float4*		img_past_0,				//9
	__global	float4*		img_past_1,				//10
	__global	float4*		img_past_2,				//11
	__global	float4*		img_past_3,				//12

	__global	float*		depth_map,				//13	// current frame depth, now stored as inv_depth
	__global	float8*		g1p,					//14	// current frame g1mem
	__global 	float8*		SE3_grad_map_cur_frame,	//15

	__global	float4*		vel_cur,				//16	// multiple past frames.
	__global	float4*		vel_past_0,				//17	// TODO, relative velocity not used yet. Will use it to modify depth map with timestep for past frames.
	__global	float4*		vel_past_1,				//18
	__global	float4*		vel_past_2,				//19
	__global	float4*		vel_past_3,				//20

	//output
	__global	float2*		Rho_,					//21	// { sum rho^2 ,  count of valid pixels used } Writen to dense patches.
	__local		float2*		local_rho,				//22	// float2 local_rho[ local_work_size/2 ]  hence sizeof( float)*local_work_size.

	__global	float2*		SE3_incr_map_,			//23
	__local		float2*		local_SE3_incr			//24
	)
{
	const uint block_size							= 32;										// or send as __private arg ? BUT as hardcoded "const uint" it can be used to size arrays etc.
	const uint se3_dof								= 6;
	const uint num_past_frames						= 4;										// 1,2,4,8,16,32,64 // variable select window of 4 frames.
	const float4 zero_f4							= {0.0f,0.0f,0.0f,0.0f};
	const float2 zero_f2							= {0.0f,0.0f};

	__global float4*	img_past[num_past_frames]	= { img_past_0, img_past_1, img_past_2, img_past_3 };
	__global float4*	vel_past[num_past_frames]	= { vel_past_0, vel_past_1, vel_past_2, vel_past_3 };

	uint  global_id_u 								= get_global_id(0);
	uint  lid 										= get_local_id(0);
	uint  group_id									= get_group_id(0);
	const uint local_size 							= get_local_size(0);

	const uint8 mipmap_params_						= mipmap_params[layer];
	uint read_offset_ 								= mipmap_params_[MiM_READ_OFFSET];
	uint read_cols_ 								= mipmap_params_[MiM_READ_COLS];
	uint read_rows_ 								= mipmap_params_[MiM_READ_ROWS];
	uint layer_pixels								= mipmap_params_[MiM_PIXELS];

	uint mm_cols									= uint_params[MM_COLS];
	uint mm_pixels									= uint_params[MM_PIXELS];

	float min_inv_depth								= fp32_params[MIN_INV_DEPTH];
	float max_inv_depth								= fp32_params[MAX_INV_DEPTH];

	float reduction									= mm_cols/read_cols_;
	uint row_length									= cols_per_row;								// blocks_cols * block_size;
	uint row_col									= fmod((float)global_id_u, row_length);
	uint block_row									= global_id_u / row_length;
	uint read_index									= read_offset_ + row_col + block_row*block_size*mm_cols;
	uint row_offset									= read_offset_/mm_cols;

	uint write_spacing								= block_size/out_block_size;
	uint write_index								= row_col/out_block_size 	+ block_row*write_spacing*mm_cols;
	uint write_index_2								= row_col/block_size 		+ block_row*mm_cols;

	float2 rho_pvt_arr[block_size]					= {zero_f2};								// pvt variable for values in this column.
	float4 rho_pvt_flt4								= zero_f4;
	float2 rho_pvt_flt2								= zero_f2;

	float8 grad_v8									= {zero_f4, zero_f4};
	float2 grad_pvt_arr[block_size*se3_dof]			= {zero_f2};								// pvt variable for values in this column.
	float4 grad_pvt_flt4_SE3[6]						= {zero_f4};
	float  grad_pvt_flt_SE3[6]						= {0.0f};
	float  grad_pvt_SE3_mag							= 0.0f;
	//float  grad_pvt_ST_mag							= 0.0f;

	float depth_weight								= 0.0f;
	float weights									= 0.0f;

	float4 grad_pvt_flt4							= zero_f4;
	float2 grad_pvt_flt2							= zero_f2;

	float2 SE3_incr_pvt_arr[block_size*se3_dof]		= {zero_f2};								// pvt variable for values in this column.
	float4 SE3_incr_pvt_flt4						= zero_f4;
	float2 SE3_incr_pvt_flt2						= zero_f2;

	float4 img_cur_pvt[block_size];																// pvt variable for values in this column.
	float8 g1p_pvt[block_size];
	float4 old_px;
	bool   intersection;

	local_rho[lid]									= zero_f2;
	for (uint se3_dim=0; se3_dim<se3_dof; se3_dim++) {
		local_SE3_incr[lid + se3_dim*local_size]	= zero_f2;
	}
																								// PATCH KERNEL //
	////////////////////////////////////////////////////////////////////////////				// transfer data from global memory.
	for (uint past_frame_idx=0; past_frame_idx</*num_past_frames*/1; past_frame_idx++){			// step though past frames ///////////////////////////////////////////////////////////////////////////////
		for (uint row_in_block=0; row_in_block<block_size; row_in_block ++){					// step through rows of the patch, /////////////////////////////////////////////////////////////
			float						u2_flt_1, 	v2_flt_1;									// current frame
			uint read_index_row 		= read_index + row_in_block * mm_cols;
			img_cur_pvt[row_in_block]	= img_cur[read_index_row];
			g1p_pvt[row_in_block]		= g1p[read_index_row];
			float inv_depth_1 			= depth_map[read_index_row];

			// Where to sample the past image frame //////
			uint index 					= read_index_row - read_offset_;
			uint v 						= index / mm_cols;
			uint u 						= fmod((float)index, mm_cols);
			float u_flt					= (float)u * reduction;														// NB this causes sparse sampling of the original space, to use the same k2k at every scale.
			float v_flt_1				= (float)v * reduction;
																													// TODO, relative velocity not used yet. Will use it to modify depth map with timestep for past frames.
			float uh2_1 				= inv_k2k[past_frame_idx][0]*u_flt 												+ inv_k2k[past_frame_idx][ 2]*1		+ inv_k2k[past_frame_idx][ 3]*inv_depth_1;		// + inv_k2k[past_frame_idx][1]*v_flt
			float vh2_1 				= inv_k2k[past_frame_idx][4]*u_flt 												+ inv_k2k[past_frame_idx][ 6]*1		+ inv_k2k[past_frame_idx][ 7]*inv_depth_1;		// + inv_k2k[past_frame_idx][5]*v_flt
			float wh2_1					= inv_k2k[past_frame_idx][8]*u_flt 	+ inv_k2k[past_frame_idx][9]*v_flt_1 		+ inv_k2k[past_frame_idx][10]*1		+ inv_k2k[past_frame_idx][11]*inv_depth_1;		//
			//float h/z  				= inv_k2k[past_frame_idx][12]*u_flt	+ inv_k2k[past_frame_idx][13]*v_flt 		+ inv_k2k[past_frame_idx][14]*1; 													// + inv_k2k[past_frame_idx][15]/z

			u2_flt_1					= (uh2_1 + inv_k2k[past_frame_idx][1]*v_flt_1 ) / ((wh2_1  )*reduction);
			v2_flt_1					= (vh2_1 + inv_k2k[past_frame_idx][5]*v_flt_1 ) / ((wh2_1  )*reduction);

			intersection 				= 	(u>2)			&& (u<=read_cols_-2)			&& (v>2)			&& (v<=read_rows_-2) 			&& \
											(u2_flt_1>2)	&& (u2_flt_1<=read_cols_-2)		&& (v2_flt_1>2) 	&& (v2_flt_1<=read_rows_-2)		&& \
											(global_id_u<=layer_pixels)		&&	(inv_depth_1>=min_inv_depth)	&& (inv_depth_1<=max_inv_depth);												// if images overlap

			rho_pvt_flt4				= zero_f4;
			float edge_weight			= 0;
			float value_sq_pvt			= 0;
			//////////////////////////////////////////////////
			if (intersection){
				edge_weight				= (1.0f - g1p_pvt[row_in_block].s3);
				// Photometric error rho ///////
				old_px					= bilinear_flt4( img_past[past_frame_idx],  u2_flt_1,  v2_flt_1,  mm_cols,  read_offset_ )	;
				rho_pvt_flt4			= (img_cur_pvt[row_in_block] - old_px) ;
				value_sq_pvt			= img_cur_pvt[row_in_block].z * old_px.z  ;
				rho_pvt_flt4.x			*= value_sq_pvt;																																		// Reduce rho hue and saturation by multiplying by old & new px value.
				rho_pvt_flt4.y			*= value_sq_pvt;
				rho_pvt_flt4.w			= 1.0f;																																					// rho.w holds pixel count.

				// Magnitude of gradient of Rho wrt SE3 rotation & translation //////
				for (uint se3_dim=0; se3_dim<se3_dof; se3_dim++) {
					grad_v8 												=  SE3_grad_map_cur_frame[ read_index_row + (se3_dim * mm_pixels) ] ;
					grad_pvt_flt4_SE3[se3_dim]								=  grad_v8.hi + grad_v8.lo;
					grad_pvt_flt4_SE3[se3_dim].w							=  1.0f;
					grad_pvt_flt_SE3[se3_dim]								= (grad_pvt_flt4_SE3[se3_dim].x + grad_pvt_flt4_SE3[se3_dim].y + grad_pvt_flt4_SE3[se3_dim].z) / 3.0f;
					grad_pvt_SE3_mag										+= grad_pvt_flt_SE3[se3_dim] * grad_pvt_flt_SE3[se3_dim];
				}
				grad_pvt_SE3_mag											= half_sqrt( grad_pvt_SE3_mag ) + FLT_EPSILON;																		// L2 norm, always +ve.		+ FLT_EPSILON; prevents div by zero.

				// SO3 rotation	//////////																																						// NB beware if inf depth, i.e. subnormal inv_depth, then div by zero error !
				depth_weight												=  fp32_params[MAX_INV_DEPTH] / ((inv_depth_1 + 0.01)*5);															// de-weight foreground for rotation, & de-weight backgroud for translation.
				weights														=  edge_weight * depth_weight;
				SE3_incr_pvt_flt2.y											=  weights;
				float delta_se3												=  delta_SE3[ 0 ];

				for (uint se3_dim=0; se3_dim<3; se3_dim++) {
					SE3_incr_pvt_flt4										= grad_pvt_flt4_SE3[se3_dim] * rho_pvt_flt4;
					SE3_incr_pvt_flt2.x										= weights * ( SE3_incr_pvt_flt4.x	+ SE3_incr_pvt_flt4.y	+ SE3_incr_pvt_flt4.z )/ (3.0f * grad_pvt_SE3_mag);		// Computes the update vector to zero Rho for this pixel.
					SE3_incr_pvt_flt2.x										= clamp( SE3_incr_pvt_flt2.x ,	-delta_se3,	+delta_se3  );															// pixelwise clamp to supress the efffect of giant steps from low gradient pixels.
					SE3_incr_pvt_flt2.y										= weights * grad_pvt_flt_SE3[se3_dim] / grad_pvt_SE3_mag;															// NB will divide   sum_SE3_incr[se3_dim] by sum weights[se3_dim] .
					SE3_incr_pvt_arr[ se3_dim*block_size + row_in_block ]	= SE3_incr_pvt_flt2 ;																								// pixelwise increment for this SE3 DoF
				}

				// ST3 translation	///////
				weights														/= depth_weight;
				depth_weight												=  inv_depth_1 / fp32_params[MAX_INV_DEPTH];																		// For ST3 emphasize foreground pixels for parallax motion: multiply pixel inv_depth by min depth in scene.
				weights														*= depth_weight;																									// NB "Office" test scene depth is in cm from approx 90 to 450cm.
				SE3_incr_pvt_flt2.y											=  weights;
				delta_se3													=  delta_SE3[ 1 ];

				for (uint se3_dim=3; se3_dim<se3_dof; se3_dim++) {
					SE3_incr_pvt_flt4										= grad_pvt_flt4_SE3[se3_dim] * rho_pvt_flt4;
					SE3_incr_pvt_flt2.x										= weights * ( SE3_incr_pvt_flt4.x 	+ SE3_incr_pvt_flt4.y	+ SE3_incr_pvt_flt4.z)/ (3.0f * grad_pvt_SE3_mag);
					SE3_incr_pvt_flt2.x										= clamp( SE3_incr_pvt_flt2.x ,	-delta_se3,	+delta_se3  );
					SE3_incr_pvt_flt2.y										= weights * grad_pvt_flt_SE3[se3_dim] / grad_pvt_SE3_mag;
					SE3_incr_pvt_arr[ se3_dim*block_size + row_in_block ]	= SE3_incr_pvt_flt2 ;
				}
			}
			barrier( CLK_GLOBAL_MEM_FENCE );

			rho_pvt_flt2.x					=  rho_pvt_flt4.x*rho_pvt_flt4.x*value_sq_pvt		+ rho_pvt_flt4.y*rho_pvt_flt4.y*value_sq_pvt	+ rho_pvt_flt4.z*rho_pvt_flt4.z;				// sum rho^2, but multiply hue and saturation by value sq TODO Hue is a problem due to wrap arround.Need to changer to the HSVgrad 8 chan colorspace.
			rho_pvt_flt2.x					*= edge_weight;																																		// Weight rho by edges. // TODO choose/ refine which edges to use.
			rho_pvt_flt2.y					=  1.0f;																																			// count the pixels.
			rho_pvt_arr[row_in_block]		+= rho_pvt_flt2;																																	// save to pvt mem for this column
		}
	}
	// Sum-reduce image, /////////////  Save intermediate size ST3 patches for depth map updates, and maximally reduced SE3 patches for pose updates. Second reduce_patch_Rho(..) kernel required for SE3 from lareger image pyramid layers, before update_k2k(..) kernel.
	uint past_frame_idx =0; // TODO remove and restore long outer loop.
	uint step;
	for ( step=1; step<block_size; step *=2){																																					// for each step size, (multiples of 2)
		for (uint block_row=0; block_row<block_size ; block_row += step){																														// step through rows in column
																						rho_pvt_arr[		block_row ]							+=rho_pvt_arr[		block_row + step ];			// sum pair of values in col,
			for (uint se3_dim=0; se3_dim<se3_dof; se3_dim++) {
																						SE3_incr_pvt_arr[	block_row + se3_dim*block_size ]	+=SE3_incr_pvt_arr[	block_row + step + se3_dim*block_size ];
			}

			if( !(fmod((float)lid,(step*2))==0) &&  (fmod((float)lid,step)==0)    ){																											// selects 2nd column, sends data
																						local_rho[			lid-step ]							= rho_pvt_arr[		block_row];
				for (uint se3_dim=0; se3_dim<se3_dof; se3_dim++) {																																// NB integer division. Hence both threads use the same index to local memory.
																						local_SE3_incr[		lid-step + se3_dim*local_size ]		= SE3_incr_pvt_arr[	block_row + se3_dim*block_size ];
				}
			}
			barrier(CLK_LOCAL_MEM_FENCE );																																						// Using barrier as a semaphore, for local mem messages between threads. This minimizes local_mem req, while allowing 2 patch sizes in output, full & ST3 map at out_block_size.

			if( (fmod((float)lid,(step*2))==0)  ){																																				// selects 1st column, adds data. Sum of patch now held in top left element of patch.
																						rho_pvt_arr[		block_row] 							+= local_rho[		lid ];
				for (uint se3_dim=0; se3_dim<se3_dof; se3_dim++) {
																						SE3_incr_pvt_arr[	block_row + se3_dim*block_size ]	+= local_SE3_incr[	lid		 + se3_dim*local_size ];
				}
			}
			barrier(CLK_LOCAL_MEM_FENCE );
		}
		// Save intermediate size ST3 patches for depth map updates, //////////
		if (step==out_block_size/2){																																							// save ST3 map at out_block_size, to use for updating depth_map and rel_vel_map
			uint frame_offset 		= write_index + past_frame_idx * 100 + 25 ;			// NB 100 works for current img size . // stacks frame ST3 maps in adjacent collumns..
			uint write_block_row	= 0;
			if( fmod((float)lid,out_block_size) == 0 ){																																			// selects columns i.e. threads within the workgroup
				for (uint block_row=0; block_row < block_size ; block_row += step*2, write_block_row++){
																						uint offset_1 				= frame_offset		+ write_block_row*mm_cols;
																						Rho_[			offset_1]	= rho_pvt_arr[		block_row ];
					for (uint se3_dim=3; se3_dim<se3_dof; se3_dim++) {																															// select only ST3
																						uint offset_2 				= offset_1			+ (se3_dim-3)*( 4+ (read_rows_/out_block_size) )*mm_cols;
																						uint offset_3 				= block_row			+ se3_dim*block_size;									// NB read_rows_/out_block_size = writre_rows
																						SE3_incr_map_[	offset_2 ]	= SE3_incr_pvt_arr[	offset_3 ];
					}
				}
			}
		}
	}
	/// Save maximally reduced SE3 32x32 patches for pose updates ////////////////////																											// Writes dense blocks. Reduces required transfer to host.
	uint write_block_row			=  0;
	if( fmod((float)lid,block_size) == 0 ){																																						// selects columns i.e. threads within the workgroup
		uint frame_offset_1 		=  write_index_2;																																			// stacks frame SE3 results vertically.
		uint block_row				=  0;
																						uint offset_2 				= frame_offset_1 + write_block_row*mm_cols;
																						Rho_[			offset_2 ]	= rho_pvt_arr[		 block_row ];
		for (uint se3_dim=0; se3_dim<se3_dof; se3_dim++) {																																		// All 6 DoF of SE3
																						uint offset_3 				= offset_2 		+ se3_dim*( 4 + (read_rows_/block_size) )*mm_cols;
																						uint offset_4				= block_row 	+ se3_dim*block_size;
																						SE3_incr_map_[	offset_3 ]	= SE3_incr_pvt_arr[  offset_4 ];
		}
	}
}


__kernel void reduce_patch_Rho(									// call just one workgroup to sum the whole image maps from the patch kernel.
	__private	uint		cols,					//0		// rows and cols in Rho result patch from  __kernel void Rho_sq(), depends on image pyramid layer.
	__private	uint 		rows,					//1
	__private	uint 		row_offset,				//2		// index of 1st pixel of the 2nd patch, i.e. spacing between patches
	__private	uint		thread_offset,			//3		// smallest 2^n > rows * cols NB rows=3, cols=4, -> 12 ->16 for layer 1.  6*8=48 -> 64 for layer 0, where base image has 640*480 pixels. NB for larger images may need a patch approach to update_SE3, to kep each SE3 DoF within
	__private	uint		mm_cols,				//4		// mip_map collumns, i.e. the width of the buffers.

	__global	float2*		Rho_,					//5		// { sum rho^2 ,  count of valid pixels used } Writen to dense patches.
	__global	float2*		SE3_incr_map_,			//6

	__local		float2*		local_Rho_,				//7		// used for sum-reduce. Need to be [groupsize/2], set in host fn.
	__local		float2*		local_SE3_incr_map_		//8
	)
{
	const	uint block_size			= 32;
	const	uint	SE3_DoF			= 6;
	uint	global_id_u				= get_global_id(0);
	float	global_id_f				= global_id_u;
	uint	lid						= get_local_id(0);
	uint	local_group_size		= get_local_size(0);
	uint	group_id				= get_group_id(0);
																								// read in global data : Rho, weights, SE3_incr
																								// NB 10x8 pactch for each SE3.
																								// Read & sum pixels in column, NB img overlap pixel count
	uint	SE3						= global_id_u / thread_offset;								// SE3 = which of 6 DoF does this thread compute.
	uint	read_col				= fmod(global_id_f, thread_offset);
	bool	in_range				= read_col < cols  &&  (SE3 < SE3_DoF);		// NB integer division.		// in_range = Thread is for a pixel in the image.
	const	float2	zero_f2			= {0.0f,0.0f};
	float2	pvt_rho					= zero_f2;
	float2	pvt_incr				= zero_f2;
	float2	pvt_rho_sum				= zero_f2;
	float2	pvt_incr_sum			= zero_f2;

	if( lid < local_group_size/2){																// Zero the local memory. NB half group size.
		local_Rho_[lid]				= zero_f2;
		local_SE3_incr_map_[lid]	= zero_f2;
	}
																								// Sum pixels in the row. // TODO summing weights seems wrong.
	if (in_range){																				// NB initially row_offset = num rows between SE3 output patches.
		uint old_row_offset			= row_offset;
		row_offset					*=(SE3 * mm_cols);											// Adjusts "row_offset" to be buffer index, given width of buffer and which SE3 DoF this thread is for.

		for(uint idx = 0; idx<=rows; idx ++){													// NB patch sum could be used for rows, BUT most iterations have too few rows to justify the ovehead.
			uint idx_2				= read_col + (idx * mm_cols);
			uint idx_3				= row_offset + idx_2;
																								// sum the columns of the fully reduced patch.
			pvt_rho_sum				+= Rho_[idx_2];												// NB incr computation in kernel Rho_sq(..) above.
			pvt_incr_sum			+= SE3_incr_map_[idx_3];
		}
	}
	barrier(CLK_GLOBAL_MEM_FENCE);//////////////////////////////////////////////////////////####################################################
	////////////////////////////////////////////////////////////////////////////////////		// sum reduce  columns of each 10x8 patch (1ayer 1), 5x5 layer 2, 3x3 layer 3, 2x2 layer 4, 1x1 layer 5.
	uint max_iter								= ceil(log2((float)cols));
	uint step									= 2;
	uint old_step								= 1;
	float col									= global_id_u - SE3*thread_offset;

	bool mod_step, mod_step_1, mod_step_2;;
	for ( uint iter=0; iter<max_iter; iter++, step*=2 ){
		uint cols_2 							= cols/step;
		mod_step								= fmod(col				, step)== 0;
		mod_step_1								= fmod((col-old_step)	, step)== 0;
		float upper_lid 						= ceil((float)lid/step);

		if( mod_step_1 && in_range  ){
			local_Rho_[				lid/step]	= pvt_rho_sum;
			local_SE3_incr_map_[	lid/step]	= pvt_incr_sum;
		}
		barrier(CLK_LOCAL_MEM_FENCE);/////////////////////////##########

		if( mod_step  && col+old_step<cols  && in_range ){
			pvt_rho_sum							+= local_Rho_[				lid/step];
			pvt_incr_sum						+= local_SE3_incr_map_[		lid/step];
		}
		barrier(CLK_LOCAL_MEM_FENCE);/////////////////////////##########

		old_step = step;
	}

	if (mod_step && in_range){
		Rho_[SE3]								= pvt_rho_sum;
		SE3_incr_map_[SE3]						= pvt_incr_sum;
	}
} // Need to end the kernel here, because cannot synchronize across workgroups.


__kernel void update_k2k(	// TODO need new kernel, for global synchronization between workgroups.  // Only one workgroup needed
	//inputs
	__private	float2		delta_SE3,				//0

	__global	float2*		Rho_,					//2		// { sum rho^2 ,  count of valid pixels used } Writen to dense patches.
	__global	float2*		SE3_incr_map_,			//3
	__global	float*		old_results,			//4		// size_of(float) * 6 * 4,  for old mag_S3, & old update as well.
	__global	float*		Pose,					//5																								(i) Need to reach zero gradient.
	__global	float*		K,						//6																								(ii) Must reject any step that makes Rho worse.
	__global	float*		inv_K,					//7																								(iii) Rho might not reach zero, but can never be negative.
	//input/output
	__global	float*		k2k						//8
	){
	// compute SE3 update	///////////////////////////////////////////////////////////////////////////////////////////////////
	uint	lid							= get_local_id(0);
	__local float local_update_vec[9];																													// NB elems 6,7,8 hold -1,0,1, for LieToP(..)

	# define UPDATE			0
	# define RHO			6
	# define RESULT			12
	# define MAG_S3			18

	if (lid<6) {
		uint	SE3						= lid;
		float old_update				= old_results[SE3 + UPDATE];																					// will be zero if 1st iteration.
		float old_Rho					= old_results[SE3 + RHO];
		float old_result				= old_results[SE3 + RESULT];
		float old_mag_S3				= old_results[SE3 + MAG_S3];

		float rho						= Rho_[SE3].x / Rho_[SE3].y; 																					// divide by number of pixels overlap, to avoid
		float update					= 0;																											// One thread per SE3 DoF. i.e. the threads to which the sum-reduce answers wwere written.
		float update_pre_clamp			= 0;

		uint  offset					= 3 * floor((float)SE3/3);																						// offset = 0 for SO3, 3 for ST3.	// compute updates. NB different for SO3 vs ST3, hence offset.
		float mag_S3					= fast_length( (float3)(SE3_incr_map_[0+offset].x,	SE3_incr_map_[1+offset].x,	SE3_incr_map_[2+offset].x) );	// fabs(local_Rho_[0+offset].x)  + fabs(local_Rho_[1+offset].x)  + fabs(local_Rho_[2+offset].x);		// float rho_ST3_mag  = fabs(local_Rho_[3].x) + fabs(local_Rho_[4].x) + fabs(local_Rho_[5].x);
		float old_mag_update			= fast_length( (float3)(old_results[0+UPDATE],		old_results[1+UPDATE],		old_results[2+UPDATE]) );
		float result 					= SE3_incr_map_[SE3].x / ( SE3_incr_map_[SE3].y + FLT_EPSILON );												// divide by sum of pixel weights => mean per pixel SE3 increment
		float rho_S3_delta				= ( old_Rho	- rho ) / old_Rho;
		uint option						= 0;

		if (old_update!=0.0f && fabs(rho_S3_delta) < 0.01){																								// step back towards the optimum ? AND reset old_results , such that we can iterate if necessary.
										update 								= -0.5 * old_results[SE3 + UPDATE];								option	=1;
										local_update_vec[SE3]				= update;
										old_results[SE3 + UPDATE]			= -1 * update;
		}else{																																			// NB reset to zero for new img pyramid layer, because old result is not valid for comparison.
			if (old_update==0.0f){
				if ( mag_S3 != 0.0f ) {	update_pre_clamp					= SE3_incr_map_[SE3].x  *  delta_SE3[ SE3/3 ] *0.1  / mag_S3;	option	=2;			// NB  (0.25^2)^(1/6) = 0.102062074... for SE3 6DoF vector length 1/4.
				}else {					update_pre_clamp					= delta_SE3[ SE3/3 ] *0.1;										option	=3;
				}																																		// TODO step size needs to decrease with layer.
			}else{
				if ( rho_S3_delta > 0.01 ){
										update_pre_clamp					= result * old_mag_update  / (old_mag_S3 - mag_S3);				option	=4;
				}else {
										update_pre_clamp					= 0;															option	=5;	// if all 6 SE3 DoF are zero, tells host to stop iterating this level of the image pyramid.
				}
			}
										update								= clamp(	update_pre_clamp,	-delta_SE3[ SE3/3 ],	+delta_SE3[ SE3/3 ] );
										local_update_vec[SE3]				= update;
										old_results[SE3 + UPDATE]			= update;
										old_results[SE3 + RHO]				= rho;
										old_results[SE3 + RESULT]			= result;
										old_results[SE3 + MAG_S3]			= mag_S3;
		}
										printf("\n__kernel void update_SE3()_1	option=%u,	SE3=,%u,	(old_update != 0.0f)=,%i,	old_update=,%f,	SE3_incr_map_[SE3].x=,%f,	SE3_incr_map_[SE3].y=,%f,	result=,%f,	Rho_[SE3].x=,%f,	/ Rho_[SE3].y=,%f,	rho=,%f,	old_Rho=,%f,	rho_S3_delta=,%f,	mag_S3=%f,	update_pre_clamp=,%f,	clamped local_update_vec[SE3]=,%f,	delta_SE3[ SE3/3 ]=,%f,", \
																				option,		SE3,		(old_update != 0.0f),		old_update,		SE3_incr_map_[SE3].x,		SE3_incr_map_[SE3].y,		result,		Rho_[SE3].x,		  Rho_[SE3].y,		rho,		old_Rho,		rho_S3_delta,		mag_S3,		update_pre_clamp,				local_update_vec[SE3], 		delta_SE3[ SE3/3 ]			);
	}																					barrier(CLK_LOCAL_MEM_FENCE);/////##########
	// compute new K2K	////////////////////////////////////////////////////////////////////////////////////////////////////////

	__local float local_K_update[ 	2* SE3_elems];																										// Buffers [32] elem, hold two 4x4 matrices.
	__local float local_pose_inv_K[ 2* SE3_elems];																										// Enable two matrix multiplications simultaneously in one work group of 32 trheads.
	__local float local_A_B[		2* SE3_elems];
	__local float local_k2k[		   SE3_elems];

	if (lid==0){
		printf("\n\n __global Pose[] = \n");
		for (uint j=0; j< 4 ; j++){ for (uint k=0; k< 4 ; k++){ 	printf(",	%f",Pose[	j*4 +k]);	} printf("\n"); } printf("\n\n");
	}																					barrier(CLK_GLOBAL_MEM_FENCE);/////##########

	if (lid < 3) {	local_update_vec[	lid + 6]		= -1.0f + lid;	}																				// sets local_update_vec[6,7,8] to -1, 0, 1
/*
// 	if (lid==0){																		// debugging hack to test LieToP etc
// 		for (int i=0; i<6; i++){ local_update_vec[ i ]	= 0.0f;}
// 		local_update_vec[	0]							= 0.00873;//50;// 0.0873; radians for SO3.  mm for z axis ST3
// 	}
*/
	if (lid < 32){	local_A_B[			lid]			= 0.0f;			}
	if (lid < 16){
					local_K_update[		lid]			= K[lid];
					local_K_update[		lid + 16]		= 0.0f;
					local_pose_inv_K[ 	lid]			= Pose[lid];
					local_pose_inv_K[ 	lid + 16]		= inv_K[lid];
					local_k2k[			lid]			= 0.0f;
	}																					barrier(CLK_LOCAL_MEM_FENCE);/////##########

	if(lid==0){printf("\n\n__kernel void update_k2k(..), \nlocal_update_vec[]={");
		for (int i=0; i<9; i++){ printf("	%f,",local_update_vec[i]);}
		printf("}\nlocal_K_update[]={");
		for (int i=0; i<2* SE3_elems; i++){printf("	%f,",local_K_update[i]);}
		printf("}\n");
	}
	LieToP( 	lid, local_update_vec,	local_K_update );								barrier(CLK_LOCAL_MEM_FENCE);/////##########

	if (lid==0){
		printf("\n\n local_K_update = \n");
		for (uint i=0; i< 2 ; i++){ for (uint j=0; j< 4 ; j++){ for (uint k=0; k< 4 ; k++){		printf(",	%f",local_K_update[		i*16 +j*4 +k]);	} printf("\n"); } printf("\n\n"); }
	}

	update_k2_kdev_fn( lid, local_K_update,	local_pose_inv_K, local_A_B, local_k2k ); 	barrier(CLK_LOCAL_MEM_FENCE);/////##########					// (local_update, local_Pose, local_K, local_inv_K, A, B, local_k2k );
	if(lid<16){ k2k[lid]								= local_k2k[lid]; }				barrier(CLK_LOCAL_MEM_FENCE);/////##########
																																						if(lid==0) {
																																							printf("\nk2k = {\n");
																																							for (uint j=0; j< 4 ; j++){ for (uint k=0; k< 4 ; k++){ 	printf(",	%f",local_k2k[	j*4 +k]);	} printf("\n"); } printf("\n\n");
																																						}
	__local float local_pose[SE3_elems];																												// update the stored pose. TODO tuck this into 2nd stage of void update_k2_kdev_fn(..)
	__local float local_update[SE3_elems];
	__local float local_new_pose[SE3_elems];

	if (lid < 16){
		local_pose[		lid]			= local_pose_inv_K[		lid];
		local_update[	lid]			= local_K_update[	16 + lid];
		local_new_pose[	lid]			= 0.0f;
	}																					barrier(CLK_LOCAL_MEM_FENCE);/////##########
	mat_mul44( lid,		local_pose,		local_update,		local_new_pose );			barrier(CLK_LOCAL_MEM_FENCE);/////##########
	if(lid<16){			Pose[lid]		= local_new_pose [lid]; }

	if (lid==0){
		printf("\n\n local_pose[] = \n");		for (uint j=0; j< 4 ; j++){ 	for (uint k=0; k< 4 ; k++){ 	printf(",	%f",local_pose[		j*4 +k]);	} printf("\n"); 	} printf("\n\n");
		printf("\n\n local_update[] = \n");		for (uint j=0; j< 4 ; j++){ 	for (uint k=0; k< 4 ; k++){ 	printf(",	%f",local_update[	j*4 +k]);	} printf("\n"); 	} printf("\n\n");
		printf("\n\n local_new_pose[] = \n");	for (uint j=0; j< 4 ; j++){ 	for (uint k=0; k< 4 ; k++){ 	printf(",	%f",local_new_pose[	j*4 +k]);	} printf("\n"); 	} printf("\n\n");
	}
}


__kernel void update_maps(  // ? integrate with patch kernel ?


	)
{
	// given global ST3 direction vector, fit depth map


	// given residual of local ST3 map, after depth update, fit rel_vel_map


}

__kernel void update_SE3(){}

//distorsion_update[..]	=  ;
//__global	float*		distorsion_update,		//6




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
	__global	float8* g1p,					//10	// keyframe_g1mem

	// outputs
	__global	float4* Rho_,					//11
	__local		float4*	local_sum_rho_sq,		//12	// 1 DoF, float4 channels
	__global 	float4*	global_sum_rho_sq		//13
	)
 {																										// find gradient wrt SE3 find global sum for each of the 6 DoF
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

	//float SE3_LM_a		= fp32_params[SE3_LM_A];													// Optimisation parameters
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

		float16 k2k_pvt		= k2k[sample];																// NB we read  k2k[1] and  k2k[2]
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
	float4 rho				= zero_v4f;
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
	bool intersection = (u>2) && (u<=read_cols_-2) && (v>2) && (v<=read_rows_-2) 	&& (u2>2) && (u2<=read_cols_-2) && (v2>2) && (v2<=read_rows_-2)  \
						&&  (global_id_u<=layer_pixels) && (inv_depth>=min_inv_depth) && (inv_depth<=max_inv_depth);

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

