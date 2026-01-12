#include "kernels__macros.h"
#include "kernels.h"

// 1st gen patch kernels  //////////////////////////////////////////////////////

__kernel void Rho_sq(								// To be launched with 1 thread per col for 32x32 patches, and an integer multiple of 32 threads.
													// Needs 16 elements of local mem per 32x32 patch, to pass data between threads in recursive square reduction.
													// Needs 32 elem array of private mem per thread.
	__private	uint		layer,					//0
	__private	uint		cols_per_row,			//1
	__private	uint		out_block_size,			//2
//	__private	float2		delta_SE3,				//3

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
	__global	float4*		vel_past_0,				//16	// TODO, relative velocity not used yet. Will use it to modify depth map with timestep for past frames.
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
	//const uint block_size							= 32;										// or send as __private arg ? BUT as hardcoded "const uint" it can be used to size arrays etc.
	//const uint num_SE3_DoF						= 6;
	//const uint num_past_frames					= 4;										// 1,2,4,8,16,32,64 // variable select window of 4 frames.
	const float4 zero_f4							= {0.0f,0.0f,0.0f,0.0f};
	const float2 zero_f2							= {0.0f,0.0f};

	__global float4*	img_past[num_past_frames]	= { img_past_0, img_past_1, img_past_2, img_past_3 };
	__global float4*	vel_past[num_past_frames]	= { vel_past_0, vel_past_1, vel_past_2, vel_past_3 };

	uint  global_id_u 								= get_global_id(0);
	uint  lid 										= get_local_id(0);
	uint  group_id									= get_group_id(0);
	const uint local_size 							= get_local_size(0);

	if (global_id_u < 1 /*num_past_frames*/){printf("\n__kernel void Rho_sq()  past_frame_num= %u,  invk2k buf = \n(%f,	%f,	%f,	%f),	\n(%f,	%f,	%f,	%f),	\n(%f,	%f,	%f,	%f),	\n(%f,	%f,	%f,	%f),	   ",\
		lid, inv_k2k[lid][0], inv_k2k[lid][1], inv_k2k[lid][2], inv_k2k[lid][3], 	inv_k2k[lid][4], inv_k2k[lid][5], inv_k2k[lid][6], inv_k2k[lid][7], 	inv_k2k[lid][8], inv_k2k[lid][9], inv_k2k[lid][10], inv_k2k[lid][11], 	inv_k2k[lid][12], inv_k2k[lid][13], inv_k2k[lid][14], inv_k2k[lid][15] );
	}

	const uint8 mipmap_params_						= mipmap_params[layer];
	uint read_offset_ 								= mipmap_params_[MiM_READ_OFFSET];
	uint read_cols_ 								= mipmap_params_[MiM_READ_COLS];
	uint read_rows_ 								= mipmap_params_[MiM_READ_ROWS];
	uint layer_pixels								= mipmap_params_[MiM_PIXELS];

	uint base_cols									= uint_params[COLS];
	uint mm_cols									= uint_params[MM_COLS];
	uint mm_pixels									= uint_params[MM_PIXELS];

	float min_inv_depth								= fp32_params[MIN_INV_DEPTH];
	float max_inv_depth								= fp32_params[MAX_INV_DEPTH];

	float reduction									= base_cols/read_cols_;
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
	float2 grad_pvt_arr[block_size*num_SE3_DoF]		= {zero_f2};								// pvt variable for values in this column.
	float4 grad_pvt_flt4_SE3[6]						= {zero_f4};

	float4 grad_pvt_flt4							= zero_f4;
	float2 grad_pvt_flt2							= zero_f2;

	float2 SE3_incr_pvt_arr[block_size*num_SE3_DoF]	= {zero_f2};								// pvt variable for values in this column.
	float4 SE3_incr_pvt_flt4						= zero_f4;
	float2 SE3_incr_pvt_flt2						= zero_f2;

	float4 img_cur_pvt[block_size];																// pvt variable for values in this column.
	float8 g1p_pvt[block_size];
	float4 old_px;
	bool   intersection;


	local_rho[lid]									= zero_f2;
	for (uint se3_dim=0; se3_dim<num_SE3_DoF; se3_dim++) {
		local_SE3_incr[lid + se3_dim*local_size]	= zero_f2;
	}
																								// PATCH KERNEL //  TODO need to transfer computation of Huber Norm weighting, Jacobian and Hessian here,
																								// because Hessian must include weights and therefore be updated if weights change.
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

			uint margin					= 4;// * reduction;
			intersection 				= 	(u>margin)			&& (u<=read_cols_-margin)			&& (v>margin)			&& (v<=read_rows_-margin)			&& \
											(u2_flt_1>margin)	&& (u2_flt_1<=read_cols_-margin)	&& (v2_flt_1>margin)	&& (v2_flt_1<=read_rows_-margin)	&& \
											(global_id_u<=layer_pixels)		&&	(inv_depth_1>=min_inv_depth)	&& (inv_depth_1<=max_inv_depth);												// if images overlap
			rho_pvt_flt4				= zero_f4;
			//////////////////////////////////////////////////
			if (intersection){

				// Photometric error rho ///////
				old_px					= bilinear_flt4( img_past[past_frame_idx],  u2_flt_1,  v2_flt_1,  mm_cols,  read_offset_ )	;
				rho_pvt_flt4			= (img_cur_pvt[row_in_block] - old_px) ;
				rho_pvt_flt4.w			= 1.0f;																																					// rho.w holds pixel count.

				// Gradient of pixel value wrt SE3 rotation & translation, taking account of current depth map //////
				SE3_incr_pvt_flt2.y											=  1;
				for (uint se3_dim=0; se3_dim<num_SE3_DoF; se3_dim++) {
					SE3_incr_pvt_flt4										= rho_pvt_flt4 		* 	SE3_grad_map_cur_frame[ read_index_row + (se3_dim * mm_pixels) ] ;
					SE3_incr_pvt_flt2.x										= SE3_incr_pvt_flt4.x;
					SE3_incr_pvt_arr[ se3_dim*block_size + row_in_block ]	= SE3_incr_pvt_flt2 ;
				}
			}
			barrier( CLK_GLOBAL_MEM_FENCE );

			rho_pvt_flt2.x					=  rho_pvt_flt4.x;																																	// Sum Rho
			rho_pvt_flt2.y					=  rho_pvt_flt4.x * rho_pvt_flt4.x;																													// Sum Rho_squared
			rho_pvt_arr[row_in_block]		+= rho_pvt_flt2;																																	// save to pvt mem for this column
		}
	}
	// Sum-reduce image, /////////////  Save intermediate size ST3 patches for depth map updates, and maximally reduced SE3 patches for pose updates. Second reduce_patch_Rho(..) kernel required for SE3 from lareger image pyramid layers, before update_k2k(..) kernel.
	uint past_frame_idx =0; // TODO remove and restore long outer loop.
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
	/// Save maximally reduced SE3 32x32 patches for pose updates ////////////////////																											// Writes dense blocks. Reduces required transfer to host.
	uint write_block_row			=  0;
	if( fmod((float)lid,block_size) == 0 ){																																						// selects columns i.e. threads within the workgroup
		uint frame_offset_1 		=  write_index_2;																																			// stacks frame SE3 results vertically.
		uint block_row				=  0;
																						uint offset_2 				= frame_offset_1 + write_block_row*mm_cols;
																						Rho_[			offset_2 ]	= rho_pvt_arr[		 block_row ];
		for (uint se3_dim=0; se3_dim<num_SE3_DoF; se3_dim++) {																																		// All 6 DoF of SE3
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

// 			printf("\n__kernel void reduce_patch_Rho. 	SE3 = %u, 	idx_3 = %u, pvt_incr_sum = (%f, %f) )", \
// 														SE3, 		idx_3, 		pvt_incr_sum.x, pvt_incr_sum.y );
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
		Rho_[			SE3	+32]			= pvt_rho_sum;	// Set away from reduction for visibility. NB written as floa2, then converted to float4 .tiff
		SE3_incr_map_[	SE3	+32]			= pvt_incr_sum;
	}
} // Need to end the kernel here, because cannot synchronize across workgroups.


__kernel void update_k2k(	// TODO need new kernel, for global synchronization between workgroups.  // Only one workgroup needed
	//inputs
	__private	float2		delta_SE3,				//0
	__private	uint		layer,					//1
	__global	float4*		Hessian_map,			//2		// holds both H_pinv and J whole image results, for each layer.
	__global	float2*		Rho_,					//3		// holds result for current layer in 1st 6 pixels										{ sum rho^2 ,  count of valid pixels used } Writen to dense patches.
	__global	float2*		SE3_incr_map_,			//4		// "					"						"
	__global	float*		old_results,			//5		// size_of(float) * 6 * 4,  for old mag_S3, & old update as well.
	__global	float*		Pose,					//6																								(i) Need to reach zero gradient.
	__global	float*		K,						//7																								(ii) Must reject any step that makes Rho worse.
	__global	float*		inv_K,					//8																								(iii) Rho might not reach zero, but can never be negative.
	//input/output
	__global	float*		k2k						//9
	){
	// compute SE3 update	///////////////////////////////////////////////////////////////////////////////////////////////////
	uint	lid							= get_local_id(0);
	__local float local_update_vec[9];																													// NB elems 6,7,8 hold -1,0,1, for LieToP(..)

	# define UPDATE			0
	# define RHO			6
	# define RESULT			12
	# define MAG_S3			18

	// Inverse compositional Lucas-Kanade update (IC-LK)
	// Load whole img H_pinv & J
	float4		J[		num_SE3_DoF];
	float4		H_pinv[	num_SE3_DoF][	num_SE3_DoF];
	for (uint i=0; i<num_SE3_DoF; i++) {									J[i] 			= Hessian_map[i 			+ layer*8*6];	}				// Each thread holds private copy of whole of J & H_pinv. NB float4, so must choose color channel, + have pixel count in J.w, H_pinv.w .
	for (uint i=0; i<num_SE3_DoF; i++) {for (uint j=0; j<num_SE3_DoF; j++)	H_pinv[i][j] 	= Hessian_map[i*6 + j + 6	+ layer*8*6];	}
/*
	Matx66f invHessian = current_frames[ current_frames_idx[0] ].invHessian[layer];
	float	sum_Rho, num_pixels;															// NB num_pixels should be only for img overlap => update each iteration.
	Matx61f	sum_Rho_J;																		// pixelwise:  Rho * J
	Matx61f	sum_J;																			// J = SE3_grad_map * img_grad
	Matx61f	pose_update	= invHessian * ( sum_Rho_J  - (sum_Rho * sum_J) )/ num_pixels;
*/
	float2	Rho				= Rho_[0];
	float	num_pixels		= Rho.s1;
	float	rho				= Rho.s0/Rho.s1;
	__local float 	pose_update_J[num_SE3_DoF];
	__local float 	pose_update_H[num_SE3_DoF*num_SE3_DoF];

	if (lid<6) {
		pose_update_J[lid]			=	SE3_incr_map_[lid].s0	-	(  rho  *  J[lid].x );													//  ( sum_Rho_J  - (sum_Rho * sum_J) )/ num_pixels				// each part needs to be divided by the correct "num_pixels" wole img, or just overlap pixels.
		pose_update_H[lid]			=	0.0f;
	}																					barrier(CLK_LOCAL_MEM_FENCE);/////##########
	if (lid<36) {																																		// multiply by pseudo-inverse of Hessian for the whole image.
		uint 	elem				=	lid/num_SE3_DoF;
		float	pose_update_		=	pose_update_J[	elem];
		float	H_elem				=	Hessian_map[	lid + 6	+ layer*8*6].x;
		pose_update_H[lid]			=	H_elem * pose_update_;
	}																					barrier(CLK_LOCAL_MEM_FENCE);/////##########
	float	pose_update				=	0.0f;
	if (lid<6) {
		for(uint i=0; i<num_SE3_DoF; i++) pose_update += pose_update_H[lid*6 +i];
	}																					barrier(CLK_LOCAL_MEM_FENCE);/////##########
	if (lid<6) {
		pose_update_J[lid]			=	pose_update;
		local_update_vec[lid]		=	-pose_update;
	}																					barrier(CLK_LOCAL_MEM_FENCE);/////##########

	if(lid==0)printf("\n__kernel void update_k2k(..) Layer=%u,  IC-LK Pose update = %f, %f, %f		%f, %f, %f", layer, pose_update_J[0], pose_update_J[1], pose_update_J[2], pose_update_J[3], pose_update_J[4], pose_update_J[5] );
	// End IC-LK //////////////////////////////////////////

/*
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
*/
	// compute new K2K	////////////////////////////////////////////////////////////////////////////////////////////////////////

	__local float local_K_update[ 	2* SE3_elems];																										// Buffers [32] elem, hold two 4x4 matrices.
	__local float local_pose_inv_K[ 2* SE3_elems];																										// Enable two matrix multiplications simultaneously in one work group of 32 trheads.
	__local float local_A_B[		2* SE3_elems];
	__local float local_k2k[		   SE3_elems];
																																						if (lid==0){
																																							printf("\n\n __global Pose[] = \n"); for (uint j=0; j< 4 ; j++){ for (uint k=0; k< 4 ; k++){ 	printf(",	%f",Pose[	j*4 +k]);	} printf("\n"); } printf("\n\n");
																																						}
																						barrier(CLK_GLOBAL_MEM_FENCE);/////##########

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
																																						if(lid==0){
																																							printf("\n\n__kernel void update_k2k(..), \nlocal_update_vec[]={"); for (int i=0; i<9; i++){ printf("	%f,",local_update_vec[i]); }
																																							printf( "}\nlocal_K_update[]={"); for (int i=0; i<2* SE3_elems; i++){ printf("	%f,",local_K_update[i]); }		printf("}\n");
																																						}
	// pose_update_J
	LieToP( 	lid,  local_update_vec,	local_K_update );								barrier(CLK_LOCAL_MEM_FENCE);/////##########
																																						if(lid==0){
																																							printf("\n\n local_K_update = \n"); for (uint i=0; i< 2 ; i++){ for (uint j=0; j< 4 ; j++){ for (uint k=0; k< 4 ; k++){ printf(",	%f",local_K_update[		i*16 +j*4 +k]); } printf("\n"); } printf("\n\n"); }
																																						}
	update_k2_kdev_fn( lid, local_K_update,	local_pose_inv_K, local_A_B, local_k2k ); 	barrier(CLK_LOCAL_MEM_FENCE);/////##########					// (local_update, local_Pose, local_K, local_inv_K, A, B, local_k2k );
	if(lid<16){ k2k[lid]								= local_k2k[lid]; }				barrier(CLK_LOCAL_MEM_FENCE);/////##########
																																						if(lid==0){
																																							printf("\nk2k = {\n"); for (uint j=0; j< 4 ; j++){ for (uint k=0; k< 4 ; k++){ 	printf(",	%f",local_k2k[	j*4 +k]);	} printf("\n"); } printf("\n\n");
																																						}
	__local float local_pose[SE3_elems];																												// update the stored pose. TODO tuck this into 2nd stage of void update_k2_kdev_fn(..)
	__local float local_update[SE3_elems];
	__local float local_new_pose[SE3_elems];

	if (lid < 16){
					local_pose[			lid]			= local_pose_inv_K[		lid];
					local_update[		lid]			= local_K_update[	16 + lid];
					local_new_pose[		lid]			= 0.0f;
	}																					barrier(CLK_LOCAL_MEM_FENCE);/////##########
	mat_mul44( 		lid,	local_pose,	local_update,	  local_new_pose );				barrier(CLK_LOCAL_MEM_FENCE);/////##########
	if(lid<16){		Pose[				lid]			= local_new_pose [lid]; }
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
 /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////end of patch based kernels
