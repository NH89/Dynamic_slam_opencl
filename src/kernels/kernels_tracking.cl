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

