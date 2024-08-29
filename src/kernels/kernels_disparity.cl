#include "kernels_macros.h"
#include "kernels.h"

__kernel void covariance(
	// inputs
	__private	uint	layer,					//0
	__private	uint	local_num_samples,		//1
	__private	uint	se3_sum_size,			//2

	__constant 	uint8*	mipmap_params,			//3
	__constant 	uint*	uint_params,			//4
	__constant  float*  fp32_params,			//5
	__global 	float4*	img_cur,				//7		// keyframe
	__global 	float4*	img_new,				//8
	__global	float2* warp,					//9		// NB keyframe GT_depth, now stored as inv_depth
	__global	float8* g1p,					//10		// keyframe_g1mem

	// outputs
	__global	float4* Rho_,					//11
	__local		float4*	local_rho,				//12
	__global 	float4*	global_sum_rho_sq		//13
	)
 {																									// find gradient wrt SE3 find global sum for each of the 6 DoF
		uint  global_id_u 	= get_global_id(0);
	float global_id_flt = global_id_u;
	uint  lid 			= get_local_id(0);
																														//if(global_id_u == 1  ){ printf("\n__kernel void se3_LK_grad (global_id_u == 1 )  chk_1"   ); }
	uint local_size 	= get_local_size(0); // / wg_divisor;
	uint group_size 	= local_size;
	uint num_groups		= get_num_groups(0); //size_t get_num_groups (uint dimindx)
	uint work_dim 		= get_work_dim();
	uint global_size	= get_global_size(0);

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
	uint v 				= global_id_u / read_cols_;																		// read_row
	uint u 				= fmod(global_id_flt, read_cols_);																// read_column
	float u_flt			= u * reduction;																				// NB this causes sparse sampling of the original space, to use the same k2k at every scale.
	float v_flt			= v * reduction;
	uint read_index 	= read_offset_  +  v  * mm_cols  + u ;
	float alpha			= img_cur[read_index].w;
                                                                                                                        // sample the images /////////////////////////////////
    uint u_minus        = u-1;
    uint u_plus         = u+1;
    uint v_minus        = v-1;
    uint v_plus         = v+1;

    uint u_sample[5]    = { u, u_minus, u, u_plus, u };
    uint v_sample[5]    = { v, v_minus, v, v_plus, v };

    float4 img_cur_sample[5];
    float4 img_new_sample[5];
                                                                                                                        // sample img_cur /////////////////////////////////
    float2 warp_        = warp[read_index];

    for (int i=0; i<5; i++){                                                                                        // wrong, we will need to sample the whole NxN patch for each of these 5 maps.
        uint read_index 	= read_offset_  +  v_sample[i]  * mm_cols  + u_sample[i] ;
        img_cur_sample[i]   = img_cur[read_index];
        img_cur_sample[i].w = alpha;
    }
                                                                                                                        // variance img_cur /////////////////////////////////
    for (int i=0; i<5; i++){


        img_cur_sample[i] * img_cur_sample[i]



    }




                                                                                                                        // disparity loop /////////////////////////////////
    #define DISPARITY_ITERATIONS 7
    for (int i=0; i< DISPARITY_ITERATIONS; i++){


        for (int i=0; i<5; i++){
            uint read_index 	= read_offset_  + ( warp_.y +  v_sample[i] )  *  mm_cols  +  warp_.x + u_sample[i] ;
            img_cur_sample[i]   = img_new[read_index];
            img_new_sample[i].w = alpha;
        }


                                                                                                                        // variance img_new /////////////////////////////////



                                                                                                                        // covariance /////////////////////////////////



                                                                                                                        // cross-correlation /////////////////////////////////



                                                                                                                        // compute optimal warp /////////////////////////////////



                                                                                                                        // clip the warp /////////////////////////////////



        barrier(CLK_LOCAL_MEM_FENCE);
                                                                                                                        // smooth/refine warp /////////////////////////////////


        barrier(CLK_LOCAL_MEM_FENCE);
                                                                                                                        // warp img new /////////////////////////////////



                                                                                                                        // save img new to global /////////////////////////////////
        barrier(CLK_LOCAL_MEM_FENCE);
    }














 }
