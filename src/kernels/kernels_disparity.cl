#include "kernels_macros.h"
#include "kernels.h"

__kernel void disparity(
	// inputs
	__private	uint	layer,					//0

	__constant 	uint8*	mipmap_params,			//1
	__constant 	uint*	uint_params,			//2
	__constant  float*  fp32_params,			//3

	__global 	float4*	img_cur,				//4		// keyframe
	__global 	float4*	img_new,				//5
	//__global	float2* warp,					//6
	__global	float8* g1p,					//6		// keyframe_g1mem
/*
														// const uint wg_divisor = 4;	//  512 * 20 * 32 / 8 =  40,960 bytes
																						// 1024 * 20 * 32 /

														// Intel(R) Iris(R) Xe Graphics :
														// Local memory size  65,536 (64KiB),
														// Max work item dimensions 3,
														// Max work item sizes 512x512x512,
														// Max work group size 512.

														// NVIDIA GeForce GTX 970M
														// Local memory size               49,152 (48KiB)
														// Max work item dimensions        3
														// Max work item sizes             1024x1024x64
														// Max work group size             1024
*/
	__local	 	float4*	local_img_cur, 			//7		// local_work_size*4*5/wg_divisor*sizeof(float)
	__local	 	float4*	local_img_new, 			//8
	__local	 	float4*	local_img_cur_sq, 		//9
	__local	 	float4*	local_img_new_sq, 		//10

	// outputs
	__global	float4* Rho_,					//11
	__global 	float4*	disparity				//12
	)
 {																														// find gradient wrt SE3 find global sum for each of the 6 DoF
	uint  global_id_orig	= get_global_id(0);
	uint  lid 				= get_local_id(0);
	uint  group_id 			= get_group_id(0);
																														//if(global_id_u == 1  ){ printf("\n__kernel void se3_LK_grad (global_id_u == 1 )  chk_1"   ); }
	const uint local_size 	= get_local_size(0); // / wg_divisor;
	const uint patch_length	= local_size;

	uint group_size 		= local_size;
	uint num_groups			= get_num_groups(0); //size_t get_num_groups (uint dimindx)
	uint work_dim 			= get_work_dim();
	//uint global_size		= get_global_size(0);

	const uint halo_width	= 2;
	uint  global_id_u		= group_id * (local_size - 2*halo_width)  + lid;												// new global_id takes acount of halo on local memory.
	float global_id_flt 	= global_id_u;

	uint8 mipmap_params_	= mipmap_params[layer];
	uint read_offset_ 		= mipmap_params_[MiM_READ_OFFSET];
	uint read_cols_ 		= mipmap_params_[MiM_READ_COLS];
	//uint read_rows_ 		= mipmap_params_[MiM_READ_ROWS];
	uint layer_pixels		= mipmap_params_[MiM_PIXELS];
{
	//uint8 mipmap_params_0	= mipmap_params[0];
	//uint read_offset_0 	= mipmap_params_0[MiM_READ_OFFSET];

	//uint base_cols		= uint_params[COLS];
	//uint margin 			= uint_params[MARGIN];
 }
	uint mm_cols			= uint_params[MM_COLS];
	uint mm_pixels			= uint_params[MM_PIXELS];
{
	//float inv_d_step 		= fp32_params[INV_DEPTH_STEP];
	//float min_inv_depth 	= fp32_params[MIN_INV_DEPTH]; // + inv_d_step;
	//float max_inv_depth 	= fp32_params[MAX_INV_DEPTH]; // - inv_d_step;

	//uint reduction			= mm_cols/read_cols_;
 }
	uint v 					= global_id_u / read_cols_;																		// read_row
	uint u 					= fmod(global_id_flt, read_cols_);																// read_column
	float u_flt				= u ;																							// NB no reduction, as we are not using K2Kin this kernel.
	float v_flt				= v ;
	uint read_index 		= read_offset_  +  v  * mm_cols  + u ;
	float alpha				= img_cur[read_index].w;
{                                                                                                                      // sample the images /////////////////////////////////
//     uint u_minus        	= u-1;
//     uint u_plus         	= u+1;
//     uint v_minus        	= v-1;
//     uint v_plus         	= v+1;
//
//     uint u_sample[5]    	= { u, u_minus, u, u_plus, u };
//     uint v_sample[5]    	= { v, v_minus, v, v_plus, v };
//
//     float4 img_cur_sample[5];
//     float4 img_new_sample[5];
 }                                                                                                                    // sample img_cur /////////////////////////////////
	if (global_id_u == 150){
		printf ("\n__kernel disparity(..) chk_1 layer=%u, global_id=%u, group_id=%u, lid=%u, u=%u, v=%u, read_index=%u, read_offset_=%u, mm_cols=%u, img_cur[read_index].x=%f, global_id_orig=%u, local_size=%u, halo_width=%u", \
		layer, global_id_u, group_id, lid, u, v, read_index, read_offset_, mm_cols, img_cur[read_index].x, global_id_orig, local_size, halo_width  );
	}

	for (int i=0; i<1+2*halo_width; i++){																				// Load local_img_patch_cur  /////////////////////////////////
		local_img_cur[lid + i*patch_length] 	= img_cur[ read_index +i*mm_cols];
	}

	for (int i=0; i<1+2*halo_width; i++){																				// Square local_img_patch_cur  /////////////////////////////////
		float4 pix_val 							= local_img_cur[lid + i*patch_length];
		local_img_cur_sq[lid + i*patch_length] 	=  pix_val *  pix_val;
	}

	float W[9] = { 1,2,1,2,4,2,1,2,1 }; 																				// 3x3 discrete Gaussian kernel
	int i_u[5] = {0,-1,0,1,0};
	int i_v[5] = {1,0,0,0,-1};

	float4 variance_curr 			= 0;
	float4 variance_new[5] 			= {0};
	float4 covariance[5] 			= {0};
	float4 cross_correlation[5] 	= {0};
	float4 disparity_pvt			= disparity[read_index];
	float  warp_u					= disparity_pvt.x;
	float  warp_v					= disparity_pvt.y;
	float  warp_incr_u 				= 0;
	float  warp_incr_v 				= 0;

	barrier(CLK_LOCAL_MEM_FENCE);////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

																														// compute variance img_cur
	//if (lid>halo_width && lid<local_size - halo_width) {	// NB original image local mem has a 1 pixel margin around it, for 3x3 samples.
		for (int j=0; j<3; j++){
			for (int k=0; k<3; k++){
				variance_curr += local_img_cur_sq [lid + j*patch_length + k] * W[(j)*3 + k];
			}
		}
	//}

	if (global_id_u == 150){
		printf ("\n__kernel disparity(..) chk_2 layer=%u, global_id=%u, group_id=%u, lid=%u, u=%u, v=%u, read_index=%u, read_offset_=%u, mm_cols=%u, img_cur[read_index].x=%f, global_id_orig=%u, local_size=%u, halo_width=%u", \
		layer, global_id_u, group_id, lid, u, v, read_index, read_offset_, mm_cols, img_cur[read_index].x, global_id_orig, local_size, halo_width  );
	}

	barrier(CLK_LOCAL_MEM_FENCE);
{
	//if (lid>halo_width  && lid=<local_size - halo_width) {
// 		if (read_index< mm_pixels) {
// 			uint j = 0, k=0;
// 			Rho_[read_index]		= local_img_cur_sq [lid + j*patch_length + k];
// 			 j=1; k=1;
// 			disparity[read_index]	= local_img_cur_sq [lid + j*patch_length + k];
//
// 			 j=2; k=2;
// 			Rho_[read_index - (650*480) ]		= local_img_cur_sq [lid + j*patch_length + k];
// 			disparity[read_index - (650*480)]	= variance_curr;
// 		}
	//}
 }
                                 // 7                                                                                       // disparity loop /////////////////////////////////
    #define DISPARITY_ITERATIONS 1
    for (int iter=0; iter< DISPARITY_ITERATIONS; iter++){

		for (int i=0; i<1+4*halo_width; i++){																			// Square local_img_patch_new  // interpollate sampling of img_new  /////////////////////////////////
			//local_img_new[lid + i*patch_length] = bilinear_flt4 ( img_new, u_flt + warp_u, v_flt + warp_v,  mm_cols,  read_offset_);
		}
		for (int i=0; i<1+4*halo_width; i++){																			// Square local_img_patch_new  /////////////////////////////////
			float4 pix_val 							= local_img_new[lid + i*patch_length];
			//local_img_new_sq[lid + i*patch_length] 	= pix_val *  pix_val;
		}
{
// 		if (read_index< mm_pixels) {
// 			uint j = 0, k=0;
// 			Rho_[read_index]					= local_img_new_sq[lid ];
// 			 j=2; k=2;
// 			disparity[read_index]				= local_img_new_sq[lid + 3*patch_length];
//
// 			 j=4; k=4;
// 			Rho_[read_index - (650*480) ]		= local_img_new_sq[lid + 4*patch_length];
// 			 j=5; k=5;
// 			disparity[read_index - (650*480)]	= local_img_new_sq[lid + 5*patch_length];
// 		}
}
        // variance img_new /////////////////////////////////
        //if (lid>halo_width && lid<local_size - halo_width) {

			for (int i=0; i<5; i++){
				for (int j=0; j<3; j++ ){
					for (int k=0; k<3; k++){
						//variance_new[i] += local_img_new_sq [ lid + (i_v[i] + j)*patch_length  +  i_u[i] + k ]    * W[j*3 + k];
					}
				}
			}
																															// covariance /////////////////////////////////
			for (int i=0; i<5; i++){
				for (int j=0, j_=-1; j<3; j++, j_++){
					for (int k=0, k_=-1; k<3; k++, k_++){
						//covariance[i] +=  local_img_cur [lid + j_*patch_length + k]  *  local_img_new [ lid + (i_v[i] + j)*patch_length + i_u[i] + k ]    * W[j*3 + k];
					}
				}
			}
																															// cross-correlation /////////////////////////////////
			for (int i=0; i<5; i++){
				//cross_correlation[i] = covariance[i] / ( variance_curr *  variance_new[i] ) ;
			}
																															// compute optimal warp /////////////////////////////////
			warp_incr_u = compute_optimum( cross_correlation[1], cross_correlation[2], cross_correlation[3] );		// TODO  do I really want float4 OR should I reduce it to float ?
			warp_incr_v = compute_optimum( cross_correlation[0], cross_correlation[2], cross_correlation[4] );
																															// clip the warp /////////////////////////////////
// 			if (read_index< mm_pixels) {
// 				Rho_[		read_index]				= warp_incr_u;
// 				disparity[	read_index]				= warp_incr_v;
// 			}

			if( iter > 5){
				warp_incr_u = clamp( warp_incr_u, -1.0f, 1.0f );
				warp_incr_v = clamp( warp_incr_v, -1.0f, 1.0f );
			}
		//}
        barrier(CLK_LOCAL_MEM_FENCE);
		if (global_id_u == 150){
			printf ("\n__kernel disparity(..) chk_3 iter=%u, layer=%u, global_id=%u, group_id=%u, lid=%u, u=%u, v=%u, read_index=%u, read_offset_=%u, mm_cols=%u, img_cur[read_index].x=%f, global_id_orig=%u, local_size=%u, halo_width=%u", \
			iter, layer, global_id_u, group_id, lid, u, v, read_index, read_offset_, mm_cols, img_cur[read_index].x, global_id_orig, local_size, halo_width  );
	}
{
// 		if (read_index< mm_pixels) {
// 			uint j = 0, k=0;
// 			Rho_[read_index]						= variance_new[1]; // local_img_new_sq [lid + j*patch_length + k];
// 			 j=1; k=1;
// 			disparity[read_index]					= variance_new[2]; // local_img_new_sq [lid + j*patch_length + k];
// 			 j=2; k=2;
// 			Rho_[read_index - (650*480) ]			= variance_new[3]; // local_img_new_sq [lid + j*patch_length + k];
// 			disparity[read_index - (650*480)]		= variance_new[4]; // variance_new[0];
// 		}


// 		if (read_index< mm_pixels) {
// 			uint i 		= 0;
// 			Rho_[		read_index]					= cross_correlation[0]; // variance_new[i];
// 			disparity[	read_index]					= cross_correlation[1]; // variance_curr;
//
// 			Rho_[		read_index - (650*480) ]	= cross_correlation[2]; // cross_correlation[i];
// 			disparity[	read_index - (650*480) ]	= cross_correlation[3]; // cross_correlation[4]; //warp_incr_u;
// 		}

// 		if (read_index< mm_pixels) {
// 			Rho_[		read_index - (650*480) ]	= warp_incr_u;
// 			disparity[	read_index - (650*480) ]	= warp_incr_v;
// 		}
}
                                                                                                                        // smooth/refine warp /////////////////////////////////
																														// TODO determine confidence based, edge preserving smoothing.
        barrier(CLK_LOCAL_MEM_FENCE);

		if (global_id_u == 150){
			printf ("\n__kernel disparity(..) chk_4 layer=%u, global_id=%u, group_id=%u, lid=%u, u=%u, v=%u, read_index=%u, read_offset_=%u, mm_cols=%u, img_cur[read_index].x=%f, global_id_orig=%u, local_size=%u, halo_width=%u", \
			layer, global_id_u, group_id, lid, u, v, read_index, read_offset_, mm_cols, img_cur[read_index].x, global_id_orig, local_size, halo_width  );
		}
                                                                                                                        // warp img new /////////////////////////////////
		warp_u -= warp_incr_u;	// TODO direcction? Why does it accumulate noise + atefacts ?
		warp_v -= warp_incr_v;
    }
																														// save img new to global /////////////////////////////////
// 	if ( ( global_id_u<layer_pixels) && (read_index< mm_pixels) && (lid>=halo_width) && (lid<local_size - halo_width) ) {
// 		float4 Rho_pvt			= local_img_cur[lid + (1+halo_width)*patch_length]  -  bilinear_flt4 ( img_new, u + warp_u, v + warp_v,  mm_cols,  read_offset_); // img_cur[read_index] - img_new[read_index]; //
// 		Rho_pvt.w				= 1.0f;
// 		//Rho_[read_index]		= Rho_pvt;
// 		float4 disparity_pvt 	= {warp_u, warp_v, 0, alpha};
// 		//disparity[read_index]	= disparity_pvt;
// 	}
 }



 __kernel void disparity2(
	// inputs
	__private	uint	layer,					//0

	__constant 	uint8*	mipmap_params,			//1
	__constant 	uint*	uint_params,			//2
	__constant  float*  fp32_params,			//3

	__global 	float4*	img_cur,				//4		// keyframe
	__global 	float4*	img_new,				//5
	//__global	float2* warp,					//6
	__global	float8* g1p,					//6		// keyframe_g1mem
/*
														// const uint wg_divisor = 4;	//  512 * 20 * 32 / 8 =  40,960 bytes
																						// 1024 * 20 * 32 /

														// Intel(R) Iris(R) Xe Graphics :
														// Local memory size  65,536 (64KiB),
														// Max work item dimensions 3,
														// Max work item sizes 512x512x512,
														// Max work group size 512.

														// NVIDIA GeForce GTX 970M
														// Local memory size               49,152 (48KiB)
														// Max work item dimensions        3
														// Max work item sizes             1024x1024x64
														// Max work group size             1024
*/
	__local	 	float4*	local_img_cur, 			//7		// local_work_size*4*5/wg_divisor*sizeof(float)
	__local	 	float4*	local_img_new, 			//8
	__local	 	float4*	local_img_cur_sq, 		//9
	__local	 	float4*	local_img_new_sq, 		//10

	// outputs
	__global	float4* Rho_,					//11
	__global 	float4*	disparity				//12
	)
 {																														// find gradient wrt SE3 find global sum for each of the 6 DoF
	uint  global_id_orig	= get_global_id(0);
	uint  lid 				= get_local_id(0);
	uint  group_id 			= get_group_id(0);
																														//if(global_id_u == 1  ){ printf("\n__kernel void se3_LK_grad (global_id_u == 1 )  chk_1"   ); }
	const uint local_size 	= get_local_size(0); // / wg_divisor;
	const uint patch_length	= local_size;

	uint group_size 		= local_size;
	uint num_groups			= get_num_groups(0); //size_t get_num_groups (uint dimindx)
	uint work_dim 			= get_work_dim();

	const uint halo_width	= 2;
	uint  global_id_u		= group_id * (local_size - 2*halo_width)  + lid;												// new global_id takes acount of halo on local memory.
	float global_id_flt 	= global_id_u;

	uint8 mipmap_params_	= mipmap_params[layer];
	uint read_offset_ 		= mipmap_params_[MiM_READ_OFFSET];
	uint read_cols_ 		= mipmap_params_[MiM_READ_COLS];
	uint layer_pixels		= mipmap_params_[MiM_PIXELS];

	uint mm_cols			= uint_params[MM_COLS];
	uint mm_pixels			= uint_params[MM_PIXELS];

	uint v 					= global_id_u / read_cols_;																		// read_row
	uint u 					= fmod(global_id_flt, read_cols_);																// read_column
	float u_flt				= u ;																							// NB no reduction, as we are not using K2Kin this kernel.
	float v_flt				= v ;
	uint read_index 		= read_offset_  +  v  * mm_cols  + u ;
	float alpha				= img_cur[read_index].w;
                                                                                                                        // sample the images /////////////////////////////////
                                                                                                                        // sample img_cur /////////////////////////////////
	if (global_id_u == 150){
		printf ("\n__kernel disparity(..) chk_1 layer=%u, global_id=%u, group_id=%u, lid=%u, u=%u, v=%u, read_index=%u, read_offset_=%u, mm_cols=%u, img_cur[read_index].x=%f, global_id_orig=%u, local_size=%u, halo_width=%u", \
		layer, global_id_u, group_id, lid, u, v, read_index, read_offset_, mm_cols, img_cur[read_index].x, global_id_orig, local_size, halo_width  );
	}

	for (int i=0; i<1+2*halo_width; i++){																				// Load local_img_patch_cur  /////////////////////////////////
		local_img_cur[lid + i*patch_length] 	= img_cur[ read_index +i*mm_cols];
	}

	for (int i=0; i<1+2*halo_width; i++){																				// Square local_img_patch_cur  /////////////////////////////////
		float4 pix_val 							= local_img_cur[lid + i*patch_length];
		local_img_cur_sq[lid + i*patch_length] 	=  pix_val *  pix_val;
	}

	float W[9] = { 1,2,1,2,4,2,1,2,1 }; 																				// 3x3 discrete Gaussian kernel
	int i_u[5] = {0,-1,0,1,0};
	int i_v[5] = {1,0,0,0,-1};

	float4 variance_curr 			= 0;
	float4 variance_new[5] 			= {0};
	float4 covariance[5] 			= {0};
	float4 cross_correlation[5] 	= {0};
	float4 disparity_pvt			= disparity[read_index];
	float  warp_u					= disparity_pvt.x;
	float  warp_v					= disparity_pvt.y;
	float  warp_incr_u 				= 0;
	float  warp_incr_v 				= 0;

	barrier(CLK_LOCAL_MEM_FENCE);////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

																														// compute variance img_cur
	//if (lid>halo_width && lid<local_size - halo_width) {	// NB original image local mem has a 1 pixel margin around it, for 3x3 samples.
		for (int j=0; j<3; j++){
			for (int k=0; k<3; k++){
				variance_curr += local_img_cur_sq [lid + j*patch_length + k] * W[(j)*3 + k];
			}
		}
	//}

	if (global_id_u == 150){
		printf ("\n__kernel disparity(..) chk_2 layer=%u, global_id=%u, group_id=%u, lid=%u, u=%u, v=%u, read_index=%u, read_offset_=%u, mm_cols=%u, img_cur[read_index].x=%f, global_id_orig=%u, local_size=%u, halo_width=%u", \
		layer, global_id_u, group_id, lid, u, v, read_index, read_offset_, mm_cols, img_cur[read_index].x, global_id_orig, local_size, halo_width  );
	}

	barrier(CLK_LOCAL_MEM_FENCE);
                                 // 7                                                                                       // disparity loop /////////////////////////////////
     #define DISPARITY_ITERATIONS 1
     for (int iter=0; iter< DISPARITY_ITERATIONS; iter++){

		 for (int i=0; i<1+4*halo_width; i++){																				// Square local_img_patch_new  // interpollate sampling of img_new  /////////////////////////////////
			local_img_new[lid + i*patch_length] = bilinear_flt4 ( img_new, u_flt + warp_u, v_flt + warp_v,  mm_cols,  read_offset_);
		}
		for (int i=0; i<1+4*halo_width; i++){																			// Square local_img_patch_new  /////////////////////////////////
			float4 pix_val 							= local_img_new[lid + i*patch_length];
			local_img_new_sq[lid + i*patch_length] 	= pix_val *  pix_val;
		}
//
//
//         // variance img_new /////////////////////////////////
//
// 																															// compute optimal warp /////////////////////////////////
 			warp_incr_u = compute_optimum( cross_correlation[1], cross_correlation[2], cross_correlation[3] );		// TODO  do I really want float4 OR should I reduce it to float ?
 			warp_incr_v = compute_optimum( cross_correlation[0], cross_correlation[2], cross_correlation[4] );
// 																															// clip the warp /////////////////////////////////
//
 			if( iter > 5){
 				warp_incr_u = clamp( warp_incr_u, -1.0f, 1.0f );
 				warp_incr_v = clamp( warp_incr_v, -1.0f, 1.0f );
 			}
//
//         barrier(CLK_LOCAL_MEM_FENCE);
 		if (global_id_u == 150){
 			printf ("\n__kernel disparity(..) chk_3 iter=%u, layer=%u, global_id=%u, group_id=%u, lid=%u, u=%u, v=%u, read_index=%u, read_offset_=%u, mm_cols=%u, img_cur[read_index].x=%f, global_id_orig=%u, local_size=%u, halo_width=%u", \
 			iter, layer, global_id_u, group_id, lid, u, v, read_index, read_offset_, mm_cols, img_cur[read_index].x, global_id_orig, local_size, halo_width  );
 		}
//
//                                                                                                                         // warp img new /////////////////////////////////
 		warp_u -= warp_incr_u;	// TODO direcction? Why does it accumulate noise + atefacts ?
 		warp_v -= warp_incr_v;
     }
																														// save img new to global /////////////////////////////////
	if ( ( global_id_u<layer_pixels) && (read_index< mm_pixels) && (lid>=halo_width) && (lid<local_size - halo_width) ) {
		float4 Rho_pvt			= local_img_cur[lid + (1+halo_width)*patch_length]  -  bilinear_flt4 ( img_new, u + warp_u, v + warp_v,  mm_cols,  read_offset_); // img_cur[read_index] - img_new[read_index]; //
		Rho_pvt.w				= 1.0f;
		Rho_[read_index]		= Rho_pvt;
		float4 disparity_pvt 	= {warp_u, warp_v, 0, alpha};
		disparity[read_index]	= disparity_pvt;
	}
 }

float compute_optimum(__private float4 A, __private float4 B, __private float4 C){
	float 	steps[3]  = {-1, 0, 1};
	float a, b, c,   d, e,   f, g,   h, i,   j, k, d2, f2, h2, prediction, optimum;																		// compute x value of the optimum of parabola, y= a*x*x + b*x + c
																																			// given samples at x=1,2,4
	d = steps[0];	e = A.x ;		// TODO  which combination of color channels ?
	f = steps[1];	g = B.x ;
	h = steps[2];	i = C.x ;

	d2 = d * d;
	f2 = f * f;
	h2 = h * h;

	j = (f2 - d2)*(f-h) - (h2 - f2)*(d-f);
	k = (g-i)*(d-f) - (e-g)*(f-h);
	a = k/j;
	b = (e-g +a*(f2-d2))  /  (d-f);
	c = e - a*d2 - b*d;

	if (a>0){																																// IF concavity leads to a minimum, use it.
		float x 		= -b /(2*a);
		prediction 		= a*(x*x) + b*x + c;
		optimum 		= x;
	}else{																																	// IF concavity leads to a maximum, pick the best sample so far.
		if (e>=i){
			prediction 	= i;
			optimum 	= h;
		}else{
			prediction 	= e;
			optimum 	= d;
		}
	}
	return optimum;
}




