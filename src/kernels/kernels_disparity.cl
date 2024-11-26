#include "kernels_macros.h"
#include "kernels.h"

/*

// __kernel void disparity(
// 	// inputs
// 	__private	uint	layer,					//0
//
// 	__constant 	uint8*	mipmap_params,			//1
// 	__constant 	uint*	uint_params,			//2
// 	__constant  float*  fp32_params,			//3
//
// 	__global 	float4*	img_cur,				//4		// keyframe
// 	__global 	float4*	img_new,				//5
// 	//__global	float2* warp,					//6
// 	__global	float8* g1p,					//6		// keyframe_g1mem
// / *
// 														// const uint wg_divisor = 4;	//  512 * 20 * 32 / 8 =  40,960 bytes
// 																						// 1024 * 20 * 32 /
//
// 														// Intel(R) Iris(R) Xe Graphics :
// 														// Local memory size  65,536 (64KiB),
// 														// Max work item dimensions 3,
// 														// Max work item sizes 512x512x512,
// 														// Max work group size 512.
//
// 														// NVIDIA GeForce GTX 970M
// 														// Local memory size               49,152 (48KiB)
// 														// Max work item dimensions        3
// 														// Max work item sizes             1024x1024x64
// 														// Max work group size             1024
// * /
// 	__local	 	float4*	local_img_cur, 			//7		// local_work_size*4*5/wg_divisor*sizeof(float)
// 	__local	 	float4*	local_img_new, 			//8
// 	__local	 	float4*	local_img_cur_sq, 		//9
// 	__local	 	float4*	local_img_new_sq, 		//10
//
// 	// outputs
// 	__global	float4* Rho_,					//11
// 	__global 	float4*	disparity				//12
// 	)
//  {																														// find gradient wrt SE3 find global sum for each of the 6 DoF
// 	uint  global_id_orig	= get_global_id(0);
// 	uint  lid 				= get_local_id(0);
// 	uint  group_id 			= get_group_id(0);
// 																														//if(global_id_u == 1  ){ printf("\n__kernel void se3_LK_grad (global_id_u == 1 )  chk_1"   ); }
// 	const uint local_size 	= get_local_size(0); // / wg_divisor;
// 	const uint patch_length	= local_size;
//
// 	uint group_size 		= local_size;
// 	uint num_groups			= get_num_groups(0); //size_t get_num_groups (uint dimindx)
// 	uint work_dim 			= get_work_dim();
// 	//uint global_size		= get_global_size(0);
//
// 	const uint halo_width	= 2;
// 	uint  global_id_u		= group_id * (local_size - 2*halo_width)  + lid;												// new global_id takes acount of halo on local memory.
// 	float global_id_flt 	= global_id_u;
//
// 	uint8 mipmap_params_	= mipmap_params[layer];
// 	uint read_offset_ 		= mipmap_params_[MiM_READ_OFFSET];
// 	uint read_cols_ 		= mipmap_params_[MiM_READ_COLS];
// 	//uint read_rows_ 		= mipmap_params_[MiM_READ_ROWS];
// 	uint layer_pixels		= mipmap_params_[MiM_PIXELS];
// {
// 	//uint8 mipmap_params_0	= mipmap_params[0];
// 	//uint read_offset_0 	= mipmap_params_0[MiM_READ_OFFSET];
//
// 	//uint base_cols		= uint_params[COLS];
// 	//uint margin 			= uint_params[MARGIN];
//  }
// 	uint mm_cols			= uint_params[MM_COLS];
// 	uint mm_pixels			= uint_params[MM_PIXELS];
// {
// 	//float inv_d_step 		= fp32_params[INV_DEPTH_STEP];
// 	//float min_inv_depth 	= fp32_params[MIN_INV_DEPTH]; // + inv_d_step;
// 	//float max_inv_depth 	= fp32_params[MAX_INV_DEPTH]; // - inv_d_step;
//
// 	//uint reduction			= mm_cols/read_cols_;
//  }
// 	uint v 					= global_id_u / read_cols_;																		// read_row
// 	uint u 					= fmod(global_id_flt, read_cols_);																// read_column
// 	float u_flt				= u ;																							// NB no reduction, as we are not using K2Kin this kernel.
// 	float v_flt				= v ;
// 	uint read_index 		= read_offset_  +  v  * mm_cols  + u ;
// 	float alpha				= img_cur[read_index].w;
// {                                                                                                                      // sample the images /////////////////////////////////
// //     uint u_minus        	= u-1;
// //     uint u_plus         	= u+1;
// //     uint v_minus        	= v-1;
// //     uint v_plus         	= v+1;
// //
// //     uint u_sample[5]    	= { u, u_minus, u, u_plus, u };
// //     uint v_sample[5]    	= { v, v_minus, v, v_plus, v };
// //
// //     float4 img_cur_sample[5];
// //     float4 img_new_sample[5];
//  }                                                                                                                    // sample img_cur /////////////////////////////////
// 	if (global_id_u == 150){
// 		printf ("\n__kernel disparity(..) chk_1 layer=%u, global_id=%u, group_id=%u, lid=%u, u=%u, v=%u, read_index=%u, read_offset_=%u, mm_cols=%u, img_cur[read_index].x=%f, global_id_orig=%u, local_size=%u, halo_width=%u", \
// 		layer, global_id_u, group_id, lid, u, v, read_index, read_offset_, mm_cols, img_cur[read_index].x, global_id_orig, local_size, halo_width  );
// 	}
//
// 	for (int i=0; i<1+2*halo_width; i++){																				// Load local_img_patch_cur  /////////////////////////////////
// 		local_img_cur[lid + i*patch_length] 	= img_cur[ read_index +i*mm_cols];
// 	}
//
// 	for (int i=0; i<1+2*halo_width; i++){																				// Square local_img_patch_cur  /////////////////////////////////
// 		float4 pix_val 							= local_img_cur[lid + i*patch_length];
// 		local_img_cur_sq[lid + i*patch_length] 	=  pix_val *  pix_val;
// 	}
//
// 	float W[9] = { 1,2,1,2,4,2,1,2,1 }; 																				// 3x3 discrete Gaussian kernel
// 	int i_u[5] = {0,-1,0,1,0};
// 	int i_v[5] = {1,0,0,0,-1};
//
// 	float4 variance_curr 			= 0;
// 	float4 variance_new[5] 			= {0};
// 	float4 covariance[5] 			= {0};
// 	float4 cross_correlation[5] 	= {0};
// 	float4 disparity_pvt			= disparity[read_index];
// 	float  warp_u					= disparity_pvt.x;
// 	float  warp_v					= disparity_pvt.y;
// 	float  warp_incr_u 				= 0;
// 	float  warp_incr_v 				= 0;
//
// 	barrier(CLK_LOCAL_MEM_FENCE);////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// 																														// compute variance img_cur
// 	//if (lid>halo_width && lid<local_size - halo_width) {	// NB original image local mem has a 1 pixel margin around it, for 3x3 samples.
// 		for (int j=0; j<3; j++){
// 			for (int k=0; k<3; k++){
// 				variance_curr += local_img_cur_sq [lid + j*patch_length + k] * W[(j)*3 + k];
// 			}
// 		}
// 	//}
//
// 	if (global_id_u == 150){
// 		printf ("\n__kernel disparity(..) chk_2 layer=%u, global_id=%u, group_id=%u, lid=%u, u=%u, v=%u, read_index=%u, read_offset_=%u, mm_cols=%u, img_cur[read_index].x=%f, global_id_orig=%u, local_size=%u, halo_width=%u", \
// 		layer, global_id_u, group_id, lid, u, v, read_index, read_offset_, mm_cols, img_cur[read_index].x, global_id_orig, local_size, halo_width  );
// 	}
//
// 	barrier(CLK_LOCAL_MEM_FENCE);
// {
// 	//if (lid>halo_width  && lid=<local_size - halo_width) {
// // 		if (read_index< mm_pixels) {
// // 			uint j = 0, k=0;
// // 			Rho_[read_index]		= local_img_cur_sq [lid + j*patch_length + k];
// // 			 j=1; k=1;
// // 			disparity[read_index]	= local_img_cur_sq [lid + j*patch_length + k];
// //
// // 			 j=2; k=2;
// // 			Rho_[read_index - (650*480) ]		= local_img_cur_sq [lid + j*patch_length + k];
// // 			disparity[read_index - (650*480)]	= variance_curr;
// // 		}
// 	//}
//  }
//                                  // 7                                                                                       // disparity loop /////////////////////////////////
//     #define DISPARITY_ITERATIONS 1
//     for (int iter=0; iter< DISPARITY_ITERATIONS; iter++){
//
// 		for (int i=0; i<1+4*halo_width; i++){																			// Square local_img_patch_new  // interpollate sampling of img_new  /////////////////////////////////
// 			//local_img_new[lid + i*patch_length] = bilinear_flt4 ( img_new, u_flt + warp_u, v_flt + warp_v,  mm_cols,  read_offset_);
// 		}
// 		for (int i=0; i<1+4*halo_width; i++){																			// Square local_img_patch_new  /////////////////////////////////
// 			float4 pix_val 							= local_img_new[lid + i*patch_length];
// 			//local_img_new_sq[lid + i*patch_length] 	= pix_val *  pix_val;
// 		}
// {
// // 		if (read_index< mm_pixels) {
// // 			uint j = 0, k=0;
// // 			Rho_[read_index]					= local_img_new_sq[lid ];
// // 			 j=2; k=2;
// // 			disparity[read_index]				= local_img_new_sq[lid + 3*patch_length];
// //
// // 			 j=4; k=4;
// // 			Rho_[read_index - (650*480) ]		= local_img_new_sq[lid + 4*patch_length];
// // 			 j=5; k=5;
// // 			disparity[read_index - (650*480)]	= local_img_new_sq[lid + 5*patch_length];
// // 		}
// }
//         // variance img_new /////////////////////////////////
//         //if (lid>halo_width && lid<local_size - halo_width) {
//
// 			for (int i=0; i<5; i++){
// 				for (int j=0; j<3; j++ ){
// 					for (int k=0; k<3; k++){
// 						//variance_new[i] += local_img_new_sq [ lid + (i_v[i] + j)*patch_length  +  i_u[i] + k ]    * W[j*3 + k];
// 					}
// 				}
// 			}
// 																															// covariance /////////////////////////////////
// 			for (int i=0; i<5; i++){
// 				for (int j=0, j_=-1; j<3; j++, j_++){
// 					for (int k=0, k_=-1; k<3; k++, k_++){
// 						//covariance[i] +=  local_img_cur [lid + j_*patch_length + k]  *  local_img_new [ lid + (i_v[i] + j)*patch_length + i_u[i] + k ]    * W[j*3 + k];
// 					}
// 				}
// 			}
// 																															// cross-correlation /////////////////////////////////
// 			for (int i=0; i<5; i++){
// 				//cross_correlation[i] = covariance[i] / ( variance_curr *  variance_new[i] ) ;
// 			}
// 																															// compute optimal warp /////////////////////////////////
// 			warp_incr_u = compute_optimum( cross_correlation[1], cross_correlation[2], cross_correlation[3] );		// TODO  do I really want float4 OR should I reduce it to float ?
// 			warp_incr_v = compute_optimum( cross_correlation[0], cross_correlation[2], cross_correlation[4] );
// 																															// clip the warp /////////////////////////////////
// // 			if (read_index< mm_pixels) {
// // 				Rho_[		read_index]				= warp_incr_u;
// // 				disparity[	read_index]				= warp_incr_v;
// // 			}
//
// 			if( iter > 5){
// 				warp_incr_u = clamp( warp_incr_u, -1.0f, 1.0f );
// 				warp_incr_v = clamp( warp_incr_v, -1.0f, 1.0f );
// 			}
// 		//}
//         barrier(CLK_LOCAL_MEM_FENCE);
// 		if (global_id_u == 150){
// 			printf ("\n__kernel disparity(..) chk_3 iter=%u, layer=%u, global_id=%u, group_id=%u, lid=%u, u=%u, v=%u, read_index=%u, read_offset_=%u, mm_cols=%u, img_cur[read_index].x=%f, global_id_orig=%u, local_size=%u, halo_width=%u", \
// 			iter, layer, global_id_u, group_id, lid, u, v, read_index, read_offset_, mm_cols, img_cur[read_index].x, global_id_orig, local_size, halo_width  );
// 	}
// {
// // 		if (read_index< mm_pixels) {
// // 			uint j = 0, k=0;
// // 			Rho_[read_index]						= variance_new[1]; // local_img_new_sq [lid + j*patch_length + k];
// // 			 j=1; k=1;
// // 			disparity[read_index]					= variance_new[2]; // local_img_new_sq [lid + j*patch_length + k];
// // 			 j=2; k=2;
// // 			Rho_[read_index - (650*480) ]			= variance_new[3]; // local_img_new_sq [lid + j*patch_length + k];
// // 			disparity[read_index - (650*480)]		= variance_new[4]; // variance_new[0];
// // 		}
//
//
// // 		if (read_index< mm_pixels) {
// // 			uint i 		= 0;
// // 			Rho_[		read_index]					= cross_correlation[0]; // variance_new[i];
// // 			disparity[	read_index]					= cross_correlation[1]; // variance_curr;
// //
// // 			Rho_[		read_index - (650*480) ]	= cross_correlation[2]; // cross_correlation[i];
// // 			disparity[	read_index - (650*480) ]	= cross_correlation[3]; // cross_correlation[4]; //warp_incr_u;
// // 		}
//
// // 		if (read_index< mm_pixels) {
// // 			Rho_[		read_index - (650*480) ]	= warp_incr_u;
// // 			disparity[	read_index - (650*480) ]	= warp_incr_v;
// // 		}
// }
//                                                                                                                         // smooth/refine warp /////////////////////////////////
// 																														// TODO determine confidence based, edge preserving smoothing.
//         barrier(CLK_LOCAL_MEM_FENCE);
//
// 		if (global_id_u == 150){
// 			printf ("\n__kernel disparity(..) chk_4 layer=%u, global_id=%u, group_id=%u, lid=%u, u=%u, v=%u, read_index=%u, read_offset_=%u, mm_cols=%u, img_cur[read_index].x=%f, global_id_orig=%u, local_size=%u, halo_width=%u", \
// 			layer, global_id_u, group_id, lid, u, v, read_index, read_offset_, mm_cols, img_cur[read_index].x, global_id_orig, local_size, halo_width  );
// 		}
//                                                                                                                         // warp img new /////////////////////////////////
// 		warp_u -= warp_incr_u;	// TODO direcction? Why does it accumulate noise + atefacts ?
// 		warp_v -= warp_incr_v;
//     }
// 																														// save img new to global /////////////////////////////////
// // 	if ( ( global_id_u<layer_pixels) && (read_index< mm_pixels) && (lid>=halo_width) && (lid<local_size - halo_width) ) {
// // 		float4 Rho_pvt			= local_img_cur[lid + (1+halo_width)*patch_length]  -  bilinear_flt4 ( img_new, u + warp_u, v + warp_v,  mm_cols,  read_offset_); // img_cur[read_index] - img_new[read_index]; //
// // 		Rho_pvt.w				= 1.0f;
// // 		//Rho_[read_index]		= Rho_pvt;
// // 		float4 disparity_pvt 	= {warp_u, warp_v, 0, alpha};
// // 		//disparity[read_index]	= disparity_pvt;
// // 	}
//  }



//  __kernel void disparity2(
// 	// inputs
// 	__private	uint	layer,					//0
//
// 	__constant 	uint8*	mipmap_params,			//1
// 	__constant 	uint*	uint_params,			//2
// 	__constant  float*  fp32_params,			//3
//
// 	__global 	float4*	img_cur,				//4		// keyframe
// 	__global 	float4*	img_new,				//5
// 	//__global	float2* warp,					//6
// 	__global	float8* g1p,					//6		// keyframe_g1mem
// / *
// 														// const uint wg_divisor = 4;	//  512 * 20 * 32 / 8 =  40,960 bytes
// 																						// 1024 * 20 * 32 /
//
// 														// Intel(R) Iris(R) Xe Graphics :
// 														// Local memory size  65,536 (64KiB),
// 														// Max work item dimensions 3,
// 														// Max work item sizes 512x512x512,
// 														// Max work group size 512.
//
// 														// NVIDIA GeForce GTX 970M
// 														// Local memory size               49,152 (48KiB)
// 														// Max work item dimensions        3
// 														// Max work item sizes             1024x1024x64
// 														// Max work group size             1024
// * /
// 	__local	 	float4*	local_img_cur, 			//7		// local_work_size*4*5/wg_divisor*sizeof(float)
// 	__local	 	float4*	local_img_new, 			//8
// 	__local	 	float4*	local_img_cur_sq, 		//9
// 	__local	 	float4*	local_img_new_sq, 		//10
//
// 	// outputs
// 	__global	float4* Rho_,					//11
// 	__global 	float4*	disparity				//12
// 	)
//  {																														// find gradient wrt SE3 find global sum for each of the 6 DoF
// 	uint  global_id_orig	= get_global_id(0);
// 	uint  lid 				= get_local_id(0);
// 	uint  group_id 			= get_group_id(0);
// 																														//if(global_id_u == 1  ){ printf("\n__kernel void se3_LK_grad (global_id_u == 1 )  chk_1"   ); }
// 	const uint local_size 	= get_local_size(0); // / wg_divisor;
// 	const uint patch_length	= local_size;
//
// 	uint group_size 		= local_size;
// 	uint num_groups			= get_num_groups(0); //size_t get_num_groups (uint dimindx)
// 	uint work_dim 			= get_work_dim();
//
// 	const uint halo_width	= 2;
// 	uint  global_id_u		= group_id * (local_size - 2*halo_width)  + lid;											// new global_id takes acount of halo on local memory.
// 	float global_id_flt 	= global_id_u;
//
// 	uint8 mipmap_params_	= mipmap_params[layer];
// 	uint read_offset_ 		= mipmap_params_[MiM_READ_OFFSET];
// 	uint read_cols_ 		= mipmap_params_[MiM_READ_COLS];
// 	uint layer_pixels		= mipmap_params_[MiM_PIXELS];
//
// 	uint mm_cols			= uint_params[MM_COLS];
// 	uint mm_pixels			= uint_params[MM_PIXELS];
//
// 	uint v 					= global_id_u / read_cols_;																	// read_row
// 	uint u 					= fmod(global_id_flt, read_cols_);															// read_column
// 	float u_flt				= u ;																						// NB no reduction, as we are not using K2Kin this kernel.
// 	float v_flt				= v ;
// 	uint read_index 		= read_offset_  +  v  * mm_cols  + u ;
// 	float alpha				= img_cur[read_index].w;
//                                                                                                                         // sample the images /////////////////////////////////
//                                                                                                                         // sample img_cur /////////////////////////////////
// 	if (global_id_u == 150){
// 		printf ("\n__kernel disparity(..) chk_1 layer=%u, global_id=%u, group_id=%u, lid=%u, u=%u, v=%u, read_index=%u, read_offset_=%u, mm_cols=%u, img_cur[read_index].x=%f, global_id_orig=%u, local_size=%u, halo_width=%u", \
// 		layer, global_id_u, group_id, lid, u, v, read_index, read_offset_, mm_cols, img_cur[read_index].x, global_id_orig, local_size, halo_width  );
// 	}
//
// 	for (int i=0; i<1+2*halo_width; i++){																				// Load local_img_patch_cur  /////////////////////////////////
// 		local_img_cur[lid + i*patch_length] 	= img_cur[ read_index +i*mm_cols];
// 	}
//
// 	for (int i=0; i<1+2*halo_width; i++){																				// Square local_img_patch_cur  /////////////////////////////////
// 		float4 pix_val 							= local_img_cur[lid + i*patch_length];
// 		local_img_cur_sq[lid + i*patch_length] 	=  pix_val *  pix_val;
// 	}
//
// 	float W[9] = { 1,2,1,2,4,2,1,2,1 }; 																				// 3x3 discrete Gaussian kernel
// 	int i_u[5] = {0,-1,0,1,0};
// 	int i_v[5] = {1,0,0,0,-1};
//
// 	float4 variance_curr 			= 0;
// 	float4 variance_new[5] 			= {0};
// 	float4 covariance[5] 			= {0};
// 	float4 cross_correlation[5] 	= {0};
// 	float4 disparity_pvt			= disparity[read_index];
// 	float  warp_u					= disparity_pvt.x;
// 	float  warp_v					= disparity_pvt.y;
// 	float  warp_incr_u 				= 0;
// 	float  warp_incr_v 				= 0;
//
// 	barrier(CLK_LOCAL_MEM_FENCE);////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// 																														// compute variance img_cur
// 	//if (lid>halo_width && lid<local_size - halo_width) {	// NB original image local mem has a 1 pixel margin around it, for 3x3 samples.
// 		for (int j=0; j<3; j++){
// 			for (int k=0; k<3; k++){
// 				variance_curr += local_img_cur_sq [lid + j*patch_length + k] * W[(j)*3 + k];
// 			}
// 		}
// 	//}
//
// 	if (global_id_u == 150){
// 		printf ("\n__kernel disparity(..) chk_2 layer=%u, global_id=%u, group_id=%u, lid=%u, u=%u, v=%u, read_index=%u, read_offset_=%u, mm_cols=%u, img_cur[read_index].x=%f, global_id_orig=%u, local_size=%u, halo_width=%u, patch_length=%u, (1+4*halo_width)=%u, (lid + 4*patch_length)=%u", \
// 		layer, global_id_u, group_id, lid, u, v, read_index, read_offset_, mm_cols, img_cur[read_index].x, global_id_orig, local_size, halo_width, patch_length, (1+4*halo_width), (lid + 4*patch_length)  );
// 	}
//
// 	barrier(CLK_LOCAL_MEM_FENCE);
// {// 	/////
// // 	if ( (lid>halo_width)  && (lid<=(local_size - halo_width) ) ){
// // 		if (read_index< mm_pixels) {
// // 			uint j = 0, k=0;
// // 			Rho_[read_index]		= local_img_cur_sq [lid + j*patch_length + k];
// // 			 j=1; k=1;
// // 			disparity[read_index]	= local_img_cur_sq [lid + j*patch_length + k];
// //
// // 			 j=2; k=2;
// // 			Rho_[read_index - (650*480) ]		= local_img_cur_sq [lid + j*patch_length + k];
// // 			disparity[read_index - (650*480)]	= variance_curr;
// // 		}
// // 	}
// // 	//////
// }
//                                  // 7																					// disparity loop /////////////////////////////////
//      #define DISPARITY_ITERATIONS 1
//      for (int iter=0; iter< DISPARITY_ITERATIONS; iter++){
//
// 		for (int i=0; i<1+4*halo_width; i++){																			// Square local_img_patch_new  // interpollate sampling of img_new  /////////////////////////////////
// 			local_img_new[lid + i*patch_length] 	= bilinear_flt4 ( img_new, u_flt + warp_u, v_flt + warp_v,  mm_cols,  read_offset_);
// 		}
// 		for (int i=0; i<1+4*halo_width; i++){																			// Square local_img_patch_new  /////////////////////////////////
// 			float4 pix_val 							= local_img_new[lid + i*patch_length];
// 			local_img_new_sq[lid + i*patch_length] 	= pix_val *  pix_val;
// 		}
// 		if (global_id_u == 150){ printf ("\n__kernel disparity(..) chk_3 "  ); }
// 		barrier(CLK_LOCAL_MEM_FENCE);
// {
// // 		/////
// // 		if (read_index< mm_pixels) {
// // 			uint j = 0, k=0;
// // 			Rho_[read_index]					= local_img_new_sq[lid ];
// // 			 j=2; k=2;
// // 			disparity[read_index]				= local_img_new_sq[lid + 3*patch_length];
// //
// // 			 j=4; k=4;
// // 			Rho_[read_index - (650*480) ]		= local_img_new_sq[lid + 4*patch_length];
// // 			 j=5; k=5;
// // 			disparity[read_index - (650*480)]	= local_img_new_sq[lid + 5*patch_length];
// //  		}
// // 		/////
// }
// //         // variance img_new /////////////////////////////////
// {
// ////
// // 		float W[9] = { 1,2,1,  2,4,2,  1,2,1 }; 																				// 3x3 discrete Gaussian kernel
// // 		int i_u[5] = {0, -1,  0,  1,  0};
// // 		int i_v[5] = {1,  0,  0,  0, -1};
// ////
// }
// 			for (int i=0; i<5; i++){		// TODO Could have local mem instead of 5 computations
// 				for (int j=0; j<3; j++ ){
// 					for (int k=0; k<3; k++){
// 						int idx = lid + (i_v[i]+j+1)*patch_length  +  i_u[i] + k ;
// 						variance_new[i] += local_img_new_sq[ idx ]    * W[j*3 + k];  // TODO unroll this loop to reduce computation
// 					}
// 				}
// 			}
// {
// // 		/////
// // 		if (read_index< mm_pixels) {
// // 			Rho_[read_index]					= variance_new[0];
// // 			disparity[read_index]				= variance_new[1];
// // 			Rho_[read_index - (650*480) ]		= variance_new[3];
// // 			disparity[read_index - (650*480)]	= variance_new[4];
// //  		}
// // 		/////
// }
// 			if (global_id_u == 150){ printf ("\n__kernel disparity(..) chk_4 "  ); }
// 																														// covariance /////////////////////////////////
// 			for (int i=0; i<5; i++){
// 				for (int j=0, j_=-1; j<3; j++, j_++){
// 					for (int k=0, k_=-1; k<3; k++, k_++){
// 						covariance[i] +=  local_img_cur [lid + j_*patch_length + k]  *  local_img_new [ lid + (i_v[i]+j+1)*patch_length + i_u[i] + k ]    * W[j*3 + k];  // TODO unroll this loop to reduce computation
// 					}
// 				}
// 			}
// // 		/////
//  		if (read_index< mm_pixels) {
// 			Rho_[read_index]					= covariance[0];
// 			disparity[read_index]				= covariance[1];
// 			Rho_[read_index - (650*480) ]		= covariance[3];
// 			disparity[read_index - (650*480)]	= covariance[4];
//   		}
// // 		/////
// 			if (global_id_u == 150){ printf ("\n__kernel disparity(..) chk_5 "  ); }
// 																														// cross-correlation /////////////////////////////////
// 			for (int i=0; i<5; i++){
// 				cross_correlation[i] = covariance[i] / ( variance_curr *  variance_new[i] ) ;
// 			}
// 			if (global_id_u == 150){ printf ("\n__kernel disparity(..) chk_6");}
// ////
// {
// 			//float4 a=cross_correlation[1];
// 			//if (global_id_u == 150){ printf ("\n__kernel disparity(..) chk_6,  cross_correlation[1]=%f,%f,%f,%f,  ", a.x, a.y, a.z, a.w ); }
//
// 			//float4 b=cross_correlation[2];
// 			//if (global_id_u == 150){ printf ("\n__kernel disparity(..) chk_6,  cross_correlation[2]=%f,%f,%f,%f,  ", b.x,b.y,b.z,b.w    ); }
//
//  			//float4 c=cross_correlation[3];
// 			//if (global_id_u == 150){ printf ("\n__kernel disparity(..) chk_6,  cross_correlation[3]=%f,%f,%f,%f,  ", c.x,c.y,c.z,c.w    ); }
//
//  			//float4 d=cross_correlation[0];
// 			//if (global_id_u == 150){ printf ("\n__kernel disparity(..) chk_6,  cross_correlation[0]=%f,%f,%f,%f,  ", d.x,d.y,d.z,d.w    ); }
//
//  			//float4 e=cross_correlation[2];
// 			//if (global_id_u == 150){ printf ("\n__kernel disparity(..) chk_6,  cross_correlation[2]=%f,%f,%f,%f,  ", e.x,e.y,e.z,e.w    ); }
//
//  			//float4 f=cross_correlation[4];
// 			//if (global_id_u == 150){ printf ("\n__kernel disparity(..) chk_6,  cross_correlation[4]=%f,%f,%f,%f,  ", f.x,f.y,f.z,f.w    ); }
// }
// 																														// compute optimal warp /////////////////////////////////
//  			warp_incr_u = 1.0; // compute_optimum( cross_correlation[1], cross_correlation[2], cross_correlation[3] );	// TODO  do I really want float4 OR should I reduce it to float ?
//  			warp_incr_v = 1.0; // compute_optimum( cross_correlation[0], cross_correlation[2], cross_correlation[4] );
// // 																														// clip the warp /////////////////////////////////
// //
// 			if (global_id_u == 150){ printf ("\n__kernel disparity(..) chk_7 "  ); }
//  			if( iter > 5){
//  				warp_incr_u = clamp( warp_incr_u, -1.0f, 1.0f );
//  				warp_incr_v = clamp( warp_incr_v, -1.0f, 1.0f );
//  			}
//  			if (global_id_u == 150){ printf ("\n__kernel disparity(..) chk_8 "  ); }
// //
//          barrier(CLK_LOCAL_MEM_FENCE);
// //  		if (global_id_u == 150){
// //  			printf ("\n__kernel disparity(..) chk_9 iter=%u, layer=%u, global_id=%u, group_id=%u, lid=%u, u=%u, v=%u, read_index=%u, read_offset_=%u, mm_cols=%u, img_cur[read_index].x=%f, global_id_orig=%u, local_size=%u, halo_width=%u", \
// //  			iter, layer, global_id_u, group_id, lid, u, v, read_index, read_offset_, mm_cols, img_cur[read_index].x, global_id_orig, local_size, halo_width  );
// //  		}
//
//  		if (global_id_u == 150){ printf ("\n__kernel disparity(..) chk_9 "  ); }
// //
// //                                                                                                                      // warp img new /////////////////////////////////
//  		warp_u -= warp_incr_u;	// TODO direcction? Why does it accumulate noise + atefacts ?
//  		warp_v -= warp_incr_v;
//      }
// 																														// save img new to global /////////////////////////////////
// 	if ( ( global_id_u<layer_pixels) && (read_index< mm_pixels) && (lid>=halo_width) && (lid<local_size - halo_width) ) {
// 		float4 Rho_pvt			= local_img_cur[lid + (1+halo_width)*patch_length]  -  bilinear_flt4 ( img_new, u + warp_u, v + warp_v,  mm_cols,  read_offset_); // img_cur[read_index] - img_new[read_index]; //
// 		Rho_pvt.w				= 1.0f;
// 		//Rho_[read_index]		= Rho_pvt;
// 		float4 disparity_pvt 	= {warp_u, warp_v, 0, alpha};
// 		//disparity[read_index]	= disparity_pvt;
// 	}
//  }

// float compute_optimum(__private float4 A, __private float4 B, __private float4 C){
// 	float 	steps[3]  = {-1, 0, 1};
// 	float a, b, c,   d, e,   f, g,   h, i,   j, k, d2, f2, h2, prediction, optimum;										// compute x value of the optimum of parabola, y= a*x*x + b*x + c
// 																														// given samples at x=1,2,4
// 	d = steps[0];	e = A.x ;		// TODO  which combination of color channels ?
// 	f = steps[1];	g = B.x ;
// 	h = steps[2];	i = C.x ;
//
// 	d2 = d * d;
// 	f2 = f * f;
// 	h2 = h * h;
//
// 	j = (f2 - d2)*(f-h) - (h2 - f2)*(d-f);
// 	k = (g-i)*(d-f) - (e-g)*(f-h);
// 	a = k/j;
// 	b = (e-g +a*(f2-d2))  /  (d-f);
// 	c = e - a*d2 - b*d;
// 																														// TODO we neet to pick maximum image correlation
// 	if (a>0){																											// IF concavity leads to a minimum, use it.
// 		float x 		= -b /(2*a);
// 		prediction 		= a*(x*x) + b*x + c;
// 		optimum 		= x;
// 	}else{																												// IF concavity leads to a maximum, pick the best sample so far.
// 		if (e>=i){
// 			prediction 	= i;
// 			optimum 	= h;
// 		}else{
// 			prediction 	= e;
// 			optimum 	= d;
// 		}
// 	}
// 	return optimum;
// }

*/

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////




 __kernel void compute_lookup_table(					// computed once at start of program	// TODO when is it possible to roll the layers together ?  i.e. when local mem is not used.
	 // inputs
	__private	uint	layer,					//0

	__constant 	uint8*	mipmap_params,			//1
	__constant 	uint*	uint_params,			//2
	__constant  float*  fp32_params,			//3

	// output
	__global 	uint4*	lookup_table			//4
){
	uint  global_id					= get_global_id(0);
	float global_id_flt 			= global_id;

	uint8 mipmap_params_			= mipmap_params[layer];
	uint read_offset_ 				= mipmap_params_[MiM_READ_OFFSET];
	uint read_cols_ 				= mipmap_params_[MiM_READ_COLS];

	uint mm_cols					= uint_params[MM_COLS];

	uint v 							= global_id / read_cols_;												// read_row
	uint u 							= fmod(global_id_flt, read_cols_);										// read_column

	uint read_index 				= read_offset_  +  v  * mm_cols  + u ;
	uint alpha						= 255;	// img_cur[read_index].w;
	uint4 lookup 					= {u,v,read_index,alpha};

	lookup_table[global_id]			= lookup;																// centre pixel
}


__kernel void warp_image(								// computed once each iteration of warping, for each layer of image pyramid
	// inputs
	__private	uint	read_offset,			//0
	__private	uint	mm_cols,				//1

	__constant 	uint*	uint_params,			//2

	__global 	float2*	warp,					//3
	__global 	uint4*	lookup_table,			//4
	__global 	float4*	new_img,				//5

	// outputs
	__global 	float4*	new_img_warped			//6
){
	uint  global_id								= get_global_id(0);
	float2 warp2								= warp[global_id];
	uint4 lookup_ref							= lookup_table[global_id];
	float u_flt 								= lookup_ref.x 	+ warp2.x;
	float v_flt 								= lookup_ref.y	+ warp2.y;
	float u_mod									= fmod(u_flt , 1);
	float v_mod									= fmod(v_flt , 1);
	uint u 										= floor(u_flt);
	uint v 										= floor(v_flt);

	uint read_index				= read_offset  +  v   * mm_cols  + u ;

	float4 warped_lower			= new_img[read_index] 			* (1-u_mod) 	+ new_img[read_index+1] 		* u_mod;
	float4 warped_upper			= new_img[read_index+mm_cols] 	* (1-u_mod) 	+ new_img[read_index+mm_cols+1] * u_mod;
	new_img_warped[global_id]	= warped_lower 					* (1-v_mod)  	+ warped_upper 					* v_mod;
}


 __kernel void img_sq(
	// inputs
	__global 	uint4*	lookup_table,			//0
	__global 	float4*	img,				//1
	// output
	__global 	float4*	img_sq					//2
){
	uint read_index						= lookup_table[get_global_id(0)].z;
	float4 pixel 						= img[read_index];
	img_sq[read_index]					= pixel*pixel;
}


 __kernel void img_variance(		// TODO ? should variuance be divided by the mean of the 3x3 patch ?
	// inputs
	__private	uint	mm_cols,				//0

	__global 	uint4*	lookup_table,			//1
	__global 	float4*	img_sq,					//2

	// output
	__global 	float4*	img_var					//3
){
	uint 	global_id					= get_global_id(0);
	uint 	read_index					= lookup_table[ global_id ].z;

	float4 	pixel 						= 0;
	float4 	var							= 0;
	float 	W[9] 						= { 1,2,1,2,4,2,1,2,1 }; 							// 3x3 discrete Gaussian kernel

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

	for (int i=0; i<9; i++ ){	var				 += img_sq[ read_index_3x3[i] ] * W[i];	}
	img_var[read_index]							 = var;
}


float compute_maximum(__private float4 A, __private float4 B, __private float4 C){
	float 	steps[3]  = {-1, 0, 1};
	float a, b, c,   d, e,   f, g,   h, i,   j, k, d2, f2, h2, prediction, optimum;										// compute x value of the optimum of parabola, y= a*x*x + b*x + c
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

	if (a<0){																											// IF concavity leads to a maximum, use it.
		float x 		= -b /(2*a);
		prediction 		= a*(x*x) + b*x + c;
		optimum 		= x;
	}else{																												// IF concavity leads to a maximum, pick the best sample so far.
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


 __kernel void compute_warp(
	// inputs
	__private	uint 	mm_size,				//0
	__private	uint	mm_cols,				//1

	__global 	uint4*	lookup_table,			//2
	__global 	float4*	curr_img,				//3
	__global 	float4*	new_img,				//4																		// NB warped version of the new image.
	__global 	float4*	curr_img_var,			//5
	__global 	float4*	new_img_var,			//6

	// output
	__global 	float4*	img_covar,				//7		// 5*float4*mm_size
	__global 	float4*	img_corr,				//8		// 5*float4*mm_size
	__global 	float2*	warp					//9
){
	uint global_id		= get_global_id(0);
	uint read_index		= lookup_table[ global_id ].z;
	float W[9] 			= { 1,2,1,2,4,2,1,2,1 }; 																		// 3x3 discrete Gaussian kernel
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
	sample_idx[0] 		=	-1;
	sample_idx[0] 		=	0;
	sample_idx[0] 		=	+1;
	sample_idx[0] 		=	+mm_cols;

	for (int j=0; j<5; j++){
		for (int i=0; i<9; i++ ){ covar[j]		+= curr_img[read_index_3x3[i] ] * new_img[read_index_3x3[i] + sample_idx[j] ] *  W[i]; }
		img_covar[read_index + j*mm_size]		= covar[j];

		float4 inv_denominator					= ( sqrt( curr_img_var[read_index] ) * sqrt( new_img_var[read_index + sample_idx[j] ]  ) );
		float4 denominator 						= isnormal(inv_denominator) ? 1/inv_denominator : 1;					// prevent div by zero

		corr[j] 								= covar[j] / denominator ;
		img_corr[read_index + j*mm_size]		= corr[j];
	}

	float2 warp2								= warp[read_index];
	float warp_u								= warp2.x + compute_maximum( corr[1], corr[2], corr[3] );
	float warp_v								= warp2.y + compute_maximum( corr[0], corr[2], corr[4] );
	warp_u										= clamp(warp_u, -1.0f, 1.0f);
	warp_v										= clamp(warp_v, -1.0f, 1.0f);											// warp increment clamped to +/-1

	float2 warp2_new							= {warp_u, warp_v};
	warp[read_index]							= warp2_new;
}

// TODO  (i) confidence map, (ii) anisotropic diffusion, (iii) Inter-Scale Disparity Refinement



////////////////////////////////////////////////////////////////////////////
//Buffers required: 10 new. (not counting curr_img and new_img)
// 	__private	uint	layer,					//0
// 	__private	uint 	mm_size,				//0
// 	__private	uint	mm_cols,				//1
// 	__private	uint	read_offset,			//0
//
// 	__constant 	uint8*	mipmap_params,			//1
// 	__constant 	uint*	uint_params,			//2
// 	__constant  float*  fp32_params,			//3
//
// 	__global 	uint4*	lookup_table			//0
// 	__global 	float2*	warp,					//1
// 	__global 	float4*	new_img,				//2
// 	__global 	float4*	new_img_warped			//3
// 	__global 	float4*	new_img_sq				//4
// 	__global 	float4*	new_img_var,			//5
//
// 	__global 	float4*	curr_img,				//6
// 	__global 	float4*	curr_img_sq				//7
// 	__global 	float4*	curr_img_var,			//8
//
// 	__global 	float4*	img_covar,				//9		// 5*float4*mm_size
// 	__global 	float4*	img_corr,				//10	// 5*float4*mm_size
// 	__global 	float2*	warp					//11
