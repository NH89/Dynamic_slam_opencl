#include "kernels__macros.h"
#include "kernels.h"

__kernel void compute_param_maps(
	__private	uint	layer,			//0
	__private	float	inv_depth,		//1

	__constant 	uint8*	mipmap_params,	//2
	__constant 	uint*	uint_params,	//3
	__constant 	float* 	SE3_k2k,		//4

	__global 	float2*	SE3_map			//5
		 )
{
	uint global_id_u 	= get_global_id(0);
	float global_id_flt = global_id_u;
	uint8 mipmap_params_= mipmap_params[layer];
	uint read_offset_ 	= mipmap_params_[MiM_READ_OFFSET];
	uint read_cols_ 	= mipmap_params_[MiM_READ_COLS];
	uint read_rows_ 	= mipmap_params_[MiM_READ_ROWS];
	if (global_id_u >= mipmap_params_[MiM_PIXELS]) return;

	uint lid 			= get_local_id(0);
	uint group_size 	= get_local_size(0);

	uint margin			= uint_params[MARGIN];
	uint mm_cols		= uint_params[MM_COLS];
	uint base_cols		= uint_params[COLS];
	float reduction		= base_cols/read_cols_;
	uint v				= global_id_u / read_cols_;													// read_row
	uint u				= fmod(global_id_flt, read_cols_);											// read_column
	float u_flt			= (float)u * reduction;															// NB this causes sparse sampling of the original space, to use the same k2k at every scale.
	float v_flt			= (float)v * reduction;
	float u2, v2;
	uint read_index 	= read_offset_  +  v  * mm_cols  + u ;

	int idx 			= layer * 6 * 16;

	bool print=false;
	if( (u_flt==10)&&(v==10) ){ print=true; }  // global_id_u==0) || (u==read_cols_/2.0f && v==read_rows_/2.0f) || (u==read_cols_ && v==read_rows_

	for (uint i=0; i<6; i++, idx+=16) {																// for each SE3 DoF
/*																									// Find new pixel position, h=homogeneous coords.
		if(global_id_u==0){
			printf("\n\n\n__kernel void compute_param_maps()          layer=%d,  SE3 i=%d,  idx=%d,  SE3_k2k=(\n(%f, %f, %f, %f),\n(%f, %f, %f, %f),\n(%f, %f, %f, %f),\n(%f, %f, %f, %f))  ",\
			layer, i, idx,\
			SE3_k2k[idx+ 0],SE3_k2k[idx+ 1],SE3_k2k[idx+ 2],SE3_k2k[idx+ 3],\
			SE3_k2k[idx+ 4],SE3_k2k[idx+ 5],SE3_k2k[idx+ 6],SE3_k2k[idx+ 7],\
			SE3_k2k[idx+ 8],SE3_k2k[idx+ 9],SE3_k2k[idx+10],SE3_k2k[idx+11],\
			SE3_k2k[idx+12],SE3_k2k[idx+13],SE3_k2k[idx+14],SE3_k2k[idx+15] );
		}
*/
		float16 k2k_ = (float16)(SE3_k2k[idx+0], 	SE3_k2k[idx+1], 	SE3_k2k[idx+2], 	SE3_k2k[idx+3],\
								 SE3_k2k[idx+4], 	SE3_k2k[idx+5], 	SE3_k2k[idx+6], 	SE3_k2k[idx+7],\
								 SE3_k2k[idx+8], 	SE3_k2k[idx+9], 	SE3_k2k[idx+10], 	SE3_k2k[idx+11],\
								 SE3_k2k[idx+12], 	SE3_k2k[idx+13], 	SE3_k2k[idx+14], 	SE3_k2k[idx+15]);

		px_k2k( k2k_,  reduction,  v,  u,  inv_depth, &u2,  &v2,  print  );
/*
		if(print==true){
			printf("\n__kernel void compute_param_maps()    u_flt=%f,  u2=%f,  v_flt=%f,   v2=%f  ", u_flt, u2 , v_flt, v2);
		}
*/
		float2 partial_gradient={ ((float)u)-u2 ,  ((float)v)-v2 }; 												// Find movement of pixel

		SE3_map[read_index + i* uint_params[MM_PIXELS]  ] = partial_gradient;

		barrier(CLK_GLOBAL_MEM_FENCE );
	}

//	if(print==true){ printf("\n");}

	// TO DO // Create a 'reproject' & 'img_grad_sum' kernels
}


__kernel void convert_depth(
	__private	uint 	invert,					//0
	__private	float 	factor,					//1

	__constant 	uint*	mipmap_params,			//2		// NB uses ony mipmap_params[layer=0]
	__constant	uint*	uint_params,			//3

	__global	float* 	depth_mem_temp,			//4
	__global	float* 	depth_mem_GT			//5
		)
{
	int global_id 		= (int)get_global_id(0);
	uint pixels 		= uint_params[PIXELS];

	uint read_offset_ 	= mipmap_params[MiM_READ_OFFSET];
	uint cols 			= uint_params[COLS];
	uint margin 		= uint_params[MARGIN];
	uint mm_cols		= uint_params[MM_COLS];

	uint base_row		= global_id/cols ;
	uint base_col		= global_id%cols ;
	uint img_row		= base_row + margin;
	uint img_col		= base_col + margin;

	uint read_index 	= read_offset_  +  base_row  * mm_cols  + base_col  ;
	uint global_id_u 	= get_global_id(0);

	if (global_id_u    >= mipmap_params[MiM_PIXELS]) return;
	float depth 		= depth_mem_temp[global_id_u]/factor;

	//if (global_id_u == 0)printf("\n__kernel void convert_depth(..) invert=%u, factor=%f, depth_mem[global_id_u]=%f,  depth=%f,  1/depth=%f   ", invert, factor, depth_mem[global_id_u], depth, 1/depth  );

	if (!(depth==0)){
		if ( invert==true ) depth_mem_GT[read_index] =  1/depth;
		else depth_mem_GT[read_index] = depth;
	}
}



/////////

__kernel void cvt_color_space_linear(																// Writes the first entry in a linear mipmap, and computes img_mean
	__global	uchar*	base,			//0															// NB for debugging the mimpam is arranged as a series below eachother with margins.
	__global	float4*	img,			//1															// This can be changed to dense packing in a linear array, to reduce memeory and data transfer requirements.

	__constant	uint*	uint_params,	//2
	__constant 	uint8*	mipmap_params,	//3

	__local		float4*	local_sum_pix,	//4
	__global	float4*	global_sum_pix	//5
		 )
{																									// NB need 32-bit uint (2**32=4,294,967,296) for index, not 16bit (2**16=65,536).
	int global_id 			= (int)get_global_id(0);
	uint pixels 			= uint_params[PIXELS];
	uint lid 				= get_local_id(0);
	uint local_size 		= get_local_size(0);
	uint group_size 		= local_size;
	uint reduction			= 1;																	// = mm_cols/read_cols_; but this is only the baselayer ofthe image pyramid.

	uint8 mipmap_params_	= mipmap_params[0];
	uint read_offset_ 		= mipmap_params_[MiM_READ_OFFSET];
	uint cols 				= uint_params[COLS];
	//uint margin 			= uint_params[MARGIN];
	uint mm_cols			= uint_params[MM_COLS];

	float R_float			= base[global_id*3]  /256.0f;
	float G_float			= base[global_id*3+1]/256.0f;
	float B_float			= base[global_id*3+2]/256.0f;

	float V 				= max(R_float, max(G_float, B_float) );
	float min_rgb 			= min(R_float, min(G_float, B_float) );
	float divisor 			= V - min_rgb;
	float S 				= (V!=0)*(V-min_rgb)/V;
	// Hue in radians
	const float Pi_3 = M_PI_F/3;

	float H = (   (V==R_float && V!=0)* 	Pi_3* ((G_float-B_float) / divisor )  \
			+     (V==G_float && V!=0)*		Pi_3*(((B_float-R_float) / divisor ) +2) \
			+     (V==B_float && V!=0)*		Pi_3*(((R_float-G_float) / divisor ) +4) \
			);																				// TO DO shift "/M_PI_F" to CPU data saving ?

	if (!(H<=2*M_PI_F && H>=0.0f) || !(S<=1.0f && S>=0.0f) || !(V<=1.0f && V>=0.0f) ) {H=S=V=0.0f;}		// to replace any NaNs

	uint base_row	= global_id/cols ;
	uint base_col	= global_id%cols ;
	uint read_index = read_offset_  +  base_row  * mm_cols  + base_col  ;							// NB 4 channels.  + margin

	float4 temp_float4  = {H/(2*M_PI_F),S,V,1.0f};													// Note how to load a float4 vector. Also H->(0,1) for display.
	if (global_id <= pixels) {
		img[read_index] 		= temp_float4;
		local_sum_pix[lid] 		= temp_float4;
	}else local_sum_pix[lid]	= 0;

	/* from https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html
	 * In case of 8-bit and 16-bit images, R, G, and B are converted to the floating-point format and scaled to fit the 0 to 1 range.
	 * V = max(R,G,B)
	 * S = (0, if V=0), otherwise (V-min(RGB))/V
	 * H = 60(G-B)/(V-min(R,G,B)  			if V=R
	 *     120 + 60(B-R)/(V-min(R,G,B))		if V=G
	 *     240 + 60(R-G)/(V-min(R,G,B))		if V=B
	 *     0								if R=G=B
	 */
	///////////////////////////////////////////////////////////										// Sum pixels in the work group, using local mem.

	int max_iter = ilogb((float)(group_size));

	for (uint iter=0; iter<=max_iter ; iter++) {	// for log2(local work group size)				// problem : how to produce one result for each mipmap layer ?
																									// NB kernels launched separately for each layer, but workgroup size varies between GPUs.
		group_size   /= 2;
		barrier(CLK_LOCAL_MEM_FENCE);																// No 'if->return' before fence between write & read local mem
		if (lid<group_size)  local_sum_pix[lid] += local_sum_pix[lid+group_size];					// local_sum_pix
	}
	if (lid==0) {
		uint group_id 			= get_group_id(0);
		uint global_sum_offset 	= 0; //read_offset_ / local_size ;		// only the base layer		// Compute offset for this layer
		uint num_groups 		= get_num_groups(0);

		float4 layer_data = {num_groups, reduction, 0.0f, 0.0f };			// Write layer data to first entry
		if (global_id == 0) {global_sum_pix[global_sum_offset] = layer_data; }
		global_sum_offset += 1+ group_id;

		if (local_sum_pix[0][3] >0){																// Using alpha channel local_sum_pix[0][3], to count valid pixels being summed.
			global_sum_pix[global_sum_offset] = local_sum_pix[0] / local_sum_pix[0][3];				// Save to global_sum_pix // Count hits, and divide group by num hits, without using atomics!
		}else global_sum_pix[global_sum_offset] = 0;
	}
}

__kernel void mipmap_linear_flt(	// Mipmap layers must be executed sesequentially				// Nvidia Geforce GPUs cannot use "half"
	__private	uint	layer,			//0															// Only used for depth_GT at present
	__constant 	uint8*	mipmap_params,	//1
	__constant 	uint*	uint_params,	//2
	__global 	float*	img,			//3
	__local	 	float*	local_img_patch	//4
		 )
{
	uint global_id_u 	= get_global_id(0);
	float global_id_flt = global_id_u;
	uint lid 			= get_local_id(0);
	uint group_size 	= get_local_size(0);
	uint patch_length	= group_size+4;

	uint8 mipmap_params_ = mipmap_params[layer];
	uint read_offset_ 	= mipmap_params_[MiM_READ_OFFSET];
	uint write_offset_ 	= mipmap_params_[MiM_WRITE_OFFSET]; 										// = read_offset_ + read_cols_*read_rows for linear MipMap.
	uint read_rows_		= mipmap_params_[MiM_READ_ROWS];
	uint write_rows_	= read_rows_ /2;
	uint read_cols_ 	= mipmap_params_[MiM_READ_COLS];
	uint write_cols_ 	= mipmap_params_[MiM_WRITE_COLS];

	uint margin 		= uint_params[MARGIN];
	uint mm_cols		= uint_params[MM_COLS];   													// whole mipmap


	uint write_row   	= global_id_u / write_cols_ ;
	uint write_column 	= fmod(global_id_flt, write_cols_);
//						if (global_id_u==1) printf("\n\n__kernel void mipmap_linear_flt():(global_id_u==1) write_row=%u, write_column=%u \n",write_row, write_column);
	uint read_row    	= 2*write_row;
	uint read_column 	= 2*write_column;

	uint read_index 	= read_offset_  +  read_row  * mm_cols  + read_column  ;					// NB 4 channels.  + margin
	uint write_index 	= write_offset_ +  write_row * mm_cols  + write_column ;					// write_cols_, use read_cols_ as multiplier to preserve images  + margin

	int in_bounds = global_id_u < mipmap_params_[MiM_PIXELS]/2 ;
	/*
	if (in_bounds == 1){
		for (int i=0, j=-2; i<5; i++, j++){															// Load local_img_patch
			local_img_patch[lid+2 + i*patch_length] = img[ read_index +j*mm_cols];
		}
		if (lid==0 || lid==1){
			for (int i=0; i<5; i++){
				local_img_patch[lid + i*patch_length] = img[ read_index +i*mm_cols -2]; //white; //
			}
		}
		if (lid==group_size-2 || lid==group_size-1){
			for (int i=0; i<5; i++){
				local_img_patch[lid+4 + i*patch_length] = img[ read_index +i*mm_cols +2]; //black; //
			}
		}
		////
		if ((write_row>write_rows_-3) ||  (write_row < 3)  ){										// Prevents blurring with black space below the image.
			for (int i=0; i<5; i++){
				local_img_patch[lid+2 + i*patch_length] = img[ read_index ];
			}
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);																	// No 'if->return' before fence between write & read local mem
	if (in_bounds == 0) return;

	float reduced_pixel = 0;
	for (int i=0; i<5; i++){
		for (int j=0; j<5; j++){
			reduced_pixel += local_img_patch[lid+j + i*patch_length]/25; 							// 5x5 box filter, rather than Gaussian
		}
	}

	if (write_column < 2 || write_column > write_cols_ -3) {
		reduced_pixel = 0;
		for (int i=0; i<5; i++){
			reduced_pixel += local_img_patch[lid+2 + i*patch_length]/5;								// prevents blur wrapping left-right.
		}
	}
	*/
	if (in_bounds != 1)	return;
	if (write_row>=write_rows_) return;
	if (global_id_u >= mipmap_params_[MiM_PIXELS]) return;											// num pixels to be written & num threads to really use.

	img[write_index] =  img[read_index];
}

