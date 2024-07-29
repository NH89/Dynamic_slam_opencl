#include "kernels_macros.h"


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
			);																				// TODO shift "/M_PI_F" to CPU data saving ?

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

__kernel void image_variance(
	__global	float4*	img_stats,		//0
	__global	float4*	img,			//1
	__constant	uint*	uint_params,	//2
	__constant 	uint8*	mipmap_params,	//3
	__local		float4*	local_sum_var,	//4
	__global	float4*	global_sum_var	//5
			  )
{
	int global_id 	= (int)get_global_id(0);
	uint pixels 	= uint_params[PIXELS];
	if (global_id > pixels) return;
	uint lid 				= get_local_id(0);
	uint local_size 		= get_local_size(0);
	uint group_size 		= local_size;
	uint reduction			= 1;																	// = mm_cols/read_cols_; but this is only the baselayer ofthe image pyramid.

	uint8 mipmap_params_ = mipmap_params[0];
	uint read_offset_ 	 = mipmap_params_[MiM_READ_OFFSET];
	//
	uint cols 		= uint_params[COLS];
	uint margin 	= uint_params[MARGIN];
	uint mm_cols	= uint_params[MM_COLS];

	uint base_row	= global_id/cols ;
	uint base_col	= global_id%cols ;
	uint img_row	= base_row + margin;
	uint img_col	= base_col + margin;
	uint img_index	= img_row*(cols + 2*margin) + img_col;   										// NB here use cols not mm_cols

	uint read_index = read_offset_  +  base_row  * mm_cols  + base_col  ;							// NB 4 channels.  + margin

	float4 variance = powr( (img[read_index]-img_stats[IMG_MEAN]), 2);								// TODO why does this cause NaNs ?

	int4 var_isnan = isnan(variance);
	if (global_id <= pixels && !var_isnan.x && !var_isnan.y && !var_isnan.z) {
		local_sum_var[lid] 		= variance;
	}else local_sum_var[lid]	= 0;
	///////////////////////// reduction
	int max_iter = ilogb((float)(group_size));

	for (uint iter=0; iter<=max_iter ; iter++) {	// for log2(local work group size)				// problem : how to produce one result for each mipmap layer ?
																									// NB kernels launched separately for each layer, but workgroup size varies between GPUs.
		group_size   /= 2;
		barrier(CLK_LOCAL_MEM_FENCE);																// No 'if->return' before fence between write & read local mem
		if (lid<group_size)  local_sum_var[lid] += local_sum_var[lid+group_size];					// local_sum_var
	}
	if (lid==0) {
		uint group_id 			= get_group_id(0);
		uint global_sum_offset 	= 0; //read_offset_ / local_size ;		// only the base layer								// Compute offset for this layer
		uint num_groups 		= get_num_groups(0);

		float4 layer_data = {num_groups, reduction, 0.0f, 0.0f };			// Write layer data to first entry
		if (global_id == 0) {global_sum_var[global_sum_offset] = layer_data; }
		global_sum_offset += 1+ group_id;

		if (local_sum_var[0][3] >0){																// Using alpha channel local_sum_var[0][3], to count valid pixels being summed.
			global_sum_var[global_sum_offset] = local_sum_var[0] / local_sum_var[0][3];				// Save to global_sum_var // Count hits, and divide group by num hits, without using atomics!
		}else global_sum_var[global_sum_offset] = 0;
	}
}

__kernel void blur_image(
	__private	uint	layer,			//0															// Intitial lyer bluring needs a different buffer to write blurred image to.
	__constant 	uint8*	mipmap_params,	//1															// => the original loading should be to an interim buffer.
	//__constant 	float* 	gaussian,		//2
	__constant 	uint*	uint_params,	//2
	__global 	float4*	img,			//3
	__global	float4* img_blurred,	//4
	__local	 	float4*	local_img_patch //5
			  )
{
	int global_id_u 		= (int)get_global_id(0);
	float global_id_flt 	= global_id_u;
	uint lid 				= get_local_id(0);
	uint group_size 		= get_local_size(0);
	uint patch_length		= group_size+4;
	uint pixels 			= uint_params[PIXELS];

	uint8 mipmap_params_	= mipmap_params[0];
	uint read_offset_ 		= mipmap_params_[MiM_READ_OFFSET];
	uint cols 				= uint_params[COLS];
	uint mm_cols			= uint_params[MM_COLS];

	uint base_row			= global_id_u/cols ;
	uint base_col			= global_id_u%cols ;
	uint read_index 		= read_offset_  +  base_row  * mm_cols  + base_col  ;

	uint read_rows_			= mipmap_params_[MiM_READ_ROWS];
	uint read_cols_ 		= mipmap_params_[MiM_READ_COLS];
	uint read_row   		= global_id_u / read_cols_ ;
	uint read_column		= fmod(global_id_flt, read_cols_);

	for (int i=0, j=-2; i<5; i++, j++){																// Load local_img_patch
		local_img_patch[lid+2 + i*patch_length] = img[ read_index +j*mm_cols];
	}
	////
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
	if ((read_row>read_rows_-3) ||  (read_row < 3)  ){												// Prevents blurring with black space below the image.
		for (int i=0; i<5; i++){
			local_img_patch[lid+2 + i*patch_length] = img[ read_index ];
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);																	// No 'if->return' before fence between write & read local mem
	float4 blurred_pixel = 0;
	for (int i=0; i<5; i++){
		for (int j=0; j<5; j++){
			blurred_pixel += local_img_patch[lid+j + i*patch_length]/25; 							// 5x5 box filter, rather than Gaussian
		}
	}
	if (read_column < 2 || read_column > read_cols_ -3) {
		blurred_pixel = 0;
		for (int i=0; i<5; i++){
			blurred_pixel += local_img_patch[lid+2 + i*patch_length]/5;								// prevents blur wrapping left-right.
		}
	}
	if (read_row>=read_rows_ || global_id_u >= pixels) return;									// num pixels to be written & num threads to really use. // mipmap_params_[MiM_PIXELS]

	blurred_pixel[3] = 1.0f;
	img_blurred[ read_index] = blurred_pixel;
}

__kernel void mipmap_linear_flt4(		// Mipmap layers must be executed sesequentially			// Nvidia Geforce GPUs cannot use "half"
	__private	uint	layer,			//0
	__constant 	uint8*	mipmap_params,	//1
	__constant 	uint*	uint_params,	//2
	__global 	float4*	img,			//3
	__local	 	float4*	local_img_patch //4
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

	uint read_row    	= 2*write_row;
	uint read_column 	= 2*write_column;

	uint read_index 	= read_offset_  +  read_row  * mm_cols  + read_column  ;					// NB 4 channels.  + margin
	uint write_index 	= write_offset_ +  write_row * mm_cols  + write_column ;					// write_cols_, use read_cols_ as multiplier to preserve images  + margin

	//float4 white = {1.0f,1.0f,1.0f,1.0f};
	//float4 black = {0.0f,0.0f,0.0f,0.0f};

	int in_bounds = global_id_u < mipmap_params_[MiM_PIXELS]/2 ;

	if (in_bounds ==1) {
		for (int i=0, j=-2; i<5; i++, j++){															// Load local_img_patch
			local_img_patch[lid+2 + i*patch_length] = img[ read_index +j*mm_cols];
		}
		////
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
	if (in_bounds ==0) return;

	float4 reduced_pixel = 0;
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

	if (write_row>=write_rows_) return;
	if (global_id_u >= mipmap_params_[MiM_PIXELS]) return;											// num pixels to be written & num threads to really use.

	reduced_pixel[3] = 1.0f;
	img[ write_index] = reduced_pixel;
}

 /*
 * 	mipmap_linear_flt(..) only used for depthmap_GT  // This version with 5x5 box blur .
__kernel void mipmap_linear_flt(	// Mipmap layers must be executed sesequentially				// Nvidia Geforce GPUs cannot use "half"
	__private	uint	layer,			//0
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

	if (write_row>=write_rows_) return;
	if (global_id_u >= mipmap_params_[MiM_PIXELS]) return;											// num pixels to be written & num threads to really use.

	img[write_index] =  reduced_pixel;
}
*/

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

__kernel void  img_grad(
	//Inputs:
	__private	uint		layer,			//0
	__constant	uint8*		mipmap_params,	//1
	__constant 	uint*		uint_params,	//2
	__constant 	float*		fp32_params,	//3
	__global 	float4*		img,			//4
	__constant 	float2*		SE3_map,		//5
	//Outputs:
	__global 	float8*		SE3_grad_map,	//6														// We keep hsv sepate at this stage, so 6*4*2=24, but float16 is the largest type, so 6*float8.
	__global 	float8*		HSV_grad		//7
		 )
{
	uint global_id_u 	= get_global_id(0);
	float global_id_flt = global_id_u;

	uint8 mipmap_params_ = mipmap_params[layer];
	uint read_offset_ 	= mipmap_params_[MiM_READ_OFFSET];
	uint read_cols_ 	= mipmap_params_[MiM_READ_COLS];
	uint read_rows_ 	= mipmap_params_[MiM_READ_ROWS];

	uint base_cols		= uint_params[COLS];
	uint margin 		= uint_params[MARGIN];
	uint mm_cols		= uint_params[MM_COLS];
	uint mm_pixels		= uint_params[MM_PIXELS];

	uint read_row    	= global_id_u / read_cols_;
	uint read_column 	= fmod(global_id_flt, read_cols_);
	uint read_index 	= read_offset_  +  read_row  * mm_cols  + read_column ;						// NB 4 channels.  + margin

	/// adapted
	int upoff			= -(read_row  >1 )*mm_cols;													//-(read_row  != 0)*mm_cols;	// up, down, left, right offsets, by boolean logic.
	int dnoff			=  (read_row  < read_rows_-2) * mm_cols;									// (read_row  < read_rows_-1) * mm_cols;
    int lfoff			= -(read_column >1);														//-(read_column != 0);
	int rtoff			=  (read_column < read_cols_-2);											// (read_column < mm_cols-1);
	uint offset			=   read_column + read_row  * mm_cols + read_offset_ ;
                                                                                                    // DTAM paper beta for optimization, (not for g1 edges): beta=0.001 while theta>0.001, else beta=0.0001
	float alphaG		= fp32_params[ALPHA_G];														// "alpha_g":0.015,//15,	// can vary 15 to 0.15  // 0.015
	float betaG 		= fp32_params[BETA_G];														// "beta_g": 1.5,			//my dtam_opencl 1.5

	float4 pu, pd, pl, pr;
	pr 					=  img[offset + rtoff];
	pl 					=  img[offset + lfoff];
	pu 					=  img[offset + upoff];
	pd 					=  img[offset + dnoff];

	float4 gx			= { (pr.x - pl.x), (pr.y - pl.y), (pr.z - pl.z), 1.0f };					// Signed img gradient in hsv
	float4 gy			= { (pd.x - pu.x), (pd.y - pu.y), (pd.z - pu.z), 1.0f };

	uint reduction		= mm_cols/read_cols_;
	uint v    			= global_id_u / read_cols_;													// read_row
	uint u 				= fmod(global_id_flt, read_cols_);											// read_column

	for (uint i=0; i<6; i++) {
		float2 SE3_px =  SE3_map[read_index + i* mm_pixels];										// SE3_map[read_index + i* uint_params[MM_PIXELS]  ] = partial_gradient;
																									// float2 partial_gradient={u_flt-u2 , v_flt-v2}; // Find movement of pixel
		float8 SE3_grad_px = {gx*SE3_px[0]  , gy*SE3_px[1] };										// NB float4 gx, gy => float8
		SE3_grad_map[read_index + i* mm_pixels]  =  SE3_grad_px ;// * rotation_wt;					// NB apply inv depth to ST3 (i=0,1,2) in tracking stage.
	}
/*
	for (uint i=3; i<6; i++) {
		float2 SE3_px =  SE3_map[read_index + i* mm_pixels];										// SE3_map[read_index + i* uint_params[MM_PIXELS]  ] = partial_gradient;
																									// float2 partial_gradient={u_flt-u2 , v_flt-v2}; // Find movement of pixel
		float8 SE3_grad_px = {gx*SE3_px[0]  , gy*SE3_px[1] };										// NB float4 gx, gy => float8
		SE3_grad_map[read_index + i* mm_pixels]  =  SE3_grad_px ;// * inv_depth ;		// inv_depth_map, only available for keyframe. Apply later in tracking.
	}
*/
	float H = img[offset][0] * 2*M_PI_F;
	float S = img[offset][1];
	float V = img[offset][2];
	float8 temp_float8  = { sin(H) , cos(H), S, V, gx[1], gy[1], gx[2], gy[2] };					// HSV_grad = { sin(H) , cos(H), S, V, gx[1], gy[1], gx[2], gy[2] };
	HSV_grad[offset] = temp_float8;
}
