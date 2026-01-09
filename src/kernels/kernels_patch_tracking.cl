#include "kernels__macros.h"
#include "kernels.h"

// old kernels - may req conversion to patch operation, lookup tables etc.



// new kernels /////////////////////////////////
// Image pyramid
__kernel void reduce_img(
	__private	uint	offset1,			//0	top left corner source image
	__private	uint	offset2,			//1	top left corner dest image

	__private	uint	buf_width,			//2 mm_cols, i.e. width of the buffer holding the image pyramid
	__private	uint	img_pixels,			//3 num rows of this level of the image pyramid
	__private	uint	img_cols,			//4

	__private	uint	img_rows,			//5
	__private	uint	patch_height,		//6 patch height
	__private	uint	stop_offset,		//7 bottom right corner of dest image

	__global 	float4*	img					//8
	)
{
	int global_id_u 			= (int)get_global_id(0);
	//if (global_id_u > img_pixels/patch_height)	return;

	uint row					= global_id_u / img_cols;
	uint col					= fmod((float)global_id_u, img_cols);

	uint read_idx 				= offset1 + col*2 + (row*2 * buf_width * patch_height);
	uint write_idx 				= offset2 + col   + (row   * buf_width * patch_height);

// 	if (global_id_u==(img_cols-1)) printf("\n__kernel void reduce_img()  read_idx=%u 	= offset1=%u + col=%u*2 + (row=%u*2 * buf_width=%u * patch_height=%u),   write_idx=%u,	img_pixels=%u,   stop_offset=%u",
// 						 													read_idx,		offset1,	col,		row,		buf_width,		patch_height,		write_idx,	img_pixels,			stop_offset);
	for (int i=0; i<patch_height; i++){
		if (write_idx >= stop_offset) return;
		img[write_idx ]			= (  img[read_idx ] + img[read_idx +1]   + img[read_idx + buf_width] + img[read_idx + buf_width +1] )  /4.0f;
		read_idx 				+= buf_width*2;
		write_idx 				+= buf_width;
		//if (global_id_u==0) printf("\n__kernel void reduce_img()  read_idx=%u  write_idx=%u", read_idx, write_idx);
	}
}




// Add blurred top layers to image pyramid
/*
// NB used 3x3 box blur and 6 iterations per level in the Python code.

/////3x3 box blur kernekls //////
__kernel void pad_image_top_bottom3(			// Apply before vertical blur
	__private	uint	offset1,			//0	top left corner
	__private	uint	offset2,			//1 bottom left corner
	__private	uint	buf_width,			//2 mm_cols, i.e. width of the buffer holding the image pyramid
	__private	uint	img_pixels,			//3 num rows of this level of the image pyramid

	__global 	float4*	img					//4
		)
{
	int global_id_u 		= (int)get_global_id(0);
	if (global_id_u > img_pixels) 	return;

	uint read_idx 			= global_id_u + offset1;					// pad top
	float4 pixel			= img[read_idx ];
	pixel.w					= 0.5;
	img[read_idx - buf_width]	= pixel;

	read_idx				= global_id_u + offset2;					// pad bottom
	pixel					= img[read_idx ];
	pixel.w					= 0.5;
	img[read_idx + buf_width]	= pixel;
}


__kernel void vertcal_blur3(
	__private	uint	offset1,			//0	top left corner
	__private	uint	buf_width,			//1 mm_cols, i.e. width of the buffer holding the image pyramid
	__private	uint	img_pixels,			//2 num rows of this level of the image pyramid
	__private	uint	img_cols,			//3

	__global 	float4*	img,				//4
	__global 	float4*	tmp_img				//5
	)
{
	int global_id_u 		= (int)get_global_id(0);
	if (global_id_u > img_pixels) 	return;

	uint row				= global_id_u / img_cols;
	uint col				= fmod((float)global_id_u, img_cols);
	uint read_idx 			= offset1 + col + (row * img_cols);

	float4 pixel			= ( img[read_idx - buf_width] + img[read_idx ] + img[read_idx + buf_width ] )/3.0f;
	pixel.w					= 1.0;
	tmp_img[read_idx ]		= pixel;
}


__kernel void pad_image_left_right3(			// Apply before horizontal blur
	__private	uint	offset1,			//0 top left corner
	__private	uint	offset2,			//1 top right corner
	__private	uint	buf_width,			//2 mm_cols, i.e. width of the buffer holding the image pyramid
	__private	uint	img_pixels,			//3 num rows of this level of the image pyramid

	__global 	float4*	img					//4
		)
{
	int global_id_u 		= (int)get_global_id(0);
	if (global_id_u > img_pixels) 	return;

	uint read_idx 			= (global_id_u * buf_width) + offset1;		// pad lhs
	float4 pixel			= img[read_idx ];
	pixel.w					= 0.5;
	img[read_idx - 1]		= pixel;

	read_idx				= (global_id_u * buf_width) + offset2;		// pad rhs
	pixel					= img[read_idx -1];
	pixel.w					= 0.5;
	img[read_idx ]			= pixel;
}


__kernel void horiz_blur3(
	__private	uint	offset1,			//0	top left corner
	__private	uint	buf_width,			//1 mm_cols, i.e. width of the buffer holding the image pyramid
	__private	uint	img_pixels,			//2 num rows of this level of the image pyramid
	__private	uint	img_cols,			//3

	__global 	float4*	img,				//4
	__global 	float4*	tmp_img				//5
	)
{
	int global_id_u 		= (int)get_global_id(0);
	if (global_id_u > img_pixels) 	return;

	uint row				= global_id_u / img_cols;
	uint col				= fmod((float)global_id_u, img_cols);
	uint read_idx 			= offset1 + col + (row * img_cols);

	float4 pixel			= ( img[read_idx - 1] + img[read_idx ] + img[read_idx + 1 ] )/3.0f;
	pixel.w					= 1.0;
	tmp_img[read_idx ]		= pixel;
}
*/
///////////////////////////////////
/////5_x_5 box blur kernekls //////

__kernel void pad_image_top_bottom2(			// Apply before vertical blur
	__private	uint	offset1,			//0	top left corner
	__private	uint	offset2,			//1 bottom left corner
	__private	uint	buf_width,			//2 mm_cols, i.e. width of the buffer holding the image pyramid
	__private	uint	img_cols,			//3 num rows of this level of the image pyramid

	__global 	float4*	img					//4
		)
{
	int global_id_u 			= (int)get_global_id(0);
	if (global_id_u > img_cols) return;

	uint read_idx 				= global_id_u + offset1;					// pad top
	float4 pixel				= img[read_idx ];
	img[read_idx - buf_width]	= pixel;
	img[read_idx - buf_width*2]	= pixel;

	read_idx					= global_id_u + offset2;					// pad bottom
	pixel						= img[read_idx ];
	img[read_idx + buf_width]	= pixel;
	img[read_idx + buf_width*2]	= pixel;
}


__kernel void vertcal_blur5(
	__private	uint	offset1,			//0	top left corner
	__private	uint	buf_width,			//1 mm_cols, i.e. width of the buffer holding the image pyramid
	__private	uint	img_pixels,			//2 num rows of this level of the image pyramid
	__private	uint	img_cols,			//3
	__private	uint	stop_offset,		//4

	__global 	float4*	img,				//5
	__global 	float4*	tmp_img				//6

	)
{
	int global_id_u 			= (int)get_global_id(0);
	//if (global_id_u > img_pixels/32 )	return;

	uint row					= global_id_u / img_cols;
	uint col					= fmod((float)global_id_u, img_cols);
	uint read_idx 				= offset1 + col + (row * 32 * buf_width);
	float4 pixel				= 0;

// 	if (global_id_u==(img_cols-1)) printf("\n__kernel void vertcal_blur5()  read_idx=%u 	offset1=%u,  col=%u,  row=%u,  buf_width=%u,     img_pixels=%u,   img_cols=%u,   stop_offset=%u",
// 																			read_idx,		offset1,		col,	row,	buf_width,		img_pixels,		img_cols,		stop_offset);

	for (int i =0; i<32; i++){
		if( read_idx > stop_offset ) return;
		pixel = ( img[read_idx - buf_width*2] + img[read_idx - buf_width] + img[read_idx ] + img[read_idx + buf_width ] + img[read_idx + buf_width*2 ] )/5.0f;

		tmp_img[read_idx ]		= pixel;
		read_idx				+= buf_width;

// 		if (global_id_u==(img_cols-1)) printf("\n__kernel void vertcal_blur5()  read_idx=%u ,  buf_width=%u,     i=%u,   stop_offset=%u",
// 																				read_idx,		buf_width,		i,		stop_offset);
	}
}


__kernel void pad_image_left_right2(			// Apply before horizontal blur
	__private	uint	offset1,			//0 top left corner
	__private	uint	offset2,			//1 top right corner
	__private	uint	buf_width,			//2 mm_cols, i.e. width of the buffer holding the image pyramid
	__private	uint	img_rows,			//3 num rows of this level of the image pyramid

	__global 	float4*	tmp_img					//4
		)
{
	int global_id_u 			= (int)get_global_id(0);
	if (global_id_u > img_rows) 	return;

	uint read_idx 				= (global_id_u * buf_width) + offset1;		// pad lhs
	float4 pixel				= tmp_img[read_idx ];

	tmp_img[read_idx - 1]		= pixel;
	tmp_img[read_idx - 2]		= pixel;

	read_idx					= (global_id_u * buf_width) + offset2;		// pad rhs
	pixel						= tmp_img[read_idx -1];

	tmp_img[read_idx ]			= pixel;
	tmp_img[read_idx +1 ]		= pixel;
}


__kernel void horiz_blur5(
	__private	uint	offset1,			//0	top left corner
	__private	uint	buf_width,			//1 mm_cols, i.e. width of the buffer holding the image pyramid
	__private	uint	img_pixels,			//2 num rows of this level of the image pyramid
	__private	uint	img_cols,			//3
	__private	uint	stop_offset,		//4

	__global 	float4*	tmp_img,			//5
	__global 	float4*	img					//6
	)
{
	int global_id_u 			= (int)get_global_id(0);
	//if (global_id_u==0){	printf("\n__kernel void horiz_blur5(..)" ); }
	//if (global_id_u > img_pixels)	return;

	uint row					= global_id_u / img_cols;
	uint col					= fmod((float)global_id_u, img_cols);
	uint read_idx				= offset1 + col + (row * 32 * buf_width);
	//if( read_idx > stop_offset ) return;

	if (global_id_u < img_cols){						// erase the previous top & bottom padding.
		img[read_idx - buf_width*2]			= zero_f4;
		img[read_idx - buf_width]			= zero_f4;

		uint tmp_idx = stop_offset + global_id_u - img_cols;
		img[tmp_idx + buf_width]			= zero_f4;
		img[tmp_idx + buf_width*2]			= zero_f4;
	}
	barrier( CLK_GLOBAL_MEM_FENCE );

	for (int i =0; i<32; i++){							// Write fully blurred img back to current imgmem.
		if( read_idx > stop_offset ) return;
		float4 pixel			= ( tmp_img[read_idx - 2] + tmp_img[read_idx - 1] + tmp_img[read_idx ] + tmp_img[read_idx + 1 ] + tmp_img[read_idx + 2 ] )/5.0f;
		img[read_idx ]			= pixel;
		read_idx				+= buf_width;
	}
}



// Image gradietnt d(value)/d(u,v)



// d(u,v)/d(se3)



// Jacobian J = d(value)/d(se3)  = d(value)/d(u,v)  *  d(u,v)/d(se3)



// Pixels weights, W = Huber norm on Rho



// Rho photometric error



// Weighted Gauss-Newton aprox Hessian = sum(Jt W J)



// Pseudo inverse or inverse of 6x6 matrix.




// Compute se3 update = WGN_inv  *  sum(W J Rho)



// Scale update wrt delta_se3  [1,1,1,theta,theta,theta]



// Apply update, ie compute new SE3 and new K2K



// Test if (SSD > old_SSD || isnan(SSD) )
/*	change pyramid layer + reduction
 *  or use separate depth maps & k, inv_k for each scale.
 */



/* Repeat using more frames, for:
 *(i) Camera matrix k & lens distortion params
 *(ii) Depth pyramid
 *(iii) Rel_vel & Accel, & camera accel & jolt
 *(iv) Reflectance & illumination map
 * (ii-iv) using
 * (a) parsimony
 * (b) local smoothing
 * (c) edge preseeving - anisotropy wrt edges
 * (d) penalize mutually cancelling information wave artifacts
 *
 *
 * NB Foveation & ability to direct attention i.e. move the fovea.
 * Also for each patch, find pixels with max gradient in u, v, uv, -uv directions, i.e. octagon sample set.
 * Use semi-sparse to accelerate lower img pyr levels.
 */
