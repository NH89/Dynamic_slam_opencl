#include "kernels_macros.h"
#include "kernels.h"

 __kernel void compute_lookup_table(					// computed once at start of program	// TODO when is it possible to roll the layers together ?  i.e. when local mem is not used.
	 // inputs
	__private	uint	layer,					//0
	__private	uint	lookup_table_offset,	//1

	__constant 	uint8*	mipmap_params,			//2
	__constant 	uint*	uint_params,			//3
	__constant  float*  fp32_params,			//4

	// output
	__global 	float4*	lookup_table			//5
){
	uint  global_id					= get_global_id(0);
	float global_id_flt 			= global_id;

	uint8 mipmap_params_			= mipmap_params[layer];
	uint read_offset_ 				= mipmap_params_[MiM_READ_OFFSET];
	uint read_cols_ 				= mipmap_params_[MiM_READ_COLS];
	uint pixels_ 					= mipmap_params_[MiM_PIXELS];

	uint mm_cols					= uint_params[MM_COLS];
// 	uint pixels						= uint_params[PIXELS];
// 	uint mm_pixels					= uint_params[MM_PIXELS];

	int v 							= global_id  / read_cols_;												// read_row
	int u 							= fmod(global_id_flt, read_cols_);										// read_column

	int read_index 					= read_offset_  +  v  * mm_cols  + u ;
	int alpha						= 255;
	float4 lookup 					= {u,v,read_index,alpha};

	if ( global_id > pixels_)		{lookup = 0;alpha=255;read_index=0;u=0;v=0;}	// NB read_index=mm_cols+3 is an unused index on the mipmap, and safe for a 3x3 sample.
// uint group_id = get_group_id(0);
// uint local_id = get_local_id(0);
// if (/*group_id==0 && local_id==0*/global_id==0) printf("  __kernel compute_lookup_table(..) layer = %u, group_id = %u, global_id = %u, lookup=%d,%d,%d,%d,  pixels_=%u, pixels=%u,  mm_pixels=%u, read_offset_=%u, global_id_offset=%u", \
// 	layer, group_id, global_id, lookup.x, lookup.y, lookup.z, lookup.w,  pixels_, pixels, mm_pixels, read_offset_, lookup_table_offset); //

	lookup_table[global_id + lookup_table_offset]	= lookup;	//read_index;//											// pixel idex in mipmap
}


__kernel void warp_image(						// Computed once each iteration of warping, for each layer of image pyramid
	// inputs									// Warp describes where to sample the new image to match the reference image.
	__private	uint	read_offset,			//0
	__private	uint	mm_cols,				//1

	__constant 	uint*	uint_params,			//2

	__global 	float2*	warp,					//3
	__global 	float4*	lookup_table,			//4
	__global 	float4*	new_img,				//5

	// outputs
	__global 	float4*	new_img_warped			//6
){
	uint	global_id			= get_global_id(0);
	float4 	lookup_ref			= lookup_table[global_id + read_offset];
	uint 	read_index			= floor(lookup_ref.z);
	float2 	warp2				= warp[read_index];
	float 	u_flt 				= lookup_ref.x 	+ warp2.x;
	float 	v_flt 				= lookup_ref.y	+ warp2.y;
	float 	u_mod				= fmod(u_flt , 1);
	float 	v_mod				= fmod(v_flt , 1);
	uint 	u 					= floor(u_flt);
	uint 	v 					= floor(v_flt);
	uint	sample_index		= read_index  +  (v * mm_cols)  + u ;

	// 2-way linear interpolation of four sample pixels.
	float4 warped_lower			= new_img[read_index] 			* (1-u_mod) 	+ new_img[read_index+1] 		* u_mod;
	float4 warped_upper			= new_img[read_index+mm_cols] 	* (1-u_mod) 	+ new_img[read_index+mm_cols+1] * u_mod;
	new_img_warped[read_index]	= warped_lower 					* (1-v_mod)  	+ warped_upper 					* v_mod;
}


 __kernel void img_sq(
	// inputs
	__private	uint	read_offset,			//0

	__global 	float4*	lookup_table,			//1
	__global 	float4*	img,					//2
	// output
	__global 	float4*	img_sq					//3
){
	uint read_index						= lookup_table[ get_global_id(0) + read_offset ].z;
	if (read_index ==0 ) return;
	float4 pixel 						= img[read_index];
	img_sq[read_index]					= pixel*pixel;
}


 __kernel void img_variance(		// TODO ? should variance be difference from the mean of the 3x3 patch ?
	// inputs
	__private	uint	read_offset,			//0
	__private	uint	mm_cols,				//1

	__global 	float4*	lookup_table,			//2
	__global 	float4*	img_sq,					//3

	// output
	__global 	float4*	img_var					//4
){
	uint 	read_index					= floor( lookup_table[ get_global_id(0) + read_offset ].z );
	if (read_index ==0 ) return;
	float4 	pixel 						= 0;
	float4 	var							= 0;
	float 	W[9] 						= { 1,2,1,2,4,2,1,2,1 }; 					// 3x3 discrete Gaussian kernel

	uint 	read_index_3x3[9];
	read_index_3x3[1]	=	read_index 			-mm_cols;
	read_index_3x3[0]	=	read_index_3x3[1] 	-1;
	read_index_3x3[2]	=	read_index_3x3[1] 	+1;

	read_index_3x3[4]	=	read_index;
	read_index_3x3[3]	=	read_index 			-1;
	read_index_3x3[5]	=	read_index 			+1;

	read_index_3x3[7]	=	read_index 			+mm_cols;
	read_index_3x3[6]	=	read_index_3x3[7]	-1;
	read_index_3x3[8]	=	read_index_3x3[7] 	+1;

// 	if (get_global_id(0)==301||get_global_id(0)==0){
// 		printf("\n__kernel img_variance(..), read_index_3x3=\n%u, %u, %u,\n%u, %u, %u,\n%u, %u, %u\n",\
// 									read_index_3x3[0],read_index_3x3[1],read_index_3x3[2],\
// 									read_index_3x3[3],read_index_3x3[4],read_index_3x3[5],\
// 									read_index_3x3[6],read_index_3x3[7],read_index_3x3[8]\
// 									);}

	for (int i=0; i<9; i++ ){	var	+= img_sq[ read_index_3x3[i] ] * W[i]; }
	img_var[read_index]	= var;
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
	__private	uint	read_offset,			//0
	__private	uint 	mm_size,				//1
	__private	uint	mm_cols,				//2

	__global 	float4*	lookup_table,			//3
	__global 	float4*	curr_img,				//4
	__global 	float4*	new_img,				//5																		// NB warped version of the new image.
	__global 	float4*	curr_img_var,			//6
	__global 	float4*	new_img_var,			//7

	// output
	__global 	float4*	img_covar,				//8		// 5*float4*mm_size
	__global 	float4*	img_corr,				//9		// 5*float4*mm_size
	__global 	float2*	warp					//10
){
	uint read_index		= lookup_table[ get_global_id(0) + read_offset ].z;
	if (read_index ==0 ) return;

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
