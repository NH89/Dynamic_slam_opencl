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

	int v 							= global_id  / read_cols_;						// read_row
	int u 							= fmod(global_id_flt, read_cols_);				// read_column

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
	float 	global_id_flt		= global_id;
	float4 	lookup_ref			= lookup_table[global_id + read_offset];
	uint 	read_index			= floor(lookup_ref.z);
	float2 	warp2				= warp[read_index];
	float	u_mod				= fmod(warp2.x, 1);
	float	v_mod				= fmod(warp2.y, 1);
	int		u_int				= floor(warp2.x);
	int		v_int				= floor(warp2.y);
	uint	sample_index		= read_index  + (v_int * mm_cols)  + u_int ;

	// 2-way linear interpolation of four sample pixels.
	float4 warped_lower			= new_img[sample_index] 		* (1-u_mod) 	+ new_img[sample_index+1] 			* u_mod;
	float4 warped_upper			= new_img[sample_index+mm_cols] * (1-u_mod) 	+ new_img[sample_index+mm_cols+1] 	* u_mod;
	new_img_warped[read_index]	= warped_lower 					* (1-v_mod)  	+ warped_upper 						* v_mod;

// 	if (0.0f == fmod(global_id_flt, 33.0f) ){
// 		printf("\n__kernel void warp_image(..): warp2=(%f,%f), global_id=%u, lookup_ref.x=%f lookup_ref.y=%f u_int=%u, u_mod=%f, read_index=%u", \
// 		warp2.x, warp2.y, global_id, lookup_ref.x, lookup_ref.y, u_int, u_mod, read_index);
// 	}
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
	uint 	read_index	= floor( lookup_table[ get_global_id(0) + read_offset ].z );
	if (read_index ==0 ) return;
	float4 	pixel 		= 0;
	float4 	var			= 0;
	float 	W[9] 		= { 1.0f/16, 2.0f/16, 1.0f/16, 2.0f/16, 4.0f/16, 2.0f/16, 1.0f/16, 2.0f/16, 1.0f/16 }; 			// 3x3 discrete Gaussian kernel

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

	for (int i=0; i<9; i++ ){	var	+= img_sq[ read_index_3x3[i] ] /* * W[i]*/; }
	img_var[read_index]	= var;
}

//////////////////////////////////////////////////////////////////////////////////////

__kernel void mean_sq_3rows(
	// inputs
	__private	uint	read_offset,			//0

	__global 	float4*	lookup_table,			//1
	__global 	float4*	img,					//2
	// output
	__global 	float4*	mean_sq_rows,			//3
	__global 	float4*	sq_mean_rows			//4
){
	uint 	read_index	= floor( lookup_table[ get_global_id(0) + read_offset ].z );
	if (read_index ==0 ) return;

	float4 mean_local				= {0,0,0,0};
	float4 mean_of_squares			= {0,0,0,0};
	float 	W[3] 					= {1/4, 2/4, 2/4};		//1x3 discrete gaussian weights, instead of equal weighting, improves insensitivity to rotation.

	for (int col=0;col<3;col++){
		float4 pix_val 				= 	img[read_index + col -1];
		mean_local 					+= 	W[col] * pix_val;
		mean_of_squares 			+=	W[col] * pix_val * pix_val;
	}
	mean_sq_rows[read_index]		= 	(mean_local * mean_local);
	sq_mean_rows[read_index]		= 	mean_of_squares;
}

__kernel void mean_sq_cols(
	// inputs
	__private	uint	read_offset,			//0
	__private	uint	mm_cols,				//1

	__global 	float4*	lookup_table,			//2
	__global 	float4*	mean_sq_rows,			//3
	__global 	float4*	sq_mean_rows,			//4
	// output
	__global 	float4*	mean,					//5
	__global 	float4*	sq_mean					//6
){
	uint 	read_index	= floor( lookup_table[ get_global_id(0) + read_offset ].z );
	if (read_index ==0 ) return;

	float4 mean_local				= {0,0,0,0};
	float4 mean_of_squares			= {0,0,0,0};
	float 	W[3] 					= {1/4, 2/4, 2/4};		//1x3 discrete gaussian weights, instead of equal weighting, improves insensitivity to rotation.

	for (int col=0;col<3;col++){
		mean_local 					+= 	W[col] * mean_sq_rows[read_index + (col-1)*mm_cols];
	}
	mean[read_index]				= 	mean_local;

	for (int col=0;col<3;col++){
		float4 pix_val 				= 	sq_mean_rows[read_index + (col-1)*mm_cols];
		mean_of_squares 			+=	W[col] * pix_val * pix_val;
	}
	sq_mean[read_index]				= 	mean_of_squares;
}

__kernel void co_mean_rows(
	// inputs
	__private	uint	read_offset,			//0
	__private	uint	mm_cols,				//1

	__global 	float4*	lookup_table,			//1
	__global 	float4*	ref_img,				//2
	__global 	float4*	warped_img,				//3
	// output
	__global 	float4*	co_mean_rows			//4
){
	uint 	read_index									= floor( lookup_table[ get_global_id(0) + read_offset ].z );
	if (read_index ==0 ) 								return;
	uint 	offset[5] 									= { -mm_cols, -1, 0, 1, mm_cols };
	float 	W[3]										= {1/4, 2/4, 2/4};
														//1x3 discrete gaussian weights, instead of equal weighting, improves insensitivity to rotation.

	for (int sample=0;sample<5;sample++){
		float4 co_mean									=	{0,0,0,0};
		for (int col=0;col<3;col++){
			float4 X 									= 	ref_img[	read_index + (col-1) ];
			float4 Y 									= 	warped_img[	read_index + (col-1) + offset[sample] ];
			co_mean 									+= 	W[col] * X * Y ;
		}
		co_mean_rows[ read_index + offset[sample] ] 	= co_mean;
	}
}

__kernel void covariance_cols(
	// inputs
	__private	uint	read_offset,			//0
	__private	uint	mm_cols,				//1

	__global 	float4*	lookup_table,			//2
	__global 	float4*	co_mean_rows,			//3

	__global 	float4*	ref_img_mean,			//4
	__global 	float4*	ref_img_sq_mean,		//5

	__global 	float4*	warped_img_mean,		//6
	__global 	float4*	warped_img_sq_mean,		//7

	// output
	__global 	float4*	correlation				//8
){
	uint 	read_index									= floor( lookup_table[ get_global_id(0) + read_offset ].z );
	if (read_index ==0 ) 								return;
	uint	offset[5] 									= { -mm_cols, -1, 0, 1, mm_cols };
	float 	W[3]										= {1/4, 2/4, 2/4};					//1x3 discrete gaussian weights, improves insensitivity to rotation.
	float4	ref_img_mean_local							= ref_img_mean[read_index];
	float4	ref_img_denominator							= sqrt( pow(ref_img_mean_local, 2)	 - ref_img_sq_mean[read_index] ) ;

	for (int sample=0;sample<5;sample++){
		float4 co_mean									= {0,0,0,0};
		for (int col=0;col<3;col++){
			co_mean 									+= 	W[col] * co_mean_rows[	read_index + (col-1) + offset[sample] ];
		}
		float4 warped_img_mean_local 					= warped_img_mean[read_index + offset[sample]];
		float4 warped_img_denominator 					= sqrt( pow( warped_img_mean_local, 2) - warped_img_sq_mean[read_index + offset[sample]] ) ;
		// TODO NB chk denominator != 0.0f

		co_mean_rows[ read_index + offset[sample] ] 	= (co_mean - ref_img_mean_local * warped_img_mean_local ) / ( ref_img_denominator * warped_img_denominator );
	}
}


///////////////////////////////////////////////////////////////////////////////////////

/*
// float compute_maximum(__private float4 A, __private float4 B, __private float4 C){
// 	float 	steps[3]  = {-1, 0, 1};
// 	float a, b, c,   d, e,   f, g,   h, i,   j, k, d2, f2, h2, prediction, optimum;										// compute x value of the optimum of parabola, y= a*x*x + b*x + c
// 																														// given samples at x=1,2,4
// 	d = steps[0];	e = A.x ;	// d=-1	// TODO  which combination of color channels ?
// 	f = steps[1];	g = B.x ;	// f=0
// 	h = steps[2];	i = C.x ;	// h=1
//
// 	d2 = d * d;		// 1
// 	f2 = f * f;		// 0
// 	h2 = h * h;		// 1
//
// 	j = (f2 - d2)*(f-h) - (h2 - f2)*(d-f);		// 0 - d-f 				= 1
// 	k = (g-i)*(d-f) - (e-g)*(f-h);				// B-C + (A-B) 			= A-C
// 	a = k/j;									// 						= A-C
// 	b = (e-g +a*(f2-d2))  /  (d-f);				// (A-B  +  A-C)/-1		= B+C-2*A
// 	c = e - a*d2 - b*d;							// A -A-C + B+C-2*A 	= B-2*A
//
// 	if (a<0){																											// IF concavity leads to a maximum, use it.
// 		float x 		= -b /(2*a);			//
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
float compute_maximum(__private float4 A, __private float4 B, __private float4 C){
	// from https://math.stackexchange.com/questions/2150199/is-there-a-method-for-estimating-the-parabolic-function-using-three-points-or-a
	float x1=-1,	x2=0,	x3=1;
	float y1=A.x,	y2=B.x,	y3=C.x;

	float k1 = y1/((x1-x2)*(x1-x3));
	float k2 = y2/((x2-x1)*(x2-x3));
	float k3 = y3/((x3-x2)*(x3-x1));

	float optimum = (k1*(x2+x3) + k2*(x1+x3) + k3*(x2+x1)) / (2*(k1+k2+k3));
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
	__global 	float2*	warp					//10	// 2*float4*mm_size // float2*
){
	uint read_index		= lookup_table[ get_global_id(0) + read_offset ].z;
 	if (read_index ==0 ) {
 		//printf("\n_kernel compute_warp(..), mm_size=%u",mm_size);
 		return;	// NB required, or it will crash.
 	}
	float W[9] 			= { 1.0f/16, 2.0f/16, 1.0f/16, 2.0f/16, 4.0f/16, 2.0f/16, 1.0f/16, 2.0f/16, 1.0f/16 };			// 3x3 discrete Gaussian kernel
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
	sample_idx[1] 		=	-1;
	sample_idx[2] 		=	0;
	sample_idx[3] 		=	+1;
	sample_idx[4] 		=	+mm_cols;

	for (int j=0; j<5; j++){
		for (int i=0; i<9; i++ ){ covar[j]		+= curr_img[read_index_3x3[i] ] * new_img[read_index_3x3[i] + sample_idx[j] ] *  W[i]; }
		img_covar[read_index + j*mm_size]		= covar[j];

		float4 inv_denominator					= ( sqrt( curr_img_var[read_index] ) * sqrt( new_img_var[read_index + sample_idx[j] ]  ) );
		float4 denominator						= isnormal(inv_denominator) ? 1/inv_denominator : 1;					// prevent div by zero

		corr[j] 								= covar[j] / denominator ;
		img_corr[read_index + j*mm_size]		= corr[j];
	}

 	float2 warp2								= warp[read_index];
 	float warp_u								= warp2.x + compute_maximum( corr[1], corr[2], corr[3] );
 	float warp_v								= warp2.y + compute_maximum( corr[0], corr[2], corr[4] );
 	warp_u										= clamp(warp_u, -1.0f, 1.0f);
 	warp_v										= clamp(warp_v, -1.0f, 1.0f);					// warp increment clamped to +/-1

	float2 warp2_new							= {warp_u, warp_v};
	warp[read_index]							= warp2_new;
}

 __kernel void propagate_warp(
	// inputs
	__private	uint	read_offset,			//0
	__private	uint 	rows_in,				//1
	__private	uint	cols_in,				//2
	__private	uint	write_offset,			//3  // rather send the mm_offset
	__private	uint	mm_cols,				//4

	__global 	float4*	lookup_table,			//5
	// input_output
	__global 	float2*	warp					//6		// 2*float4*mm_size // float2*
){
	uint 	global_id		= get_global_id(0);
	float 	global_id_flt 	= global_id;
	float4 	lookup_in		= lookup_table[ global_id + read_offset  ];
	uint read_col			= lookup_in.x;
	uint read_row			= lookup_in.y;
	uint read_index			= lookup_in.z;

	uint row 				= global_id / cols_in;
	uint col 				= fmod( global_id_flt, cols_in);
	uint write_index		= col*2 + row*2*mm_cols +  lookup_table[ write_offset ].z;  //

	float2 warp_in			= warp[read_index];
	float2 warp_in_up		= warp[read_index - mm_cols];
	float2 warp_in_left		= warp[read_index - 1];
	float2 warp_in_right	= warp[read_index + 1];
	float2 warp_in_down		= warp[read_index + mm_cols];

	float2 warp_out_ld, warp_out_rd, warp_out_lu, warp_out_ru;

	if (global_id < rows_in*cols_in) {
		/*if (col==0 || row==0 || col==cols_in || row== rows_in ){		// If at margin: copy directly,
			warp_out_ld = warp_in;
			warp_out_rd = warp_in;
			warp_out_lu = warp_in;
			warp_out_ru = warp_in;
		}else*/{																					// else: interpolate with neighbors.
			warp_out_ld = warp_in*0.5f +  warp_in_left *0.25f +  warp_in_down*0.25f;
			warp_out_rd = warp_in*0.5f +  warp_in_right*0.25f +  warp_in_down*0.25f;
			warp_out_lu = warp_in*0.5f +  warp_in_left *0.25f +  warp_in_up  *0.25f;
			warp_out_ru = warp_in*0.5f +  warp_in_right*0.25f +  warp_in_up  *0.25f;
		}
		float2 flt2_ones	= {col, row}; //{1.0f, 1.0f}; //{read_index,write_index}; //

		warp[write_index]				= warp_out_lu;	// flt2_ones; //
		warp[write_index +1]			= warp_out_ru;
		warp[write_index + mm_cols]		= warp_out_ld;//warp_out_lu;
		warp[write_index + mm_cols +1]	= warp_out_rd;//warp_out_ru;

		if (fmod(global_id_flt,333.0f) ==0.0f ) {
			printf("\n_kernel propagate_warp(..), global_id=%u, read_offset=%u, rows_in=%u, cols_in=%u, write_offset=%u, write_index=%u, mm_cols=%u, read_col=%u, read_row=%u, col=%u, row=%u",\
			global_id, read_offset, rows_in, cols_in, write_offset, write_index, mm_cols, read_col, read_row, col, row);
		}
	}
	//barrier(CLK_GLOBAL_MEM_FENCE);
 }

/*
// 	float4 warp_u_f4							= corr[1] - corr[3];
// 	//float4 warp_v								= corr[0] - corr[4];
// 	warp_u_f4.w									= 1;
// 	//warp_v.w									= 1;
// 	float4 warp_uv_f4							= {warp_u, warp_v, 0.0f, 1.0f};

// 	warp[read_index]							= warp_u_f4;			//warp2_new;
// 	warp[read_index + mm_size]					= warp_uv_f4;
*/

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
