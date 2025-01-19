#include "kernels_macros.h"
#include "kernels.h"

__constant float4 zero_f4				= {0.0f,0.0f,0.0f,0.0f};

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

	if ( global_id >= pixels_)		{lookup = (float4)(0.0f,0.0f,0.0f,0.0f);} /*alpha=255;read_index=0;u=0;v=0;*/ // NB read_index=mm_cols+3 is an unused index on the mipmap, and safe for a 3x3 sample.
// uint group_id = get_group_id(0);
// uint local_id = get_local_id(0);
// if (/*group_id==0 && local_id==0*/global_id==0) printf("  __kernel compute_lookup_table(..) layer = %u, group_id = %u, global_id = %u, lookup=%d,%d,%d,%d,  pixels_=%u, pixels=%u,  mm_pixels=%u, read_offset_=%u, global_id_offset=%u", \
// 	layer, group_id, global_id, lookup.x, lookup.y, lookup.z, lookup.w,  pixels_, pixels, mm_pixels, read_offset_, lookup_table_offset); //

	lookup_table[global_id + lookup_table_offset]	= lookup;	//read_index;//											// pixel idex in mipmap
}


__kernel void set_warp_new_image(						// Computed once each iteration of warping, for each layer of image pyramid
	// inputs									// Warp describes where to sample the new image to match the reference image.
	__private	uint	read_offset,			//0
	__private	uint	layer,					//1
	__private	float	reduction,				//2

	__constant 	uint8*	mipmap_params,			//3
	__constant 	uint*	uint_params,			//4
	__constant  float*  fp32_params,			//5

	__global	float16*k2k,					//6		// keyframe2K[3]
	__global 	float4*	lookup_table,			//7
	__global	float* 	depth_map,				//8
	// output
	__global 	float2*	warp					//9
){
	uint global_id_uint = get_global_id(0);
	float4 	lookup_ref			= lookup_table[global_id_uint + read_offset];
	uint 	read_index			= floor(lookup_ref.z);
	if (read_index ==0 ) return;

	float	u_flt				= lookup_ref.x*reduction;	// NB reduction is used to enlarge the coords to layer zero, before k2k transform.
	float	v_flt				= lookup_ref.y*reduction;
	float 	inv_depth 			= depth_map[read_index]; 	//1.0f;// mid point max-min inv depth	// Find new pixel position, h=homogeneous coords.//inv dept  //depth_index

	int sample					= 0;
	float16 k2k_pvt				= k2k[sample];															// NB we read  k2k[1] and  k2k[2]
	float uh2 					= k2k_pvt[0]*u_flt + k2k_pvt[1]*v_flt + k2k_pvt[2]*1 + k2k_pvt[3]*inv_depth;
	float vh2 					= k2k_pvt[4]*u_flt + k2k_pvt[5]*v_flt + k2k_pvt[6]*1 + k2k_pvt[7]*inv_depth;
	float wh2 					= k2k_pvt[8]*u_flt + k2k_pvt[9]*v_flt + k2k_pvt[10]*1+ k2k_pvt[11]*inv_depth;
	//float h/z  				= k2k_pvt[12]*u_flt + k2k_pvt[13]*v + k2k_pvt[14]*1; // +k2k_pvt[15]/z

	float u2_flt				= ((uh2)/(wh2*reduction));	// + warp2.x;  	// NB Ideally we should have scaled versions of k2k, to avoid using reduction.
	float v2_flt				= ((vh2)/(wh2*reduction));	// + warp2.y;	// NB need float u,v to compute interpolation.

	if (fmod(u_flt,40.0f)<1.0f && fmod(v_flt,20.0f)<1.0f ){
		uint mm_cols			= uint_params[MM_COLS];
		uint new_read_index		= read_index + floor(v2_flt-lookup_ref.y) * mm_cols  + floor(u2_flt-lookup_ref.x);
		float2 new_img_warp		= {0.0f,0.0f};
		if (new_read_index < uint_params[MM_PIXELS] && new_read_index > 0) {warp[new_read_index] 	= new_img_warp;}

		printf("\n__kernel set_warp_new_image(..)  global_id_uint=%u, read_index=%u, new_read_index=%u,  lookup_ref.x=%f lookup_ref.y=%f, uh2=%f, vh2=%f, wh2=%f, reduction=%f, layer=%u, u2_flt=%f, v2_flt=%f, (u2_flt-lookup_ref.x)=%f, (v2_flt-lookup_ref.y)=%f ",\
												   global_id_uint,    read_index,    new_read_index,     lookup_ref.x,   lookup_ref.y,    uh2,    vh2,    wh2,    reduction,    layer,    u2_flt,    v2_flt,    (u2_flt-lookup_ref.x),    (v2_flt-lookup_ref.y) );
		v2_flt 					= 0.0f;
	}
	float2 new_img_warp			= {u2_flt, v2_flt}; // or just read from the warp, not the u,v from lookup_table
													// Used to set up warp field for new image.
													// This allows efficient iteration of warp merged with SE3 tracking.
	warp[read_index] 			= new_img_warp;		// TODO NB will need to separate warp from SE3 _before_ doing this again.
													// Need to pass old_k2k and new_k2k, for new image within keyframe.
													// For New keyframe ? use old_k2k from old keyframe, and new_k2k from new keframe.
													// TODO NB Problem how to predict warp for new frame, from warp from old frame ?
													// Perhaps integrate the extra warp into depth & vel maps at the end of each frame.
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
	__global 	float4*	sq_mean_rows			//3
){
	uint 	read_index	= floor( lookup_table[ get_global_id(0) + read_offset ].z );
	if (read_index ==0 ) return;

	const float 	W[3] 			= {1.0f/4, 2.0f/4, 1.0f/4};		//1x3 discrete gaussian weights, instead of equal weighting, improves insensitivity to rotation.
	float4 mean_of_squares			= {0,0,0,0};
	float4 pix_val;

	__attribute__((opencl_unroll_hint))
	for (int col=0;col<3;col++){
		pix_val 					= 	img[read_index + col -1];
		mean_of_squares 			+=	W[col] * pix_val * pix_val;
	}
	sq_mean_rows[read_index]		= 	mean_of_squares;
}

__kernel void mean_sq_cols(
	// inputs
	__private	uint	read_offset,			//0
	__private	uint	mm_cols,				//1

	__global 	float4*	lookup_table,			//2
	__global 	float4*	sq_mean_rows,			//3
	// output
	__global 	float4*	sq_mean					//4
){
	uint 	read_index	= floor( lookup_table[ get_global_id(0) + read_offset ].z );
	if (read_index ==0 ) return;

	float4 mean_of_squares			= {0,0,0,0};
	float 	W[3] 					= {1.0f/4, 2.0f/4, 1.0f/4};		//1x3 discrete gaussian weights, instead of equal weighting, improves insensitivity to rotation.

	__attribute__((opencl_unroll_hint))
	for (int col=0;col<3;col++){
		mean_of_squares 			+=	W[col] * sq_mean_rows[read_index + (col-1)*mm_cols];
	}
	sq_mean[read_index]				= 	mean_of_squares;
}

__kernel void co_mean_rows(
	// inputs
	__private	uint	read_offset,			//0
	__private	uint 	mm_size,				//1
	__private	uint	mm_cols,				//2

	__global 	float4*	lookup_table,			//3
	__global 	float4*	ref_img,				//4
	__global 	float4*	warped_img,				//5
	// output
	__global 	float4*	co_mean_rows			//6
){
	uint 	read_index									= floor( lookup_table[ get_global_id(0) + read_offset ].z );
	if (read_index ==0 ) 								return;
	const uint 	offset[5] 								= { -mm_cols, -1, 0, 1, mm_cols };
	const float 	W[3]								= {1.0f/4, 2.0f/4, 1.0f/4};
														//1x3 discrete gaussian weights, instead of equal weighting, improves insensitivity to rotation.
	float4 X[3];
	__attribute__((opencl_unroll_hint))
	for (int col=0;col<3;col++){
			X[col] 									= 	W[col] * ref_img[	read_index + (col-1) ];
	}

	__attribute__((opencl_unroll_hint))
	for (int sample=0;sample<5;sample++){
		float4 co_mean									=	{0,0,0,0};
		__attribute__((opencl_unroll_hint))
		for (int col=0;col<3;col++){
			float4 Y 									= 	warped_img[	read_index + (col-1) + offset[sample] ];
			co_mean 									+= 	X[col] * Y ;
		}
		co_mean_rows[ read_index + sample*mm_size ] 	= co_mean;
	}
}

float2 compute_maximum(__private float4 A, __private float4 B, __private float4 C){
	// from https://math.stackexchange.com/questions/2150199/is-there-a-method-for-estimating-the-parabolic-function-using-three-points-or-a
	float x1=-1,	x2=0,	x3=1;
	float y1=A.x,	y2=B.x,	y3=C.x;			// TODO  which choice of channels ?

	float k1 		= y1/((x1-x2)*(x1-x3));
	float k2 		= y2/((x2-x1)*(x2-x3));
	float k3 		= y3/((x3-x2)*(x3-x1));

	float optimum_x = (k1*(x2+x3) + k2*(x1+x3) + k3*(x2+x1)) / (2*(k1+k2+k3));
	optimum_x 		= clamp(optimum_x, -1.0f, 1.0f);										// warp increment clamped to +/-1

	float a1 		= optimum_x - x1;
	float a2 		= optimum_x - x2;
	float a3 		= optimum_x - x3;

	float optimum_y = k1*a2*a3 + k2*a1*a3 + k3*a1*a2;

	float2 optimum	= {optimum_x, optimum_y};
	return optimum;
}

__kernel void covariance_cols(
	// inputs
	__private	uint	read_offset,			//0
	__private	uint 	mm_size,				//1
	__private	uint	mm_cols,				//2

	__global 	float4*	lookup_table,			//3
	__global 	float4*	co_mean_rows,			//4
	__global 	float4*	ref_img_sq_mean,		//5
	__global 	float4*	warped_img_sq_mean,		//6

	// output
	__global 	float4*	correlation,			//7
	__global 	float2*	warp,					//8	// 2*float4*mm_size // float2*
	__global 	float2*	confidence				//9
){
	uint 	read_index							= floor( lookup_table[ get_global_id(0) + read_offset ].z );
	if (read_index ==0 ) 						return;
	uint	offset[5] 							= { -mm_cols, -1, 0, 1, mm_cols };
	float 	W[3]								= {1.0f/4, 2.0f/4, 1.0f/4};					//1x3 discrete gaussian weights, improves insensitivity to rotation.
	float4	ref_img_denominator					= sqrt( ref_img_sq_mean[read_index] ) ;  	// /*pow(ref_img_mean_local, 2)	 -*/ TODO what if ref_img_sq_mean > ref_img_mean ?

	float4	corr[5]								= {0};
	bool 	unsafe								= false;

	__attribute__((opencl_unroll_hint))
	for (int sample=0;sample<5;sample++){
		float4 co_mean							= {0,0,0,0};
		__attribute__((opencl_unroll_hint))
		for (int col=0;col<3;col++){
			co_mean 							+= 	W[col] * co_mean_rows[	read_index + (col-1)*mm_cols + sample*mm_size ];
		}
		float4 warped_img_denominator 			= sqrt( warped_img_sq_mean[read_index + offset[sample]] ) ;
		corr[sample] 							= co_mean / ( ref_img_denominator  *  warped_img_denominator  );
		//corr[sample] 							= select( corr[sample], zero_f4, (isless(corr[sample],zero_f4) || isgreater(corr[sample],zero_f4)) );

  		if( !isfinite(corr[sample].x) )	{	corr[sample].x = 0.0f;	unsafe=true;	}
  		if( !isfinite(corr[sample].y) )	{	corr[sample].y = 0.0f;	unsafe=true;	}
  		if( !isfinite(corr[sample].z) )	{	corr[sample].z = 0.0f;	unsafe=true;	}

		corr[sample].w 							= 1.0f;
		correlation[read_index + sample*mm_size]= corr[sample];
	}

	float2 warp2								= warp[read_index];
 	float2 opt_u								= compute_maximum( corr[1], corr[2], corr[3] );
	float2 opt_v								= compute_maximum( corr[0], corr[2], corr[4] );
 	float warp_u								= warp2.x + opt_u.x;
 	float warp_v								= warp2.y + opt_v.x;

	float2 warp2_new							= {warp_u, warp_v};
	warp[read_index]							= warp2_new;

	float2 InitConf  							= confidence[read_index];
	float2 ConfHV 								= {opt_u.y, opt_v.y};
	InitConf									= ConfHV + 0.75f * (InitConf - ConfHV);

	//	if (unsafe)	InitConf						= (float2)(0.0f,0.0f);
	confidence[read_index]						= InitConf;
}
// NB confidence should always be 0<conf<1 , even if warp is -ve
// Confidence is derived from the maximum_correlation value.

__kernel void regularize_warp(
	// inputs
	__private	uint	read_offset,			//0
	__private	uint	mm_cols,				//1

	__global 	float4*	lookup_table,			//2
	__global 	float2*	confidence_buf,			//3
	// output
	__global 	float2*	warp					//4

){
	uint 	read_index							= floor( lookup_table[ get_global_id(0) + read_offset ].z );
	if 		(read_index ==0 ) 					return;
	uint	offset[5] 							= { -mm_cols, -1, 0, 1, mm_cols };
	float2 	Disp								= {0.0f, 0.0f};
	float2	Conf 								= {0.0f, 0.0f};
	float2	Warp								= {0.0f, 0.0f};

	__attribute__((opencl_unroll_hint))
	for (int sample=0;sample<5;sample++){
		Conf	=  confidence_buf[read_index + offset[sample]];
		Warp	=  warp[read_index + offset[sample]];
		Disp	+= Warp * Conf ;
	}
	warp[read_index] = Disp;
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
	float2 opt_u								= compute_maximum( corr[1], corr[2], corr[3] );
	float2 opt_v								= compute_maximum( corr[0], corr[2], corr[4] );
 	float warp_u								= warp2.x + opt_u.x;
 	float warp_v								= warp2.y + opt_v.x;
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
