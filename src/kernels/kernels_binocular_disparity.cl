#include "kernels_macros.h"
#include "kernels.h"

__constant float4 zero_f4				= {0.0f,0.0f,0.0f,0.0f};
__constant float4 ones_f4				= {1.0f,1.0f,1.0f,1.0f};


 __kernel void compute_lookup_table(					// computed once at start of program	// TODO when is it possible to roll the layers together ?  i.e. when local mem is not used.
	 // inputs
	__private	uint	layer,					//0
	__private	uint	lookup_table_offset,	//1

	__constant 	uint8*	mipmap_params,			//3
	__constant 	uint*	uint_params,			//4
	__constant  float*  fp32_params,			//5

	// output
	__global 	float4*	lookup_table			//6
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

__kernel void disparity_load_frame(
	__private	uint	read_offset,			//0
	__private	uint 	baseImage_size,			//1

	__global 	float4*	lookup_table,			//2
	__global 	uchar*	basemem,				//3

	// outputs
	__global 	float4*	new_img					//4
){
	uint 	global_id_uint 		= get_global_id(0);
	float4 	lookup_ref			= lookup_table[global_id_uint];
	uint 	write_index			= floor(lookup_ref.z);
	if ( write_index ==0 ) return;

	float R_float				= basemem[global_id_uint*3]  /256.0f;
	float G_float				= basemem[global_id_uint*3+1]/256.0f;
	float B_float				= basemem[global_id_uint*3+2]/256.0f;

	float4 pixel				= { R_float, G_float, B_float, 1.0f };
	new_img[write_index]		= pixel;
}

__kernel void set_warp_new_image(						// Computed once each iteration of warping, for each layer of image pyramid
	// inputs									// Warp describes where to sample the new image to match the reference image.
	__private	uint	read_offset,			//0
	__private	uint	layer,					//1
	__private	float	reduction,				//2
	__private	uint	mm_cols,				//3

	__constant 	uint*	uint_params,			//4
	__constant  float*  fp32_params,			//5

	__global	float16*k2k,					//6		// keyframe2K[3]
	__global 	float4*	lookup_table,			//7
	__global	float* 	depth_map,				//8
	// output
	__global 	float2*	warp					//9
){
	uint 	global_id_uint 		= get_global_id(0);
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

	float2 new_img_warp			= {(u2_flt-lookup_ref.x), (v2_flt-lookup_ref.y)}; // or just read from the warp, not the u,v from lookup_table
													// Used to set up warp field for new image.
													// This allows efficient iteration of warp merged with SE3 tracking.
	warp[read_index] 			= new_img_warp;		// TODO NB will need to separate warp from SE3 _before_ doing this again.
													// Need to pass old_k2k and new_k2k, for new image within keyframe.
													// For New keyframe ? use old_k2k from old keyframe, and new_k2k from new keframe.
													// TODO NB Problem how to predict warp for new frame, from warp from old frame ?
													// Perhaps integrate the extra warp into depth & vel maps at the end of each frame.
	//}
}


__kernel void warp_image(						// Computed once each iteration of warping, for each layer of image pyramid
	// inputs									// Warp describes where to sample the new image to match the reference image.
	__private	uint	read_offset,			//0
	__private	uint	mm_cols,				//1
	__private	uint	layer,					//2

	__constant 	uint8*	mipmap_params,			//3

	__global 	float2*	warp,					//4
	__global 	float4*	lookup_table,			//5
	__global 	float4*	new_img,				//6

	// outputs
	__global 	float4*	new_img_warped			//7
){
	uint	global_id			= get_global_id(0);
	float 	global_id_flt		= global_id;

	uint8 mipmap_params_ 		= mipmap_params[layer];
	uint read_cols_ 			= mipmap_params_[MiM_READ_COLS];
	uint read_rows_ 			= mipmap_params_[MiM_READ_ROWS];

	float4 	lookup_ref			= lookup_table[global_id + read_offset];
	uint 	read_index			= floor(lookup_ref.z);
	float2 	warp2				= warp[read_index];
	int		u_int				= floor(warp2.x);
	int		v_int				= floor(warp2.y);
	float	u_mod				= warp2.x - u_int;
	float	v_mod				= warp2.y - v_int;
	uint	sample_index		= read_index  + (v_int * mm_cols)  + u_int ;

	//if(sample_index>0 && sample_index < uint_params[MM_PIXELS] ){
	float 	u 	= lookup_ref.x + warp2.x;
	float 	v 	= lookup_ref.y + warp2.y;
	if( v>=0 && v<read_rows_  && u>=0 && u <= read_cols_){
		// 2-way linear interpolation of four sample pixels.
		float4 warped_lower			= new_img[sample_index] 			* (1-u_mod)  +  new_img[sample_index+1] 			* u_mod; // mix( new_img[sample_index], 			new_img[sample_index+1], 			u_mod );  //
		float4 warped_upper			= new_img[sample_index+mm_cols] 	* (1-u_mod)  +  new_img[sample_index+mm_cols+1] 	* u_mod; // mix( new_img[sample_index+mm_cols] , 	new_img[sample_index+mm_cols+1],	u_mod );  //
		new_img_warped[read_index]	= warped_lower 						* (1-v_mod)  +  warped_upper 						* v_mod; // mix( warped_lower , 					warped_upper ,						v_mod );  //

		//float4	test 				= { (float)u_int, (float)v_int, u_mod, v_mod };
		//new_img_warped[read_index]	=	test;
	}
// 	if (0.0f == fmod(global_id_flt, 33.0f) ){
// 		printf("\n__kernel void warp_image(..): warp2=(%f,%f), global_id=%u, lookup_ref.x=%f lookup_ref.y=%f u_int=%u, u_mod=%f, read_index=%u", \
// 		warp2.x, warp2.y, global_id, lookup_ref.x, lookup_ref.y, u_int, u_mod, read_index);
// 	}
}


__kernel void correlation_one_step(		// TODO add local memory for efficiency
	// inputs
	__private	uint	read_offset,			//0
	__private	uint	mm_cols,				//1
	__private	uint 	mm_size,				//2
	//local
	__local		float4*	local_ref_img,			//3		// 3*(group_size + 2) * sizeof(float4)
	__local 	float4*	local_warped_img,		//4		// 5*(group_size + 4) * sizeof(float4)
	//global
	__global 	float4*	lookup_table,			//5
	__global 	float4*	ref_img,				//6
	__global 	float4*	warped_img,				//7
	//outputs
	__global 	float4*	covariance,				//8
	__global 	float4*	correlation				//9
			  ){
	// From YouTube Template Matching by  Correlation | Image Processing I, Columbia Univ.
	// N_tf[i,j] = Sum_m,n( f[m,n] * t[m-i,n-j] ) / sqrt(Sum_m,n( f^2[m,n] ) * sqrt(Sum_m,n( t^2[m-i,n-j] ) )

	uint 	read_index							= floor( lookup_table[ get_global_id(0) + read_offset ].z );
	if (read_index ==0 ) return;
	uint 	lid									= get_local_id(0);
	uint 	local_size							= get_local_size(0);
	uint 	local_ref_mem_width					= local_size + 2;
	uint 	local_warpmem_width					= local_size + 4;
	uint 	gid									= get_group_id(0);
	// TODO check local mem is large enough

	{
	// 	// Zero local_ref_mem
// 	float4 black_f4 = { 0.0, 0.0, 0.0, 1.0 };
// 	for( int idx = lid; idx < 5*(local_size+4); idx +=local_size) local_ref_img[idx] = black_f4;
//
// 	// debug zero margins of local mem
// 	float4 orange_f4 = { 0.0f, 0.5f, 1.0f, 0.8f};
// 	float4 blue_f4	 = { 1.0f, 0.5f, 0.0f, 0.8f};
// 	if(lid==0){
// 		for(int row=0; row<5; row++){
// 			local_warped_img[lid+row*local_warpmem_width]						= orange_f4;
// 			local_warped_img[lid+row*local_warpmem_width + 1]					= orange_f4;
// 			local_warped_img[lid+row*local_warpmem_width + 2 + local_size ]		= blue_f4;
// 			local_warped_img[lid+row*local_warpmem_width + 3 + local_size ]		= blue_f4;
// 		}
// 		for(int row=0; row<3; row++){
// 			local_ref_img[lid+row*local_ref_mem_width]				= orange_f4;
// 			local_ref_img[lid+(1+row)*local_ref_mem_width -2 ]		= blue_f4;
// 		}
// 	}
// 	barrier(CLK_LOCAL_MEM_FENCE);
	// 	//// end debug
	}

	for(int step=-1; step<2; step++){  // -1
		local_ref_img[1+lid+(1+step)*local_ref_mem_width]		= ref_img[read_index+step*mm_cols];
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	if (lid<6){
		const int local_index[6]				= { (0), 								(local_ref_mem_width),		(2*local_ref_mem_width),	\
													(local_ref_mem_width-1),			(2*local_ref_mem_width-1),	(3*local_ref_mem_width-1)	};

		const int offset[6]						= { (-1-mm_cols),						(-2), 						(-3+mm_cols), 						\
													(-5-mm_cols+local_ref_mem_width), 	(-6+local_ref_mem_width), 	(-7+mm_cols+local_ref_mem_width)	};

		local_ref_img[ local_index[lid] ]		= ref_img[ read_index + offset[lid] ];
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	for(uint step=0; step<5; step++){
		uint local_index						= 2 + lid		+ step		*local_warpmem_width;
		uint global_index						= read_index 	+ (step-2)	*mm_cols;
		local_warped_img[local_index]			= warped_img[global_index];
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (lid < 16){
		const int local_index[16]				= { 										(local_warpmem_width),				(2*local_warpmem_width),				(3*local_warpmem_width), 															\
													(1),									(local_warpmem_width+1),			(2*local_warpmem_width+1),				(3*local_warpmem_width+1),					(4*local_warpmem_width+1),				\
													(2+local_size),							(local_warpmem_width+2+local_size),	(2*local_warpmem_width+2+local_size),	(3*local_warpmem_width+2+local_size),		(4*local_warpmem_width+2+local_size),	\
																							(local_warpmem_width+3+local_size), (2*local_warpmem_width+3+local_size),	(3*local_warpmem_width+3+local_size) 												};

		const int offset[16]					= { 										(-2-mm_cols),						(-3), 									(-4+mm_cols), 																		\
													(-4-2*mm_cols),							(-5-mm_cols),						(-6), 									(-7+mm_cols), 								(-8+2*mm_cols),							\
													(-10-2*mm_cols+local_warpmem_width),	(-11-mm_cols+local_warpmem_width),	(-12+local_warpmem_width),				(-13+mm_cols+local_warpmem_width),			(-14+2*mm_cols+local_warpmem_width),	\
																							(-13-mm_cols+local_warpmem_width),	(-14+local_warpmem_width), 				(-15+mm_cols+local_warpmem_width)													};

 		local_warped_img[ local_index[lid] ]	= warped_img[ read_index + offset[lid] ];
 	}
	barrier(CLK_LOCAL_MEM_FENCE);
	{
		{
// 	////////////
//	uint gid = get_group_id(0);
// 	if ( gid > 50) {
// 		covariance[read_index]	= 	local_ref_img[lid];
// 		correlation[read_index]	=	local_warped_img[lid];
//
// 		if(lid==0){
// 			float4 green_f4 			= { 0.0f, 0.5f, 0.0f, 1.0f};
// 			covariance[read_index]		= green_f4;
// 			correlation[read_index]		= green_f4;
// 		}
// 	}
//
// 	if (gid == 0 ) {
// 		if(lid==0){
// 			float4 red_f4 				= { 0.0f, 0.0f, 1.0f, 1.0f};
// 			covariance[read_index]		+= red_f4;
// 			correlation[read_index]		+= red_f4;
// 		}
// 	}
//
// 	if (!(gid == 11 || gid == 24 )) return;
		}
		{
//	float4 temp 				= { 0.0f, -0.1f, 0.0f, 0.2f};

// 	for (int i=0; i<5; i++){
// 		correlation[read_index+ i*mm_cols]	= local_warped_img[lid+ i*local_warpmem_width] - temp;
// 	}
// 	for (int i=0; i<3; i++){
// 		covariance[read_index+ i*mm_cols]	= local_ref_img[lid+ i*local_ref_mem_width] - temp;
// 	}
		}
		{
// 	if(lid==0){
// 			for(uint row=0; row <5 ; row++){
// 				for(uint col=0;	col<local_warpmem_width  ; col++){
// 					correlation[read_index+ col + row*mm_cols]	= local_warped_img[lid+ col + row*local_warpmem_width] ;
// 				}
// 			}
//
// 			for(uint row=0; row <3 ; row++){
// 				for(uint col=0;	col<local_ref_mem_width  ; col++){
// 					covariance[read_index+ col + row*mm_cols]	= local_ref_img[lid+ col + row*local_ref_mem_width];
// 				}
// 			}
//
// 			//float4 temp 				= { 0.0f, 1.0f, 0.5f, 1.0f};
// 			covariance[read_index]								= orange_f4;
// 			covariance[read_index + local_size]					= orange_f4;
// 			covariance[read_index + local_ref_mem_width-1]		= orange_f4;
//
// 			correlation[read_index]								= orange_f4;
// 			correlation[read_index+1]							= orange_f4;
//
// 			correlation[read_index+ local_size]					= orange_f4;
// 			correlation[read_index+ local_warpmem_width-1]		= orange_f4;
// 	}
		}

	//return;
	////////////
	}

	uint 	index;
	float4  pvt_ref_pixel[9];
	float4  pvt_pixel[9];
	float4  pvt_correlation[5];
	float4  pvt_covariance;
	float4  pvt_sum_sq_ref_patch;
	float4  pvt_sum_sq_warped_patch;

	uint	write_index 						= read_index;
	uint	increment[5]						= { -local_warpmem_width , -1 , 0 , 1 , local_warpmem_width };		//{ -mm_cols , -1 , 0 , 1 , mm_cols };

	index										= 0;
	pvt_sum_sq_ref_patch						= zero_f4;

	__attribute__((opencl_unroll_hint))
	for (int col=0;col<3;col++){
		__attribute__((opencl_unroll_hint))
		for (int row=0;row<3;row++){
				pvt_ref_pixel[index]			= local_ref_img[ lid + col + row*local_ref_mem_width ];	//	ref_img[read_index+col+row*mm_cols];
				pvt_sum_sq_ref_patch			+= pvt_ref_pixel[index] * pvt_ref_pixel[index];
				index++;
		}
	}
	pvt_sum_sq_ref_patch						=  sqrt(pvt_sum_sq_ref_patch) / 3.0f;

	__attribute__((opencl_unroll_hint))
	for (int sample=0;sample<5;sample++){

		index 									= 0;
		pvt_correlation[sample]					= zero_f4;
		pvt_covariance							= zero_f4;
		pvt_sum_sq_warped_patch					= zero_f4;

		__attribute__((opencl_unroll_hint))
		for (int col=1;col<4;col++){    // int col=1;col<4;col++
			__attribute__((opencl_unroll_hint))
			for (int row=1;row<4;row++){
				pvt_pixel[index]				= local_warped_img[ lid + col + row*local_warpmem_width + increment[sample] ];	//	warped_img[ read_index + col + row*mm_cols + increment[sample] ];
				pvt_sum_sq_warped_patch			+= pvt_pixel[index] * pvt_pixel[index];

				pvt_covariance					+= (pvt_ref_pixel[index] * pvt_pixel[index]);
				index++;
			}
		}
		pvt_sum_sq_warped_patch					= sqrt(pvt_sum_sq_warped_patch) / 3.0f;
		pvt_covariance							/= 9.0f;

		float4 denominator 						= (pvt_sum_sq_ref_patch * pvt_sum_sq_warped_patch ); //clamp( (pvt_sum_sq_ref_patch * pvt_sum_sq_warped_patch )  , 0.001f, 100.0f );
		{
		//pvt_correlation[ sample ]				/= denominator;

// 		float4  X,Y,Z,W;
//
// 		X.x	= pvt_correlation[ sample ].x;
// 		X.y = denominator.x;
// 		X.z = X.x / X.y;
// 		X.w = 1.0f;
//
// 		Y.x = pvt_correlation[ sample ].y;
// 		Y.y = denominator.y;
// 		Y.z = Y.x / Y.y;
// 		Y.w = 1.0f;
//
// 		Z.x = pvt_correlation[ sample ].z;
// 		Z.y = denominator.z;
// 		Z.z = Z.x / Z.y;
// 		Z.w = 1.0f;
//
// 		W.x = pvt_correlation[ sample ].w;
// 		W.y = denominator.w;
// 		W.z = W.x / W.y;
// 		W.w = 1.0f;
//
//
// 		float4 pixel;
// 		pixel.x = pvt_pixel[0].x;
// 		pixel.y = pvt_pixel[4].x;
// 		pixel.z = pvt_pixel[8].x;
// 		pixel.w = 1.0f;
//
// 		float4 covar;
// 		covar.x  = pvt_covariance.x; //pvt_ref_pixel[8].z ;
// 		covar.y  = pvt_covariance.y; //pvt_pixel[8].z;
// 		covar.z  = pvt_covariance.z;
// 		covar.w  = 1.0f;

// 		if ( !isnormal(pvt_correlation[sample].x) )	{ pvt_correlation[sample].x	= -0.5f; }		// not needed if clamped to (-1 <-> +1)
// 		if ( !isnormal(pvt_correlation[sample].y) )	{ pvt_correlation[sample].y	= -0.5f; }
// 		if ( !isnormal(pvt_correlation[sample].z) )	{ pvt_correlation[sample].z	= -0.5f; }

// 		pvt_correlation[sample] 				= clamp( pvt_correlation[sample], -1.0f, 1.0f );

// 		pvt_correlation[ sample ].w 			= 1.0f;
		}

		//if ( gid > 50) {

			covariance[  write_index ]				= pvt_covariance;  // FIXME corrupted local_ref_img margins  // local_ref_img[lid +2 + 2*local_size ]; //
			// // float4 base
			//correlation[ write_index ]				= denominator; // FIXME alpha!=1.0f    //pvt_sum_sq_ref_patch; //pvt_sum_sq_warped_patch; //pvt_covariance ; //covar; //X; //pixel; // pvt_correlation[ sample ] ;  // denominator ; //

			pvt_correlation[ sample ]				= pvt_covariance / denominator ;
			correlation[ write_index ]				= pvt_correlation[ sample ];


			{
	// 		pvt_correlation[ sample ].x				= pvt_covariance.x  * base.x ;
	// 		pvt_correlation[ sample ].y				= pvt_covariance.y  * base.y ;
	// 		pvt_correlation[ sample ].z				= pvt_covariance.z  * base.z ;
	// 		pvt_correlation[ sample ].w				= 1.0f;
	//
	// 		correlation[ write_index ]				= pvt_correlation[ sample ];
			}
// 			if (lid==0){
// 				float4 green_f4 					= { 0.0f, 0.5f, 0.0f, 1.0f};
// 				covariance[  write_index + 3]		= green_f4 ;
// 				correlation[ write_index + 3]		= green_f4 ;
// 				for (int i=0; i<9; i++){
// 					covariance[  write_index + 4 + i]	= pvt_ref_pixel[i];
// 					correlation[ write_index + 4 + i]	= pvt_pixel[i];
// 				}
// 			}
// 			if (lid==local_size-1){
// 				float4 blue_f4 						= { 1.0f, 0.0f, 0.0f, 1.0f};
// 				covariance[  write_index - 3]		= blue_f4 ;
// 				correlation[ write_index - 3]		= blue_f4 ;
// 				for (int i=0; i<9; i++){
// 					covariance[  write_index - 4 - i]	= pvt_ref_pixel[i];
// 					correlation[ write_index - 4 - i]	= pvt_pixel[i];
// 				}
// 			}
		//}

// 		if ( gid == 11 || gid == 24 ){
// 			float4 temp 				= { 0.0f, -0.1f, 0.0f, 0.2f};
//
// 			for (int i=0; i<5; i++){
// 				correlation[read_index+ i*mm_cols]	= local_warped_img[lid+ i*local_warpmem_width]; // - temp;
// 			}
// 			for (int i=0; i<3; i++){
// 				covariance[read_index+ i*mm_cols]	= local_ref_img[lid+ i*local_ref_mem_width]; // - temp;
// 			}
// 		}

		write_index								+= mm_size;
	}

	// TODO add warp & confidence

}


__kernel void correlation_2nd_step(						// Could merge with covariance_3cols(..). Also computes new Warp and Confidence values.
	// inputs
	__private	uint	read_offset,			//0
	__private	uint	mm_cols,				//1
	__private	uint 	mm_size,				//2

	__global 	float4*	lookup_table,			//3
	__global	float4* covariance,				//4		// sizeof(float4) * 5 * mm_size
	//output
	__global	float4*	correlation,			//5		// sizeof(float4) * 5 * mm_size
	__global 	float2*	warp,					//6		// 2*float4*mm_size // float2*
	__global 	float*	confidence				//7

 ){
	uint 	read_index							= floor( lookup_table[ get_global_id(0) + read_offset ].z );
	if (read_index ==0 ) return;
	float4 	pvt_correlation[5] 					= {zero_f4, zero_f4, zero_f4, zero_f4, zero_f4};
	float4 	pvt_covariance[5]					= {zero_f4, zero_f4, zero_f4, zero_f4, zero_f4};
	float4 	pvt_denominator[5]					= {zero_f4, zero_f4, zero_f4, zero_f4, zero_f4};

	uint	write_index							= read_index;
	uint	increment[5]						= { -mm_cols , -1 , 0 , 1 , mm_cols };

	__attribute__((opencl_unroll_hint))
	for (uint sample=0;sample<5;sample++){
		pvt_denominator[sample]					= clamp( correlation[read_index + increment[sample] ], 0.001f, 100.0f );
	}
	__attribute__((opencl_unroll_hint))
	for (uint sample=0;sample<5;sample++){
		pvt_covariance[sample]					= covariance[read_index  + increment[sample] ];
	}
	__attribute__((opencl_unroll_hint))
	for (uint sample=0;sample<5;sample++){
		pvt_correlation[sample]					= pvt_covariance[sample] / pvt_denominator[sample]; // clamp((ones_f4 / denominator), 0.001f, 100.0f )

		correlation[read_index + increment[sample] ] 	= pvt_correlation[sample];
	}


}


												// Rows then Columns kernels reduces computation & memoory reads,
												// especially as patch size increases.

__kernel void mean_3rows(						// square the pixels to accentuate high values in each channel
												// Compute mu_X for the 3x3 patch, and then Standard Deviation sd =(X-mu_X)
	// inputs
	__private	uint	read_offset,			//0

	__global 	float4*	lookup_table,			//1
	__global 	float4*	img,					//2
	// output
	__global 	float4*	sq_mean_rows			//3
){
	uint 	read_index					= floor( lookup_table[ get_global_id(0) + read_offset ].z );
	if (read_index ==0 ) return;

	float4 sum_of_squares				= zero_f4;

	__attribute__((opencl_unroll_hint))
	for (int col=-1;col<2;col++){
		sum_of_squares 					+=	img[read_index + col]; 			//pown( img[read_index + col], 2);						// pixel squared
	}
	sum_of_squares.w					= 3.0f;
	sq_mean_rows[read_index]			= sum_of_squares/3.0f;
}

__kernel void mean_3cols(
	// inputs
	__private	uint	read_offset,			//0
	__private	uint	mm_cols,				//1

	__global 	float4*	lookup_table,			//2
	__global 	float4*	sq_mean_rows,			//3
	__global 	float4*	img,					//4
	// output
	__global 	float4*	mean,					//5
	__global	float4* diff					//6		// standard deviation of this pixel.
){
	uint 	read_index					= floor( lookup_table[ get_global_id(0) + read_offset ].z );
	if (read_index ==0 ) return;

	float4 pvt_mean						= zero_f4;

	__attribute__((opencl_unroll_hint))
	for (int col=-1;col<2;col++){ 					// mad_sat -> non-wrap-around integer result. // read_index + col*mm_cols
		pvt_mean 						+=	sq_mean_rows[ mad_sat( (int)col , (int)mm_cols , (int)read_index ) ];
	}
	pvt_mean.w							=  3.0f;
	pvt_mean 							/= 3.0f;
	mean[read_index]					= pvt_mean;

	float4 pvt_diff						= img[read_index] - pvt_mean;
	pvt_diff.w							= 1.0f;
	diff[read_index]					= pvt_diff;
}

												// TODO not clear that the sigma kernels are needed, may just use sd from mean_sq kernels.
__kernel void sigma_3rows(						// sigma_X = Expectation[ X - mu_X ],  the Standard Deviation for the 3x3 patch.
												// Here I am using mu_x for the 3x3 patch arround each pixel, then taking the mean SD of the 3x3 patch arround the pixel.
												// i.e. different mu for each pixel, and a 5x5 region of influence due to the different mu for each pixel.
	// inputs
	__private	uint	read_offset,			//0

	__global 	float4*	lookup_table,			//1
	__global 	float4*	diff,					//2
	// output
	__global 	float4*	sigma_rows				//3
){
	uint 	read_index					= floor( lookup_table[ get_global_id(0) + read_offset ].z );
	if (read_index ==0 ) return;

	float4 sum_diff_squared				= zero_f4;
	float4 pvt_diff;

	__attribute__((opencl_unroll_hint))
	for (int col=-1;col<2;col++){
			pvt_diff					= diff[read_index + col];
			sum_diff_squared 			+= pvt_diff * pvt_diff;
	}
	sum_diff_squared.w					= 3.0f;
	sigma_rows[read_index]				= sum_diff_squared/3.0f;
}


__kernel void sigma_3cols(
	// inputs
	__private	uint	read_offset,			//0
	__private	uint	mm_cols,				//1

	__global 	float4*	lookup_table,			//2
	__global 	float4*	sigma_rows,				//3
	// output
	__global 	float4*	SD						//4		// standard deviation of this 3x3 patch of pixels
){
	uint 	read_index					= floor( lookup_table[ get_global_id(0) + read_offset ].z );
	if (read_index ==0 ) return;

	float4 mean_of_sigma				= zero_f4;

	__attribute__((opencl_unroll_hint))
	for (int col=-1;col<2;col++){ 					// mad_sat -> non-wrap-around integer result. // read_index + col*mm_cols
		mean_of_sigma	 				+=	sigma_rows[ mad_sat( (int)col , (int)mm_cols , (int)read_index ) ];
	}
	mean_of_sigma.w						= 3.0f;
	SD[read_index]						= sqrt( mean_of_sigma/3.0f );
}


__kernel void covariance_3rows( 				// ref_img SD * warped_img SD, for each of 5 sample pixel alignments.
												// sd_X_sd_Y_rows holds 5 maps, one for each sample.
	// inputs
	__private	uint	read_offset,			//0
	__private	uint	mm_cols,				//1
	__private	uint 	mm_size,				//2

	__global 	float4*	lookup_table,			//3
	__global	float4* diff_ref_img,			//4
	__global	float4* diff_warped_img,		//5
	//output
	__global	float4*	covariance_rows			//6		// sizeof(float4) * 5 * mm_size
){
	uint 	read_index							= floor( lookup_table[ get_global_id(0) + read_offset ].z );
	if (read_index ==0 ) return;
	float4 	temp_val, pvt_diff_ref[3];

	__attribute__((opencl_unroll_hint))
	for (int col=0;col<3;col++){
		pvt_diff_ref[col]							= diff_ref_img[read_index + col -1];							// store pixelwise (X- mu_x)
	}

	uint	write_index 						= read_index;
	uint	increment[5]						= { -mm_cols , -1 , 0 , 1 , mm_cols };

	__attribute__((opencl_unroll_hint))
	for (int sample=0;sample<5;sample++){
		temp_val								= zero_f4;

		__attribute__((opencl_unroll_hint))
		for (int col=0;col<3;col++){
			temp_val							+= pvt_diff_ref[col] * diff_warped_img[ read_index + col -1 + increment[sample] ];	// co-variance
		}
		temp_val.w								= 3.0f;
		covariance_rows[ write_index ]			= temp_val/3.0f;
		write_index								+= mm_size;
	}
}


__kernel void covariance_3cols(
	// inputs
	__private	uint	read_offset,			//0
	__private	uint	mm_cols,				//1
	__private	uint 	mm_size,				//2

	__global 	float4*	lookup_table,			//3
	__global	float4* covariance_rows,		//4
	//output
	__global	float4*	covariance				//5		// sizeof(float4) * 5 * mm_size
 ){
	uint 	read_index					= floor( lookup_table[ get_global_id(0) + read_offset ].z );
	if (read_index ==0 ) return;
	float4 	temp_val;

	uint	write_index 				= read_index;
	uint	increment[5]				= { -mm_cols , -1 , 0 , 1 , mm_cols };

	__attribute__((opencl_unroll_hint))
	for (int sample=0;sample<5;sample++){
		temp_val						= zero_f4;

		__attribute__((opencl_unroll_hint))
		for (int col=-1;col<2;col++){
			temp_val					+= covariance_rows[ mad_sat( (int)col , (int)mm_cols , (int)write_index )  ];
		}
		temp_val.w						= 3.0f;
		covariance[ write_index ]		= temp_val/3.0f;
		write_index						+= mm_size;
	}
}


float2 compute_maximum(__private float4 A, __private float4 B, __private float4 C){
	// from https://math.stackexchange.com/questions/2150199/is-there-a-method-for-estimating-the-parabolic-function-using-three-points-or-a
	// Also incorporating ug_stereomatcher Matachlib.cu  PolyDisparity kenel concavity chk & confidence setting.
	float x1=-1,	x2=0,	x3=1;
	float y1=A.x,	y2=B.x,	y3=C.x;			// TODO  which choice of channels ?

	float k1 		= y1/((x1-x2)*(x1-x3));
	float k2 		= y2/((x2-x1)*(x2-x3));
	float k3 		= y3/((x3-x2)*(x3-x1));

	float optimum_x, optimum_y;
	float concavity = mad(y2, -2.0f, (y1+y3) );

	if ( concavity  <0.0f) {														// correct concavity, use parabolic curvature
		optimum_x = (k1*(x2+x3) + k2*(x1+x3) + k3*(x2+x1)) / (2*(k1+k2+k3));
		optimum_x = clamp(optimum_x, -1.0f, 1.0f);										// ### warp increment clamped to +/-1
		float a1 		= optimum_x - x1;
		float a2 		= optimum_x - x2;
		float a3 		= optimum_x - x3;
		optimum_y = k1*a2*a3 + k2*a1*a3 + k3*a1*a2;
	}else{																			// wrong concavity, use linear gradient
		optimum_x = (1.0f-y2) * 0.5f*(y1-y3);
		optimum_x = clamp(optimum_x, -1.0f, 1.0f);										// ### warp increment clamped to +/-1
		optimum_y = 1.0f;
	}
																					// NB intuitively, "confidence in warp" = curvature of fit * correlation .

	if (optimum_y >1.0f){															// if predicted fit impossibly good
		float d = optimum_y - y2;
		if(d>1e-10){																// if predicted increase in fit is non-negligible
			optimum_x = optimum_x * ((1.0 - y2) / d);								// scale the increment to reach correlation = 1.0f
		}
		optimum_y = 1.0f;
	}else{
		optimum_y = mad(0.3f , optimum_y , 0.7f);
	}

	float2 optimum	= {optimum_x, optimum_y};
	return optimum;
}


__kernel void correlation(						// Could merge with covariance_3cols(..). Also computes new Warp and Confidence values.
	// inputs
	__private	uint	read_offset,			//0
	__private	uint	mm_cols,				//1
	__private	uint 	mm_size,				//2

	__global 	float4*	lookup_table,			//3
	__global	float4* sd_ref_img,				//4
	__global	float4* sd_warped_img,			//5
	__global	float4* covariance,				//6		// sizeof(float4) * 5 * mm_size
	//output
	__global	float4*	correlation,			//7		// sizeof(float4) * 5 * mm_size
	__global 	float2*	warp,					//8		// 2*float4*mm_size // float2*
	__global 	float*	confidence				//9

 ){
	uint 	read_index							= floor( lookup_table[ get_global_id(0) + read_offset ].z );
	if (read_index ==0 ) return;
	float4 	pvt_correlation[5] 					= {zero_f4, zero_f4, zero_f4, zero_f4, zero_f4};

	uint	write_index							= read_index;
	uint	increment[5]						= { -mm_cols , -1 , 0 , 1 , mm_cols };

	__attribute__((opencl_unroll_hint))
	for (uint sample=0;sample<5;sample++){
		pvt_correlation[sample]					= covariance[write_index] / sqrt(sd_ref_img[read_index] * sd_warped_img[read_index + increment[sample] ] );      // clamp( (covariance[write_index] /  sd_ref_img[read_index] * sd_warped_img[write_index]), -1.0f , 1.0f ) ;  // ### clamped to 0.0 < correlation < 1.0

		if ( !isnormal(pvt_correlation[sample].x) )	{ pvt_correlation[sample].x	= 0.111f; }		// not needed if clamped to (-1 <-> +1)
		if ( !isnormal(pvt_correlation[sample].y) )	{ pvt_correlation[sample].y	= 0.111f; }
		if ( !isnormal(pvt_correlation[sample].z) )	{ pvt_correlation[sample].z	= 0.111f; }
////sd_warped_img[read_index + increment[sample] ];// (sd_ref_img[read_index] );//* sd_warped_img[read_index + increment[sample] ] ); //covariance[write_index]; //
		correlation[write_index]				=  pvt_correlation[sample];
		write_index								+= mm_size;
	}

	float old_conf  							= confidence[read_index];
	float2 warp2								= warp[read_index];
 	float2 opt_u								= pvt_correlation[1].x - pvt_correlation[3].x; // compute_maximum( pvt_correlation[1], pvt_correlation[2], pvt_correlation[3] );//		// does the work of PolyDisparity kernel in UG code.
	float2 opt_v								= pvt_correlation[0].x - pvt_correlation[4].x; //compute_maximum( pvt_correlation[0], pvt_correlation[2], pvt_correlation[4] );//

	float2 warp_new;
	warp_new.x									=  opt_u.x + warp2.x;	// TODO currently only using channel .x of the correlation, from .x of the images.
 	warp_new.y									=  opt_v.x + warp2.y;	// Need to decide how to blend the channels.
	warp[read_index]							= warp_new;

	confidence[read_index]						= (0.75f*old_conf + 0.25f * opt_u.y * opt_v.y); // clamp( (0.75f*old_conf + 0.25f * opt_u.y * opt_v.y), 0.0f, 1.0f);						// ### clamped to 0.0 < confidence < 1.0
}


__kernel void regularize_warp(					// Use local mem to avoid repeat loading of same data by different threads. // TODO handle margins.
												// TODO ideal might be 8x8 tiled, rather than strips.
	// inputs
	__private	uint	read_offset,			//0
	__private	uint	mm_cols,				//1

	__local		float2*	local_warp,				//2		2 rows + 1px halo at each end, middle row only.
	__local		float*	local_confidence,		//3		2 rows + 1px halo at each end, middle row only.

	__global 	float4*	lookup_table,			//4
	__global 	float2*	warp,					//5
	__global 	float*	confidence_buf,			//6
	// outputs
	__global 	float2*	new_warp,				//7
	__global 	float*	new_confidence			//8
){
	uint 	read_index							= floor( lookup_table[ get_global_id(0) + read_offset ].z );
	if 		(read_index ==0 ) 					return;
	uint 	lid									= get_local_id(0);
	uint 	lid_1								= lid + 1;
	// load local buffers
	// load 3 rows
	int patch_length 							= 2 + get_local_size(0);

	__attribute__((opencl_unroll_hint))
	for (int i=0; i<3; i++){  local_warp[      lid_1 + i*patch_length]		= warp[           read_index + (i-1) * mm_cols]; }

	__attribute__((opencl_unroll_hint))
	for (int i=0; i<3; i++){  local_confidence[lid_1 + i*patch_length]		= confidence_buf[ read_index + (i-1) * mm_cols]; }

	// load margins. NB we only need the middle row.
	int pix_offset[2] = {-1, patch_length};
	if (lid <2){
		local_warp[       lid_1 + pix_offset[lid] + patch_length ]			= warp[           read_index + pix_offset[lid] ];
		local_confidence[ lid_1 + pix_offset[lid] + patch_length ]			= confidence_buf[ read_index + pix_offset[lid] ];
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	// 5-way weighted sum of neighbopurs (i) conf*input,  and (ii) conf
	uint	increment[5]						= { 0 , patch_length-1 , patch_length , patch_length+1 , 2*patch_length }; // where to sample local mem relative to lid_1 = lid+1.
	float2	sum_warp 							= {0.0f, 0.0f};
	float	sum_conf 							= 0.0f;
	float	pvt_conf							= 0.0f;
	__attribute__((opencl_unroll_hint))
	for (int sample=0;sample<5;sample++){
		pvt_conf								= 	local_confidence[lid_1 + increment[sample] ];
		sum_warp								+=  local_warp[lid_1 + increment[sample] ] * pvt_conf;
		sum_conf								+=	pvt_conf;
	}
	new_warp[read_index]						=	sum_warp / sum_conf;
	new_confidence[read_index]					=	sum_conf / 5.0f;
}



 __kernel void propagate_warp(					// Propagates warp to the next layer of the image pyramid, by linear interpolation. // TODO handle right and bottom margins.
	// inputs
	__private	uint	read_offset,			//0
	__private	uint 	rows_in,				//1
	__private	uint	cols_in,				//2
	__private	uint	write_offset,			//3  // rather send the mm_offset
	__private	uint	mm_cols,				//4

	__local		float2*	local_warp,				//5		// 2 rows + 1px halo at right end for both rows.

	__global 	float4*	lookup_table,			//6
	// input_output
	__global 	float2*	warp					//7		// 2*float4*mm_size // float2* // NB reading & writing to a different regions of the same buffer.
){
	uint 	global_id			= get_global_id(0);
	float 	global_id_flt 		= global_id;
	float4 	lookup_in			= lookup_table[ global_id + read_offset  ];
	uint read_col				= lookup_in.x;
	uint read_row				= lookup_in.y;
	uint read_index				= lookup_in.z;

	uint row 					= global_id / cols_in;
	uint col 					= fmod( global_id_flt, cols_in);
	uint write_index			= col*2 + row*2*mm_cols +  lookup_table[ write_offset ].z;

	uint 	lid					= get_local_id(0);
	// load local buffers
	// load 3 rows
	uint patch_length 			= 1 + get_local_size(0);

	__attribute__((opencl_unroll_hint))
	for (int i=0; i<2; i++){  local_warp[      lid + i*patch_length]		= warp[          read_index + i* mm_cols]; }

	// load margins. NB we only need the middle row.
	int pix_offset[2] = {patch_length, 2*patch_length};
	if (lid <2){
		local_warp[  pix_offset[lid] ]			= warp[ read_index + pix_offset[lid] ];
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	float2 warp_in00					= local_warp[lid] ;
	float2 warp_in10					= local_warp[lid+1] ;
	float2 warp_in01					= local_warp[lid+patch_length] ;
	float2 warp_in11					= local_warp[lid+patch_length+1] ;

	warp[write_index]					= warp_in00;
	float2 warp_out10					= mix(warp_in00, warp_in10, 0.5f);
	warp[write_index +1]				= warp_out10;
	warp[write_index + mm_cols]			= mix(warp_in00, warp_in01, 0.5f);
	warp[write_index + mm_cols +1]		= mix( mix(warp_in10, warp_in11, 0.5f), warp_out10, 0.5f);

	if (fmod(global_id_flt,333.0f) ==0.0f ) {
		printf("\n_kernel propagate_warp(..), global_id=%u, lid=%u,	read_offset=%u, rows_in=%u, cols_in=%u, write_offset=%u, write_index=%u, mm_cols=%u, read_col=%u, read_row=%u, col=%u, row=%u",\
		global_id, lid, read_offset, rows_in, cols_in, write_offset, write_index, mm_cols, read_col, read_row, col, row);
	}
 }


//////////////////////////////////////////////////////////////////////////// TODO update this list
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
