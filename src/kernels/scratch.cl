
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
		printf("\n_kernel compute_warp(..), mm_size=%u",mm_size);
		return;
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
		float4 denominator 						= isnormal(inv_denominator) ? 1/inv_denominator : 1;					// prevent div by zero

		corr[j] 								= covar[j] / denominator ;
		img_corr[read_index + j*mm_size]		= corr[j];
	}

 	float2 warp2								= warp[read_index];
 	float warp_u								= /*warp2.x +*/ compute_maximum( corr[1], corr[2], corr[3] );
 	float warp_v								= /*warp2.y +*/ compute_maximum( corr[0], corr[2], corr[4] );
 	warp_u										= clamp(warp_u, -1.0f, 1.0f);
 	warp_v										= clamp(warp_v, -1.0f, 1.0f);					// warp increment clamped to +/-1

	float2 warp2_new							= {warp_u, warp_v};
	warp[read_index]							= warp2_new;
}
