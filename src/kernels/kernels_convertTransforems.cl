#include "kernels_macros.h"
#include "kernels.h"

#define	SO3_x		0
#define	SO3_y		1
#define	SO3_z		2

#define	ST3_x		4
#define	ST3_y		5
#define	ST3_z		6

#define	ONE			6
#define	ZERO		7

// #define	ONE_		1
// #define	MINUS_ONE_	-1

#define	COS_THETA				0
#define ONE_MINUS_COS_THETA		1
#define	SIN_THETA				2
#define	ZERO_VAR				3
#define ONE_VAR					4
#define MINUS_ONE_VAR			5

void LieToP( uint lid,	__local float SE3[9],	__local float Pose[32/*16*/] ){

	const uint LtoP[16][6] = { /*	Indices for SE3[8] and vars[4], to compose the elements of 4x4 SE3 transformation matrix, from SE3 Lie vector.	*/\
								{ ONE_VAR		, ONE,		COS_THETA, 		SO3_x, SO3_x, ONE_MINUS_COS_THETA 	},	/*	1*1   *cos(theta)    +    w_x w_x (1 - cos_theta)	= SE3[6] * vars[0]   +   SE3[0] * SE3[0] * vars[1]  */\
								{ ONE_VAR		, SO3_z,	SIN_THETA, 		SO3_y, SO3_y, ONE_MINUS_COS_THETA	},	/*	1*w_z *sin(theta)    +    w_x w_y (1 - cos_theta) 	*/\
								{ ONE_VAR		, SO3_y,	SIN_THETA, 		SO3_x, SO3_z, ONE_MINUS_COS_THETA	},	/*	1*w_y *sin(theta)    +    w_x w_z (1 - cos_theta)	*/\
								{ ONE_VAR		, ONE,		ST3_x, 			ZERO,  ZERO,  ZERO_VAR				},	/*	t_x	*/\
	\
								{ ONE_VAR		, SO3_z,	SIN_THETA,		SO3_x, SO3_y, ONE_MINUS_COS_THETA	},	/*	1*w_z *sin(theta)    +    w_x w_y (1 - cos_theta)	*/\
								{ ONE_VAR		, ONE, 		COS_THETA,		SO3_y, SO3_y, ONE_MINUS_COS_THETA	},	/*	1*1   *cos(theta)    +    w_y w_y (1 - cos_theta)	*/\
								{ MINUS_ONE_VAR	, SO3_x, 	SIN_THETA, 		SO3_y, SO3_z, ONE_MINUS_COS_THETA	},	/*	-1*w_x*sin(theta)    +    w_y w_z (1 - cos_theta)	*/\
								{ ONE_VAR		, ONE,		ST3_y, 			ZERO,  ZERO,  ZERO_VAR				},	/*	t_y	*/\
	\
								{ MINUS_ONE_VAR	, SO3_y,	SIN_THETA,		SO3_x, SO3_z, ONE_MINUS_COS_THETA	},	/*	-1*w_y *sin(theta)   +    w_x w_z (1 - cos_theta)	*/\
								{ ONE_VAR		, SO3_x,	SIN_THETA,		SO3_y, SO3_z, ONE_MINUS_COS_THETA	},	/*	1*w_x  *sin(theta)   +    w_y w_z (1 - cos_theta)	*/\
								{ ONE_VAR		, ONE,		COS_THETA,		SO3_z, SO3_z, ONE_MINUS_COS_THETA	},	/*	1*1    *cos(theta)   +    w_z w_z (1 - cos_theta)	*/\
								{ ONE_VAR		, ONE,		ST3_z,			ZERO,  ZERO,  ZERO_VAR				},	/*	t_z	*/\
	\
								{ ONE_VAR		, ZERO,		ZERO_VAR,		ZERO,  ZERO,  ZERO_VAR				},	/*	0	*/\
								{ ONE_VAR		, ZERO,		ZERO_VAR,		ZERO,  ZERO,  ZERO_VAR				},	/*	0	*/\
								{ ONE_VAR		, ZERO,		ZERO_VAR,		ZERO,  ZERO,  ZERO_VAR				},	/*	0	*/\
								{ ONE_VAR		, ZERO,		ZERO_VAR,		ONE,   ONE,   ONE_VAR				},	/*	1	*/\
	};

	float3	So3 			= (float3)( SE3[0], SE3[1], SE3[2] );
	float	theta 			= fast_length(So3);
	float	cos_theta 		= cos(theta);
	float	one_cos_theta	= 1.0f - cos_theta;
	float	sin_theta		= sin(theta);

	float	vars[6];
	vars[0]					= cos_theta;
	vars[1]					= one_cos_theta;
	vars[2]					= sin_theta;
	vars[3]					= 0;
	vars[4]					= 1;
	vars[5]					= -1;

	const uint * L2P;
	if (lid < 16){
		L2P					= LtoP[lid];
		Pose[lid +16]		= vars[L2P[0]] * SE3[L2P[1]] * vars[L2P[2]]   +   SE3[L2P[3]] * SE3[L2P[4]] * vars[L2P[5]] ;

		uint idx_L2P = fmod((float)lid, 6);
		uint idx_SE3 = fmod((float)lid, 16);

		printf( "\nLieToP(..) lid=%u,  theta=%f,  cos_theta=%f,  one_cos_theta=%f,   sin_theta=%f,   L2P[%u]={%u},  SE3[%u]={%f},  Pose[lid +16]=%f ", \
			lid,  theta,  cos_theta,  one_cos_theta,  sin_theta, \
			idx_L2P,  L2P[ idx_L2P ], \
			idx_SE3,  SE3[ idx_SE3 ], \
			Pose[lid +16] );

	}																												// L2P[1], L2P[2],   L2P[3], L2P[4], L2P[5],   // SE3[1], SE3[2],   SE3[3], SE3[4], SE3[5],   SE3[6], SE3[7], SE3[8],

	if (lid==0) {printf("\nSE3[]="); for(uint i=0; i<9; i++) printf(",	%f",SE3[i]);}

}


void update_k2_kdev_fn(
	uint lid,
	__local float local_K_update[ 	2* 	SE3_elems],
	__local float local_pose_inv_K[ 2* 	SE3_elems],
	__local float local_A_B[ 		2* 	SE3_elems],
	__local float local_k2k[			SE3_elems]
){																													// K2K   =   K * pose * update * inv_K   =    (K * pose) * (update * inv_K);
	uint elem 						= fmod((float)lid, SE3_elems);
	uint offset						= (lid/ SE3_elems)  * SE3_elems;
	uint col						= fmod((float)elem, 4.0f);
	uint row						= elem / (uint)4;
	/*
	if (lid==0){
		printf("\n\n local_K_update = \n");
		for (uint i=0; i< 2 ; i++){ for (uint j=0; j< 4 ; j++){ for (uint k=0; k< 4 ; k++){ 	printf(",	%f",local_K_update[		i*16 +j*4 +k]);	} printf("\n"); } printf("\n\n"); }

		printf("\n\n local_pose_inv_K = \n");
		for (uint i=0; i< 2 ; i++){ for (uint j=0; j< 4 ; j++){ for (uint k=0; k< 4 ; k++){ 	printf(",	%f",local_pose_inv_K[	i*16 +j*4 +k]);	} printf("\n"); } printf("\n\n"); }

		printf("\n\n local_A_B = \n");
		for (uint i=0; i< 2 ; i++){ for (uint j=0; j< 4 ; j++){ for (uint k=0; k< 4 ; k++){ 	printf(",	%f",local_A_B[			i*16 +j*4 +k]);	} printf("\n"); } printf("\n\n"); }

		printf("\n\n local_k2k = \n");
		for (uint i=0; i< 1 ; i++){ for (uint j=0; j< 4 ; j++){ for (uint k=0; k< 4 ; k++){ 	printf(",	%f",local_k2k[			i*16 +j*4 +k]);	} printf("\n"); } printf("\n\n"); }
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	*/
	if(lid<32){
		for (uint i =0; i<4; i++){
			local_A_B[ lid ]		+=	local_K_update[ offset + row * 4 + i ] 		* local_pose_inv_K[ offset + i * 4  + col  ] ;

			//printf("\n updatek2k() lid=%u, elem=%u, offset=%u, col=%u, row=%u,  local_K_update[ offset + row * 4 + i ]=[ %u ]= %f,  local_pose_inv_K[ offset + row * i + col  ]=[ %u ]= %f  product= %f ", \
				lid, elem, offset, col, row, (offset + row * 4 + i), local_K_update[ offset + row * 4 + i ],  (offset + row * i + col),  local_pose_inv_K[ offset + row * i + col  ],  (local_K_update[offset+row*4+i] * local_pose_inv_K[offset+row*i+col]) );

		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	if (lid<16){
		for (uint i =0; i<4; i++){
			local_k2k[lid]			+=	local_A_B[ row * 4 + i ] 					* local_A_B[ SE3_elems + i * 4 + col ] ;		// SE3_elems = 16
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);

// 	if (lid==0){
// 		printf("\n\n local_K_update = \n");
// 		for (uint i=0; i< 2 ; i++){ for (uint j=0; j< 4 ; j++){ for (uint k=0; k< 4 ; k++){ 	printf(",	%f",local_K_update[		i*16 +j*4 +k]);	} printf("\n"); } printf("\n\n"); }
//
// 		printf("\n\n local_pose_inv_K = \n");
// 		for (uint i=0; i< 2 ; i++){ for (uint j=0; j< 4 ; j++){ for (uint k=0; k< 4 ; k++){ 	printf(",	%f",local_pose_inv_K[	i*16 +j*4 +k]);	} printf("\n"); } printf("\n\n"); }
//
// 		printf("\n\n local_A_B = \n");
// 		for (uint i=0; i< 2 ; i++){ for (uint j=0; j< 4 ; j++){ for (uint k=0; k< 4 ; k++){ 	printf(",	%f",local_A_B[			i*16 +j*4 +k]);	} printf("\n"); } printf("\n\n"); }
//
// 		printf("\n\n local_k2k = \n");
// 		for (uint i=0; i< 1 ; i++){ for (uint j=0; j< 4 ; j++){ for (uint k=0; k< 4 ; k++){ 	printf(",	%f",local_k2k[			i*16 +j*4 +k]);	} printf("\n"); } printf("\n\n"); }
// 	}

}


void mat_mul44( uint lid,	__local float local_A[16],		__local float local_B[16],		__local float local_C[16] ){

	uint elem 						= fmod((float)lid, SE3_elems);
	uint col						= fmod((float)elem, 4.0f);
	uint row						= elem / (uint)4;

	for (uint i =0; i<4; i++){
		if (lid<16){
			local_C[lid]			+=	local_A[ row * 4 + i ] 						* local_B[ i * 4 + col ] ;

			printf("\nmat_mul44(..)	lid=%u,	elem=%u,	col=%u,	row=%u	i=%u,	local_C[lid](%f)			+=	local_A[ row * 4 + i ](%f) 						* local_B[ i * 4 + col ](%f)", \
				lid,	elem,	col, row,	i,	local_C[lid],	local_A[ row * 4 + i ],		 local_B[ i * 4 + col ]		);
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		if (lid==0) printf("\n");
		barrier(CLK_LOCAL_MEM_FENCE);
	}
}
