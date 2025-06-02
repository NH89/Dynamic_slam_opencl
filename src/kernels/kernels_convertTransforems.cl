#include "kernels_macros.h"
#include "kernels.h"

#define	SO3_x		0
#define	SO3_y		1
#define	SO3_z		2

#define	ST3_x		4
#define	ST3_y		5
#define	ST3_z		6

#define	ONE			7
#define	ZERO		8

#define	ONE_		1
#define	MINUS_ONE_	-1

#define	COS_THETA				0
#define ONE_MINUS_COS_THETA		1
#define	SIN_THETA				2
#define	ZERO_VAR				3
#define ONE_VAR					4

void LieToP( uint lid, __local float SE3[9], __local float Pose[32/*16*/] ){

	const uint LtoP[16][6] = { /*	Indices for SE3[8] and vars[4], to compose the elements of 4x4 SE3 transformation matrix, from SE3 Lie vector.	*/\
								{ ONE_		, ONE,		COS_THETA, 		SO3_x, SO3_x, ONE_MINUS_COS_THETA 	},	/*	1*1   *cos(theta)    +    w_x w_x (1 - cos_theta)	= SE3[6] * vars[0]   +   SE3[0] * SE3[0] * vars[1]  */\
								{ ONE_		, SO3_z,	SIN_THETA, 		SO3_y, SO3_y, ONE_MINUS_COS_THETA	},	/*	1*w_z *sin(theta)    +    w_x w_y (1 - cos_theta) 	*/\
								{ ONE_		, SO3_y,	SIN_THETA, 		SO3_x, SO3_z, ONE_MINUS_COS_THETA	},	/*	1*w_y *sin(theta)    +    w_x w_z (1 - cos_theta)	*/\
								{ ONE_		, ONE,		ST3_x, 			ZERO,  ZERO,  ZERO_VAR				},	/*	t_x	*/\
	\
								{ ONE_		, SO3_z,	SIN_THETA,		SO3_x, SO3_y, ONE_MINUS_COS_THETA	},	/*	1*w_z *sin(theta)    +    w_x w_y (1 - cos_theta)	*/\
								{ ONE_		, ONE, 		COS_THETA,		SO3_y, SO3_y, ONE_MINUS_COS_THETA	},	/*	1*1   *cos(theta)    +    w_y w_y (1 - cos_theta)	*/\
								{ MINUS_ONE_, SO3_x, 	SIN_THETA, 		SO3_y, SO3_z, ONE_MINUS_COS_THETA	},	/*	-1*w_x*sin(theta)    +    w_y w_z (1 - cos_theta)	*/\
								{ ONE_		, ONE,		ST3_y, 			ZERO,  ZERO,  ZERO_VAR				},	/*	t_y	*/\
	\
								{ MINUS_ONE_, SO3_y,	SIN_THETA,		SO3_x, SO3_z, ONE_MINUS_COS_THETA	},	/*	-1*w_y *sin(theta)   +    w_x w_z (1 - cos_theta)	*/\
								{ ONE_		, SO3_x,	SIN_THETA,		SO3_y, SO3_z, ONE_MINUS_COS_THETA	},	/*	1*w_x  *sin(theta)   +    w_y w_z (1 - cos_theta)	*/\
								{ ONE_		, ONE,		COS_THETA,		SO3_z, SO3_z, ONE_MINUS_COS_THETA	},	/*	1*1    *cos(theta)   +    w_z w_z (1 - cos_theta)	*/\
								{ ONE_		, ONE,		ST3_z,			ZERO,  ZERO,  ZERO_VAR				},	/*	t_z	*/\
	\
								{ ONE_		, ZERO,		ZERO_VAR,		ZERO,  ZERO,  ZERO_VAR				},	/*	0	*/\
								{ ONE_		, ZERO,		ZERO_VAR,		ZERO,  ZERO,  ZERO_VAR				},	/*	0	*/\
								{ ONE_		, ZERO,		ZERO_VAR,		ZERO,  ZERO,  ZERO_VAR				},	/*	0	*/\
								{ ONE_		, ZERO,		ZERO_VAR,		ONE,   ONE,   ONE_VAR				},	/*	1	*/\
	};

	float3	So3 			= (float3)( SE3[0], SE3[1], SE3[2] );
	float	theta 			= fast_length(So3);
	float	cos_theta 		= cos(theta);
	float	one_cos_theta	= 1.0f - cos_theta;
	float	sin_theta		= sin(theta);

	float	vars[5];
	vars[0]					= cos_theta;
	vars[1]					= one_cos_theta;
	vars[2]					= sin_theta;
	vars[3]					= 0;
	vars[4]					= 1;

	const uint * L2P;
	if (lid < 16){
		L2P					= LtoP[lid];
		Pose[lid +16]		= L2P[0] * SE3[L2P[1]] * vars[L2P[2]]   +   SE3[L2P[3]] * SE3[L2P[4]] * vars[L2P[5]] ;
	}
}


void update_k2k(
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

	if(lid<32){
		for (uint i =0; i<4; i++){
			local_A_B[ lid ]		+=	local_K_update[ offset + row * 4 + i ] 		* local_pose_inv_K[ offset + row * i + col  ] ;
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	if (lid<16){
		for (uint i =0; i<4; i++){
			local_k2k[lid]			+=	local_A_B[ row * 4 + i ] 					* local_A_B[ SE3_elems + row * 4 + i ] ;
		}
	}
}

