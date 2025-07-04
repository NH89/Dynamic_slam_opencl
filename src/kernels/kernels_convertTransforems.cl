#include "kernels_macros.h"
#include "kernels.h"

#define	SO3_x		0
#define	SO3_y		1
#define	SO3_z		2

#define	ST3_x		3
#define	ST3_y		4
#define	ST3_z		5

#define MINUS_ONE	6
#define	ZERO		7
#define	ONE			8

// #define	ONE_		1
// #define	MINUS_ONE_	-1

#define	COS_THETA				0
#define ONE_MINUS_COS_THETA		1
#define	SIN_THETA				2
#define	ZERO_VAR				3
#define ONE_VAR					4
#define MINUS_ONE_VAR			5

void LieToP( uint lid,	__local float SE3[9],	__local float Pose[32/*16*/] ){
	// NB SO3 3x3 mat is derived by Rodrigues Formula.
	// SO3 rotation matrix =  e^A	= 				Identity		+	A * sin_theta / theta		+	A^2  *  ( 1 - cos_theta ) / theta^2    // We use this version because it has fewer terms.
	//								= 	cos_theta * Identity		+	A * sin_theta / theta		+	B    *  ( 1 - cos_theta ) / theta^2
	// w.xyz = so3 rotation vector
	// theta = /w.xyz/ , i.e. pythagorean length of rotation vector, aka magnitude.
	// A = {{ 0 , -w.z, w.y }, { w.z, 0, -w.x }, { -w.y, w.x, 0}}

	// A^2 = {{-z^2-y^2,  yx,  zx}, {xy,  -z^2-x^2,  zy}, {xz,  yz,  -y^2-x^2}}

	const float identity44[16]	= {1,0,0,0,    0,1,0,0,    0,0,1,0,    0,0,0,1};

	const uint LtoP[16][9] = { /*	Indices for SE3[8] and vars[4], to compose the elements of 4x4 SE3 transformation matrix, from SE3 Lie vector.	*/\
								/*  Identity +	A * sin_theta / theta		+	A^2  * 											( 1 - cos_theta ) / theta^2   */
								{ ONE_VAR		, ONE,		ONE_VAR, 			MINUS_ONE_VAR,	SO3_z,	SO3_z,	SO3_y,	SO3_y,		ONE_MINUS_COS_THETA	},	/*	1	+		0   						+    0										= SE3[6] * vars[0]   +   SE3[0] * SE3[0] * vars[1]  */\
								{ MINUS_ONE_VAR	, SO3_z,	SIN_THETA, 			ONE_VAR,		ZERO,	ZERO,	SO3_y,	SO3_x,		ONE_MINUS_COS_THETA	},	/*	0	+		-1*w_z*(sin_theta)/theta    +    w_x w_y (1 - cos_theta)/ theta^2 	*/\
								{ ONE_VAR		, SO3_y,	SIN_THETA, 			ONE_VAR,		ZERO,	ZERO,	SO3_z,	SO3_x,		ONE_MINUS_COS_THETA	},	/*	0	+		1*w_y *(sin_theta)/theta    +    w_x w_z (1 - cos_theta)/ theta^2	*/\
								{ ONE_VAR		, ONE,		ST3_x, 				ZERO_VAR,		ZERO,	ZERO,	ZERO,	ZERO,		ZERO_VAR			},	/*	0	+		t_x							+    0									*/\
	\
								{ ONE_VAR		, SO3_z,	SIN_THETA,			ONE_VAR,		ZERO,	ZERO,	SO3_x,	SO3_y,		ONE_MINUS_COS_THETA	},	/*	0	+		1*w_z *(sin_theta)/theta    +    w_x w_y (1 - cos_theta)/ theta^2	*/\
								{ ONE_VAR		, ONE,		ONE_VAR,			MINUS_ONE_VAR,	SO3_z,	SO3_z,	SO3_x,	SO3_x,		ONE_MINUS_COS_THETA	},	/*	1	+		0    						+    0									*/\
								{ MINUS_ONE_VAR	, SO3_x, 	SIN_THETA, 			ONE_VAR,		ZERO,	ZERO,	SO3_z,	SO3_z,		ONE_MINUS_COS_THETA	},	/*	0	+		-1*w_x*(sin_theta)/theta    +    w_y w_z (1 - cos_theta)/ theta^2	*/\
								{ ONE_VAR		, ONE,		ST3_y, 				ZERO_VAR,		ZERO,	ZERO,	ZERO,	ZERO,		ZERO_VAR			},	/*	0	+		t_y							+    0									*/\
	\
								{ MINUS_ONE_VAR	, SO3_y,	SIN_THETA,			ONE_VAR,		ZERO,	ZERO,	SO3_x,	SO3_z,		ONE_MINUS_COS_THETA	},	/*	0	+		-1*w_y *(sin_theta)/theta   +    w_x w_z (1 - cos_theta)/ theta^2	*/\
								{ ONE_VAR		, SO3_x,	SIN_THETA,			ONE_VAR,		ZERO,	ZERO,	SO3_y,	SO3_z,		ONE_MINUS_COS_THETA	},	/*	0	+		1*w_x  *(sin_theta)/theta   +    w_y w_z (1 - cos_theta)/ theta^2	*/\
								{ ONE_VAR		, ONE,		ONE_VAR,			MINUS_ONE_VAR,	SO3_x,	SO3_x,	SO3_y,	SO3_y,		ONE_MINUS_COS_THETA	},	/*	1	+		0  							+    0									*/\
								{ ONE_VAR		, ONE,		ST3_z,				ZERO_VAR,		ZERO,  ZERO,	ZERO,	ZERO,		ZERO_VAR			},	/*	0	+		t_z							+    0									*/\
	\
								{ ZERO_VAR		, ZERO,		ZERO_VAR,			ZERO_VAR,		ZERO,  ZERO,	ZERO,	ZERO,		ZERO_VAR			},	/*	0	+		0							+    0									*/\
								{ ZERO_VAR		, ZERO,		ZERO_VAR,			ZERO_VAR,		ZERO,  ZERO,	ZERO,	ZERO,		ZERO_VAR			},	/*	0	+		0							+    0									*/\
								{ ZERO_VAR		, ZERO,		ZERO_VAR,			ZERO_VAR,		ZERO,  ZERO,	ZERO,	ZERO,		ZERO_VAR			},	/*	0	+		0							+    0									*/\
								{ ONE_VAR		, ONE,		ONE_VAR,			ZERO_VAR,		ZERO,  ZERO,	ZERO,	ZERO,		ZERO_VAR			},	/*	1	+		0							+    0									*/\
	};

	float3	So3 			= (float3)( SE3[0], SE3[1], SE3[2] );
	float	theta 			= fast_length(So3);
	if (lid==0) {printf("\n\nSE3[]="); for(uint i=0; i<9; i++) printf(",	%f",SE3[i]);}

	if (theta < FLT_EPSILON) {				// If theta is near to zero, rotation is "identity"
		if (lid < 16){
													Pose[lid +16]	=	identity44[lid];
			if ( fmod((float)lid,4)==3 && lid<12)	Pose[lid +16]	+=	SE3[3 + lid/4 ];
		}
	}else{
		float	inv_theta		= 1.0f / theta;
		float	cos_theta 		= cos(theta);
		float	one_cos_theta	= (1.0f - cos_theta)	* inv_theta * inv_theta ;
		float	sin_theta		= sin(theta)			* inv_theta ;
		float	vars[6];
				vars[0]			= cos_theta;
				vars[1]			= one_cos_theta;	//  ( 1 - cos_theta ) / theta^2
				vars[2]			= sin_theta;		//  sin_theta         / theta
				vars[3]			=  0;
				vars[4]			=  1;
				vars[5]			= -1;

		//const uint * L2P;
		if (lid < 16){
			const uint * L2P	= LtoP[lid];	// Rodrigues formula for 3x3 rotation, + ST3 for 4x4 SE3 matrix
									/* Identity	+	A * sin_theta / theta			+	A^2  *  																					( 1 - cos_theta ) / theta^2   */
			Pose[lid +16]		= 	vars[L2P[0]] * SE3[L2P[1]] * vars[L2P[2]] 		+	vars[L2P[3]] * ( SE3[L2P[4]] * SE3[L2P[5]]  +  SE3[L2P[6]] * SE3[L2P[7]] * SE3[L2P[8]] ) 	* vars[L2P[9]]  ;

			if (lid == 0) printf( "\nLieToP(..) lid=%u,  theta=%f,  cos_theta=%f,  ( 1 - cos_theta ) / theta^2 =%f,   sin_theta / theta =%f,    ", \
				lid,  theta,  cos_theta,  one_cos_theta,  sin_theta );

			printf( "\nLieToP(..) lid=%u,     vars[( %u )L2P[0]]( %f ) * SE3[( %u )L2P[1]]( %f ) * vars[( %u )L2P[2]]( %f )		+	vars[( %u )L2P[3]]( %f ) * ( SE3[( %u )L2P[4]]( %f ) * SE3[( %u )L2P[5]]( %f )  +  SE3[( %u )L2P[6]]( %f ) * SE3[( %u )L2P[7]]( %f ) * SE3[( %u )L2P[8]]( %f ) ) 	* vars[( %u )L2P[9]]( %f )            =  Pose[lid +16](%f)",\
									lid,	L2P[0],	vars[L2P[0]],	L2P[1],	SE3[L2P[1]],	L2P[2],	vars[L2P[2]],				L2P[3],	vars[L2P[3]],			L2P[4],	SE3[L2P[4]],		L2P[5], SE3[L2P[5]],		L2P[6],	SE3[L2P[6]],	L2P[7],	SE3[L2P[7]],		L2P[8],	SE3[L2P[8]],			L2P[9],	vars[L2P[9]],			 		Pose[lid +16]\
			);
		}
	}
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

			//printf("\nmat_mul44(..)	lid=%u,	elem=%u,		col=%u,	row=%u	i=%u,	local_C[lid](%f)			+=	local_A[ row * 4 + i ](%f) 			* local_B[ i * 4 + col ](%f)", \
				lid,	elem,	col, row,	i,	local_C[lid],	local_A[ row * 4 + i ],		 local_B[ i * 4 + col ]		);
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		if (lid==0) printf("\n");
		barrier(CLK_LOCAL_MEM_FENCE);
	}
}
