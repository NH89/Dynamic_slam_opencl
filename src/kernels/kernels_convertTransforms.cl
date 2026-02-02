#include "kernels__macros.h"
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
/*
			//printf("\n updatek2k() lid=%u, elem=%u, offset=%u, col=%u, row=%u,  local_K_update[ offset + row * 4 + i ]=[ %u ]= %f,  local_pose_inv_K[ offset + row * i + col  ]=[ %u ]= %f  product= %f ", \
				lid, elem, offset, col, row, (offset + row * 4 + i), local_K_update[ offset + row * 4 + i ],  (offset + row * i + col),  local_pose_inv_K[ offset + row * i + col  ],  (local_K_update[offset+row*4+i] * local_pose_inv_K[offset+row*i+col]) );
*/
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	if (lid<16){
		for (uint i =0; i<4; i++){
			local_k2k[lid]			+=	local_A_B[ row * 4 + i ] 					* local_A_B[ SE3_elems + i * 4 + col ] ;		// SE3_elems = 16
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
/*
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
*/
}

void matmul_44x41_single_thread(float16 k2k,  float4 px_in, float px_out[4], bool print_  ){ // NB could use float4 for each k2k row, then dot4(row,px_in)

	px_out[0]	= k2k.s0*px_in.s0  + k2k.s1*px_in.s1 +  k2k.s2*px_in.s2  + k2k.s3*px_in.s3;
	px_out[1]	= k2k.s4*px_in.s0  + k2k.s5*px_in.s1 +  k2k.s6*px_in.s2  + k2k.s7*px_in.s3;

	px_out[2]	= k2k.s8*px_in.s0  + k2k.s9*px_in.s1 +  k2k.sa*px_in.s2  + k2k.sb*px_in.s3;
	px_out[3]	= k2k.sc*px_in.s0  + k2k.sd*px_in.s1 +  k2k.se*px_in.s2  + k2k.sf*px_in.s3;

	if(print_==true){
			printf("\n\n__device_fn matmul_44x41   px_in=(%f, %f, %f, %f),  px_out[0-3]=(%f, %f, %f, %f)",\
									px_in.s0, px_in.s1, px_in.s2, px_in.s3,   px_out[0], px_out[1], px_out[2], px_out[3] );
	}
	px_out[0]		/= px_out[3];
	px_out[1]		/= px_out[3];
}

void px_k2k( float16 k2k_,  float reduction,  uint v,  uint u,  float inv_depth, float *u2_flt_1,  float *v2_flt_1,  bool print_  ){

	float u_flt		= (float)u * reduction;
	float v_flt		= (float)v * reduction;

	float4 px_in	= (float4)( u_flt, v_flt, inv_depth, 1.0f );
	float px_out[4]	= { 0.0f, 0.0f, 0.0f, 0.0f };

	matmul_44x41_single_thread( k2k_, px_in, px_out, print_);

	*u2_flt_1		= px_out[0]/reduction;
	*v2_flt_1		= px_out[1]/reduction;

	if(print_==true){
			printf("\n\n__device_fn px_k2k()    u_flt=%f,  u2=%f,  v_flt=%f,   v2=%f,  k2k_.s0=%f,   inv_depth_1=%f", \
			u_flt,  *u2_flt_1,  v_flt, *v2_flt_1,  k2k_.s0,  inv_depth );
	}
}

void mat_mul44( uint lid,	__local float local_A[16],		__local float local_B[16],		__local float local_C[16] ){

	uint elem 						= fmod((float)lid, SE3_elems);
	uint col						= fmod((float)elem, 4.0f);
	uint row						= elem / (uint)4;

	for (uint i =0; i<4; i++){
		if (lid<16){
			local_C[lid]			+=	local_A[ row * 4 + i ] 						* local_B[ i * 4 + col ] ;
/*
			//printf("\nmat_mul44(..)	lid=%u,	elem=%u,		col=%u,	row=%u	i=%u,	local_C[lid](%f)			+=	local_A[ row * 4 + i ](%f) 			* local_B[ i * 4 + col ](%f)", \
				lid,	elem,	col, row,	i,	local_C[lid],	local_A[ row * 4 + i ],		 local_B[ i * 4 + col ]		);
*/
		}
		barrier(CLK_LOCAL_MEM_FENCE);
/*
		if (lid==0) printf("\n");
		barrier(CLK_LOCAL_MEM_FENCE);
*/
	}
}

/*
 * This GPU Hessiam matrix approach does not work.
 * NB the Hessian = the n-D curvature of the photomentric fit. It is used in Lucas-Kanade type fitting, e.g. the Inverse Compositional variant which we use.
 * 1) The Hessian matrix is not Hermitian, so Cholesky decomposition does not work.
 * 2) Generally the Hessian need not be fully invertible
 *		Consequently LDU decomposition and Moore-Penrose pseudo-inverse are needed.
 * 3) LDU decomposition involves multiple conditional branching, so is not easily adapted to GPU,
 * 4) There are good CPU libraries that (a) are very fast for 6x6 Hessian matricies, (b) can easily handle very much larger matrices.
 *
void cholesky_4x4(
	uint lid,
	__local float* mat_in, // [20] with 1st row zero, 4x5 holding 4x4
	__local float* mat_out // [16]
){
	const uint idx[16][4][4]={
		{ {1,1, 1,1}, {0,0, 0,0}, {0,0, 0,0}, {0,0, 0,0} },	//	l11*l11
		{ {2,1, 1,1}, {0,0, 0,0}, {0,0, 0,0}, {0,0, 0,0} },	//	l21*l11
		{ {3,1, 1,1}, {0,0, 0,0}, {0,0, 0,0}, {0,0, 0,0} },	//	l31*l11
		{ {4,1, 1,1}, {0,0, 0,0}, {0,0, 0,0}, {0,0, 0,0} },	//	l41*l11

		{ {2,1, 1,1}, {0,0, 0,0}, {0,0, 0,0}, {0,0, 0,0} },	//	l21*l11
		{ {2,1, 2,1}, {2,2, 2,2}, {0,0, 0,0}, {0,0, 0,0} },	//	l21*l21 + l22*l22
		{ {2,1, 3,1}, {2,2, 3,2}, {0,0, 0,0}, {0,0, 0,0} },	//	l21*l31 + l32*l22
		{ {2,1, 4,1}, {2,2, 4,2}, {0,0, 0,0}, {0,0, 0,0} },	//	l21*l41 + l42*l22

		{ {3,1, 1,1}, {0,0, 0,0}, {0,0, 0,0}, {0,0, 0,0} },	//	l31*l11
		{ {3,1, 2,1}, {3,2, 2,2}, {0,0, 0,0}, {0,0, 0,0} },	//	l31*l21 + l32*l22
		{ {3,1, 3,1}, {3,2, 3,2}, {3,3, 3,3}, {0,0, 0,0} },	//	l31*l31 + l32*l32 + l33*l33
		{ {3,1, 4,1}, {3,2, 4,2}, {3,3, 4,3}, {0,0, 0,0} },	//	l31*l41 + l32*l42 + l33*l43

		{ {4,1, 1,1}, {0,0, 0,0}, {0,0, 0,0}, {0,0, 0,0} },	//	l41*l11
		{ {4,1, 2,1}, {4,2, 2,2}, {0,0, 0,0}, {0,0, 0,0} },	//	l41*l21 + l42*l22
		{ {4,1, 3,1}, {4,2, 3,2}, {4,3, 3,3}, {0,0, 0,0} },	//	l41*l31 + l42*l32 + l43*l33
		{ {4,1, 4,1}, {4,2, 4,2}, {4,3, 4,3}, {4,4, 4,4} }	//	l41*l41 + l42*l42 + l43*l43 +l44*l44
	};
	float pvt_mat_out=0;
	float pvt_mat_in[16];

	if (lid<16){
		#pragma unroll
		for (int i=0; i<16; i++) pvt_mat_in[i] = mat_in[i];
		barrier(CLK_LOCAL_MEM_FENCE);

		#pragma unroll
		for (int i=0; i<4; i++){
			pvt_mat_out += pvt_mat_in[ 4*idx[lid][i][0]] +  pvt_mat_in[idx[lid][i][1]] 		*	pvt_mat_in[ 4*idx[lid][i][2]] +  pvt_mat_in[idx[lid][i][3]]  ;
		}
	}
	mat_out[lid]	= pvt_mat_out;
}

void pseudo_inverse_4x4(
	uint lid,
	__local float* mat_in,
	__local float* mat_out
){
	__local float	L_in[20];
	__local float	L[16];
	if (lid<4)		L[lid]		= 0.0f;
	if(lid<16)		L[lid+4]	= mat_in[lid];
	cholesky_4x4(lid, L_in, L);
	barrier(CLK_LOCAL_MEM_FENCE);

	__local float M1[32];
	__local float M2[32];

	if(lid<16){
		M1[lid] 		= L[lid];  // eliminate this
		M1[lid+16] 		=


}
*/
