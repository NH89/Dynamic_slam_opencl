#ifndef KERNELS_H
#define KERNELS_H

/* NB GPU limits
 * For Intel iRIS Xe
// Max number of constant args                     8
// Max constant buffer size                        4294959104 (4GiB)
NB shoud use these for things that never change during runtime, not for variables constant in a particular kernel but not another.
*/

#include "kernels__macros.h"

__constant float2 zero_f2				= {0.0f,0.0f};
__constant float4 zero_f4				= {0.0f,0.0f,0.0f,0.0f};
__constant float8 zero_f8				= {0.0f,0.0f,0.0f,0.0f, 0.0f,0.0f,0.0f,0.0f};

__constant float4 ones_f4				= {1.0f,1.0f,1.0f,1.0f};

__constant float  const_sqrt_flt_min	= 0x1.0p-63f;										// 2^(-63) is sqrt of FLT_MIN = 2^(-126)

__constant uint block_size				= BLOCK_SIZE;										// or send as __private arg ? BUT as hardcoded "const uint" it can be used to size arrays etc.
__constant uint num_SE3_DoF				= NUM_SE3_DOF;
__constant uint num_ST3_DoF				= NUM_ST3_DOF;

__constant uint num_current_frames		= NUM_CURR_FRAMES;									// 1,2,4,8,16,32,64 // variable select window of 4 frames.

__constant uint num_depth_steps			= NUM_DEPTH_STEPS;


// Declarations of local device functions used by the kernels.

// Tau_HSV_grad(I) := distance in 8D space [ sin(Hue ), cos(Hue ), Saturation , (Saturation)/dx , (Saturation)/dy , Value , (Value)/dx , (Vallue)/dy ]
float Tau_HSV_grad (float8 B, float8 c);

float8 Tau_HSV_grad_8chan (float8 B, float8 c);

///////////////////// Interpolation /////////////////////

float8 nearest_neigbour (float8* img, float u_flt, float v_flt, int cols, int read_offset_, uint reduction);

float8 bilinear (__global float8* img, float u_flt, float v_flt, int cols, int read_offset_, uint reduction);

float trilinear (__global float* vol, float u_flt, float v_flt, float layer_flt, int mm_pixels, int cols, int read_offset_, uint reduction);

float8 bilinear_SE3_grad (__global float8* img, float u_flt, float v_flt, int cols, int read_offset_); //, , uint reduction, int i, int mm_pixels);

float4 bilinear_flt4 (__global float4* img, float u_flt, float v_flt, int cols, int read_offset_);                                   // Used in tracking

void bilinear_SE3_grad_weight (float4 weights[6], __global float8* SE3_grad_map_cur_frame, int read_index, __global float8* SE3_grad_map_new_frame, float u2_flt, float v2_flt, int cols, int read_offset_, uint reduction, uint mm_pixels, float alpha );

float bilinear_grad_weight (__global float8* HSV_grad, int read_index, float u2_flt, float v2_flt, int cols, int read_offset_, uint reduction);

//float compute_optimum(__private float4 a, __private float4 b, __private float4 c);

void compute_minimum( float rho_sq_0, float rho_sq_1, float rho_sq_2, float inv_depth_0, float inv_depth_1, float inv_depth_2, float * prediction, float * optimum );


inline void atomic_maxf(															  				// from https://ingowald.blog/2018/06/24/float-atomics-in-opencl/
	volatile 	__global 	float 	*g_val,
							float 	myValue
){
	float cur = FLT_MIN;
	while (myValue > (cur = *g_val)) 		myValue 	= atomic_xchg( g_val,  fmax(cur,myValue) );
}

inline void atomic_minf(															  				// from https://ingowald.blog/2018/06/24/float-atomics-in-opencl/
	volatile 	__global 	float 	*g_val,
							float 	myValue
){
	float cur = FLT_MAX;
	while (myValue > (cur = *g_val)) 		myValue 	= atomic_xchg( g_val,  fmin(cur,myValue) );
}

// __constant const int test_var = 1;

////////////////////////  convertTransforms.cl ///////////////////////

void LieToP( uint lid, __local float SE3[9], __local float Pose[32/*16*/] );

#define SE3_elems  16
void update_k2_kdev_fn(
	uint lid,
	__local float local_K_update[	2* SE3_elems],
	__local float local_pose_inv_K[ 2* SE3_elems],
	__local float local_A_B[		2* SE3_elems],
	__local float local_k2k[		   SE3_elems]
	);


void matmul_44x41_single_thread(float16 k2k,  float4 px_in, float px_out[4],  bool print_   );

void px_k2k( float16 k2k_,  float reduction,  uint v,  uint u,  float inv_depth_1, float *u2_flt_1,  float *v2_flt_1,  bool print_  );

void mat_mul44( uint lid,	__local float local_A[16],		__local float local_B[16],		__local float local_C[16] );

#endif /*KERNELS_H*/
