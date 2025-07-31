#ifndef KERNELS_H
#define KERNELS_H

__constant float2 zero_f2				= {0.0f,0.0f};
__constant float4 zero_f4				= {0.0f,0.0f,0.0f,0.0f};
__constant float8 zero_f8				= {0.0f,0.0f,0.0f,0.0f, 0.0f,0.0f,0.0f,0.0f};

__constant float4 ones_f4				= {1.0f,1.0f,1.0f,1.0f};

__constant float  const_sqrt_flt_min	= 0x1.0p-63f; // 2^(-63) is sqrt of FLT_MIN = 2^(-126)

__constant uint block_size				= 32;										// or send as __private arg ? BUT as hardcoded "const uint" it can be used to size arrays etc.
__constant uint se3_dof					= 6;
__constant uint num_past_frames			= 4;										// 1,2,4,8,16,32,64 // variable select window of 4 frames.


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

float compute_optimum(__private float4 a, __private float4 b, __private float4 c);


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

void mat_mul44( uint lid,	__local float local_A[16],		__local float local_B[16],		__local float local_C[16] );

#endif /*KERNELS_H*/
