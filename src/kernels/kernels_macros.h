#ifndef KERNEL_MACROS_H
#define KERNEL_MACROS_H

#define MAX_INV_DEPTH		0	// fp32_params indices, 		for DTAM mapping algorithm.
#define MIN_INV_DEPTH		1
#define INV_DEPTH_STEP		2
#define ALPHA_G				3
#define BETA_G				4	//  __kernel void CacheG4
#define EPSILON 			5	//  __kernel void UpdateQD		// epsilon = 0.1
#define SIGMA_Q 			6									// sigma_q = 0.0559017
#define SIGMA_D 			7
#define THETA				8
#define LAMBDA				9	//  __kernel void UpdateA2
#define SCALE_EAUX			10
#define SE3_LM_A			11	// LM damped least squares parameters for SE3 tracking
#define SE3_LM_B			12

#define PIXELS				0	// uint_params indices, 		when launching one kernel per layer. 	Constant throughout program run.
#define ROWS				1	// baseimage
#define COLS				2
#define COSTVOL_LAYERS		3
#define MARGIN				4
#define MM_PIXELS			5	// whole mipmap
#define MM_ROWS				6
#define MM_COLS				7

#define MiM_PIXELS			0	// for mipmap_buf, 				when launching one kernel per layer. 	Updated for each layer.
#define MiM_READ_OFFSET		1	// for ths layer, 				start of image data
#define MiM_WRITE_OFFSET	2
#define MiM_READ_COLS		3	// cols without margins
#define MiM_WRITE_COLS		4
// #define MiM_GAUSSIAN_SIZE	5	// filter box size
#define MiM_READ_ROWS		6	// rows without margins
#define MiM_WRITE_ROWS		7

#define IMG_MEAN			0	// for img_stats
#define IMG_VAR 			1	//


#define TRACKING_NUM_COLOR_CHANNELS  4
#define TRACKING_NUM_SAMPLES         2                          // Just 2 _additional_ Rho samples



#endif /*KERNEL_MACROS_H*/
