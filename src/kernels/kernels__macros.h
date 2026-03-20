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
#define OLD_THETA			13  // replaces DynamicSLAM::old_theta

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

#define TRACKING_TOT_SAMPLES		4
#define MAX_MIPMAP_LAYERS			10//8							// Determines max image size, for img pyr apex < 10x10. 10k=>10, 8k=>9, 4k=>8, 2k=>7, SD(640x480)=>6 (2^6=64)
#define NUM_SE3_DOF					6
#define NUM_ST3_DOF					3

#define BLOCK_SIZE					32
#define OUT_BLOCK_SIZE				4
#define NUM_CURR_FRAMES				5

#define NUM_DEPTH_STEPS				16//8

// LOCAL VERBOSITY FOR EACH KERNEL FILE ##############################################
// These remove the "txt" code before compilation, so it will not take time in execution.
	/////////////////////////////////////// KERNELS_MAPPING.CL
    #define VK_MAPPING(txt)					/*txt*/					// to comment out,  /*txt*/

	/////////////////////////////////////// KERNELS_TRACKING.CL
    #define VK_TRACKING(txt)				/*txt*/

	/////////////////////////////////////// KERNELS_LOAD_IMAGE.CL
    #define VK_LOAD_IMAGE(txt)				/*txt*/

	/////////////////////////////////////// KERELS_PHOTOMETRIC_COST.CL
    #define VK_PHOTOMETRIC_COST(txt)		/*txt*/

	/////////////////////////////////////// KERNELS_SETUP_KEYFRAME.CL
    #define VK_SETUP_KEYFRAME(txt)			/*txt*/


#endif /*KERNEL_MACROS_H*/
