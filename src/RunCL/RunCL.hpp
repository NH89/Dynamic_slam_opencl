#ifndef RUNCL_H
#define RUNCL_H

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_HPP_MINIMUM_OPENCL_VERSION		300
#define CL_TARGET_OPENCL_VERSION			300  // defined as 120, ie OpenCL 1.2 in CMakeLists.txt
#define CL_HPP_TARGET_OPENCL_VERSION		300  // OpenCL 2.0

#include "../utils/verbosity.hpp"

#include <jsoncpp/json/json.h>

#include "../utils/conf_params.hpp"
#include "../utils/print_functions.hpp"
#include "../utils/CV_chk.hpp"
#include "../utils/time_utils.hpp"

#include "../kernels/kernels_macros.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc_c.h> 				// req for types e.g. CV_BGR2GRAY
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui.hpp>

#include <boost/format.hpp>

#include <CL/opencl.hpp>

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <string>
#include <fstream>
#include <iostream>
#include <filesystem>							// C++17 TODO <replace boost/filesystem>
#include <chrono>								// For measuring time of execution.


using namespace std::chrono;
const uint tracking_num_colour_channels = TRACKING_NUM_COLOR_CHANNELS;
const uint tracking_num_samples 		= TRACKING_NUM_SAMPLES;			// One more on host, for original Rho sample.
const uint tracking_tot_samples 		= 4;
const uint max_mipmap_layers 			= 8;
const uint num_SE3_DoF					= 6;

using namespace std;
class RunCL
{
public:
	//#include "opencl_utils.hpp"

	RunCL( Json::Value obj_ );
	Json::Value 		obj;

	cv::Mat 			resultsMat;																																// used to insert images for multiple iterations, and variables for comparison. Size set in itialization, from cnf.json data.
	int					verbosity;
	bool				tiff, png, vtp;
	std::vector<cl_platform_id> 	m_platform_ids;
	cl_context			m_context;
	cl_device_id		m_device_id;
	cl_command_queue	m_queue, uload_queue, dload_queue, track_queue;
	cl_program			m_program;

	// Kernels
	cl_kernel			disparity_kernel;
	cl_kernel			convert_depth_kernel, invert_depth_kernel, transform_depthmap_kernel, transform_costvolume_kernel;
	cl_kernel 			depth_cost_vol_kernel, cost_kernel, cache3_kernel, cache4_kernel, updateQD_kernel, updateG_kernel, updateA_kernel, measureDepthFit_kernel;
	cl_kernel			cvt_color_space_kernel, cvt_color_space_linear_kernel, sum_image_variance_kernel, blur_image_kernel;
	cl_kernel 			reduce_kernel, mipmap_float4_kernel, mipmap_float_kernel, img_grad_kernel, se3_rho_sq_kernel, comp_param_maps_kernel;
	cl_kernel			se3_lk_grad_kernel, atomic_test1_kernel, atomic_test2_kernel;
	cl_kernel			compute_lookup_table_kernel, warp_image_kernel, img_sq_kernel, img_variance_kernel, compute_warp_kernel;
	
	// GPU Buffers
	cl_mem 				basemem, imgmem,  imgmem_blurred, gxmem, gymem, k_map_mem, dist_map_mem, SE3_grad_map_mem, SE3_incr_map_mem;
	cl_mem				cdatabuf, temp_cdatabuf, cdatabuf_8chan, hdatabuf, temp_hdatabuf, dbg_databuf;
	cl_mem 				dmem, amem, qmem, qmem2, lomem, himem, mean_mem, img_sum_buf, depth_mem_temp, depth_mem_GT;												// 'depth_mem_temp' is use to load & prepare data for depth_mem_GT and transform_depthmap

	cl_mem				k2kbuf, invk2kbuf, SO3_k2kbuf, SE3_k2kbuf, fp32_param_buf, uint_param_buf, mipmap_buf, gaussian_buf, img_stats_buf;
	cl_mem 				SE3_map_mem, SE3_rho_map_mem, se3_sum_rho_sq_mem, SE3_weight_map_mem;
	cl_mem 				pix_sum_mem, var_sum_mem, se3_sum_mem, se3_sum2_mem, se3_weight_sum_mem;
	cl_mem 				keyframe_imgmem, keyframe_imgmem_HSV_grad, keyframe_depth_mem, keyframe_g1mem, keyframe_SE3_grad_map_mem, keyframe_depth_mem_GT;
	cl_mem				HSV_grad_mem, dmem_disparity, dmem_disparity_sum;
	//cl_mem				binocular_disparity, binocular_rho;
	//cl_mem				atomic_test1_buf, atomic_test2_buf;
	cl_mem				lookup_table_buf,   curr_img_buf,  curr_img_sq_buf, curr_img_var_buf,    new_img_buf, new_img_warped_buf, new_img_sq_buf, new_img_var_buf,    img_covar_buf, img_corr_buf, warp_buf;
	
	//
	cv::Mat 			baseImage, key_frame;

	size_t  			global_work_size, mm_global_work_size, local_work_size, image_size_bytes, image_size_bytes_C1, mm_size_bytes_C1;
	size_t 				mm_size_bytes_C3, mm_size_bytes_C4, mm_size_bytes_C8, mm_size_bytes_half4, mm_vol_size_bytes;
	size_t 				so3_sum_size, so3_sum_size_bytes, mm_se3_sum_size, se3_sum_size_bytes, se3_sum2_size_bytes, pix_sum_size, pix_sum_size_bytes;
	size_t 				d_disp_sum_size, d_disp_sum_size_bytes;
	uint				se3_sum_size;

	cl_device_id 		deviceId;
	
	size_t				img_stats_size_bytes 	= sizeof(float)*8*4*2;
	float				img_stats[8*4*2]		= {0};		// 8 layers, 4 channels, 2 variables.
	size_t 				num_threads[8]			= {0};
	size_t 				lookup_table_offset[8] 	= {0};
	uint 				MipMap[8*8]				= {0};
	uint				uint_params[8]			= {0};
	
	float				fp32_params[16]			= {0};
	float				fp32_so3_k2k[9]			= {0};
	float 				fp32_k2keyframe[16]		= {0};
	
	uint	 			mm_num_reductions;				//	
	int 				mm_gaussian_size;				//	
	int 				mm_margin;						//	
	int 				mm_height;						//	
	int 				mm_width;						//	
	int 				mm_layerstep;					//	
	int 				fp16_size;
	uint 				mm_start;						//	
	int 				mm_stop;
	int 				baseImage_width;				//	
	int 				baseImage_height;				//	
	int 				layerstep;						//	
	int 				costVolLayers;					//	
	int 				baseImage_type;					//	
	int 				mm_Image_type;					//	
	
	int 				dataset_frame_num;				//	Frame number in dataset, set in constructor from json file. Incremented in Dynamic_slam::nextFrame.
	int 				costvol_frame_num;				//	Frame number in the cost volume. Set = 0 in RunCL::initializeDepthCostVol(..) . Incremented in Dynamic_slam::nextFrame(..)
	int 				keyFrameCount			= 0;	//	used in saving data to file. Incremented in Dynamic_slam::initialize_new_keyframe(..)
	int 				save_index				= 0;	//	Set in RunCL::initializeDepthCostVol(), and RunCL::updateDepthCostVol(), to save_index = keyFrameCount*1000 + costvol_frame_num;
	
	int 				QD_count 				= 0; 	//	Incremented in RunCL::updateQD(..) Set = 0 in Dynmaic_slam::initialize_new_keyfrme(..)  & in Dynamic_slam::nextFrame()
	int 				A_count					= 0;	//	Incremented in RunCL::updateA(..), ditto
	int 				G_count					= 0;	//	Incremented in RunCL::updateG(..), ditto
	
	cv::Size 			baseImage_size, mm_Image_size;
	std::map< std::string, std::filesystem::path > paths;

	///////////////////////////////////// RunCL_class.cpp


	void testOpencl();
	void getDeviceInfoOpencl(cl_platform_id platform);
	int  convertToString(const char *filename, std::string& s);
	void createQueues();
	void createAndBulidProgramFromSource(cl_device_id *devices);
	void createKernels();


	void mipmap_call_kernel(cl_kernel kernel_to_call, cl_command_queue queue_to_call, uint start, uint stop, bool layers_sequential, const size_t local_work_size);						// Call kernels on mipmap: start,stop allow running specific layers.

	void mipmap_call_kernel(cl_kernel kernel_to_call, cl_command_queue queue_to_call, bool layers_sequential=false){ mipmap_call_kernel( kernel_to_call,  queue_to_call, mm_start, mm_stop, layers_sequential, local_work_size); }

	//void mipmap_call_kernel(cl_kernel kernel_to_call, cl_command_queue queue_to_call){ mipmap_call_kernel( kernel_to_call,  queue_to_call, mm_start, mm_stop, false, local_work_size); } // , true

	void layer_call_kernel(cl_kernel kernel_to_call,  cl_command_queue queue_to_call, uint layer, const size_t local_work_size);

	void initialize_fp32_params();
	void initialize_RunCL( cv::Mat baseImage_ );																						// Setting up buffers & mipmap parameters
	void allocatemem();

	void CleanUp();																														// Exit...
	void exit_(int res);
	~RunCL();

	/////////////////////////////////////// RunCL_disparity.cpp

	void compute_lookup_table(uint start, uint stop);
	// void binocular_disparity_copy_buffers(  );
	void warp_image(   uint layer, uint iter);
	void img_sq(       uint layer, uint iter, cl_mem img_buf,    cl_mem img_sq_buf,  std::string folder);
	void img_variance( uint layer, uint iter, cl_mem img_sq_buf, cl_mem img_var_buf, std::string folder);
	void compute_warp( uint layer, uint iter);


	/////////////////////////////////////// RunCL_DownloadAndSave.cpp

	void createFolders();																												// Called by RunCL(..) constructor, above.
	void ReadOutput(uchar* outmat) ;
	void ReadOutput(uchar* outmat, cl_mem buf_mem, size_t data_size, size_t offset=0) ;
	void saveCostVols(float max_range);

	void Store_keyframe();																												// Required for Save_vtk(..), used for amem, demem etc.
	void Save_vtk(cv::Mat mat, cv::Mat keyframe, std::filesystem::path folder );

	// void DownloadAndSave_lookuptable(cl_mem buffer, std::string count, std::filesystem::path folder_tiff, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range );

	void DownloadAndSave(cl_mem buffer, std::string count, std::filesystem::path folder, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range=1 );
	void DownloadAndSave_2Channel_volume(cl_mem buffer, std::string count, std::filesystem::path folder_tiff, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range, uint vol_layers );
	
	void DownloadAndSave_3Channel(cl_mem buffer, std::string count, std::filesystem::path folder_tiff, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range=1, uint offset=0, bool exception_tiff=false ){
		cv::Mat bufImg;
		DownloadAndSave_3Channel(buffer, count, folder_tiff, image_size_bytes,  size_mat,  type_mat,  show,  &bufImg,  max_range, offset, exception_tiff );
	}
	void DownloadAndSave_3Channel(cl_mem buffer, std::string count, std::filesystem::path folder_tiff, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, cv::Mat *bufImg, float max_range=1, uint offset=0, bool exception_tiff=false );
	void DownloadAndSave_3Channel_volume(cl_mem buffer, std::string count, std::filesystem::path folder, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range, uint vol_layers,  bool exception_tiff=false, float iter=0, bool display=false );

	void PrepareResults_3Channel(cl_mem buffer, size_t image_size_bytes, cv::Size size_mat, int type_mat, cv::Mat *bufImg, float max_range /*=1*/, uint offset /*=0*/ );
	void PrepareResults_3Channel_volume(cl_mem buffer, size_t image_size_bytes, cv::Size size_mat, int type_mat, float max_range, uint vol_layers,  float iter);

	void DownloadAndSave_6Channel(cl_mem buffer, std::string count, std::filesystem::path folder_tiff, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range, uint offset=0);
	void DownloadAndSave_6Channel_volume(cl_mem buffer, std::string count, std::filesystem::path folder, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range, uint vol_layers );
	
	void DownloadAndSave_8Channel(cl_mem buffer, std::string count, std::map< std::string, std::filesystem::path > folder_tiff, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range /*=1*/, uint offset /*=0*/);
	void DownloadAndSave_8Channel_volume(cl_mem buffer, std::string count, std::filesystem::path folder, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range, uint vol_layers );
	
	void DownloadAndSave_HSV_grad(cl_mem buffer, std::string count, std::filesystem::path folder_tiff, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range, uint offset=0 );
	
	void SaveMat(cv::Mat temp_mat, int type_mat, std::filesystem::path folder_tiff, bool show, float max_range, std::string mat_name, std::string count);
	void SaveMat_1chan(cv::Mat temp_mat, int type_mat, std::filesystem::path folder_tiff, bool show, float max_range, std::string mat_name, std::string count);
	void DownloadAndSaveVolume(cl_mem buffer, std::string count, std::filesystem::path folder, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range, bool exception_tiff=false );

	////////////////////////////////////// RunCL_load_image.cpp

	void precom_param_maps(float SO3_k2k[6*16]);																						// Image loading & preparation
	void loadFrame(cv::Mat image);
	void cvt_color_space();
	void sum_image_variance();
	void blur_image();
	void mipmap_linear();
	void img_gradients();
	
	void load_GT_depth(cv::Mat GT_depth, bool invert);																					// Depthmap loading & preparation
	void convert_depth(uint invert, float factor);
	void mipmap_depthmap(cl_mem depthmap_);
	
	/////////////////////////////////////// RunCL_tracking.cpp
	void update_tracking_depthmap(cl_mem depthmap_);
	//void initialize_tracking_depthmap(float initial_depth);
	void se3_rho_sq( const uint local_num_samples,  const uint start_sample_idx,  float Rho_sq_results[tracking_tot_samples][max_mipmap_layers][tracking_num_colour_channels],    const float count[4], uint start, uint stop, float k2k_3_16_[tracking_tot_samples][16]  ); //float k2k_[16]  );
	void se3_rho_sq( 								float Rho_sq_results[tracking_tot_samples][max_mipmap_layers][tracking_num_colour_channels], 	const float count[4], uint start, uint stop, float k2k_3_16_[tracking_tot_samples][16]  );				// Tracking
	void estimateSE3_LK(float local_k2k[16], float SE3_results[max_mipmap_layers][num_SE3_DoF][tracking_num_colour_channels], float SE3_weights_results[max_mipmap_layers][num_SE3_DoF][tracking_num_colour_channels], float Rho_sq_results[max_mipmap_layers][tracking_num_colour_channels], int count, uint start, uint stop);

	void read_Rho_sq(float Rho_sq_results[max_mipmap_layers][tracking_num_colour_channels], int offset=0);

	void read_se3_weights(float SE3_weights_results[max_mipmap_layers][num_SE3_DoF][tracking_num_colour_channels]);
	void read_se3_incr(float SE3_results[max_mipmap_layers][num_SE3_DoF][tracking_num_colour_channels]);

	void writeToResultsMat(cv::Mat *bufImg  , uint column_of_images , uint row_of_images );
	void tracking_result(string result);
	void estimateCalibration();																											// Camera calibration
	void RelativeVel_Map();																												// RelativeVelMap - placeholder...
	void atomic_test1();
	void atomic_test2();

	/////////////////////////////////////// RunCL_mapping.cpp
	void swap_costvol_pointers();
	void transform_depthmap( /*cv::Matx44f K2K_*/ float K2K_arry[16], cl_mem depthmap_);																		// Cost volume
	void transform_costvolume(/*cv::Matx44f K2K_*/ float K2K_arry[16]);						//, cl_mem old_cdata_mem =cdatabuf,  cl_mem new_cdata_mem =new_cdatabuf, cl_mem old_hdata_mem =hdatabuf,  cl_mem new_hdata_mem =new_hdatabuf );
	
	void initializeFirstDepthCostVol( float default_depth );
	void initializeDepthCostVol( cl_mem key_frame_depth_map_src);		// Depth costvol functions
	void updateDepthCostVol(cv::Matx44f K2K_, int count);
	void updateQD(float epsilon, float theta, float sigma_q, float sigma_d);
	void updateG(int count);
	void updateA(float lambda, float theta);
	void computeSigmas(float epsilon, float theta, float L, float &sigma_d, float &sigma_q);
	
	void measureDepthFit();

	void SpatialCostFns();																												// SIRFS cost functions
	void ParsimonyCostFns();
	void ExhaustiveSearch();

	//////////////////////////////////////
	void _clEnqueueNDRangeKernel(
	cl_command_queue _queue,
	cl_kernel        kernel,
	cl_uint          work_dim,
	const size_t *   global_work_offset,
	const size_t *   global_work_size,
	const size_t *   local_work_size,
	string 			fname
	);

	void _cl_flush_finish(cl_command_queue	_queue,  string fname);

	int  waitForEventAndRelease(cl_event *event);

	void cl_mem_swap_ptr(cl_mem buf1, cl_mem buf2);

	void _clSetKernelArg(cl_kernel kernel,  cl_uint arg_index,  size_t arg_size, const void* arg_value, string fname);

	void _clEnqueueWriteBuffer(
		cl_command_queue    command_queue,
		cl_mem              buffer,
		cl_bool             blocking_write,
		size_t              offset,
		size_t              size,
		const void*         ptr,
		string 				fname
	);

	void _clEnqueueFillBuffer(
		cl_command_queue    command_queue,
		cl_mem              buffer,
		const void*         pattern,
		size_t              pattern_size,
		size_t              offset,
		size_t              size,
		string 				fname
	);

	void _clEnqueueCopyBuffer(
		cl_command_queue command_queue,
		cl_mem src_buffer,
		cl_mem dst_buffer,
		size_t src_offset,
		size_t dst_offset,
		size_t size,
		string fname
	);

	void _clCreateBuffer(
		cl_context          context,
		cl_mem_flags        flags,
		size_t              size,
		void*               host_ptr,
		cl_int*             errcode_ret,
		cl_mem 				memobj,
		string 				fname
	);

	void _clReleaseMemObject(cl_mem memobj);

	void _clReleaseKerne(cl_kernel kernel);

	string checkerror(int input);

};
#endif /*RUNCL_H*/
