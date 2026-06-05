#ifndef RUNCL_H
#define RUNCL_H

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_HPP_MINIMUM_OPENCL_VERSION		300
#define CL_TARGET_OPENCL_VERSION			300  // defined as 120, ie OpenCL 1.2 in CMakeLists.txt
#define CL_HPP_TARGET_OPENCL_VERSION		300  // OpenCL 2.0

#include "../utils/verbosity.hpp"

#include <jsoncpp/json/json.h>

#include "../utils/convertTransforms.hpp"
#include "../utils/conf_params.hpp"
#include "../utils/print_functions.hpp"
#include "../utils/CV_chk.hpp"
#include "../utils/time_utils.hpp"

#include "../kernels/kernels__macros.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc_c.h> 			// req for types e.g. CV_BGR2GRAY
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui.hpp>

#include <eigen3/Eigen/QR>						// For (pseudo)inverse of hessian
#include <eigen3/Eigen/Dense>

#include <CL/opencl.hpp>

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <string>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <chrono>								// For measuring time of execution.

using namespace std::chrono;
constexpr uint tracking_num_colour_channels = TRACKING_NUM_COLOR_CHANNELS;
constexpr uint tracking_num_samples 		= TRACKING_NUM_SAMPLES +1;				// One more on host, for original Rho sample.
constexpr uint tracking_tot_samples 		= TRACKING_TOT_SAMPLES;
constexpr uint max_mipmap_layers 			= MAX_MIPMAP_LAYERS;					// Determines max image size, for img pyr apex < 10x10. 10k=>10, 8k=>9, 4k=>8, 2k=>7, SD(640x480)=>6 (2^6=64).
																					// Insufficient layers would reduce tracking robustness, due to more pixels in apex of image pyramid.
constexpr uint max_num_DoF					= MAX_NUM_DOF;
constexpr uint num_SE3_DoF					= NUM_SE3_DOF;
constexpr uint num_camera_matrix_DoF		= NUM_CAMERA_MATRIX_DOF;
constexpr uint num_lens_distortion_DoF		= NUM_LENS_DISTOTION_DOF;

constexpr uint block_size					= BLOCK_SIZE;							// or send as __private arg ? BUT as hardcoded "const uint" it can be used to size arrays etc.
constexpr uint out_block_size				= OUT_BLOCK_SIZE;
//constexpr uint num_current_frames				= NUM_PAST_FRAMES;					// 1,2,4,8,16,32,64 // variable select window of 4 frames.  /*num_past_frames*/
constexpr uint num_current_frames			= NUM_CURR_FRAMES;
static constexpr uint max_patches_per_layer = 2^max_mipmap_layers * 2^max_mipmap_layers; //

#define FLOAT_16_EYE 	{1.0f, 0.0f, 0.0f, 0.0f,	0.0f, 1.0f, 0.0f, 0.0f,		0.0f, 0.0f, 1.0f, 0.0f,		0.0f, 0.0f, 0.0f, 1.0f}
constexpr float identity_flt16[16]			= FLOAT_16_EYE;
constexpr float zero_flt					= 0;
constexpr uint  zero_uint					= 0;
constexpr cl_uint8  zero_uint8				= (cl_uint8){{0}};

constexpr uint	patch_size					= 32;	// Set global patch size from device parameters // generally: device_work_size_multiple = patch_size * integer, eg 32, 64, 128
constexpr size_t cl_flt16_size				= sizeof(cl_float16);

constexpr size_t	num_depth_steps			= NUM_DEPTH_STEPS;




using namespace std;
class RunCL
{
public:
	RunCL( Json::Value obj_ );
	Json::Value 		obj;
	int					verbosity;
	bool				tiff, png, vtp;
	float				max_depth;

	//OpenCL
	std::vector<cl_platform_id> 	m_platform_ids;
	cl_context			m_context;
	cl_device_id		m_device_id;
	cl_command_queue	m_queue, uload_queue, dload_queue, track_queue;
	cl_program			m_program;

	// Kernels
	// RunCL_autocalibration
	cl_kernel			comp_cam_and_lens_maps_kernel, patch_cam_and_lens_Hessian_kernel;
	// old kernels
	cl_kernel			convert_depth_kernel;
	cl_kernel			cvt_color_space_kernel, cvt_color_space_linear_kernel, cvt_image_kernel;
	cl_kernel 			mipmap_float_kernel,  comp_SE3_param_maps_kernel;
	// 1st gen patch kernels ?
	cl_kernel			rho_sq_to_0_kernel, rho_sq_from_0_kernel, reduce_patch_Rho_kernel, update_k2k_kernel;	// TO DO declare, create, release kernel in Run_cl.h etc.
	// RunCL_patchslam.cpp
	cl_kernel			compute_patch_lookup_table_kernel, patch_img_grad_kernel, patch_hessian_reduce_kernel;
	// RunCL_patch_tracking.cpp
	cl_kernel			pad_image_top_bottom2_kernel, vertcal_blur5_kernel, pad_image_left_right2_kernel, horiz_blur5_kernel, reduce_img_kernel;
	// RunCL_depth.cpp
	cl_kernel			update_depth_2_kernel, regularize_depth_kernel, enlarge_layer_float_kernel, use_inferred_depthmap_kernel, use_GT_depthmap_kernel;

	// GPU Buffers																	// static = same for all instances of class Dynamic_slam.
	cl_mem				fp32_param_buf=nullptr,					uint_param_buf=nullptr,								mipmap_buf=nullptr;
	cl_mem				img_stats_buf=nullptr,					patch_lookup_table_buf=nullptr;

	cl_mem				SE3_rho_map_mem=nullptr,				SE3_weight_map_mem=nullptr;
	cl_mem				SE3_map_mem=nullptr,					SE3_incr_map_mem=nullptr,							SE3_hessian_map_mem=nullptr;//, SE3_jacobian_map_mem;
	cl_mem				depth_mem_temp=nullptr,					depth_mem_GT=nullptr;								// 'depth_mem_temp' is use to load & prepare data for depth_mem_GT and transform_depthmap

	cl_mem				camera_matrix_map_mem=nullptr,			camera_matrix_grad_map_mem=nullptr,					camera_matrix_hessian_map_mem=nullptr;
	cl_mem				lens_distortion_map_mem=nullptr,		distorsion_update_buf=nullptr;

	cl_mem				pix_sum_mem=nullptr,					var_sum_mem=nullptr;
	cl_mem				HSV_grad_mem=nullptr,					ST3_img_grad_mem=nullptr,							img_grad_mem=nullptr,								img_edge_mem=nullptr;//, g1mem;

	cl_mem				SE3_k2kbuf=nullptr;
	cl_mem				old_results_buf=nullptr,				K_buf=nullptr, 										inv_K_buf=nullptr;

	cl_mem				basemem=nullptr, 						imgmem_blurred=nullptr;
	cl_mem 				imgmem[num_current_frames]={nullptr},	depth_mem[num_current_frames]={nullptr}, 			velmap[num_current_frames]={nullptr},				SE3_grad_map_mem[num_current_frames]={nullptr};
	cl_mem				k2kbuf[num_current_frames]={nullptr},	cur_frames_k2kbuf[num_current_frames]={nullptr},	cur_frames_st3buf[num_current_frames]={nullptr};
	cl_mem				pose_buf[num_current_frames]={nullptr};

	// current frames
	struct frame{
		int 			dataset_frame_num					= 0;
		uint			frame_count							= 0;
		uint			frame_data_index					= 0;					// Index of this frame in the recycled arrays of cl_mem buffers above: imgmem[], 	velmap[]
		////////////////////////////////////////////////////////////////////
		cl_mem			img_buf								= nullptr;
		cl_mem			SE3_grad_buf						= nullptr;
		cl_mem			depth_buf							= nullptr;
		cl_mem			r_vel_buf							= nullptr;

		cl_mem			pose_buf							= nullptr;
		cl_mem			k2k_buf_from_0						= nullptr;				// used from new frame for depth, cam & lens calib, relative to current depth map
		cl_mem			k2k_buf_to_0						= nullptr;				// used in tracking new frame, relative to depth map of ??
		////////////////////////////////////////////////////////////////////
		Matx44f			pose_gt								= Matx44f::eye();
		Matx44f			pose_from_start						= Matx44f::eye();
		Matx44f			pose_from_0							= Matx44f::eye();
		Matx44f			pose_to_0							= Matx44f::eye();

		Matx44f			K									= Matx44f::eye();		// camera intrinsic matrix
		Matx44f			inv_K								= Matx44f::eye();
		Matx44f			k2k_from_0							= Matx44f::eye();
		Matx44f			k2k_to_0							= Matx44f::eye();
		////////////////////////////////////////////////////////////////////
		Matx66f			inv_SE3_Hessian[			max_mipmap_layers]	= { Matx66f::eye() };
		Matx55d			inv_camera_matrix_Hessian[	max_mipmap_layers]	= { Matx55d::eye() };
		Matx55d			inv_lens_distortion_Hessian[max_mipmap_layers]	= { Matx55d::eye() };
	};
	std::array<	frame, 				num_current_frames	> 	current_frames;			// Needs to be initialized after the buffers are created.
	uint	current_frames_idx[		num_current_frames]		= {4,3,2,1,0};			// NB always access via:    current_frames[  current_frames_idx[ idx ]].img_buf   or   runcl.current_frames[ runcl.current_frames_idx[0] ].img_buf...
	uint	new_current_frames_idx[	num_current_frames]		= {4,3,2,1,0};			// Must be set correctly, because it will be swaped to current_frames_idx[.idx.]


	// variables
	cv::Mat				baseImage;
						// workgroup counter and offsets	### ? unused ?				// Assuming 32x32 patches. NB some GPUs may hold multipler patches pers workgroup, especially at the higher layers.
	uint				wg_counter[max_mipmap_layers]						 =  {0};	// 10k = 10240x4320  => 10240/2^10=10, 4320/2^10=4.21.., so 10 reductions to img pyr apex <10x10.		// Workgroups per layer
	uint				wg_offsets[max_mipmap_layers][max_patches_per_layer] = {{0}};	// 10k = 10240x4320  => 320x135=43200 (32x32)patches,	NB >75% unused, BUT avoids calloc & free.		// Workgroup start idx, for each layer
																						// Requires 432000*sizeof(uint) = 1,728,000bytes on 32bit, or 3,456,000bytes on 64bitsystem.
																						// Can be reduced by reducing MAX_MIPMAP_LAYERS, and => max image size.

	size_t  			global_work_size, mm_global_work_size, local_work_size, image_size_bytes, image_size_bytes_C1, mm_size_bytes_C1;
	size_t				kernel_work_size_multiple, device_work_size_multiple;
	size_t 				mm_size_bytes_C3, mm_size_bytes_C4, mm_size_bytes_C8, mm_size_bytes_half4, mm_vol_size_bytes;
	size_t 				so3_sum_size, so3_sum_size_bytes, mm_se3_sum_size, se3_sum_size_bytes, se3_sum2_size_bytes, pix_sum_size, pix_sum_size_bytes;
	uint				se3_sum_size;

	cl_device_id 		deviceId;
	
	static const uint	img_stats_size									= max_mipmap_layers*4*2;										// 8 layers, 4 channels, 2 variables.
	size_t				img_stats_size_bytes							= sizeof(float)*img_stats_size;
	float				img_stats[				img_stats_size]			= {0};

	size_t 				num_threads[			max_mipmap_layers]		= {0};
	size_t 				lookup_table_offset[	max_mipmap_layers]		= {0};	//	### ? unused ?

	cl_uint8			depthmap_params[		max_mipmap_layers]		= {zero_uint8};// depthmap params for depth inference kernels. NB dense packed img layers with margin.
	uint				patch_depthmap_width[	max_mipmap_layers ]		= {0};//TODO	could make these a struct, or an object.	//  ### ? unused ?
	uint				patch_depthmap_offset[	max_mipmap_layers ]		= {0};//													//  ### ? unused ?
	uint 				depth_save_offset[		max_mipmap_layers ]		= {0};//													//  ### ? unused ?

	uint				MipMap[					max_mipmap_layers	*8]	= {0};
	uint				uint_params[			8]						= {0};
	float				fp32_params[			16]						= {0};
	
	uint	 			mm_num_reductions;				//	
	uint				mm_num_blur_layers;				//
	int 				mm_gaussian_size;				//	
	int 				mm_margin;						//	
	int 				mm_height;						//	
	int 				mm_width;						//	
	int 				mm_layerstep;					//	

	uint 				mm_start;						//
	int 				mm_stop;
	int 				baseImage_width;				//	
	int 				baseImage_height;				//	
	int 				layerstep;						//	
	int 				costVolLayers;					//	
	int 				baseImage_type;					//	
	int 				mm_Image_type;					//	

	int 				dataset_frame_num		= 0;	//	Frame number in dataset, set in constructor from json file. Incremented in Dynamic_slam::nextFrame.
	uint				frame_count				= 0;

	cv::Size 			baseImage_size, mm_Image_size;
	std::map< std::string, std::filesystem::path > paths;



	///////////////////////////////////// RunCL_class.cpp

	void testOpencl();
	void getDeviceInfoOpencl(				cl_platform_id platform);
	int  convertToString(					const char *filename, std::string& s);
	void createQueues();
	void createAndBulidProgramFromSource(	cl_device_id *devices);
	void createKernels();

	void set_cam_bufs( 						cv::Matx44f k,  cv::Matx44f inv_k,  cv::Matx44f pose,  cv::Matx44f k2k );
	void set_all_cam_bufs(					cv::Matx44f k,  cv::Matx44f inv_k,  cv::Matx44f pose,  cv::Matx44f k2k );
	void set_cam_bufs( 						cv::Matx44f k,  cv::Matx44f inv_k,  cv::Matx44f pose,  cv::Matx44f k2k,	uint frame_idx );

	void mipmap_call_kernel(				cl_kernel kernel_to_call, cl_command_queue queue_to_call, uint start, uint stop, bool layers_sequential, const size_t local_work_size);						// Call kernels on mipmap: start,stop allow running specific layers.
	void mipmap_call_kernel(				cl_kernel kernel_to_call, cl_command_queue queue_to_call, bool layers_sequential=false){ mipmap_call_kernel( kernel_to_call,  queue_to_call, mm_start, mm_stop, layers_sequential, local_work_size); }

	void initialize_fp32_params();
	void initialize_patch_depthmap_offset();
	void initialize_RunCL( 					cv::Mat baseImage_ );													// Setting up buffers & mipmap parameters
	void set_mimpmap_offsets();
	void allocatemem();

	void CleanUp();																									// Exit...
	void exit_(int res);
	~RunCL();


	///////////////////////////////////// RunCL_autocalibration.cpp
	void precomp_cam_and_lens_maps(			 uint layer, cl_float16 SE3_k2k[ num_camera_matrix_DoF +1 ], cl_mem map_mem, uint num_vars, string calling_fn);
	void patch_cam_and_lens_Hessian(		 uint layer, cl_mem param_map_mem, cl_mem param_grad_map_mem, cl_mem param_hessian_map_mem  );
	void patch_cam_and_lens__hessian_reduce( uint layer, cl_mem param_hessian_map_mem, Matx55d &inv_Hessian);


	///////////////////////////////////// RunCL_current_frames.cpp
	Matx44f update_pose_bufs_cur_frames( Matx44f new_pose_0to1 );
	void initialize_current_frame( 		int idx);
	void initialize_new_frame (frame old_frame1);
	void initialize_current_frames();
	void update_current_frames_idx();
	void update_44f_buf(				Matx44f matrix44f,		cl_mem matrix_buf,	string fname );
	//	void test_update_current_frames_idx( uint num_iter );


	/////////////////////////////////////// RunCL_DownloadAndSave.cpp

	void 			createFolders();																				// Called by RunCL(..) constructor, above.
	void 			ReadOutput(				uchar* outmat, cl_mem buf_mem, size_t data_size, size_t offset=0) ;
	void 			ReadOutputRect(			uchar* outmat, cl_mem buf_mem, size_t data_size, size_t offset/*=0*/, size_t cols, size_t margin);

	vector<Matx44f> ReadOutput_44f_vec( 	cl_mem buf_mem, size_t offset=0);
	vector<Matx66f> ReadOutput_66f_vec( 	cl_mem buf_mem, size_t offset=0);
	vector<Matx16f> ReadOutput_16f_vec( 	cl_mem buf_mem, size_t offset=0);

	Matx44f 		ReadOutput_44f( 		cl_mem buf_mem, size_t offset=0);
	Matx66f 		ReadOutput_66f( 		cl_mem buf_mem, size_t offset=0);
	Matx16f 		ReadOutput_16f( 		cl_mem buf_mem, size_t offset=0);
	Matx61f 		ReadOutput_61f( 		cl_mem buf_mem, size_t offset=0);

	void Save_vtk(							cv::Mat mat, cv::Mat keyframe, std::filesystem::path folder );
	void Save_vtk_depth(					cl_mem depth_buf, cl_mem rho_buf, std::filesystem::path folder, 		uint layer, uint depth_iter_per_layer  );

	void Save_pcd_depth(					cl_mem depth_buf, cl_mem rho_buf, std::filesystem::path folder, 		size_t image_size_bytes, cv::Size size_mat, uint offset_rho_bytes, uint offset_depth_bytes, uint layer, float scale );
	void Save_csv_mat(						cv::Mat mat, std::filesystem::path folder, uint layer );
	void SavePoints_asciiPLY (				cv::Mat mat, cv::Mat mat_depth, std::filesystem::path folder, 			uint layer );

	void DownloadAndSave(					cl_mem buffer, std::string count, std::filesystem::path folder, 		size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range=1 );
	void DownloadAndSave_2Channel(			cl_mem buffer, std::string count, std::filesystem::path folder_tiff, 	size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range, uint offset );
	void DownloadAndSave_2Channel_volume(	cl_mem buffer, std::string count, std::filesystem::path folder_tiff, 	size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range, uint vol_layers );
	
	void DownloadAndSave_3Channel(			cl_mem buffer, std::string count, std::filesystem::path folder_tiff, 	size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range=1, uint offset=0, bool exception_tiff=false ){
		cv::Mat bufImg;
		DownloadAndSave_3Channel(			buffer, count, folder_tiff, image_size_bytes,  size_mat,  type_mat,  	show,  &bufImg,  max_range, offset, exception_tiff );
	}
	void DownloadAndSave_3Channel(			cl_mem buffer, std::string count, std::filesystem::path folder_tiff, 	size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, cv::Mat *bufImg, float max_range=1, uint offset=0, bool exception_tiff=false );
	void DownloadAndSave_3Channel_volume(	cl_mem buffer, std::string count, std::filesystem::path folder, 		size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range, uint vol_layers,  bool exception_tiff=false, float iter=0, bool display=false );

	void DownloadAndSave_6Channel(			cl_mem buffer, std::string count, std::filesystem::path folder_tiff, 	size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range, uint offset=0);
	void DownloadAndSave_6Channel_volume(	cl_mem buffer, std::string count, std::filesystem::path folder, 		size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range, uint vol_layers );
	
	void DownloadAndSave_HSV_grad(			cl_mem buffer, std::string count, std::filesystem::path folder_tiff, 	size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range, uint offset=0 );
	
	void SaveMat(							cv::Mat temp_mat, int type_mat,   std::filesystem::path folder_tiff, 	bool show, float max_range, std::string mat_name, std::string count);
	void DownloadAndSaveVolume(				cl_mem buffer, std::string count, std::filesystem::path folder, 		size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range, bool exception_tiff=false );

	void DownloadAndSaveDepthUpdate(		uint layer, uint offset_rho, uint offset_depth, string fname );

	////////////////////////////////////// RunCL_load_image.cpp
	void use_inferred_depthmap();
	void loadFrame(							cv::Mat image);
	void cvt_image();
	void cvt_color_space();
	void patch_img_gradients();
	
	void load_GT_depth(						cv::Mat GT_depth, bool invert);															// Depthmap loading & preparation
	void convert_depth(						uint invert, float factor);
	void mipmap_depthmap(					cl_mem depthmap_);
	

	////////////////////////////////////// RunCL_patch_slam.cpp
	// variables
	const uint	patch_size												= 32;
	size_t		device_max_workitem_sizes[			3]					= {0};
	cl_uint		device_max_compute_units								=  0;
	cl_ulong	device_local_mem_size									=  0;

	uint		patch_kernel_workgroup_size								=  0;
	size_t		patch_lookup_table_offset[			max_mipmap_layers]	= {0};
	uint		patch_local_work_size[				max_mipmap_layers]	= {0};		// for kernels not using local mem. Otherwise must compute local_mem(available / req per patch) * patch_size(32)
	uint		patch_num_threads[					max_mipmap_layers]	= {0};
	uint		patch_cols_per_row[					max_mipmap_layers]	= {0};

	size_t		patch_img_gradients_workgroup_size[	max_mipmap_layers]								=   {0};
	uint		patch_hessian_cols[ 				max_mipmap_layers]								=   {0};
	uint		patch_hessian_rows[ 				max_mipmap_layers]								=   {0};
	uint		patch_hessian_start_idx[ 			max_mipmap_layers][num_SE3_DoF][num_SE3_DoF]	= {{{0}}};
	uint		patch_ST3_hessian_start_idx[		max_mipmap_layers][3][3]						= {{{0}}};


	// functions
	void	initialize_patch_params();
	void	compute_patch_lookup_table( );

	void	patch_img_gradients_set_params();
	void	patch_img_gradients( 					uint layer);					// NB this version uses patch_lookup_table.
	void	patch_SE3_hessian_reduce (				uint layer);
	void	patch_cam_matrix_hessian_reduce(		uint layer);


	////////////////////////////////////// RunCL_patch_tracking.cpp
	void	precomp_SE3_param_maps (				float SE3_k2k[max_mipmap_layers*num_SE3_DoF*16],  cl_mem map_mem,  uint num_vars,  string calling_fn);		// Image loading & preparation
	void	build_img_pyramid( 						std::string folder );

	void	blur_image_layer( 						uint layer );
	void	pad_image_top_bottom2(					uint layer);
	void	vertcal_blur5(							uint layer);
	void	pad_image_left_right2(					uint layer);
	void	horiz_blur5(							uint layer);

	void	reduce_img( 							uint layer, std::string folder );
	void 	copy_translate_img( 					uint layer );


	// 1st gen,  Patch based kernels /////////////////////////////
	struct Rho_sq_params{
		size_t		kernel_workgroup_size;
		size_t		device_max_workitem_sizes[3];
		cl_uint		device_max_compute_units;
		size_t		max_workgroup_size;
	} rho_sq_params;

	struct Rho_result {
		cl_float2	Rho										= {{0}};
		float		param_incr_arry[	max_num_DoF*2]		=  {0};
	} se3_rho_result, camera_matrix_result, lens_distortion_result;

	void rho_sq_set_params( 			uint out_block_size);
	void rho_sq_to_0 (					uint out_block_size, uint iter, uint frame_idx, uint layer, cl_mem k2k_buf, uint num_DoF, string calling_fn);
	void rho_sq_from_0(					uint out_block_size, uint iter, uint frame_idx, uint layer, 			    uint num_DoF, string calling_fn);
	void reduce_patch_Rho(				uint out_block_size, uint iter, 				uint layer,					uint num_DoF);
	void get_rho_result( 				Rho_result &rho_result, 						uint layer,					uint num_DoF);

	void update_k2k_cpu( 				uint layer );
	void update_k2k( 					uint layer, float delta_theta, float delta, Matx44f GT_pose );


	/////////////////////////////////////// RunCL_tracking.cpp


	void SpatialCostFns();																												// SIRFS cost functions
	void ParsimonyCostFns();
	void ExhaustiveSearch();

	////////////////////////////////////// RunCL_depth.cpp
	void update_depth( 					uint out_block_size, uint layer);
	void update_depth_2( 				uint out_block_size, uint layer);
	void regularize_depth(				uint write_layer );
	void propagate_depth_next_layer(	uint layer);
	void use_inferred_depthmap(			uint write_layer );
	void use_GT_depthmap(				uint write_layer );


	//////////////////////////////////////
	void _cl_flush_finish(				cl_command_queue	_queue,  string fname);
	int  waitForEventAndRelease(		cl_event *event);
	void cl_mem_swap_ptr(				cl_mem buf1, cl_mem buf2);
	void _clSetKernelArg(				cl_kernel kernel,  cl_uint arg_index,  size_t arg_size, const void* arg_value, string fname);
	void _clReleaseMemObject(			cl_mem memobj);
	void _clReleaseKerne(				cl_kernel kernel);
	string checkerror(					int input);

	void _clEnqueueNDRangeKernel(
		cl_command_queue    _queue,
		cl_kernel           kernel,
		cl_uint             work_dim,
		const size_t *      global_work_offset,
		const size_t *      global_work_size,
		const size_t *      local_work_size,
		string              fname
	);

	void _clEnqueueWriteBuffer(
		cl_command_queue    command_queue,
		cl_mem              buffer,
		cl_bool             blocking_write,
		size_t              offset,
		size_t              size,
		const void*         ptr,
		string              fname
	);

	void _clEnqueueFillBuffer(
		cl_command_queue    command_queue,
		cl_mem              buffer,
		const void*         pattern,
		size_t              pattern_size,
		size_t              offset,
		size_t              size,
		string              fname
	);

	void _clEnqueueCopyBuffer(
		cl_command_queue    command_queue,
		cl_mem              src_buffer,
		cl_mem              dst_buffer,
		size_t              src_offset,
		size_t              dst_offset,
		size_t              size,
		string              fname
	);

	void _clCreateBuffer(
		cl_context          context,
		cl_mem_flags        flags,
		size_t              size,
		void*               host_ptr,
		cl_mem              memobj,
		string              fname
	);
};
#endif /*RUNCL_H*/
