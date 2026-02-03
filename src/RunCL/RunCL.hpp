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
constexpr uint num_SE3_DoF					= NUM_SE3_DOF;
constexpr uint block_size					= BLOCK_SIZE;							// or send as __private arg ? BUT as hardcoded "const uint" it can be used to size arrays etc.
constexpr uint out_block_size				= OUT_BLOCK_SIZE;
constexpr uint num_past_frames				= NUM_PAST_FRAMES;						// 1,2,4,8,16,32,64 // variable select window of 4 frames.
static constexpr uint max_patches_per_layer = 2^max_mipmap_layers * 2^max_mipmap_layers; //

#define FLOAT_16_EYE 	{1.0f, 0.0f, 0.0f, 0.0f,	0.0f, 1.0f, 0.0f, 0.0f,		0.0f, 0.0f, 1.0f, 0.0f,		0.0f, 0.0f, 0.0f, 1.0f}
constexpr float zero						= 0;
constexpr uint	patch_size					= 32;	// Set global patch size from device parameters // generally: device_work_size_multiple = patch_size * integer, eg 32, 64, 128

using namespace std;
class RunCL
{
public:
	RunCL( Json::Value obj_ );
	Json::Value 		obj;

	int					verbosity;
	bool				tiff, png, vtp;
	std::vector<cl_platform_id> 	m_platform_ids;
	cl_context			m_context;
	cl_device_id		m_device_id;
	cl_command_queue	m_queue, uload_queue, dload_queue, track_queue;
	cl_program			m_program;

	// Kernels
	// old kernels
	cl_kernel			convert_depth_kernel;
	cl_kernel			cvt_color_space_kernel, cvt_color_space_linear_kernel, sum_image_variance_kernel, sample_image_variance_kernel, blur_image_kernel;
	cl_kernel 			mipmap_float4_kernel, mipmap_float_kernel,  comp_param_maps_kernel;
	// 1st gen patch kernels ?
	cl_kernel			rho_sq_kernel, reduce_patch_Rho_kernel, update_k2k_kernel;	// TODO declare, create, release kernel in Run_cl.h etc.
	// RunCL_patchslam.cpp
	cl_kernel			compute_patch_lookup_table_kernel, patch_img_grad_kernel, patch_hessian_reduce_kernel;
	// RunCL_patch_tracking.cpp
	cl_kernel			pad_image_top_bottom2_kernel, vertcal_blur5_kernel, pad_image_left_right2_kernel, horiz_blur5_kernel, reduce_img_kernel;

	// GPU Buffers
	static const uint 	num_current_frames	= 5;																												// static = same for all instances of class Dynamic_slam.
	cl_mem 				imgmem[num_current_frames], 	velmap[num_current_frames], 	depth_mem, 	g1mem;
	/////////////////////////////


	struct frame{
		int 			frame_num		= 0;
		uint			frame_data_index= 0;				// Index of this frame in the recycled arrays of cl_mem buffers above: imgmem[], 	velmap[]

		cl_mem			img_buf			= nullptr;
		cl_mem			depth_buf		= nullptr;
		cl_mem			r_vel_buf		= nullptr;

		Matx44f			pose_gt			= Matx44f::eye() ;
		float			pose[16]		= FLOAT_16_EYE;		// Absolute pose of the frame. i.e. relative to initial frame. Computed from the local sample frames. Will req adjustment at loop closure.
		Matx44f			pose_0to1		= Matx44f::eye() ;

		Matx44f			K				= Matx44f::eye() ;	// camera intrinsic matrix
		Matx44f			inv_K			= Matx44f::eye() ;

		float			k2k_0to1_est[16]				= FLOAT_16_EYE;				// Reprojection matrix to the current img. Estimated, then fitted for each new frame, also with updates of camera inrinsic mattix.
		float			k2k[num_past_frames][16]		= {FLOAT_16_EYE};

		Matx16f			Jacobian[max_mipmap_layers]		= { Matx16f::zeros() };
		Matx66f			invHessian[max_mipmap_layers]	= { Matx66f::eye() };
	};

	std::array<frame, num_current_frames> 					current_frames;			// Needs to be initialized after the buffers are created.
	uint current_frames_idx[num_current_frames]				= {4,3,2,1,0};			// NB always access via:    current_frames[  current_frames_idx[ idx ]].img_buf   or   runcl.current_frames[ runcl.current_frames_idx[0] ].img_buf...
	uint new_current_frames_idx[num_current_frames]			= {4,3,2,1,0};			// Must be set correctly, because it will be swaped to current_frames_idx[.idx.]

	const float identity_flt16[16]	=	FLOAT_16_EYE;

	void initialize_current_frames(){
		for (uint idx = 0; idx < num_current_frames; idx++){
			current_frames[idx].frame_num			= -1;
			current_frames[idx].frame_data_index	= idx;							// Initialized with cl_mem buffers in order. This will change with update_current_frames_idx().

			current_frames[idx].img_buf				= imgmem[idx];
			current_frames[idx].r_vel_buf			= velmap[idx];					// velocity _relative_ to the camera.
			//current_frames[idx].depth_buf			= depth_mem[idx];

			current_frames[idx].pose_gt				= Matx44f::eye();
			for(uint i=0; i<16; i++){
				current_frames[idx].pose[i]			= identity_flt16[i];
				current_frames[idx].k2k_0to1_est[i]	= identity_flt16[i];
			}
			for(uint i=0; i<max_mipmap_layers; i++){
				current_frames[idx].invHessian[i]	= Matx66f::eye();
			}
		}
	}

	void update_current_frames_idx(){												// Call immediately _before_ loading new frame.
		int frame_count = dataset_frame_num;
		uint mod_16		= fmod(frame_count,16); // NB fastest way would be a nested if sequence, using bit shift to test the last bit.
		uint mod_8  	= fmod(mod_16,8);
		uint mod_4		= fmod(mod_8,4);
		uint mod_2		= fmod(mod_4,2);

		if (mod_8==0){																//cout<<"\n(mod_16==0) ";
			new_current_frames_idx[0] = current_frames_idx[4];
			new_current_frames_idx[1] = current_frames_idx[0];
			new_current_frames_idx[2] = current_frames_idx[1];
			new_current_frames_idx[3] = current_frames_idx[2];
			new_current_frames_idx[4] = current_frames_idx[3];
		}else if (mod_4==0){														//cout<<"\n(mod_8==0) ";
			new_current_frames_idx[0] = current_frames_idx[3];
			new_current_frames_idx[1] = current_frames_idx[0];
			new_current_frames_idx[2] = current_frames_idx[1];
			new_current_frames_idx[3] = current_frames_idx[2];
			new_current_frames_idx[4] = current_frames_idx[4];
		}else if (mod_2==0){														//cout<<"\n(mod_4==0) ";
			new_current_frames_idx[0] = current_frames_idx[2];
			new_current_frames_idx[1] = current_frames_idx[0];
			new_current_frames_idx[2] = current_frames_idx[1];
			new_current_frames_idx[3] = current_frames_idx[3];
			new_current_frames_idx[4] = current_frames_idx[4];
		}else{																		//cout<<"\n(mod_2==0) ";
			new_current_frames_idx[0] = current_frames_idx[1];
			new_current_frames_idx[1] = current_frames_idx[0];
			new_current_frames_idx[2] = current_frames_idx[2];
			new_current_frames_idx[3] = current_frames_idx[3];
			new_current_frames_idx[4] = current_frames_idx[4];
		}
		swap( new_current_frames_idx, current_frames_idx);

		// for(uint i=0; i<max_mipmap_layers; i++){
		// 	current_frames[ current_frames_idx[0] ].invHessian[i]	=  Matx66f::eye();
		// }

		current_frames[ current_frames_idx[0] ]		=	current_frames[ current_frames_idx[1] ];	// NB This will initialize the new frame with the values from the previous frame.

		return;
	};

	void test_update_current_frames_idx(uint num_iter){
		cout << "\n\n RunCL::test_update_current_frames_idx(uint "<<num_iter<<")";
		for (uint iter=0; iter<=num_iter; iter++){
			dataset_frame_num++;
			update_current_frames_idx();
			current_frames[current_frames_idx[0]].frame_data_index = dataset_frame_num;			//iter;
																																		cout<<"\niter="<<iter;
																																		for (uint idx=0; idx<5; idx++){
																																			cout<<"\t\t current_frames_idx["<<idx<<"]="<<current_frames_idx[idx]
																																			<<", frame="<< current_frames[current_frames_idx[idx]].frame_data_index<<",";
																																		}
																																		cout << flush;
		}
	}

	cl_mem				basemem, imgmem_blurred, SE3_grad_map_mem, SE3_incr_map_mem;
	cl_mem				depth_mem_temp, depth_mem_GT;																					// 'depth_mem_temp' is use to load & prepare data for depth_mem_GT and transform_depthmap

	cl_mem				k2kbuf, SE3_k2kbuf, fp32_param_buf, uint_param_buf, mipmap_buf, img_stats_buf;
	cl_mem				SE3_map_mem, SE3_rho_map_mem, SE3_weight_map_mem;
	cl_mem				pix_sum_mem, var_sum_mem;
	cl_mem				HSV_grad_mem;

	// buffers for patch kernel based Dynamic_slam
	cl_mem				patch_lookup_table_buf;
	cl_mem				SE3_hessian_pinv_map_mem,		SE3_jacobian_map_mem;
	cl_mem				pose_buf, pose_update_buf,	distorsion_update_buf,		old_results_buf,				K_buf, inv_K_buf;

	//
	cv::Mat				baseImage, key_frame;
																						// Assuming 32x32 patches. NB some GPUs may hold multipler patches pers workgroup, especially at the higher layers.
	uint				wg_counter[max_mipmap_layers] = {0};							// 10k = 10240x4320  => 10240/2^10=10, 4320/2^10=4.21.., so 10 reductions to img pyr apex <10x10.		// Workgroups per layer
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
	size_t 				lookup_table_offset[	max_mipmap_layers]		= {0};

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

	cv::Size 			baseImage_size, mm_Image_size;
	std::map< std::string, std::filesystem::path > paths;



	///////////////////////////////////// RunCL_class.cpp

	void testOpencl();
	void getDeviceInfoOpencl(cl_platform_id platform);
	int  convertToString(const char *filename, std::string& s);
	void createQueues();
	void createAndBulidProgramFromSource(cl_device_id *devices);
	void createKernels();

	void set_cam_bufs( cv::Matx44f k,  cv::Matx44f inv_k,  cv::Matx44f pose,  cv::Matx44f k2k );

	void mipmap_call_kernel(cl_kernel kernel_to_call, cl_command_queue queue_to_call, uint start, uint stop, bool layers_sequential, const size_t local_work_size);						// Call kernels on mipmap: start,stop allow running specific layers.
	void mipmap_call_kernel(cl_kernel kernel_to_call, cl_command_queue queue_to_call, bool layers_sequential=false){ mipmap_call_kernel( kernel_to_call,  queue_to_call, mm_start, mm_stop, layers_sequential, local_work_size); }

	void initialize_fp32_params();
	void initialize_RunCL( cv::Mat baseImage_ );																						// Setting up buffers & mipmap parameters
	void set_mimpmap_offsets();
	//void free_wg_offsets();
	void allocatemem();

	void CleanUp();																														// Exit...
	void exit_(int res);
	~RunCL();


	/////////////////////////////////////// RunCL_DownloadAndSave.cpp

	void createFolders();																												// Called by RunCL(..) constructor, above.
	void ReadOutput(uchar* outmat, cl_mem buf_mem, size_t data_size, size_t offset=0) ;

	vector<Matx44f> ReadOutput_44f_vec( cl_mem buf_mem, size_t offset=0);
	vector<Matx66f> ReadOutput_66f_vec( cl_mem buf_mem, size_t offset=0);
	vector<Matx16f> ReadOutput_16f_vec( cl_mem buf_mem, size_t offset=0);

	Matx44f ReadOutput_44f( cl_mem buf_mem, size_t offset=0);
	Matx66f ReadOutput_66f( cl_mem buf_mem, size_t offset=0);
	Matx16f ReadOutput_16f( cl_mem buf_mem, size_t offset=0);
	Matx61f ReadOutput_61f( cl_mem buf_mem, size_t offset=0);

	void Save_vtk(cv::Mat mat, cv::Mat keyframe, std::filesystem::path folder );

	void DownloadAndSave(cl_mem buffer, std::string count, std::filesystem::path folder, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range=1 );
	void DownloadAndSave_2Channel_volume(cl_mem buffer, std::string count, std::filesystem::path folder_tiff, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range, uint vol_layers );
	
	void DownloadAndSave_3Channel(cl_mem buffer, std::string count, std::filesystem::path folder_tiff, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range=1, uint offset=0, bool exception_tiff=false ){
		cv::Mat bufImg;
		DownloadAndSave_3Channel(buffer, count, folder_tiff, image_size_bytes,  size_mat,  type_mat,  show,  &bufImg,  max_range, offset, exception_tiff );
	}
	void DownloadAndSave_3Channel(cl_mem buffer, std::string count, std::filesystem::path folder_tiff, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, cv::Mat *bufImg, float max_range=1, uint offset=0, bool exception_tiff=false );
	void DownloadAndSave_3Channel_volume(cl_mem buffer, std::string count, std::filesystem::path folder, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range, uint vol_layers,  bool exception_tiff=false, float iter=0, bool display=false );

	void DownloadAndSave_6Channel(cl_mem buffer, std::string count, std::filesystem::path folder_tiff, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range, uint offset=0);
	void DownloadAndSave_6Channel_volume(cl_mem buffer, std::string count, std::filesystem::path folder, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range, uint vol_layers );
	
	void DownloadAndSave_HSV_grad(cl_mem buffer, std::string count, std::filesystem::path folder_tiff, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range, uint offset=0 );
	
	void SaveMat(cv::Mat temp_mat, int type_mat, std::filesystem::path folder_tiff, bool show, float max_range, std::string mat_name, std::string count);
	//void SaveMat_1chan(cv::Mat temp_mat, int type_mat, std::filesystem::path folder_tiff, bool show, float max_range, std::string mat_name/*, std::string count*/);
	void DownloadAndSaveVolume(cl_mem buffer, std::string count, std::filesystem::path folder, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range, bool exception_tiff=false );


	////////////////////////////////////// RunCL_load_image.cpp

	void precomp_param_maps (float SE3_k2k[max_mipmap_layers*num_SE3_DoF*16]);																						// Image loading & preparation
	void update_tracking_depthmap(cl_mem depthmap_);
	void loadFrame(cv::Mat image);
	void cvt_color_space();
	void sum_image_variance();
	void sample_image_variance();
	void mipmap_linear(cl_mem image_buf, std::string folder);
	void patch_img_gradients();
	
	void load_GT_depth(cv::Mat GT_depth, bool invert);																					// Depthmap loading & preparation
	void convert_depth(uint invert, float factor);
	void mipmap_depthmap(cl_mem depthmap_);
	

	////////////////////////////////////// RunCL_patch_slam.cpp

	const uint	patch_size										= 32;
	size_t		device_max_workitem_sizes[	3]					= {0};
	cl_uint		device_max_compute_units						=  0;
	cl_ulong	device_local_mem_size							=  0;

	uint		patch_kernel_workgroup_size						=  0;
	size_t		patch_lookup_table_offset[	max_mipmap_layers]	= {0};
	uint		patch_local_work_size[		max_mipmap_layers]	= {0};		// for kernels not using local mem. Otherwise must compute local_mem(available / req per patch) * patch_size(32)
	uint		patch_num_threads[			max_mipmap_layers]	= {0};
	uint		patch_cols_per_row[			max_mipmap_layers]	= {0};

	void	initialize_patch_params();
	void	compute_patch_lookup_table( );

	size_t	patch_img_gradients_workgroup_size[	max_mipmap_layers]	= {0};
	void	patch_img_gradients_set_params();
	void	patch_img_gradients( uint layer);								// NB this version uses patch_lookup_table.

	void	patch_hessian_reduce(uint layer);
	uint	patch_hessian_cols[ 			max_mipmap_layers]	= {0};
	uint	patch_hessian_rows[ 			max_mipmap_layers]	= {0};
	uint	patch_hessian_start_idx[ 		max_mipmap_layers][num_SE3_DoF][num_SE3_DoF]	= {{{0}}};
	uint	patch_ST3_hessian_start_idx[	max_mipmap_layers][3][3]						= {{{0}}};

	void	patch_hessian_reduce();

	////////////////////////////////////// RunCL_patch_tracking.cpp
	void	build_img_pyramid( std::string folder );

	void	blur_image_layer( uint layer );
	void	pad_image_top_bottom2(uint layer);
	void	vertcal_blur5(uint layer);
	void	pad_image_left_right2(uint layer);
	void	horiz_blur5(uint layer);

	void	reduce_img( uint layer, std::string folder );
	void 	copy_translate_img( uint layer );


	// 1st gen,  Patch based kernels /////////////////////////////
	void rho_sq( uint out_block_size, uint iter, uint layer);
	void reduce_patch_Rho ( uint out_block_size, uint iter, uint layer );

	struct rho_result{
      cl_float2	Rho					= {{0}};
      float		SE3_incr_arry[6*2]	= {0};
      float		Hessian[6*6]		= {0};
    } se3_rho_result;

	void update_k2k_cpu( uint layer );
	void update_k2k( uint layer, float delta_theta, float delta, Matx44f GT_pose );


	/////////////////////////////////////// RunCL_tracking.cpp
	void update_k2k_buf(	float k2k_3_16_[16], 	float pose_arry[16]  );
	void update_k2k_buf( Matx44f k2k, Matx44f pose );

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
		cl_mem 				memobj,
		string 				fname
	);

	void _clReleaseMemObject(cl_mem memobj);

	void _clReleaseKerne(cl_kernel kernel);

	string checkerror(int input);

};
#endif /*RUNCL_H*/
