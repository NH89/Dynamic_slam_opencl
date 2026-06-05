#pragma once

#include "../RunCL/RunCL.hpp"

#include <vector>
#include <string>
#include <fstream>
#include <filesystem>
#include <set>

#include "../utils/convertAhandaPovRayToStandard.hpp"

#define Rx	0	// TO DO  correct to match Python code and convention ( st(3), so(3) )
#define Ry  1
#define Rz	2
#define Tx	3
#define Ty	4
#define Tz	5

namespace fs = std::filesystem;

class Dynamic_slam
{
  public:
    ~Dynamic_slam();
    Dynamic_slam(Json::Value obj_);
    Json::Value             obj;
    int                     verbosity;
    RunCL                   runcl;
    bool					GT_available						= false;
    bool					use_artif_pose_error				= false;
    bool					use_GT_pose							= false;
    bool					use_conf_camera_matx				= false;
    bool					use_GT_camera_matx					= false;
    bool					invert_GT_depth						= false;
    bool					initialize_keyframe_from_GT 		= false;
	bool					initialize_tracking_from_GT_depth 	= false;

    //uint                    SE_iter_per_layer;
    uint                    SE3_stop_layer;
    uint                    SE3_start_layer;
    uint                    SE_iter;

    // data files
    std::string             rootpath;
    fs::path                root;
    std::vector<fs::path>   txt;
    std::vector<fs::path>   png;
    std::vector<fs::path>   depth;

    // camera & pose params
    cv::Matx44f initial_K, inv_initial_K;
	float f;								//= fmaxf(	obj["cameraMatrix"][0].asFloat(),	obj["cameraMatrix"][4].asFloat()	);	// focal length in pixels.
	float delta[max_mipmap_layers];			//= obj["min_depth"].asFloat() /f ;														// ST3_delta * (Translation to cause 1 pixel of parallax at min_depth)
	float delta_theta[max_mipmap_layers];	//= 1/f																					// SO3_delta_theta * (Rotation to cause 1 pixel of rotation flow)
	float cos_theta[max_mipmap_layers];		//= cos(delta_theta);
	float sin_theta[max_mipmap_layers];		//= sin(delta_theta);
	float delta_depth[max_mipmap_layers];	//= f*2.0f / ( min_depth * fmaxf(  obj["cameraMatrix"][2].asFloat(),	obj["cameraMatrix"][5].asFloat() )  );
	Matx16f deltas_matx[max_mipmap_layers];	// used to multiply SE3 update results.



    struct pose_datum{      // default intitialization, if instatiated with " ... = {}; "
      cv::Matx16f           keyframe2pose_algebra   = {0} ;

      cv::Matx44f           K                       = Matx44f::eye() ;             // camera intrinsic matrix
      cv::Matx44f           inv_K                   = Matx44f::eye() ;
      cv::Matx44f           pose                    = Matx44f::eye() ;             // pose in global coords. (not pose2pose from prev_frame, nor from keyframe) ?
      cv::Matx44f           inv_pose                = Matx44f::eye() ;
      cv::Matx44f           prev_pose2pose          = Matx44f::eye() ;
      cv::Matx44f           K2K                     = Matx44f::eye() ;             // Kamera to Kamera reprojection.
                                                                                //cv::Matx44f           pose_from_start         = MATX44F_EYE ;             // pose2pose_accumulated
      // lens distortion params
    };

    struct frame_datum{     // default intitialization, if instatiated with " ... = {}; "
      uint                  keyframe_index          = 0 ;                       // Index within this vector< >, of the keyframe for this frame.
      pose_datum            frame_data              = {};
      pose_datum            frame_data_GT           = {};
      pose_datum            error_data              = {};
    };
/*
    struct keyframe_datum{  // default intitialization, if instatiated with " ... = {}; "
      uint                  first_frame_index       = 0 ;
      cv::Mat               reference_image;
      cv::Mat               depthmap;
      frame_datum           frame_data              = {};
    };
*/
    std::vector<frame_datum>     frame_data;			// (frame_data start, old, current, key_frame) are now indices of elements in the vector.
//    std::vector<keyframe_datum>  keyframe_data;		// TO DO remove keyframes ?

    // GT data loading ?
    cv::Mat image, depth_GT, cameraMatrix;				// TO DO should these be Matx ?   , projection   NB cameraMatrix => K_GT
    cv::Mat R,     T;
    cv::Mat old_R, old_T;

    float SE3_k2k[ max_mipmap_layers*num_SE3_DoF*16 ];	// used for param_maps, minimal steps in SE3


    // functions ////////////////////////////////////////

    // return the filenames of all files that have the specified extension
    // in the specified directory and all subdirectories
    void get_all(const fs::path& root , const string& ext, vector<fs::path>& ret ) {
      if (!fs::exists(root))        return;
      if (fs::is_directory(root))   {
        typedef std::set<std::filesystem::path> Files;
        Files files;
        fs::recursive_directory_iterator it0(root);
        fs::recursive_directory_iterator endit0;
        std::copy(it0, endit0, std::inserter(files, files.begin()));
        Files::iterator it= files.begin();
        Files::iterator endit= files.end();

        while(it != endit)
        {
          if (fs::is_regular_file(*it) && (*it).extension() == ext)
          {
            ret.push_back(*it);
          }
          ++it;
        }
      }
    }

    /////////////////////////////////////// Dynamic_slam_class.cpp
    void initialize_resultsMat();
    void initialize_camera_intrinsic_matrix();
    void initialize_camera_vec();

    int  nextFrame();
    void getFrame();

    /////////////////////////////////////// Dynamic_slam_ground_truth.cpp
    void getFrameData_vec( frame_datum &datum );
    void set_artif_pose_error();
    void use_GT_pose_vec();

    /////////////////////////////////////// Dynamic_slam_tracking.cpp
    void precompute_SE3_buffers();
    void generate_SE3_deltas();
    void generate_SE3_k2k_vec( float _SE3_k2k[  max_mipmap_layers* num_SE3_DoF *16  ] );
    void estimate_tracking();

    //////////////////////////////////// Dynamic_slam_patch_slam.cpp
	void estimate_depth();
        // Not yet deveoped
    void SpatialCostFns();
    void ParsimonyCostFns();
    void ExhaustiveSearch();
    // To add :
    // depth from previous frame estimates
    // rel vel & accel -> deformation
    // SIRFS, specularity...SPMP

    //////////////////////////////////// Dynamic_slam_autocalibration.cpp
    void estimate_calibration();
    void precompute_cam_matrix_buffers(  uint layer, uint frame_idx );
    void generate_camera_matrix_k2k_vec( cv::Matx44f K, cv::Matx44f pose,  cl_float16 _camera_matrix_k2k[ num_camera_matrix_DoF+1 ] );
    void estimate_camera_matrix(		 uint layer);

	void estimate_lens_distortion();
    void precompute_lens_distortion_buffers( uint frame_idx );

    ///////////////////////////////////// Dynamic_slam_results.cpp
    void report_GT_pose_error();
    void print_pose_datum(      Dynamic_slam::pose_datum datum );
    void print_frame_datum(     Dynamic_slam::frame_datum datum );
    void print_frame_data_vector(       uint start,     uint stop,  vector<Dynamic_slam::frame_datum>       frame_data_vector,      string vector_name );
        // Code profiling
    typedef std::chrono::_V2::system_clock::time_point time_pt;
    void getNextFrameProfile(time_pt step_0, time_pt step_1, time_pt step_2, time_pt step_3, time_pt step_4, time_pt step_5 );

};
