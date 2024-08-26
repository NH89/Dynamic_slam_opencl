#pragma once

#include "../RunCL/RunCL.hpp"

#include <vector>
#include <string>
#include <fstream>
#include <filesystem>
#include <set>

#include "../utils/convertTransforms.hpp"
#include "../utils/convertAhandaPovRayToStandard.hpp"


#define Rx	0
#define Ry  1
#define Rz	2
#define Tx	3
#define Ty	4
#define Tz	5

#define MAX_LAYERS  6

namespace fs = std::filesystem;  //::std::filesystem;


class Dynamic_slam
{
  public:
    ~Dynamic_slam();
    Dynamic_slam(Json::Value obj_);
    Json::Value             obj;
    bool                    invert_GT_depth = false;

    RunCL                   runcl;

    int                     verbosity;
    uint                    SE_iter_per_layer;
    uint                    SE3_stop_layer;
    uint                    SE3_start_layer;
    uint                    SE_iter;
    float                   SE_factor;
    float                   SE3_Rho_sq_threshold[5][3];
    float                   SE3_update_dof_weights[6];
    float                   SE3_update_layer_weights[5];

    // image parameters
    cv::Size                base_image_size;
    int                     base_image_type;

    // data files
    std::string             rootpath;
    fs::path                root;
    std::vector<fs::path>   txt;
    std::vector<fs::path>   png;
    std::vector<fs::path>   depth;

    // camera & pose params
    const cv::Matx44f       Matx44f_zero = {0,0,0,0,  0,0,0,0,  0,0,0,0,  0,0,0,0};   //  = cv::Matx44f::zeros();//
    const cv::Matx44f       Matx44f_eye  = {1,0,0,0,  0,1,0,0,  0,0,1,0,  0,0,0,1};
    #define                 MATX44F_EYE    {1,0,0,0,  0,1,0,0,  0,0,1,0,  0,0,0,1}

    #define D_K             0
    #define D_inv_K         1
    #define D_pose          2
    #define D_inv_pose      3
    #define D_keyframe2pose 4
    #define D_K2K           5

    std::vector<cv::Matx44f > Mat_pose_vec;

    struct pose_datum{      // default intitialization, if instatiated with " ... = {}; "
      cv::Matx16f           keyframe2pose_algebra   = {0} ;

      cv::Matx44f           K                       = MATX44F_EYE ;             // camera intrinsic matrix
      cv::Matx44f           inv_K                   = MATX44F_EYE ;
      cv::Matx44f           pose                    = MATX44F_EYE ;             // pose in global coords. (not pose2pose from prev_frame, nor from keyframe) ?
      cv::Matx44f           inv_pose                = MATX44F_EYE ;
      cv::Matx44f           keyframe2pose           = MATX44F_EYE ;
      cv::Matx44f           K2K                     = MATX44F_EYE ;             //
                                                                                //cv::Matx44f           pose_from_start         = MATX44F_EYE ;             // pose2pose_accumulated
      // lens distortion params
    };

    struct frame_datum{     // default intitialization, if instatiated with " ... = {}; "
      uint                  keyframe_index          = 0 ;                       // Index within this vector< >, of the keyframe for this frame.
      pose_datum            frame_data              = {};
      pose_datum            frame_data_GT           = {};
      pose_datum            error_data              = {};
    };

    struct keyframe_datum{  // default intitialization, if instatiated with " ... = {}; "
      uint                  first_frame_index       = 0 ;
      cv::Mat               reference_image;
      cv::Mat               depthmap;
      frame_datum           frame_data              = {};
    };

    std::vector<frame_datum>     frame_data;           // (frame_data start, old, current, key_frame) are now indices of elements in the vector.
    std::vector<keyframe_datum>  keyframe_data;

    // GT data loading ?
    cv::Mat image, depth_GT, cameraMatrix;        // TODO should these be Matx ?   , projection   NB cameraMatrix => K_GT
    cv::Mat R,     T;
    cv::Mat old_R, old_T;

    float SE3_k2k[6*16];                          // used for param_maps, minimal steps in SE3

    // functions ////////////////////////////////////////
    /////////////////////////////////////// Dynamic_slam_class.cpp
    void initialize_resultsMat();
    void initialize_camera_vec();

    int  nextFrame();
    void use_GT_pose_vec();
    void getFrame();
    void getFrameData_vec();

    // Code profiling
    typedef std::chrono::_V2::system_clock::time_point time_pt;
    void getNextFrameProfile(time_pt step_0, time_pt step_1, time_pt step_2, time_pt step_3, time_pt step_4, time_pt step_5, time_pt step_6, time_pt step_7, time_pt step_8);

    // Not yet deveoped
    void estimateCalibration();
    void SpatialCostFns();
    void ParsimonyCostFns();
    void ExhaustiveSearch();

    // Result
    void getResult();                         // called at end of main(). Currentlyshows mapping params and saves amem & dmem depth maps.
    void print_pose_vectors(uint start, uint stop);

    /////////////////////////////////////// Dynamic_slam_keyframe.cpp
    void initialize_keyframe_vec(  );

    void initialize_keyframe();
    void initialize_keyframe_from_GT();
    void initialize_keyframe_from_tracking();
    void initialize_new_keyframe();

    /////////////////////////////////////// Dynamic_slam_mapping.cpp
    void optimize_depth();
    void updateDepthCostVol();                 // Built forwards. Updates keframe only when needed.
    void buildDepthCostVol_fast_peripheral();  // Higher levels only, built on current frame.
    void updateQD();
    void cacheGValues();
    bool updateA();

    /////////////////////////////////////// Dynamic_slam_tracking.cpp
    void report_GT_pose_error();
    void display_frame_resluts();

    void artificial_pose_error_vec();
    void predictFrame_vec();

    void generate_SE3_k2k_vec( float _SE3_k2k[6*16] );

    void update_k2k(int case_idx, float k2k_4_16[tracking_tot_samples][16]  );
    void update_k2k(Matx16f update_);
    void update_k2k(Matx16f update_,  Matx44f local_keyframe_pose2pose);

    void update_k2k_4(float steps[3], cv::Matx16f update_,  Matx44f K,  Matx44f keyframe2pose,  Matx44f inv_K,  Matx44f keyframe_k2k,  float local_k2k_4_16[4][16] );

    void compute_optimum( float steps[3], float Rho_sq_results_3[tracking_num_samples][8][tracking_num_colour_channels], int layer, int channel, float *prediction, float *optimum, float *stepsize );

    void estimateSE3();                         // version with adaptive step and halting
  
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

    ///////////////////////////////////// Dynamic_slam_pose_vec_print_fns.cpp
    void print_pose_datum(      Dynamic_slam::pose_datum datum );
    void print_frame_datum(     Dynamic_slam::frame_datum datum );
    void print_keyframe_datum(  Dynamic_slam::keyframe_datum datum );

    void print_frame_data_vector(       uint start,     uint stop,  vector<Dynamic_slam::frame_datum>       frame_data_vector,      string vector_name );
    void print_keyframe_data_vector(    uint start,     uint stop,  vector<Dynamic_slam::keyframe_datum>    keyframe_data_vector,   string vector_name );


  private:
    float old_theta, theta, thetaStart, thetaStep, thetaMin, epsilon, lambda, sigma_d, sigma_q;     // DTAM depthmap smoothing & optimization parameters

};
