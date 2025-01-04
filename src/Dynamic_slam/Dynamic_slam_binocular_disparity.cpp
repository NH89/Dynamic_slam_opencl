#include "Dynamic_slam.hpp"
#include <fstream>
#include <iomanip>   // for std::setprecision, std::setw

using namespace cv;
using namespace std;

void Dynamic_slam::binocular_reference_frame(){
	string fname = "RunCL::binocular_reference_frame( )";
	int local_verbosity_threshold = V_DYNAMIC_SLAM_BINOCULAR_REFERENCE_FRAME;
																												if(verbosity>local_verbosity_threshold){ cout << "\f Dynamic_slam::reference_frame() ######################################"<<flush;}
	// load image into buffers
	runcl._clEnqueueCopyBuffer( runcl.m_queue, runcl.keyframe_imgmem, 	runcl.curr_img_buf, 			0, 0, runcl.mm_size_bytes_C4, 	fname);

	// prepare reference frame
	uint iter = 0;
	for (int layer = runcl.mm_stop ; layer >=0 ; layer-- ){
		runcl.img_sq( 		layer, iter, runcl.curr_img_buf, 		runcl.curr_img_sq_buf,  "curr_img_sq_buf");	//RunCL::img_sq(	uint start, uint stop, uint layer , cl_mem img_buf,    cl_mem img_sq_buf);
		runcl.img_variance( layer, iter, runcl.curr_img_sq_buf, 	runcl.curr_img_var_buf, "curr_img_var_buf");
	}
	// predict warp ? from previous keyframe


																												if(verbosity>local_verbosity_threshold){ cout << "\n\nDynamic_slam::reference_frame() finished ############################\f"<<flush;}
}

void Dynamic_slam::binocular_reference_frame2(){
	string fname = "RunCL::binocular_reference_frame( )";
	int local_verbosity_threshold = V_DYNAMIC_SLAM_BINOCULAR_REFERENCE_FRAME;
																												if(verbosity>local_verbosity_threshold){ cout << "\f Dynamic_slam::reference_frame() ######################################"<<flush;}
	// load image into buffers
	runcl._clEnqueueCopyBuffer( runcl.m_queue, runcl.keyframe_imgmem, 	runcl.curr_img_buf, 			0, 0, runcl.mm_size_bytes_C4, 	fname);

	// prepare reference frame
	uint iter = 0;
	for (int layer = runcl.mm_stop ; layer >=0 ; layer-- ){
		runcl.mean_sq_3rows(   uint layer, uint iter, cl_mem img_sq_buf, cl_mem img_var_buf, std::string folder);
		runcl.mean_sq_cols(    uint layer, uint iter, cl_mem img_sq_buf, cl_mem img_var_buf, std::string folder);
	}
	// predict warp ? from previous keyframe


																												if(verbosity>local_verbosity_threshold){ cout << "\n\nDynamic_slam::reference_frame() finished ############################\f"<<flush;}
}


void Dynamic_slam::binocular_disparity(){
	string fname = "RunCL::binocular_disparity( )";
	int local_verbosity_threshold = V_DYNAMIC_SLAM_BINOCULAR_DISPARITY;
																												if(verbosity>local_verbosity_threshold){ cout << "\n Dynamic_slam::disparity() ######################################"<<flush;}
	// load image into buffers
	runcl._clEnqueueCopyBuffer( runcl.m_queue, runcl.imgmem, 	runcl.new_img_buf, 			0, 0, runcl.mm_size_bytes_C4, 	fname);

	// predict warp from previous frame ?


	// compute warp for this frame  // can be until (layer > runcl.mm_start) but takes 3x longer.
	for (int layer = runcl.mm_stop ; layer > runcl.mm_start+1 ; layer-- ){										if(verbosity>local_verbosity_threshold){ cout << "\n Dynamic_slam::disparity() chk_1 layer="<<layer<<", runcl.mm_start="<<runcl.mm_start<<flush;}
		for (int iter = 0 ; iter < 5   ; iter++ ){																if(verbosity>local_verbosity_threshold){ cout << "\n Dynamic_slam::disparity() chk_2 layer="<<layer<<", iter="<<iter<<"_"<<flush;}
			runcl.warp_image(   layer, iter );
			runcl.img_sq( 		layer, iter, runcl.new_img_warped_buf, 	runcl.new_img_sq_buf,  "new_img_sq_buf");	//RunCL::img_sq(		uint start, uint stop, uint layer , cl_mem img_buf,    cl_mem img_sq_buf);
			runcl.img_variance( layer, iter, runcl.new_img_sq_buf, 		runcl.new_img_var_buf, "new_img_var_buf");	//RunCL::img_variance(	uint start, uint stop, uint layer , cl_mem img_sq_buf, cl_mem img_var_buf);
			runcl.compute_warp( layer, iter );																		//RunCL::compute_warp(	uint start, uint stop, uint layer, uint iter );
		}
		if (layer>runcl.mm_start){ runcl.propagate_warp( layer ); }
	}
																												if(verbosity>local_verbosity_threshold){ cout << "\n\nDynamic_slam::disparity() finished ############################\f"<<flush;}
}

void Dynamic_slam::binocular_disparity2(){
	string fname = "RunCL::binocular_disparity( )";
	int local_verbosity_threshold = V_DYNAMIC_SLAM_BINOCULAR_DISPARITY;
																												if(verbosity>local_verbosity_threshold){ cout << "\n Dynamic_slam::disparity() ######################################"<<flush;}
	// load image into buffers
	runcl._clEnqueueCopyBuffer( runcl.m_queue, runcl.imgmem, 	runcl.new_img_buf, 			0, 0, runcl.mm_size_bytes_C4, 	fname);

	// predict warp from previous frame ?


	// compute warp for this frame  // can be until (layer > runcl.mm_start) but takes 3x longer.
	for (int layer = runcl.mm_stop ; layer > runcl.mm_start+1 ; layer-- ){										if(verbosity>local_verbosity_threshold){ cout << "\n Dynamic_slam::disparity() chk_1 layer="<<layer<<", runcl.mm_start="<<runcl.mm_start<<flush;}
		for (int iter = 0 ; iter < 5   ; iter++ ){																if(verbosity>local_verbosity_threshold){ cout << "\n Dynamic_slam::disparity() chk_2 layer="<<layer<<", iter="<<iter<<"_"<<flush;}
			runcl.warp_image(   layer, iter );
			runcl.mean_sq_3rows(   uint layer, uint iter, cl_mem img_sq_buf, cl_mem img_var_buf, std::string folder);
			runcl.mean_sq_cols(    uint layer, uint iter, cl_mem img_sq_buf, cl_mem img_var_buf, std::string folder);
			runcl.co_mean_rows(    uint layer, uint iter, cl_mem img_sq_buf, cl_mem img_var_buf, std::string folder);
			runcl.covariance_cols( uint layer, uint iter, cl_mem img_sq_buf, cl_mem img_var_buf, std::string folder);

		}
		if (layer>runcl.mm_start){ runcl.propagate_warp( layer ); }
	}
																												if(verbosity>local_verbosity_threshold){ cout << "\n\nDynamic_slam::disparity() finished ############################\f"<<flush;}
}
