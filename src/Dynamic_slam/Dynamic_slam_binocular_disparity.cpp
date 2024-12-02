#include "Dynamic_slam.hpp"
#include <fstream>
#include <iomanip>   // for std::setprecision, std::setw

using namespace cv;
using namespace std;

void Dynamic_slam::binocular_reference_frame(){
	int local_verbosity_threshold = V_DYNAMIC_SLAM_BINOCULAR_REFERENCE_FRAME;
																												if(verbosity>local_verbosity_threshold){ cout << "\f Dynamic_slam::reference_frame() ######################################"<<flush;}
	// load images into buffers



	// prepare reference frame
	uint iter = 0;
	for (int layer = runcl.mm_stop ; layer >=0 ; layer-- ){
		runcl.img_sq( 		layer, iter, runcl.curr_img_buf, 		runcl.curr_img_sq_buf,  "curr_img_sq_buf");	//RunCL::img_sq(	uint start, uint stop, uint layer , cl_mem img_buf,    cl_mem img_sq_buf);
		runcl.img_variance( layer, iter, runcl.curr_img_sq_buf, 	runcl.curr_img_var_buf, "curr_img_var_buf");
	}
	// predict warp ?


																												if(verbosity>local_verbosity_threshold){ cout << "\n\nDynamic_slam::reference_frame() finished ############################\f"<<flush;}
}


void Dynamic_slam::binocular_disparity(){
	int local_verbosity_threshold = V_DYNAMIC_SLAM_BINOCULAR_DISPARITY;
																												if(verbosity>local_verbosity_threshold){ cout << "\f Dynamic_slam::disparity() ######################################"<<flush;}
	for (int layer = runcl.mm_stop ; layer >=0 ; layer-- ){
		for (int iter = 0 ; iter < 5   ; iter++ ){
			runcl.warp_image(   layer, iter );
			runcl.img_sq( 		layer, iter, runcl.new_img_buf, 	runcl.new_img_sq_buf,  "new_img_sq_buf");	//RunCL::img_sq(		uint start, uint stop, uint layer , cl_mem img_buf,    cl_mem img_sq_buf);
			runcl.img_variance( layer, iter, runcl.new_img_sq_buf, 	runcl.new_img_var_buf, "new_img_var_buf");	//RunCL::img_variance(	uint start, uint stop, uint layer , cl_mem img_sq_buf, cl_mem img_var_buf);
			runcl.compute_warp( layer, iter );																	//RunCL::compute_warp(	uint start, uint stop, uint layer, uint iter );
		}
	}
																												if(verbosity>local_verbosity_threshold){ cout << "\n\nDynamic_slam::disparity() finished ############################\f"<<flush;}
}


