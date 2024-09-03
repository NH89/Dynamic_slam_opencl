#include "Dynamic_slam.hpp"
#include <fstream>
#include <iomanip>   // for std::setprecision, std::setw

using namespace cv;
using namespace std;

void Dynamic_slam::disparity(){
	int local_verbosity_threshold = V_DYNAMIC_SLAM_OPTIMIZE_DEPTH;//verbosity_mp["Dynamic_slam::optimize_depth"];
																																			if(verbosity>local_verbosity_threshold){ cout << "\f Dynamic_slam::disparity() ######################################"<<flush;}
    uint  	layer 	= SE3_start_layer;
	runcl.disparity(  layer-2, layer );

																																			if(verbosity>local_verbosity_threshold){ cout << "\n\nDynamic_slam::disparity() finished ############################\f"<<flush;}
}
