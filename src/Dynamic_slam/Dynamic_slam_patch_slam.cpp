#include "Dynamic_slam.hpp"
#include <fstream>
#include <iomanip>   // for std::setprecision, std::setw

using namespace cv;
using namespace std;

void Dynamic_slam::estimate_depth(){
	string fname = "Dynamic_slam::estimate_depth()";
	int 	local_verbosity_threshold 		= V_DYNAMIC_SLAM_ESTIMATE_DEPTH;//verbosity_mp["Dynamic_slam::estimateSE3"];
																																		if(verbosity>local_verbosity_threshold) {
																																			cout << "\n\nDynamic_slam::estimate_depth() chk_0"
																																			<<"  ##############################################################"<< flush;
																																		}
	//float default_inv_depth				= 0.007f;																						// half the max inv depth, i.e. twice the min depth.
	//runcl._clEnqueueFillBuffer( runcl.uload_queue, runcl.depth_mem,	&default_inv_depth, sizeof(float), 0, runcl.mm_size_bytes_C1, fname ); // TODO  remove this, temporary for testing tracking and mapping given GT poses.
	runcl._clEnqueueFillBuffer( runcl.uload_queue, runcl.depth_mem_temp,	&zero_flt, sizeof(float), 0, runcl.mm_size_bytes_C1, fname );


	for (int layer=3; layer>=0; layer--){	// NB must start at least 2 layers below apex of image pyramid. Uses img grad fom 2 layers higher.
																																		if(verbosity>local_verbosity_threshold) {
																																			cout << "\nDynamic_slam::estimate_depth()  layer= "<<
																																			layer << endl <<flush;
																																			/*
																																			size_t depthUpdate_bytes	=	runcl.mm_size_bytes_C1;
																																			cv::Size depthUpdate_size	=	cv::Size( runcl.mm_Image_size.width, runcl.mm_Image_size.height/2.0f );
																																			size_t offset_depth_bytes	=	0;

																																			stringstream 	ss;
																																			ss << "ds-framenum"<<runcl.dataset_frame_num<<"_img_layer"<<layer<<"_out_bock_size"<<out_block_size<<"_"<<fname;
																																			ss	<<"_test__";

																																			bool 			show				= false;
																																			float 			max_range			= -1;
																																			bool 			old_tiff			= runcl.tiff;
																																							runcl.tiff				= true;

																																			runcl.DownloadAndSave_2Channel( runcl.depth_mem_temp,  ss.str( ), runcl.paths.at( "depth_mem_temp"),	depthUpdate_bytes,   depthUpdate_size,	CV_32FC2, show, max_range,	offset_depth_bytes );	cout<<"\nDownloadAndSaveDepthUpdate chk_4  "<<flush;

																																			runcl.tiff = old_tiff;
																																			*/
																																			//  // For debugging, get a larger, finer Rho map
																																			// uint	out_block_size		= 2;
																																			// uint	layer_				= 0;
																																			// for( uint frame_index=0; frame_index<num_current_frames; frame_index++){
																																			// 	runcl.rho_sq( out_block_size, 10+frame_index, frame_index, layer_, runcl.cur_frames_k2kbuf );
																																			// }
																																		}
		// uint out_block_size = 4;					// NB constexpr uint out_block_size	= OUT_BLOCK_SIZE	 4
		//runcl.update_depth( out_block_size, layer);			// LK
		runcl.update_depth_2( out_block_size, layer);		// cost_vol & Glasgow type optimization, on depth from ST3 given transpose.
		// anisotropic smoothing
		runcl.regularize_depth( layer );
		// parsimony of orientation, plane, curvature ?
		// if( layer>0){
		// 	runcl.propagate_depth_next_layer(layer-1);
		// }


		// NB this kernel would be faster if it used 1 thread per depth patch, ie 16 pixels. ... Maybe not. The existing method uses half as many threads, BUT benefits from contiguious reads of data.
	}
	// copy depth to tracking depth map ### TODO
	uint depth_layer = 0;																													// i.e. layer of depth map used for tracking. Currently has to be 0.
	if(	initialize_tracking_from_GT_depth == false){	runcl.use_inferred_depthmap( depth_layer );	}
}




// ## Regularize Maps : AbsDepth, GradDepth, SurfNormal, RelVel,
void Dynamic_slam::SpatialCostFns(){
	//int local_verbosity_threshold = V_DYNAMIC_SLAM_SPATIALCOSTFNS;//verbosity_mp["Dynamic_slam::SpatialCostFns"];
// # Spatial cost functions
// see CostVol::updateQD(..), RunCL::updateQD(..) & __kernel void UpdateQD(..)

}

void Dynamic_slam::ParsimonyCostFns(){
	//int local_verbosity_threshold = V_DYNAMIC_SLAM_PARSIMONYCOSTFNS;//verbosity_mp["Dynamic_slam::ParsimonyCostFns"];
// # Parsimony cost functions : NB Bin sort pixels to find non-spatial neighbours
// see SIFS for priors & Morphogenesis for BinSort

}

void Dynamic_slam::ExhaustiveSearch(){
	//int local_verbosity_threshold = V_DYNAMIC_SLAM_EXHAUSTIVESEARCH;//verbosity_mp["Dynamic_slam::ExhaustiveSearch"];
// # Update A : exhaustive search on cost vol with cost fns -> update maps.
// see CostVol::updateA(..), RunCL::updateA(..) & __kernel void UpdateA2(..)

}
