#include "Dynamic_slam.hpp"
#include <fstream>
#include <iomanip>   // for std::setprecision, std::setw

using namespace cv;
using namespace std;

void Dynamic_slam::optimize_depth(){
	int local_verbosity_threshold = V_DYNAMIC_SLAM_OPTIMIZE_DEPTH;//verbosity_mp["Dynamic_slam::optimize_depth"];
																																			if(verbosity>local_verbosity_threshold){ cout << "\f Dynamic_slam::optimize_depth()"<<flush;}
	bool doneOptimizing;
	int  opt_count				= 0;
	int  max_opt_count			= obj["max_opt_count"].asInt();
	int  max_inner_opt_count 	= obj["max_inner_opt_count"].asInt();

	runcl.QD_count				= 0;
	runcl.A_count				= 0;
	runcl.G_count				= 0;
	runcl.initialize_fp32_params();											// reset params for update QD & A

	cacheGValues();															// initialize "keyframe_g1mem"" map of image edges.
	do{
																			cerr << "\nDynamic_slam::optimize_depth()  max_opt_count="<<max_opt_count<<",   opt_count="<<opt_count<<flush;
		for (int i = 0; i < max_inner_opt_count; i++) updateQD();			// Optimize Q, D   (primal-dual)		/ *5* /
		doneOptimizing = updateA();											// Optimize A      (pointwise exhaustive search)
		opt_count ++;
	} while (!doneOptimizing && (opt_count<max_opt_count));

	int outer_iter = obj["regularizer_outer_iter"].asInt();
	int inner_iter = obj["regularizer_inner_iter"].asInt();
	for (int i=0; i<outer_iter ; i++){
		for (int j=0; j<inner_iter ; j++){
			SpatialCostFns();
			ParsimonyCostFns();
		}
		ExhaustiveSearch();
	}
																																			if(verbosity>local_verbosity_threshold){ cout << "\n\nDynamic_slam::optimize_depth() finished\f"<<flush;}
}

///

void Dynamic_slam::updateDepthCostVol(){																							// Built forwards. Updates keframe only when needed.
	int local_verbosity_threshold = V_DYNAMIC_SLAM_UPDATEDEPTHCOSTVOL;//verbosity_mp["Dynamic_slam::updateDepthCostVol"];// -1;
																																			if(verbosity>local_verbosity_threshold){ cout << "\n Dynamic_slam::buildDepthCostVol()_chk 0,  runcl.dataset_frame_num="<<runcl.dataset_frame_num << flush;}
// # Build depth cost vol on current image, using image array[6] in MipMap buffer, plus RelVelMap,
// with current camera params & DepthMap if bootstrapping, otherwise with params for each frame.
// NB 2*(1+7) = 14 layers on MipMap DepthCostVol: for model & pyramid, ID cetntre layer plus 7 samples, i.e. centre +&- 3 layers.

	// ############ From pseudocode.cpp
	// 5) (i)data exchange between mapping and tracking ?
	//    (ii)NB choice of relative scale - how many frames?
	//    (iii)Boostrapping ? (a) initially assume spherical middle distance.
	//                        (b) initial values of other maps ?

	// 6) Need to rotate costvol forward to new frame & decay the old cost.
	//    Need to build cost vol around depth range estimate from higher image columnm layer - ie 32 layers in cost vol , not 256 layers
	//    therefore cost vol depth varies per pixel.

// Select naive depth map
// See CostVol::updateCost(..), RunCL::calcCostVol(..) &  __kernel void BuildCostVolume2
																																			// int count: Iteration of for loop in this function. Here used to count num imgages use in costvol.
	cv::Matx44f K2K_;
	if( obj["use_GT_pose"].asBool() == false ){
		K2K_ = frame_data.back().frame_data.K2K;
																																			if(verbosity>local_verbosity_threshold){
																																				cout << "\n Dynamic_slam::buildDepthCostVol()_chk 1, "
																																				<< "\nobj['use_GT_pose'].asBool()="<<obj["use_GT_pose"].asBool() << "   false,"
																																				<< "  K2K_ = frame_data.back().frame_data.K2K; " << flush;
																																				PRINT_MATX44F(K2K_,);
																																				PRINT_MATX44F(frame_data.back().frame_data.keyframe2pose, );
																																				PRINT_MATX16F(frame_data.back().frame_data.keyframe2pose_algebra , );
																																				PRINT_MATX44F(frame_data.back().frame_data.K2K, );
																																				PRINT_MATX44F(frame_data.back().frame_data.K,);
																																				PRINT_MATX44F(frame_data.back().frame_data.inv_K,);
																																				PRINT_MATX44F(frame_data.back().frame_data.K * frame_data.back().frame_data.inv_K,);
																																			}
	}else{
		K2K_ = frame_data.back().frame_data_GT.K2K;
																																			if(verbosity>local_verbosity_threshold){
																																				cout << "\n Dynamic_slam::buildDepthCostVol()_chk 1, "
																																				<< "\nobj['use_GT_pose'].asBool()="<<obj["use_GT_pose"].asBool() << "   true,"
																																				<< "  K2K_ = frame_data.back().frame_data_GT.K2K; " << flush;
																																				PRINT_MATX44F(K2K_,);
																																				PRINT_MATX44F(frame_data.back().frame_data_GT.keyframe2pose, );
																																				PRINT_MATX16F(frame_data.back().frame_data_GT.keyframe2pose_algebra , );
																																				PRINT_MATX44F(frame_data.back().frame_data_GT.K2K, );
																																				PRINT_MATX44F(frame_data.back().frame_data_GT.K,);
																																				PRINT_MATX44F(frame_data.back().frame_data_GT.inv_K,);
																																				PRINT_MATX44F(frame_data.back().frame_data_GT.K * frame_data.back().frame_data_GT.inv_K,);
																																			}
	}
	// keyframe_K2K_GT; 					//TODO K2K; 		// needs keyframe_K2K from keyframe. 						// camera-to-camera transform for this image to the keyframe of this cost vol.
	//bool image_ = runcl.frame_bool_idx; 																									// Index to correct img pyramid buffer on device.

	runcl.updateDepthCostVol( K2K_, runcl.costvol_frame_num ); 																				// NB in DTAM_opencl : void RunCL::calcCostVol(float* k2k,  cv::Mat &image)
																																			// in  void Dynamic_slam::estimateSE3() above : runcl.estimateSE3(SE3_reults, Rho_sq_results, iter, 0, 8);
																																			// -> mipmap_call_kernel( se3_grad_kernel, m_queue, start, stop );
}

void Dynamic_slam::buildDepthCostVol_fast_peripheral(){																						// Higher levels only, built on current frame.
	int local_verbosity_threshold = V_DYNAMIC_SLAM_BUILDDEPTHCOSTVOL_FAST_PERIPHERAL;//verbosity_mp["Dynamic_slam::buildDepthCostVol_fast_peripheral"];// 1;
																																			if(verbosity>local_verbosity_threshold){ cout<<"\nDynamic_slam::buildDepthCostVol_fast_peripheral_chk0, " << flush;}



}

/*void Dynamic_slam::computeSigmas(float epsilon, float theta, float L, float &sigma_d, float &sigma_q ){
		float mu	= 2.0*std::sqrt((1.0/theta)*epsilon) /L;
		sigma_d		= mu / (2.0/ theta);
		sigma_q		= mu / (2.0*epsilon);
}
*/

void Dynamic_slam::updateQD(){
	int local_verbosity_threshold = V_DYNAMIC_SLAM_UPDATEQD;//verbosity_mp["Dynamic_slam::updateQD"];// 1;
																																			if(verbosity>local_verbosity_threshold) cout<<"\nDynamic_slam::updateQD_chk0, epsilon="<<runcl.fp32_params[EPSILON]<<" theta="<<runcl.fp32_params[THETA]<<flush;
	runcl.computeSigmas(runcl.fp32_params[EPSILON], runcl.fp32_params[THETA], obj["L"].asFloat(), runcl.fp32_params[SIGMA_D], runcl.fp32_params[SIGMA_Q] );
																																			if(verbosity>local_verbosity_threshold) cout<<"\nDynamic_slam::updateQD_chk1, epsilon="<<runcl.fp32_params[EPSILON]<<" theta="<<runcl.fp32_params[THETA]\
																																								<<" sigma_q="<<runcl.fp32_params[SIGMA_Q]<<" sigma_d="<<runcl.fp32_params[SIGMA_D]<<flush;
	runcl.updateQD(runcl.fp32_params[EPSILON], runcl.fp32_params[THETA], runcl.fp32_params[SIGMA_Q], runcl.fp32_params[SIGMA_D]);
																																			if(verbosity>local_verbosity_threshold) cout<<"\nDynamic_slam::updateQD_chk3, epsilon="<<runcl.fp32_params[EPSILON]<<" theta="<<runcl.fp32_params[THETA]\
																																								<<" sigma_q="<<runcl.fp32_params[SIGMA_Q]<<" sigma_d="<<runcl.fp32_params[SIGMA_D]<<flush;
}

void Dynamic_slam::cacheGValues()
{
	int local_verbosity_threshold = V_DYNAMIC_SLAM_CACHEGVALUES;//verbosity_mp["Dynamic_slam::cacheGValues"];// 1;
																																			if(verbosity>local_verbosity_threshold) {cout<<"\nDynamic_slam::cacheGValues()" <<flush;}
	runcl.updateG(runcl.G_count);
}

bool Dynamic_slam::updateA(){
	int local_verbosity_threshold = V_DYNAMIC_SLAM_UPDATEA;//verbosity_mp["Dynamic_slam::updateA"];// 1;
																																			if(verbosity>local_verbosity_threshold) cout<<"\nDynamic_slam::updateA "<<flush;
	if (/*theta*/runcl.fp32_params[THETA] < 0.001 && /*old_theta*/runcl.fp32_params[OLD_THETA] > 0.001){  cacheGValues(); /*old_theta*/runcl.fp32_params[OLD_THETA] = /*theta*/runcl.fp32_params[THETA]; }		// If theta falls below 0.001, then G must be recomputed.
	// bool doneOptimizing = (theta <= thetaMin);


	runcl.updateA( runcl.fp32_params[LAMBDA], runcl.fp32_params[THETA]/*,  runcl.mm_start, runcl.mm_stop*/ );

	runcl.measureDepthFit();


	runcl.fp32_params[THETA] *= obj["thetaStep"].asFloat();

	//return doneOptimizing;
																																			if(verbosity>local_verbosity_threshold) cout<<"\nDynamic_slam::updateA finished"<<flush;
	return false;
}
