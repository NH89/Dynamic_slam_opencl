#include "Dynamic_slam.hpp"

#include <iostream>
#include <fstream>
#include <iomanip>   // for std::setprecision, std::setw

using namespace cv;
using namespace std;
/*
void Dynamic_slam::initialize_keyframe(){
	if( obj["initialize_keyframe_from_GT"].asBool() ){  	// j_params.bool_mp["initialize_keyframe_from_GT"];
		initialize_keyframe_from_GT();
	}else{
		initialize_keyframe_from_tracking();
	}
}

void Dynamic_slam::initialize_keyframe_from_GT(){																							// GT depth map is for current GT pose.
	int local_verbosity_threshold = ;//verbosity_mp["Dynamic_slam::initialize_keyframe_from_GT"];// -1;
																																			if(verbosity>local_verbosity_threshold){ cout << "\n\nDynamic_slam::initialize_keyframe_from_GT()_chk 0" << flush;}
	keyframe_pose_GT 		= pose_GT;
	keyframe_inv_pose_GT 	= getInvPose(keyframe_pose_GT);
	keyframe_inv_K_GT		= generate_invK_(K_GT);

	keyframe_old_pose		= old_pose_GT;
	keyframe_old_K			= old_K_GT;

	//keyframe_pose 			= pose_GT;
	keyframe_K				= K_GT;
	keyframe_inv_pose 		= inv_pose_GT;
	keyframe_inv_K			= inv_K_GT;

	keyframe_K2K 			= K2K_GT;						// TODO chk wrt when this is called and what values it would hold.
															// d_slam.predictFrame() sets keyframe_K2K  = K * pose * keyframe_inv_pose * inv_old_K.
															// d_slam.getFrameData() sets K2L_GT = old_K_GT * old_pose_GT * inv_pose_GT * inv_K_GT

	keyframe_pose2pose 		= pose2pose_GT;					// d_slam.predictFrame() sets keyframe_pose2pose = pose2pose
															// d_slam.getFrameData() sets pose2pose_GT = old_pose_GT * inv_pose_GT

	runcl.initializeDepthCostVol( runcl.depth_mem_GT );

	initialize_new_keyframe();
}

void Dynamic_slam::initialize_keyframe_from_tracking(){																						// NB need to transform depth map from previous keyfrae to current pose.
	int local_verbosity_threshold = ;//verbosity_mp["Dynamic_slam::initialize_keyframe_from_tracking"];// -1;
	keyframe_pose_GT 		= pose_GT;
	keyframe_inv_pose_GT 	= getInvPose(keyframe_pose_GT);
	keyframe_inv_K_GT		= generate_invK_(K_GT);

	keyframe_old_pose		= keyframe_pose;
	keyframe_old_K			= keyframe_K;
// / *
	//keyframe_pose 			= pose;
	//keyframe_K				= K;
	//keyframe_inv_pose 		= inv_pose;
	//keyframe_inv_K			= inv_K;

	// keyframe_K2K   		// d_slam.predictFrame() sets 	keyframe_K2K  		= K * pose * keyframe_inv_pose * inv_old_K.					// Projects keyframe pixel to current frame
	// keyframe_pose2pose	// d_slam.predictFrame() sets 	keyframe_pose2pose 	= pose2pose
// * /
	cv::Matx44f inv_pose2pose = getInvPose( keyframe_pose2pose );																			// cv::Matx44f Dynamic_slam::getInvPose(cv::Matx44f pose)
	cv::Matx44f forward_keyframe2K  = K * inv_pose2pose * inv_K;	// TODO auto calibration												// Projects new keyframe pixel to previous keyframe
																																			if(verbosity>local_verbosity_threshold){
																																				cout<<"\n\nDynamic_slam::initialize_keyframe_from_tracking,  dataset_frame_num="<<runcl.dataset_frame_num<<flush;
																																				PRINT_MATX16F( PToLie(keyframe_pose2pose), );
																																				PRINT_MATX16F( PToLie(inv_pose2pose), );
																																				PRINT_MATX44F(K,);
																																				PRINT_MATX44F(keyframe_pose2pose,);
																																				PRINT_MATX44F(inv_pose2pose,);
																																				PRINT_MATX44F(inv_K,);
																																				PRINT_MATX44F(forward_keyframe2K,);
																																			}
	if( obj["initialize_tracking_from_GT_depth"].asBool() ){  																				// j_params.bool_mp["initialize_keyframe_from_GT"];
																																			if(verbosity>local_verbosity_threshold){
																																				cout<<"\nobj[\"Dynamic_slam::initialize_keyframe_from_tracking(),  initialize_tracking_from_GT_depth\"].asBool() = true "<<flush;
																																			}
		runcl.transform_depthmap( forward_keyframe2K, runcl.keyframe_depth_mem );															// NB runcl.transform_depthmap(..) must be used _before_ initializing the new cost_volume, because it uses keyframe_basemem.
		runcl.initializeDepthCostVol( runcl.keyframe_depth_mem );  																			// TODO  Need to boostrap from blank depthmap. ######################
		initialize_new_keyframe();
	}else{
																																			if(verbosity>local_verbosity_threshold){
																																				cout<<"\nobj[\"Dynamic_slam::initialize_keyframe_from_tracking(),  initialize_tracking_from_GT_depth\"].asBool() = false "<<flush;
																																			}
		//runcl.transform_depthmap(forward_keyframe2K, runcl.amem );		// runcl.depth_mem );																	// Sets new depth_mem used in tracking.
		runcl.swap_costvol_pointers();																										// Swaps old cdatabuf and hdatabuf to temp_cdatabuf and temp_hdatabuf.
		runcl.initializeDepthCostVol( runcl.amem );		// runcl.depth_mem );																// Zeros buffers: cdatabuf, hdatabuf, lomem, himem) Also copies  runcl.depth_mem -> runcl.keyframe_depth_mem
		initialize_new_keyframe();																											// runcl.initialize_fp32_params();  runcl.keyFrameCount++; runcl.dataset_frame_num++;
		//runcl.transform_costvolume( forward_keyframe2K );
	}
	keyframe_pose2pose =  pose2pose; 																										// reset to predct next frame relative to the new keyframe.
}

void Dynamic_slam::initialize_new_keyframe(){
	int local_verbosity_threshold = ;//verbosity_mp["Dynamic_slam::initialize_new_keyframe"];// -1;
																																			if(verbosity>local_verbosity_threshold){ cout << "\n\nDynamic_slam::initialize_new_keyframe()_chk 0,  runcl.dataset_frame_num = "<< runcl.dataset_frame_num << flush;}
	runcl.initialize_fp32_params();
	//runcl.QD_count 	= 0; 	// TODO NB these are reset in Dynamic_slam::nextFrame()
	//runcl.A_count 	= 0;
	//runcl.G_count 	= 0;

	//cacheGValues();			// TODO may not be needed here.
								// TODO   keyframe_K2K_GT, keyframe_K2K etc ?
	runcl.keyFrameCount++;
	//runcl.dataset_frame_num++;
}
*/
///////////

void Dynamic_slam::initialize_keyframe_vec(){
	int local_verbosity_threshold = V_DYNAMIC_SLAM_INITIALIZE_KEYFRAME;//verbosity_mp["Dynamic_slam::initialize_new_keyframe"];// -1;
																																			if(verbosity>local_verbosity_threshold){ cout << "\n\nDynamic_slam::initialize_keyframe_vec()_chk 0,  runcl.dataset_frame_num = "<< runcl.dataset_frame_num << flush;}
	keyframe_datum 							new_keyframe;
	new_keyframe.frame_data  				=	frame_data.back();																			// Copy current tracking frame to the new keyframe.
	keyframe_data.push_back( 				new_keyframe );
	keyframe_data.back().first_frame_index	=	frame_data.size();
																																			if(verbosity>local_verbosity_threshold){ cout << "\n\nDynamic_slam::initialize_keyframe_vec()_chk 1, "
																																				<< "keyframe_data.back().first_frame_index = "<< keyframe_data.back().first_frame_index << flush;
																																			}
	if ( keyframe_data.size() > 1 ){

		cv::Matx44f inv_pose2pose 			=  getInvPose( frame_data.back().frame_data.keyframe2pose ); //    keyframe_pose2pose );		// cv::Matx44f Dynamic_slam::getInvPose(cv::Matx44f pose)
		cv::Matx44f forward_keyframe2K  	=  frame_data.back().frame_data.K * inv_pose2pose * frame_data.back().frame_data.inv_K;			// Projects new keyframe pixel to previous keyframe
		runcl.transform_depthmap( 			forward_keyframe2K, runcl.amem );																// Sets new depth_mem used in tracking.
		runcl.swap_costvol_pointers();																										// Swaps   cdatabuf<->temp_cdatabuf ,  hdatabuf<->temp_hdatabuf.
		runcl.initializeDepthCostVol( 		runcl.amem );																					// Also copies  	imgmem 						-> keyframe_imgmem
																																			//					HSV_grad_mem 				-> keyframe_imgmem_HSV_grad

																																			//	runcl.amem 	=	key_frame_depth_map_src 	-> keyframe_depth_mem
																																			//					depth_mem_GT 				-> keyframe_depth_mem_GT
																																			//					SE3_grad_map_mem 			-> keyframe_SE3_grad_map_mem,

																																			// Zeros buffers: 	dbg_databuf, cdatabuf, hdatabuf, img_sum_buf,
																																			// 					dmem, amem, qmem, qmem2, lomem, himem.
		runcl.transform_costvolume( 		forward_keyframe2K );

	}else{																																	// IF starting a new vector<keframe_datum>, i.e. begining of program.
		runcl.initializeDepthCostVol( 		runcl.amem );																					// Zeros  	amem, cdatabuf, hdatabuf etc..
		runcl.swap_costvol_pointers();																										// Swaps	cdatabuf<->temp_cdatabuf ,  hdatabuf<->temp_hdatabuf.
		runcl.initializeDepthCostVol( 		runcl.amem );																					// Copies	amem	=	key_frame_depth_map_src -> keyframe_depth_mem
		float initial_depth  = (runcl.fp32_params[MAX_INV_DEPTH] + runcl.fp32_params[MIN_INV_DEPTH])/2.0;									// Fills depth_mem buffer with mid depth.
		runcl.initialize_tracking_depthmap(initial_depth);
	}
																																			if(verbosity>local_verbosity_threshold){ cout << "\n\nDynamic_slam::initialize_keyframe_vec()_chk 2, " << flush; }
	if( obj["initialize_tracking_from_GT_depth"].asBool() == true ){
		runcl.update_tracking_depthmap( runcl.depth_mem_GT );																				// copies buffer runcl.depth_mem_GT to keyframe_depth_mem
	}
	if(  obj["initialize_keyframe_from_GT"].asBool() == true ){
		keyframe_data.back().frame_data.frame_data = keyframe_data.back().frame_data.frame_data_GT;											// copies frame_data_GT to frame_data for the new keyframe
	}
																																			// Save keyframe "amem" and "reference_image" /////////////////////////////////////
	keyframe_data.back().depthmap			= cv::Mat::zeros(		runcl.uint_params[MM_ROWS], 	runcl.uint_params[MM_ROWS],  CV_32FC4	);	// Instantiate and zero CV::Mat
	keyframe_data.back().reference_image	= cv::Mat::zeros(		runcl.uint_params[MM_ROWS], 	runcl.uint_params[MM_ROWS],  CV_32FC4	);

	runcl.ReadOutput(  keyframe_data.back().depthmap.data, 			runcl.amem,  	runcl.image_size_bytes );								// Saves "amem" and "reference_image" buffers to CV::Mat in current elem of vector<> Keyframe_data.
	runcl.ReadOutput(  keyframe_data.back().reference_image.data, 	runcl.imgmem,	runcl.image_size_bytes );								// These can be used for loop closure later on.

	runcl.initialize_fp32_params();												// reset parameters											// runcl.initialize_fp32_params();  runcl.keyFrameCount++; runcl.dataset_frame_num++;
	runcl.keyFrameCount++;
																																			if(verbosity>local_verbosity_threshold){ cout << "\n\nDynamic_slam::initialize_keyframe_vec() Finished ###########################" << flush;}
}
