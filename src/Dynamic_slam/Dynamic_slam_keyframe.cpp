#include "Dynamic_slam.hpp"

#include <iostream>
#include <fstream>
#include <iomanip>   // for std::setprecision, std::setw

using namespace cv;
using namespace std;

void Dynamic_slam::initialize_keyframe_vec(  ){
	string fname = "Dynamic_slam::initialize_keyframe_vec()";
	int local_verbosity_threshold = V_DYNAMIC_SLAM_INITIALIZE_KEYFRAME;
																																			if(verbosity>local_verbosity_threshold){ cout << "\n\nDynamic_slam::initialize_keyframe_vec()_chk 0,  runcl.dataset_frame_num = "
																																				<< runcl.dataset_frame_num << flush;
																																				PRINT_MATX44F( frame_data.back().frame_data.inv_pose , );
																																			}
	keyframe_datum 												new_keyframe;
	Matx44f old_keyframe2pose 									= frame_data.back().frame_data_GT.keyframe2pose;

	frame_data.back().frame_data_GT.keyframe2pose 				= Matx44f_eye ;																// Set current frame to be its own keyframe. TODO These variables should be redundant in keyframe. Rather copy only the valid variables. ? Reduce the keyframe struct ?
	frame_data.back().frame_data.keyframe2pose 					= Matx44f_eye ;
	frame_data.back().frame_data_GT.keyframe2pose_algebra		= PToLie( frame_data.back().frame_data_GT.keyframe2pose );
	frame_data.back().frame_data.keyframe2pose_algebra			= PToLie( frame_data.back().frame_data.keyframe2pose );

	float default_depth 										= ( runcl.fp32_params[MAX_INV_DEPTH] + runcl.fp32_params[MIN_INV_DEPTH] ) /2.0;
/*
	// keyframe_datum 												new_keyframe;
	// //Matx44f old_keyframe2pose 								= frame_data.back().frame_data_GT.keyframe2pose;
	// frame_datum  old_frame_datum								= frame_data.back();
 //
	// frame_data.back().frame_data_GT								= pose_datum{};																// Set current frame to be its own keyframe, by zeroing it and transfering the pose.
	// frame_data.back().frame_data								= pose_datum{};
 //
	// frame_data.back().frame_data_GT.pose						= old_frame_datum.frame_data_GT.pose;
	// frame_data.back().frame_data_GT.inv_pose					= old_frame_datum.frame_data_GT.inv_pose;
 //
	// frame_data.back().frame_data.pose							= old_frame_datum.frame_data.pose;
	// frame_data.back().frame_data.inv_pose						= old_frame_datum.frame_data.inv_pose;
																																			// TODO These variables should be redundant in keyframe. Rather copy only the valid variables. ? Reduce the keyframe struct ?
*/
	new_keyframe.frame_data  									=	frame_data.back();														// Copy current tracking frame to the new keyframe.
	keyframe_data.push_back( 									new_keyframe );
	keyframe_data.back().first_frame_index						=	frame_data.size();
																																			if(verbosity>local_verbosity_threshold){ cout << "\n\nDynamic_slam::initialize_keyframe_vec()_chk 1, "
																																				<< "keyframe_data.back().first_frame_index = "<< keyframe_data.back().first_frame_index
																																				<< ",\t keyframe_data.size()="<<keyframe_data.size()<< flush;
																																				PRINT_MATX44F( keyframe_data.back().frame_data.frame_data.inv_pose , );
																																			}
	if ( keyframe_data.size() > 1 ){
																																			if(verbosity>local_verbosity_threshold){ cout << "\n\nDynamic_slam::initialize_keyframe_vec()_chk 1.1,  depth=runcl.amem"<< flush;
																																				// print_pose_vectors(0,10);
																																				PRINT_MATX44F( old_keyframe2pose ,  );
																																				//PRINT_MATX44F( old_frame_datum.frame_data_GT.keyframe2pose ,  );
																																			}
/*
		//cv::Matx44f inv_pose2pose 			=  getInvPose( old_frame_datum.frame_data.keyframe2pose, verbosity );
		// 																																	PRINT_MATX44F( inv_pose2pose, );
		// 																																	PRINT_MATX44F( frame_data.back().frame_data.K, );
		// 																																	PRINT_MATX44F( frame_data.back().frame_data.inv_K, );
*/
		// cv::Matx44f forward_keyframe2K  	=  frame_data.back().frame_data.K * inv_pose2pose * frame_data.back().frame_data.inv_K;			// Projects new keyframe pixel to previous keyframe
		 cv::Matx44f forward_keyframe2K  	= frame_data.back().frame_data.K2K.inv() ;  // old_frame_datum.frame_data.K2K; //.inv() ;
		float forward_keyframe2K_f16[16];
		Matx44f_To_float16arry( 			forward_keyframe2K, forward_keyframe2K_f16 );
																																			PRINT_FLOAT_16( forward_keyframe2K_f16, forward_keyframe2K );
		//runcl.transform_depthmap( 			forward_keyframe2K_f16, runcl.amem );				 											// Sets new depth_mem used in tracking.
		runcl.swap_costvol_pointers();																										// Swaps   cdatabuf<->temp_cdatabuf ,  hdatabuf<->temp_hdatabuf.
		// runcl.initializeDepthCostVol( 		runcl.amem );																					// Also copies  	imgmem 						-> keyframe_imgmem
																																			//					HSV_grad_mem 				-> keyframe_imgmem_HSV_grad
		runcl.initializeFirstDepthCostVol(	default_depth );
/*
																																			//	runcl.amem 	=	key_frame_depth_map_src 	-> keyframe_depth_mem
																																			//					depth_mem_GT 				-> keyframe_depth_mem_GT
																																			//					SE3_grad_map_mem 			-> keyframe_SE3_grad_map_mem,

																																			// Zeros buffers: 	dbg_databuf, cdatabuf, hdatabuf, img_sum_buf,
																																			// 					dmem, amem, qmem, qmem2, lomem, himem.
		//runcl.transform_costvolume( 		forward_keyframe2K_f16 );
*/
	}else{																																	// IF starting a new vector<keframe_datum>, i.e. begining of program.
																																			if(verbosity>local_verbosity_threshold){ cout << "\n\nDynamic_slam::initialize_keyframe_vec()_chk 1.2,  default_depth="<<default_depth<< flush; }
		runcl.initializeFirstDepthCostVol(	default_depth );																				// Zeros  	amem, cdatabuf, temp_cdatabuf,  etc..
																																			if(verbosity>local_verbosity_threshold){ cout << "\n\nDynamic_slam::initialize_keyframe_vec()_chk 1.3"<< flush;
																																				stringstream ss;	ss << fname << "_chk1.3_frame_num" << runcl.dataset_frame_num << "_estimateSE3_LK_";
																																				runcl.DownloadAndSave( 	runcl.keyframe_depth_mem,   	ss.str( ), 	runcl.paths.at( "keyframe_depth_mem"), runcl.mm_size_bytes_C1,   runcl.mm_Image_size,   CV_32FC1, 	false , 1); // runcl.fp32_params[MAX_INV_DEPTH]
																																			}
	}
																																			if(verbosity>local_verbosity_threshold){ cout << "\n\nDynamic_slam::initialize_keyframe_vec()_chk 2, " << flush; }
	if( obj["initialize_tracking_from_GT_depth"].asBool() == true ){																		if(verbosity>local_verbosity_threshold){ cout << "\n\nDynamic_slam::initialize_keyframe_vec()_chk 2.1, depth = runcl.depth_mem_GT " << flush; }
		runcl.update_tracking_depthmap( 							runcl.depth_mem_GT );													// copies buffer runcl.depth_mem_GT to keyframe_depth_mem
	}
	if( obj["initialize_keyframe_from_GT"].asBool() == true ){																				if(verbosity>local_verbosity_threshold){ cout << "\n\nDynamic_slam::initialize_keyframe_vec()_chk 2.2, frame_data = frame_data_GT"  << flush; }
		keyframe_data.back().frame_data.frame_data 					= keyframe_data.back().frame_data.frame_data_GT;						// copies frame_data_GT to frame_data for the new keyframe
	}
																																			// Save keyframe "amem" and "reference_image" /////////////////////////////////////
	keyframe_data.back().depthmap			= cv::Mat::zeros(		runcl.uint_params[MM_ROWS], 	runcl.uint_params[MM_ROWS],  CV_32FC4	);	// Instantiate and zero CV::Mat
	keyframe_data.back().reference_image	= cv::Mat::zeros(		runcl.uint_params[MM_ROWS], 	runcl.uint_params[MM_ROWS],  CV_32FC4	);

	runcl.ReadOutput(  keyframe_data.back().depthmap.data, 			runcl.amem,  	runcl.image_size_bytes );								// Saves "amem" and "reference_image" buffers to CV::Mat in current elem of vector<> Keyframe_data.
	runcl.ReadOutput(  keyframe_data.back().reference_image.data, 	runcl.imgmem,	runcl.image_size_bytes );								// These can be used for loop closure later on.
																																			if(verbosity>local_verbosity_threshold){ cout << "\n\nDynamic_slam::initialize_keyframe_vec()_chk 3"<< flush;
																																				stringstream ss;	ss << fname << "_chk3_frame_num" << runcl.dataset_frame_num << "_estimateSE3_LK_";
																																				runcl.DownloadAndSave( 	runcl.keyframe_depth_mem,   	ss.str( ), 	runcl.paths.at( "keyframe_depth_mem"), runcl.mm_size_bytes_C1,   runcl.mm_Image_size,   CV_32FC1, 	false , 1); // runcl.fp32_params[MAX_INV_DEPTH]
																																			}
	runcl.initialize_fp32_params();												// reset parameters											// runcl.initialize_fp32_params();  runcl.keyFrameCount++; runcl.dataset_frame_num++;
	binocular_reference_frame();
	runcl.keyFrameCount++;
																																			if(verbosity>local_verbosity_threshold){ cout << "\n\nDynamic_slam::initialize_keyframe_vec() Finished ###########################" << flush;}
}
