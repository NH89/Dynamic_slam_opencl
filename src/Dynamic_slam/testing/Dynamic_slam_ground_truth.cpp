#include "../Dynamic_slam.hpp"


void Dynamic_slam::getFrameData_vec(){  // Dynamic_slam::initialize_camera_vec(),  Dynmaic_slam::nextFrame()
	int local_verbosity_threshold = V_DYNAMIC_SLAM_GETFRAMEDATA;//verbosity_mp["Dynamic_slam::getFrameData"];
																																			if(verbosity>local_verbosity_threshold) cout << "\n Dynamic_slam::getFrameData_vec_chk 0.  runcl.dataset_frame_num = "<< runcl.dataset_frame_num
																																				<< "\t###################################" << flush;
	std::string str 						= txt[runcl.dataset_frame_num].c_str();																					// grab .txt file from array of files (e.g. "scene_00_0000.txt")
    char        *ch 						= new char [str.length()+1];
    std::strcpy (ch, str.c_str());
	cv::Mat T_alt;
	convertAhandaPovRayToStandard_2( obj,  ch, R, T, cameraMatrix );
    convertAhandaPovRayToStandard( obj,  ch, R, T, cameraMatrix );
	delete [] ch; //free(ch);
																																			if(verbosity>local_verbosity_threshold) {
																																				cout << "\n Dynamic_slam::getFrameData_vec_chk 1";
																																				cout << "\n\n R = \n" << R;
																																				cout << "\n\n T = \n" << T;
																																				cout << "\n\n cameraMatrix = \n" << cameraMatrix;
																																				PRINT_MATX44F(frame_data.back().frame_data.keyframe2pose,);		// correct here
																																				cout << endl << flush;
																																			}
	cv::Matx44f K_GT = cv::Matx44f::zeros();
	for (int i=0; i<3; i++){
		for (int j=0; j<3; j++){
			K_GT.operator()(i,j) 			= cameraMatrix.at<float>(i,j);
																																			cout << ", " <<  cameraMatrix.at<float>(i,j);
		}
	}K_GT.operator()(3,3) = 1;																												// Orthographic camera, See notes in convertTransforms.cpp , cv::Matx44f generate_invK_(cv::Matx44f K_, int verbosity){..}
																																			// 4x4 perspective matrix is not invertable for points at infinity. We correct ortho->perspective in the kernel by dividing by Z.

	pose_datum datum 						= {};																							// default initialization.
	datum.K									= K_GT;
    datum.inv_K								= generate_invK_(K_GT, verbosity);
    datum.pose								= getPose(R,T, verbosity);
    datum.inv_pose							= getInvPose(datum.pose, verbosity);
																																			if(verbosity>local_verbosity_threshold) {cout << "\n Dynamic_slam::getFrameData_vec_chk 2, "
																																				<<"\truncl.dataset_frame_num="<<runcl.dataset_frame_num
																																				<<"\tframe_data.size()="<<frame_data.size()
																																				<<endl<<flush;}
	if (runcl.dataset_frame_num > 0){
																																			if(verbosity>local_verbosity_threshold) cout << "\n Dynamic_slam::getFrameData_vec_chk 2.1,  (runcl.dataset_frame_num > 0)"<<flush;
		uint 			index 				= frame_data.back().keyframe_index;
		cv::Matx44f		invPose_index		= keyframe_data[index].frame_data.frame_data_GT.inv_pose ;  					//   getInvPose( keyframe_data[index].frame_data.frame_data_GT.keyframe2pose, verbosity);
		datum.keyframe2pose					= datum.pose 	*    invPose_index;
																																			PRINT_MATX44F(invPose_index  		,Dynamic_slam::getFrameData_vec()  );
																																			PRINT_MATX44F(datum.pose  			,Dynamic_slam::getFrameData_vec()  );
																																			PRINT_MATX44F(datum.keyframe2pose  	,Dynamic_slam::getFrameData_vec()  );

		datum.K2K							= datum.K 		* datum.keyframe2pose 	* frame_data[index].frame_data_GT.inv_K;
		datum.keyframe2pose_algebra			= PToLie(datum.keyframe2pose);
	}else{																																	if(verbosity>local_verbosity_threshold) cout << "\n Dynamic_slam::getFrameData_vec_chk 2.2,  (runcl.dataset_frame_num <= 0)"<<flush;
		datum.keyframe2pose					= MATX44F_EYE;
		datum.K2K							= MATX44F_EYE;
		datum.keyframe2pose_algebra			= {0,0,0,  0,0,0} ;
	}
	//frame_data.back().keyframe_index		= runcl.dataset_frame_num;
	frame_data.back().frame_data_GT			= datum;
	// frame_data.back().frame_data			= datum;						// TODO if(use GT),  but move it out to Dynamic_slam::next_frame()
																																			if ( runcl.baseImage.empty() ) {cerr << "\nDynamic_slam::getFrameData_vec():   Error runcl.baseImage.empty() "<<flush;  runcl.exit_(1); }
	int r 									= runcl.baseImage.rows;
	int c 									= runcl.baseImage.cols;
	depth_GT 								= loadDepthAhanda(obj, depth[runcl.dataset_frame_num].string(), r,c,cameraMatrix);

	runcl.load_GT_depth(depth_GT, invert_GT_depth);
																																			if(verbosity>local_verbosity_threshold) {
																																				cout << "\n### Dynamic_slam::getFrameData_vec_chk 2.3";
																																				print_pose_datum( frame_data.back().frame_data_GT );
																																			}
																																			if(verbosity>local_verbosity_threshold) cout << "\n Dynamic_slam::getFrameData_vec_chk Finished ######################################"<<flush;
}

void Dynamic_slam::use_GT_pose_vec(){
	int local_verbosity_threshold = V_DYNAMIC_SLAM_USE_GT_POSE;//verbosity_mp["Dynamic_slam::use_GT_pose"];// -1;
																																			if(verbosity>local_verbosity_threshold) cout << "\n Dynamic_slam::use_GT_pose_chk_0,"<<flush;
	frame_data.back().frame_data = frame_data.back().frame_data_GT;
	for (int i=0; i<16; i++){ runcl.fp32_k2keyframe[i] = frame_data.back().frame_data.K2K.operator()(i/4, i%4);}
																																			if(verbosity>local_verbosity_threshold){
																																				PRINT_MATX44F(frame_data.back().frame_data.keyframe2pose,);
																																				PRINT_FLOAT_16(runcl.fp32_k2keyframe,);
																																			}
}

void Dynamic_slam::artificial_pose_error_vec(){
	int local_verbosity_threshold = V_DYNAMIC_SLAM_ARTIFICIAL_POSE_ERROR;//verbosity_mp["Dynamic_slam::artificial_pose_error"];													if(verbosity>local_verbosity_threshold){ cout << "\n\nDynamic_slam::artificial_pose_error() chk_0"<<flush; }
	Matx16f pose_step_algebra;
	for (int SE3=0; SE3<6; SE3++)  pose_step_algebra.operator()(0,SE3) = obj["Artif_pose_err_algebra"][SE3].asFloat();
	Matx44f poseStep 	= LieToP_Matx(pose_step_algebra);																					if(verbosity>local_verbosity_threshold){
																																				PRINT_MATX44F(poseStep,);
																																				PRINT_MATX16F(pose_step_algebra,);
																																				PRINT_MATX16F(PToLie( frame_data.back().frame_data.keyframe2pose ),True);  }

	frame_data.back().frame_data.keyframe2pose = frame_data.back().frame_data.keyframe2pose * poseStep;										if(verbosity>local_verbosity_threshold){
																																				PRINT_MATX16F(PToLie(frame_data.back().frame_data.keyframe2pose), Start); 	}

	frame_data.back().frame_data.K2K 	= frame_data.back().frame_data.K  * frame_data.back().frame_data.keyframe2pose  *  frame_data.back().frame_data.inv_K;
																																			if(verbosity>local_verbosity_threshold){
																																				cout << "\n\n##Dynamic_slam::artificial_pose_error() : frame_data.back().frame_data.K2K, New" << endl << flush;
																																				PRINT_MATX44F(frame_data.back().frame_data.K ,);
																																				PRINT_MATX44F(frame_data.back().frame_data.keyframe2pose ,);
																																				PRINT_MATX44F(frame_data.back().frame_data.inv_K ,);

																																				PRINT_MATX44F(frame_data.back().frame_data.K2K, New);			//  huge values !
																																				PRINT_FLOAT_16(runcl.fp32_k2keyframe,Old);
																																			}// Add error of one step in the 2nd SE3 DoF.

	for (int i=0; i<16; i++){ runcl.fp32_k2keyframe[i] = frame_data.back().frame_data.K2K.operator()(i/4, i%4);  }							if(verbosity>local_verbosity_threshold){
																																				PRINT_FLOAT_16(runcl.fp32_k2keyframe,New);
																																				cout << "\nDynamic_slam::artificial_pose_error()_finish ##############################################" << flush;	}
}
