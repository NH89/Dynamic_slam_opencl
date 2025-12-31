#include "../Dynamic_slam.hpp"


void Dynamic_slam::getFrameData_vec(){  // Dynamic_slam::initialize_camera_vec(),  Dynmaic_slam::nextFrame()
	int local_verbosity_threshold = V_DYNAMIC_SLAM_GETFRAMEDATA;//verbosity_mp["Dynamic_slam::getFrameData"];
																																			if(verbosity>local_verbosity_threshold) cout << "\n Dynamic_slam::getFrameData_vec_chk 0.  runcl.dataset_frame_num = "<< runcl.dataset_frame_num
																																				<< "\t###################################" << flush;
																																			cout<<"\nruncl.dataset_frame_num="<<runcl.dataset_frame_num<<flush;

	std::string str 						= txt[runcl.dataset_frame_num].c_str();															cout<<"chk1 "<<flush;						// grab .txt file from array of files (e.g. "scene_00_0000.txt")
	char		*ch 						= new char [str.length()+1];																	cout<<"chk2 "<<flush;
	std::strcpy (ch, str.c_str());																											cout<<"chk3 "<<flush;
	cv::Mat		T_alt;																														cout<<"chk4 "<<flush;
	convertAhandaPovRayToStandard_2( obj,  ch, R, T, cameraMatrix );	cout<<"chk5 "<<flush;	// TODO  which of these 2 versions of convertAhandaPovRayToStandard() is correct ?
	convertAhandaPovRayToStandard(   obj,  ch, R, T, cameraMatrix );	cout<<"chk6 "<<flush;
	delete [] ch; 														cout<<"chk7 "<<flush;	//free(ch);
																																			if(verbosity>local_verbosity_threshold) {
																																				cout << "\n Dynamic_slam::getFrameData_vec_chk 1";
																																				cout << "\n\n R = \n" << R;
																																				cout << "\n\n T = \n" << T;
																																				cout << "\n\n cameraMatrix = \n" << cameraMatrix;
																																				PRINT_MATX44F(frame_data.back().frame_data.keyframe2pose,);		// correct here
																																				cout << endl << flush;
																																			}
	cv::Matx44f K_GT						= cv::Matx44f::zeros();
	for (int i=0; i<3; i++){
		for (int j=0; j<3; j++){
			K_GT.operator()(i,j)			= cameraMatrix.at<float>(i,j);
																																			cout << ", " <<  cameraMatrix.at<float>(i,j);
		}
	}K_GT.operator()(3,3) = 1;																												// Orthographic camera, See notes in convertTransforms.cpp , cv::Matx44f generate_invK_(cv::Matx44f K_, int verbosity){..}
																																			// 4x4 perspective matrix is not invertable for points at infinity. We correct ortho->perspective in the kernel by dividing by Z.
	pose_datum datum 						= {};																							// default initialization.
	datum.K									= K_GT;
	datum.inv_K								= generate_invK_(K_GT, verbosity);
	datum.pose								= getPose(R,T, verbosity);																		PRINT_MATX44F( datum.pose, );
	datum.inv_pose							= getInvPose(datum.pose, verbosity);															PRINT_MATX44F( datum.inv_pose, );

	runcl.current_frames[ runcl.current_frames_idx[0] ].pose_gt		= datum.pose;															cout << "\n\n runcl.current_frames_idx[0-5] = ";
																																			for(int i=0; i<5; i++){ cout<< runcl.current_frames_idx[i] << ",  "; }	cout << flush;
																																			for(int i=0; i<5; i++){	cout<<"\ni="<<i<<"  ";
																																					PRINT_MATX44F(	runcl.current_frames[ runcl.current_frames_idx[i] ].pose_gt,   );
																																			}
	Matx44f pose_frame0to1_gt	= runcl.current_frames[ runcl.current_frames_idx[1] ].pose_gt	*	datum.inv_pose;							PRINT_MATX44F( pose_frame0to1_gt, );
	Matx16f	artif_error			= {0.10f, 0.0f, 0.0f,		2.0f, 0.0f, 0.0f };																PRINT_MATX16F( artif_error,);
	Matx44f artif_error_matx	= LieToP_Matx( artif_error );																				PRINT_MATX44F( artif_error_matx,);
	Matx44f pose_frame0to1		= pose_frame0to1_gt * artif_error_matx;																		PRINT_MATX44F( pose_frame0to1,);
	Matx44f k2k_0to1			= datum.K			* pose_frame0to1	* datum.K.inv();													PRINT_MATX44F( k2k_0to1,);
	runcl.update_k2k_buf(		k2k_0to1, pose_frame0to1);																					// NB Rotation is in Radians. Translation is in world units. Translation is depth range dependent.

	Matx44f pose_error_1		= pose_frame0to1_gt				* pose_frame0to1.inv();														PRINT_MATX44F( pose_error_1, pose_frame0to1_gt		* pose_frame0to1.inv()	);
	Matx44f pose_error_2		= pose_frame0to1.inv()			* pose_frame0to1_gt;														PRINT_MATX44F( pose_error_2, pose_frame0to1.inv()	* pose_frame0to1_gt		);
	Matx44f pose_error_3		= pose_frame0to1_gt.inv()		* pose_frame0to1;															PRINT_MATX44F( pose_error_3, pose_frame0to1_gt.inv()* pose_frame0to1		);
	Matx44f pose_error_4		= pose_frame0to1_gt				* pose_frame0to1.inv();														PRINT_MATX44F( pose_error_4, pose_frame0to1_gt		* pose_frame0to1.inv()	);

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/*Below is code for older Dynamic_slam::frame_data.    Above is for RunCL::current_frames[..] and direct write to k2kbuf & pose_buf.	if(verbosity>local_verbosity_threshold) {cout << "\n Dynamic_slam::getFrameData_vec_chk 2, "
																																				<<"\truncl.dataset_frame_num="<<runcl.dataset_frame_num
																																				<<"\tframe_data.size()="<<frame_data.size()
																																				<<endl<<flush;}
	if (runcl.dataset_frame_num > 0){
																																			if(verbosity>local_verbosity_threshold) cout << "\n Dynamic_slam::getFrameData_vec_chk 2.1,  (runcl.dataset_frame_num > 0)"<<flush;
		uint 			index 				= frame_data.back().keyframe_index;																cout<<"chk1 "<<flush; cout<<"keyframe_data["<<index<<"].frame_data.frame_data_GT.inv_pose"<<flush;
		cv::Matx44f		invPose_index		= keyframe_data[index].frame_data.frame_data_GT.inv_pose;										cout<<"chk2 "<<flush;
		datum.keyframe2pose					= datum.pose	* invPose_index;																cout<<"chk3 "<<flush;
																																			PRINT_MATX44F(invPose_index  		,Dynamic_slam::getFrameData_vec()  );
																																			PRINT_MATX44F(datum.pose  			,Dynamic_slam::getFrameData_vec()  );
																																			PRINT_MATX44F(datum.keyframe2pose  	,Dynamic_slam::getFrameData_vec()  );
		datum.K2K							= datum.K		* datum.keyframe2pose	* frame_data[index].frame_data_GT.inv_K;
		datum.keyframe2pose_algebra			= PToLie(datum.keyframe2pose);
	}else{																																	if(verbosity>local_verbosity_threshold) cout << "\n Dynamic_slam::getFrameData_vec_chk 2.2,  (runcl.dataset_frame_num <= 0)"<<flush;
		datum.keyframe2pose					= MATX44F_EYE;
		datum.K2K							= MATX44F_EYE;
		datum.keyframe2pose_algebra			= {0,0,0,  0,0,0} ;
	}
	frame_data.back().frame_data_GT			= datum;						// TODO if(use GT),  but move it out to Dynamic_slam::next_frame()
																																			if ( runcl.baseImage.empty() ) {cerr << "\nDynamic_slam::getFrameData_vec():   Error runcl.baseImage.empty() "<<flush;  runcl.exit_(1); }
	int r 									= runcl.baseImage.rows;
	int c 									= runcl.baseImage.cols;
	depth_GT 								= loadDepthAhanda(obj, depth[runcl.dataset_frame_num].string(), r,c,cameraMatrix);

	runcl.load_GT_depth(depth_GT, invert_GT_depth);
																																			if(verbosity>local_verbosity_threshold) {
																																				cout << "\n### Dynamic_slam::getFrameData_vec_chk 2.3";
																																				print_pose_datum( frame_data.back().frame_data_GT );
																																			}
*/
																																			if(verbosity>local_verbosity_threshold) cout << "\n Dynamic_slam::getFrameData_vec_chk Finished ######################################"<<flush;
}

void Dynamic_slam::use_GT_pose_vec(){
	int local_verbosity_threshold = V_DYNAMIC_SLAM_USE_GT_POSE;//verbosity_mp["Dynamic_slam::use_GT_pose"];// -1;
																																			if(verbosity>local_verbosity_threshold) cout << "\n Dynamic_slam::use_GT_pose_chk_0,"<<flush;

	float pose_arry[16];
	Matx44f_To_float16arry(		runcl.current_frames[  runcl.current_frames_idx[0]  ].pose_gt,			pose_arry );
	runcl.update_k2k_buf(		runcl.current_frames[  runcl.current_frames_idx[0]  ].k2k_0to1_est,		pose_arry );
																																			if(verbosity>local_verbosity_threshold){
																																				//PRINT_MATX44F(frame_data.back().frame_data.keyframe2pose,);
																																				//PRINT_FLOAT_16(runcl.fp32_k2keyframe,);
																																				cout << "\nDynamic_slam::use_GT_pose()_finish ##############################################\n\n" << flush;
																																			}
}

/* old artificial_pose_error_vec()
void Dynamic_slam::artificial_pose_error_vec(){	// TODO if(GT_available==true){}else{}
	int local_verbosity_threshold = V_DYNAMIC_SLAM_ARTIFICIAL_POSE_ERROR;//verbosity_mp["Dynamic_slam::artificial_pose_error"];
																																			if(verbosity>local_verbosity_threshold){
																																				cout << "\n\n##Dynamic_slam::artificial_pose_error()_chk_0 ##########################################" << endl << flush;
																																			}
	Matx16f pose_step_algebra;
	for (int SE3=0; SE3<6; SE3++)  pose_step_algebra.operator()(0,SE3) = obj["Artif_pose_err_algebra"][SE3].asFloat();
	Matx44f poseStep 	= LieToP_Matx(pose_step_algebra);																					if(verbosity>local_verbosity_threshold){
																																				cout << "\n\n##Dynamic_slam::artificial_pose_error()_chk_1"<<flush;
																																				PRINT_MATX44F(poseStep,);
																																				PRINT_MATX16F(pose_step_algebra,);
																																				PRINT_MATX16F(PToLie(poseStep), );
																																				PRINT_MATX16F(PToLie( frame_data.back().frame_data.keyframe2pose ),True);

																																				PRINT_MATX44F(frame_data.back().frame_data.keyframe2pose , );
																																				PRINT_MATX16F(frame_data.back().frame_data.keyframe2pose_algebra, Start);
																																			}
	// cv::Matx44f           pose                    = MATX44F_EYE ;             // pose in global coords. (not pose2pose from prev_frame, nor from keyframe) ?
	// cv::Matx44f           inv_pose                = MATX44F_EYE ;
	// cv::Matx44f           keyframe2pose           = MATX44F_EYE ;
	// cv::Matx44f           K2K                     = MATX44F_EYE ;
																																				cout << "\n\n##Dynamic_slam::artificial_pose_error()_chk_1.5"<<flush;
																																				PRINT_MATX44F(frame_data.back().frame_data.pose , );
	Matx44f	old							= frame_data.back().frame_data.pose;

	frame_data.back().frame_data.pose	= poseStep	*	frame_data.back().frame_data.pose;	// TODO which side to multiply from ?
																																				PRINT_MATX44F(frame_data.back().frame_data.pose , );
	Matx44f test1						= old.inv() * 	frame_data.back().frame_data.pose;
	Matx44f test2						= old 		* 	frame_data.back().frame_data.pose.inv();			// Correct, negative.
	Matx44f test3						= frame_data.back().frame_data.pose 		* old.inv();			// Correct
	Matx44f test4						= frame_data.back().frame_data.pose.inv()	* old;
																																				PRINT_MATX44F( test1, );
																																				PRINT_MATX44F( test2, );
																																				PRINT_MATX44F( test3, );
																																				PRINT_MATX44F( test4, );

	frame_data.back().frame_data.inv_pose						= getInvPose(frame_data.back().frame_data.pose, verbosity);
																																				PRINT_MATX44F(frame_data.back().frame_data.inv_pose , );

	uint 			index 										= frame_data.back().keyframe_index;
	cv::Matx44f		invPose_index								= keyframe_data[index].frame_data.frame_data_GT.inv_pose ;  				// getInvPose( keyframe_data[index].frame_data.frame_data_GT.keyframe2pose, verbosity);
	frame_data.back().frame_data.keyframe2pose					= frame_data.back().frame_data.pose 	* invPose_index;

	frame_data.back().frame_data.K2K							= frame_data.back().frame_data.K 		* frame_data.back().frame_data.keyframe2pose 	* frame_data[index].frame_data_GT.inv_K;
	frame_data.back().frame_data.keyframe2pose_algebra			= PToLie( frame_data.back().frame_data.keyframe2pose );
																																			if(verbosity>local_verbosity_threshold){
																																				cout << "\n\n##Dynamic_slam::artificial_pose_error()_chk_2"<<flush;
																																				PRINT_MATX44F(frame_data.back().frame_data.keyframe2pose , );
																																				PRINT_MATX44F(frame_data.back().frame_data.inv_pose , );
																																				PRINT_MATX44F(keyframe_data[index].frame_data.frame_data_GT.inv_pose , );

																																				PRINT_MATX16F(frame_data.back().frame_data.keyframe2pose_algebra, Start);
																																				PRINT_MATX44F(frame_data.back().frame_data.K2K , );
																																			}

	frame_data.back().frame_data.K2K 	= frame_data.back().frame_data.K  * frame_data.back().frame_data.keyframe2pose  *  frame_data.back().frame_data.inv_K;
																																			if(verbosity>local_verbosity_threshold){
																																				cout << "\n\n##Dynamic_slam::artificial_pose_error()_chk_3 : frame_data.back().frame_data.K2K, New" << endl << flush;
																																				PRINT_MATX44F(frame_data.back().frame_data.K ,);
																																				PRINT_MATX44F(frame_data.back().frame_data.keyframe2pose ,);
																																				PRINT_MATX44F(frame_data.back().frame_data.inv_K ,);

																																				PRINT_MATX44F(frame_data.back().frame_data.K2K, New);			//  huge values !
																																				PRINT_FLOAT_16(runcl.fp32_k2keyframe,Old);
																																			}// Add error of one step in the 2nd SE3 DoF.

	for (int i=0; i<16; i++){ runcl.fp32_k2keyframe[i] = frame_data.back().frame_data.K2K.operator()(i/4, i%4);  }
	float pose_arry[16];
	Matx44f_To_float16arry( frame_data.back().frame_data.keyframe2pose, pose_arry );
	runcl.update_k2k_buf( runcl.fp32_k2keyframe, pose_arry );
																																			if(verbosity>local_verbosity_threshold){
																																				PRINT_FLOAT_16(runcl.fp32_k2keyframe,New);
																																				cout << "\nDynamic_slam::artificial_pose_error()_finish ##############################################\n\n" << flush;	}
}
*/
