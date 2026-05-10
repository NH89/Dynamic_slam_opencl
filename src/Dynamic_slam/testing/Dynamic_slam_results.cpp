#include "../Dynamic_slam.hpp"

void Dynamic_slam::print_pose_datum( Dynamic_slam::pose_datum datum ){
      PRINT_MATX44F(    datum.K                       ,K                        );              // camera intrinsic matrix
      PRINT_MATX44F(    datum.inv_K                   ,inv_k                    );
      PRINT_MATX44F(    datum.pose                    ,pose                     );              // pose in abs coords. (not pose2pose from prev_frame, nor from keyframe) ?
      PRINT_MATX44F(    datum.inv_pose                ,inv_pose                 );
      PRINT_MATX44F(    datum.prev_pose2pose           ,keyframe2pose            );
      PRINT_MATX16F(    datum.keyframe2pose_algebra   ,keyframe2pose_algebra    );				// SE(3) algebra.
      PRINT_MATX44F(    datum.K2K                     ,K2K                      );
}

void Dynamic_slam::print_frame_datum( Dynamic_slam::frame_datum datum ){
    cout << "\n keyframe_index = " << datum.keyframe_index ;									// Index within this vector< >, of the keyframe for this frame.
    cout << "\n\f frame_data: ++++++++++++++++++++++++++++++++++++++++++++";		print_pose_datum( datum.frame_data		);
    cout << "\n\f frame_data_GT ++++++++++++++++++++++++++++++++++++++++++";		print_pose_datum( datum.frame_data_GT	);
    cout << "\n\f error_data +++++++++++++++++++++++++++++++++++++++++++++";		print_pose_datum( datum.error_data		);
}

void Dynamic_slam::print_frame_data_vector(       uint start,     uint stop,  vector<Dynamic_slam::frame_datum>       frame_data_vector,  string vector_name ){
    cout << "\n\f Dynamic_slam::print_frame_data_vector :  vector<Dynamic_slam::frame_datum> " << vector_name << " : ##############################################################################";
    if (stop > frame_data_vector.size() ) stop = frame_data_vector.size();

    for (int i = start; i<stop; i++){
		if (i>start)cout << "\n\f";
        cout << "\n\n\n\n  Element = " << i << "  ##############################################################";
        print_frame_datum( frame_data_vector[i] );
    }
    cout << "\n\n Dynamic_slam::print_frame_data_vector :  vector<Dynamic_slam::frame_datum> Finished " << vector_name << " : ##############################################################################\n\f";
}

void Dynamic_slam::report_GT_pose_error(){																																		// An expensive function, use only for debugging.
	string fname="Dynamic_slam::report_GT_pose_error()";
	//int local_verbosity_threshold = V_DYNAMIC_SLAM_REPORT_GT_POSE_ERROR;

	cout << "\n\f void Dynamic_slam::report_GT_pose_error() ###################################################################### " << flush;
	frame_data.back().error_data.K							= frame_data.back().frame_data.K						*	frame_data.back().frame_data_GT.K.inv();						// NB we use the more expensive general matrix inverse from opencv,
	frame_data.back().error_data.inv_K						= frame_data.back().frame_data.inv_K					*	frame_data.back().frame_data_GT.inv_K.inv();					// to verify that the specialist pose and intrinsic matrix inverses are correct.
	frame_data.back().error_data.pose						= frame_data.back().frame_data.pose						*	frame_data.back().frame_data_GT.pose.inv();
	frame_data.back().error_data.inv_pose					= frame_data.back().frame_data.inv_pose					*	frame_data.back().frame_data_GT.inv_pose.inv();
	frame_data.back().error_data.prev_pose2pose				= frame_data.back().frame_data.prev_pose2pose			*	frame_data.back().frame_data_GT.prev_pose2pose.inv();
	frame_data.back().error_data.K2K						= frame_data.back().frame_data.K2K						*	frame_data.back().frame_data_GT.K2K.inv();
	frame_data.back().error_data.keyframe2pose_algebra		= PToLie(frame_data.back().error_data.prev_pose2pose );

	print_frame_datum( frame_data.back()   );																																	// Print the whole set for frame_data, frame_data_GT, and error_data.

	cout << "\n void Dynamic_slam::report_GT_pose_error() Finished ######################################################################\n\f" << flush;
}

void Dynamic_slam::getNextFrameProfile(time_pt step_0, time_pt step_1, time_pt step_2, time_pt step_3, time_pt step_4, time_pt step_5 ){
	stringstream ss;
	ss 	<<"\nExecution times:(microseconds)####################################################"
		<<"\nCPU fns()                                     "	<<  duration_cast<microseconds>(step_1 - step_0).count()
		<<"\ngetFrame()...................................."	<<  duration_cast<microseconds>(step_2 - step_1).count()
		<<"\nestimate_tracking()                           "	<<  duration_cast<microseconds>(step_3 - step_2).count()
		<<"\nestimate_depth().............................."	<<  duration_cast<microseconds>(step_4 - step_3).count()
		<<"\nestimate_calibration()........................"	<<  duration_cast<microseconds>(step_5 - step_4).count()


		<<"\nTotal                                         "	<<  duration_cast<microseconds>(step_5 - step_0).count()
		<<"\n##################################################################################";
	cout << ss.str() << flush;
}
