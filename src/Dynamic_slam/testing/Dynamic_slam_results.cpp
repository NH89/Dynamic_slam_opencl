#include "../Dynamic_slam.hpp"

void Dynamic_slam::print_pose_datum( Dynamic_slam::pose_datum datum ){
      PRINT_MATX44F(    datum.K                       ,K                        );              // camera intrinsic matrix
      PRINT_MATX44F(    datum.inv_K                   ,inv_k                    );
      PRINT_MATX44F(    datum.pose                    ,pose                     );              // pose in abs coords. (not pose2pose from prev_frame, nor from keyframe) ?
      PRINT_MATX44F(    datum.inv_pose                ,inv_pose                 );
      PRINT_MATX44F(    datum.keyframe2pose           ,keyframe2pose            );
      PRINT_MATX16F(    datum.keyframe2pose_algebra   ,keyframe2pose_algebra    );				// SE(3) algebra.
      PRINT_MATX44F(    datum.K2K                     ,K2K                      );
}

void Dynamic_slam::print_frame_datum( Dynamic_slam::frame_datum datum ){
    cout << "\n keyframe_index = " << datum.keyframe_index ;									// Index within this vector< >, of the keyframe for this frame.
    cout << "\f frame_data: ++++++++++++++++++++++++++++++++++++++++++++";		print_pose_datum( datum.frame_data		);
    cout << "\f frame_data_GT ++++++++++++++++++++++++++++++++++++++++++";		print_pose_datum( datum.frame_data_GT	);
    cout << "\f error_data +++++++++++++++++++++++++++++++++++++++++++++";		print_pose_datum( datum.error_data		);
}

void Dynamic_slam::print_keyframe_datum( Dynamic_slam::keyframe_datum datum ){
    cout << "\n first_frame_index = " << datum.first_frame_index ;
    cout << "\n ";      print_frame_datum( datum.frame_data  );
}

void Dynamic_slam::print_frame_data_vector(       uint start,     uint stop,  vector<Dynamic_slam::frame_datum>       frame_data_vector,  string vector_name ){
    cout << "\f Dynamic_slam::print_frame_data_vector :  vector<Dynamic_slam::frame_datum> " << vector_name << " : ##############################################################################";
    if (stop > frame_data_vector.size() ) stop = frame_data_vector.size();

    for (int i = start; i<stop; i++){
		if (i>start)cout << "\f";
        cout << "\n\n\n\n  Element = " << i << "  ##############################################################";
        print_frame_datum( frame_data_vector[i] );
    }
    cout << "\n\n Dynamic_slam::print_frame_data_vector :  vector<Dynamic_slam::frame_datum> Finished " << vector_name << " : ##############################################################################\f";
}

void Dynamic_slam::print_keyframe_data_vector(    uint start,     uint stop,  vector<Dynamic_slam::keyframe_datum>    keyframe_data_vector,  string vector_name  ){
    cout << "\f Dynamic_slam::print_keyframe_data_vector():  vector<Dynamic_slam::keyframe_datum> " << vector_name << " : ###########################################################################";
    if (stop > keyframe_data_vector.size() ) stop = keyframe_data_vector.size();

    for (int i = start; i<stop; i++){
		if (i>start)cout << "\f";
        cout << "\n\n\n\n  Element = " << i << "  ##############################################################";
        print_keyframe_datum( keyframe_data_vector[i] );
    }
    cout << "\n\n Dynamic_slam::print_keyframe_data_vector():  vector<Dynamic_slam::keyframe_datum> Finished " << vector_name << " : ###########################################################################\f";
}

void Dynamic_slam::print_pose_vectors(uint start, uint stop){
	print_frame_data_vector( 	start, stop, frame_data, 		"dynamic_slam.frame_data" 		);
	print_keyframe_data_vector(	start, stop, keyframe_data, 	"dynamic_slam.keyframe_data" 	);
}


void Dynamic_slam::report_GT_pose_error(){																																	// An expensive function, use only for debugging.
	string fname="Dynamic_slam::report_GT_pose_error()";
	int local_verbosity_threshold = V_DYNAMIC_SLAM_REPORT_GT_POSE_ERROR;

	cout << "\f void Dynamic_slam::report_GT_pose_error() ###################################################################### " << flush;
	frame_data.back().error_data.K							=		frame_data.back().frame_data.K				*    frame_data.back().frame_data_GT.K.inv();				// NB we use the more expensive general matrix inverse from opencv,
	frame_data.back().error_data.inv_K						=		frame_data.back().frame_data.inv_K			*    frame_data.back().frame_data_GT.inv_K.inv();			// to verify that the specialist pose and intrinsic matrix inverses are correct.
	frame_data.back().error_data.pose						=		frame_data.back().frame_data.pose			*    frame_data.back().frame_data_GT.pose.inv();
	frame_data.back().error_data.inv_pose					=		frame_data.back().frame_data.inv_pose		*    frame_data.back().frame_data_GT.inv_pose.inv();
	frame_data.back().error_data.keyframe2pose				=		frame_data.back().frame_data.keyframe2pose	*    frame_data.back().frame_data_GT.keyframe2pose.inv();
	frame_data.back().error_data.K2K						=		frame_data.back().frame_data.K2K			*    frame_data.back().frame_data_GT.K2K.inv();
	frame_data.back().error_data.keyframe2pose_algebra		=		LieSub( frame_data.back().frame_data.keyframe2pose_algebra,		frame_data.back().frame_data_GT.keyframe2pose_algebra );   // Verify correctness of algebras, and Lie functions.

	print_frame_datum( frame_data.back()   );																																// Print the whole set for frame_data, frame_data_GT, and error_data.

	cout << "\n void Dynamic_slam::report_GT_pose_error() Finished ######################################################################\f" << flush;
}


void Dynamic_slam::initialize_resultsMat(){	// need to take img pyramid layer 2 of output, or read layer num from .json .
	int local_verbosity_threshold = V_DYNAMIC_SLAM_INITIALIZE_RESULTSMAT;//verbosity_mp["Dynamic_slam::initialize_resultsMat"];
																																			if(verbosity>local_verbosity_threshold) cout << "\n\n Dynamic_slam::initialize_resultsMat()_chk 1" << flush;
	uint reduction 		= obj["sample_layer"].asUInt();
	uint SE_iter 		= obj["SE_iter"].asUInt();
	int rows 			= 7 * ( runcl.MipMap[reduction*8 + MiM_READ_ROWS] +  runcl.mm_margin );
	int cols 			= SE_iter * ( runcl.MipMap[reduction*8 + MiM_READ_COLS] +  runcl.mm_margin );										if(verbosity>local_verbosity_threshold) cout << "\n Dynamic_slam::initialize_resultsMat()_chk 3: rows = "	<<rows<<", cols = "<<cols<< flush;
	runcl.resultsMat 	= cv::Mat::zeros ( rows, cols , CV_8UC4);																			if(verbosity>local_verbosity_threshold) cout << ",  runcl.resultsMat.size() = "<< runcl.resultsMat.size() 	<< flush;
																																			if(verbosity>local_verbosity_threshold) cout << "\n Dynamic_slam::initialize_resultsMat()_chk  finished\n" 	<< flush;
}

void Dynamic_slam::getResult(){
	int local_verbosity_threshold = V_DYNAMIC_SLAM_GETRESULT;//verbosity_mp["Dynamic_slam::getResult"];
	if(verbosity>local_verbosity_threshold){
		cout<<"\nDynamic_slam::getResult()  ################################################\n"<<flush;

		stringstream ss;
		ss << "getResult"<< runcl.save_index << "_QD_count_" << runcl.QD_count <<"_epsilon_"<<epsilon<<"_sigmaQ_"<<sigma_q<<"_D_"<<sigma_d<<"_theta_"<<theta;
																																				//int this_count = save_index * 1000 + QD_count;
																																				//ss << save_index << "_QD_count_" << QD_count;
		cv::Size q_size( runcl.mm_Image_size.width, 2* runcl.mm_Image_size.height ); 			// 2x sized for qx and qy.
		runcl.DownloadAndSave(runcl.dmem,   ss.str(), runcl.paths.at("dmem"),    runcl.mm_size_bytes_C1 , runcl.mm_Image_size , CV_32FC1, false ,    runcl.fp32_params[MAX_INV_DEPTH]  );
		runcl.DownloadAndSave(runcl.amem,   ss.str(), runcl.paths.at("amem"),    runcl.mm_size_bytes_C1 , runcl.mm_Image_size , CV_32FC1, false ,    runcl.fp32_params[MAX_INV_DEPTH]  );

		//DownloadAndSave(qmem,   ss.str(), paths.at("qmem"),  2*mm_size_bytes_C1 , q_size        , CV_32FC1, false , -1*fp32_params[MAX_INV_DEPTH]  );
		//DownloadAndSave(qmem2,   ss.str(), paths.at("qmem2"),2*mm_size_bytes_C1 , q_size        , CV_32FC1, false , -1*fp32_params[MAX_INV_DEPTH]  );  // 1/uint_params[MM_PIXELS]
		cout<<"\nDynamic_slam::getResult()_finished ################################################\n"<<flush;
	}
};

void Dynamic_slam::getNextFrameProfile(time_pt step_0, time_pt step_1, time_pt step_2, time_pt step_3, time_pt step_4, time_pt step_5, time_pt step_6, time_pt step_7, time_pt step_8){
	stringstream ss;
	ss 	<<"\nExecution times:(microseconds)####################################################"
		<<"\ngetFrameData()                                    "	<<  duration_cast<microseconds>(step_1 - step_0).count()
		<<"\npredictFrame()...................................."	<<  duration_cast<microseconds>(step_2 - step_1).count()
		<<"\nuse_GT_pose()                                     "	<<  duration_cast<microseconds>(step_3 - step_2).count()
		<<"\ngetFrame()........................................"	<<  duration_cast<microseconds>(step_4 - step_3).count()
		<<"\nartificial_pose_error()                           "	<<  duration_cast<microseconds>(step_5 - step_4).count()
		<<"\nestimateSE3_LK()                                  "	<<  duration_cast<microseconds>(step_6 - step_5).count()
		<<"\nreport_GT_pose_error() &  display_frame_resluts() "	<<  duration_cast<microseconds>(step_7 - step_6).count()
		<<"\nupdateDepthCostVol()                              "	<<  duration_cast<microseconds>(step_8 - step_7).count()
		<<"\nTotal                                             "	<<  duration_cast<microseconds>(step_8 - step_0).count()
		<<"\n##################################################################################";
	cout << ss.str() << flush;
}
