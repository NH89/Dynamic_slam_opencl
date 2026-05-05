#include "Dynamic_slam.hpp"
#include <fstream>
#include <iomanip>   // for std::setprecision, std::setw

using namespace cv;
using namespace std;

void Dynamic_slam::estimateSLAM(){																										// Adaptive step size LM tracking and halting
	string fname = "Dynamic_slam::estimateSLAM()";
	int 	local_verbosity_threshold 		= V_DYNAMIC_SLAM_ESTIMATE_SLAM;//verbosity_mp["Dynamic_slam::estimateSE3"];
																																		if(verbosity>local_verbosity_threshold) {
																																			cout << "\fDynamic_slam::estimate_SLAM() chk_0"
																																			<<"  ##############################################################"<< flush;
																																		}


	estimate_tracking();																									cout<<"\nDynamic_slam::estimateSLAM() chk_0.1"<<flush;
	Matx44f pose_temp 	= runcl.update_pose_bufs_cur_frames( );																cout<<"\nDynamic_slam::estimateSLAM() chk_0.2"<<flush;				// NB these two lines are req because nextFrame() calls  runcl.set_cam_bufs(..), using frame_data.
	frame_data.back().frame_data.pose 	= pose_temp ;																		cout<<"\nDynamic_slam::estimateSLAM() chk_0.3"<<flush;

	// for( uint frame_index=0; frame_index<num_current_frames; frame_index++){													cout<<"\nDynamic_slam::estimateSLAM() chk_0.5 frame_index = "<<frame_index<<endl<<flush;
	// 	uint layer_ = 0;
	// 	runcl.rho_sq( out_block_size, 10+frame_index, frame_index, layer_, runcl.cur_frames_k2kbuf );	//iter, frame_idx, layer, cl_mem k2k_buf	);
	// }
	float16arry_To_Matx44f(	 &runcl.current_frames[	runcl.current_frames_idx[0]	].k2k_0to1_est[0]	, frame_data.back().frame_data.K2K );
																																		if(verbosity>local_verbosity_threshold) {
																																			cout << "\fDynamic_slam::estimate_SLAM() chk_1"<< flush;
																																			PRINT_MATX44F(frame_data.back().frame_data.pose,);
																																			PRINT_MATX44F(frame_data.back().frame_data.K2K, );
																																		}
	estimate_depth();
																																		if( verbosity>local_verbosity_threshold ){
																																			cout << "\nDynamic_slam::estimate_SLAM() finished  ###########################"<<flush;
																																		}
}

void Dynamic_slam::estimate_tracking(){
	string fname = "Dynamic_slam::estimate_tracking()";
	int 	local_verbosity_threshold 		= V_DYNAMIC_SLAM_ESTIMATE_TRACKING;//verbosity_mp["Dynamic_slam::estimateSE3"];
																																		if(verbosity>local_verbosity_threshold) {
																																			cout << "\fDynamic_slam::estimate_tracking() chk_0"
																																			<<"  ##############################################################"<< flush;
																																		}
	int		layer 				= SE3_start_layer;
	uint 	frame_idx			= 1;
	uint 	out_block_size 		= 4;
	float	old_sum_rho_sq		= FLT_MAX-1;
	float	factor				= -2.0f;
	Matx44f	old_pose			= Matx44f::eye();
	Matx44f	old_k2k				= Matx44f::eye();
	Matx44f newPose				= runcl.ReadOutput_44f( runcl.pose_buf );
	Matx44f newK2K				= runcl.ReadOutput_44f( runcl.k2kbuf, cl_flt16_size ); 													// offset = current_frames_idx * cl_flt16_size
																																		//for (uint out_block_size = 4/*32*/; out_block_size > 2; out_block_size /=2){
	for (uint iter = 0; iter<SE_iter; iter++){
		auto step_0 = high_resolution_clock::now();
																																		if(verbosity>local_verbosity_threshold) {
																																			cout << "\nDynamic_slam::estimate_tracking() chk_1: layer="<<layer
																																			<<", out_block_size="<<out_block_size<<",  iter="<<iter<<",  ###########################"<<flush;
																																			uint	out_block_size		= 2;
																																			uint	layer				= 0;
																																			runcl.rho_sq( out_block_size, iter, frame_idx, layer, runcl.k2kbuf	);	// For debugging, get a larger, finer Rho map
																																			PRINT_MATX44F( old_k2k, ); PRINT_MATX44F( old_pose, );
																																		}
		runcl.rho_sq( 			out_block_size, iter, frame_idx,  	(uint)layer,  runcl.k2kbuf );
		runcl.reduce_patch_Rho( out_block_size, iter, 	(uint)layer );
		runcl.update_k2k_cpu( 							(uint)layer );																	// frame_data_GT.keyframe2pose for comparision only.
		float		sum_rho		=	runcl.se3_rho_result.Rho.x;																			// currently .x colour channel only.
		float		sum_rho_sq	=	runcl.se3_rho_result.Rho.y;
		if( isnan(sum_rho_sq) ){
																			cout << "\nisnan(sum_rho_sq)" <<flush;
			break;
		}else if(sum_rho_sq > old_sum_rho_sq     ){																						// Rho, photometric error, got worse not better
			if(layer<=0) {break;}																										// Reached bottom of image pyramid.
			else {
																			cout << "\nsum_rho_sq > old_sum_rho_sq = "<< old_sum_rho_sq;
				if(factor<-1.0f){																										// End amplified steps
					factor = -1.0f;
																			cout << ",  factor -2.0f -> -1.0f";
				}else {
					layer --;																											// Step down to lower layer of image pyramid
																			cout << "\nlayer = "	<<	layer;
					old_sum_rho_sq			=	FLT_MAX-1;																				// Re-set old_sum_rho_sq for new layer
				}
																																		PRINT_MATX44F( old_k2k, ); PRINT_MATX44F( old_pose, );
				runcl.update_k2k_buf(		old_k2k,		old_pose);																	// Re-set to previous pose.
																			cout << endl << flush;
			}
		}else{
			old_sum_rho_sq			=	sum_rho_sq;
			old_pose				=	newPose;
			old_k2k					=	newK2K;

			float		num_pixels	=	runcl.se3_rho_result.SE3_incr_arry[1];																		// TO DO move numpixels to SE3_incr.w   & reduce SE3_incr_map_mem from float8 tro float4
			Matx16d		SE3_incr;	for (int i=0;	i<6; i++){	SE3_incr.operator()(i)	=	runcl.se3_rho_result.SE3_incr_arry[i*2];  };
																																		if( verbosity>local_verbosity_threshold ){
																																			cout << "\nDynamic_slam::estimate_tracking() chk_4: ,  ###########################"<<
																																			"\n sum_rho = "			<< sum_rho		<<
																																			",	sum_rho_sq	= "		<< sum_rho_sq	<<
																																			",	num_pixels = "		<< num_pixels	<< endl<<flush;
																																			PRINT_MATX16F( SE3_incr, );
																																		}
			Matx44f		pose		=	runcl.ReadOutput_44f(	runcl.pose_buf );
			Matx44f		invK		=	runcl.ReadOutput_44f(	runcl.inv_K_buf);
			Matx44f		K			=	runcl.ReadOutput_44f(	runcl.K_buf	 );
																																		if( verbosity>local_verbosity_threshold ){
																																			PRINT_MATX44F( pose,	from pose_buf );	PRINT_MATX16F( PToLie(pose),);
																																			PRINT_MATX44F( invK,	);
																																			PRINT_MATX44F( K,		);
																																			PRINT_MATX44F( K * invK,		);
																																			PRINT_MATX44F( invK * K,		);
																																		}
			Matx66d	invH			=	runcl.current_frames[ runcl.current_frames_idx[0] ].invHessian[layer];
			Matx16d pose_update_cpu	=	SE3_incr * invH;	// Matx_16fmul66f( SE3_incr, invH);  //										// Double precision is required
																																		if( verbosity>local_verbosity_threshold ){
																																			cout << "\nSE3_incr="			<<SE3_incr			<<endl<<flush;
																																			cout << "\ninvH="				<<invH				<<endl<<flush;
																																			cout << "\npose_update_cpu="	<<pose_update_cpu	<<endl<<flush;
																																			PRINT_MATX66F( invH, );
																																			PRINT_MATX16F( pose_update_cpu, );
																																			Matx44f	pose_old	= runcl.ReadOutput_44f( runcl.pose_buf );	PRINT_MATX44F( pose_old, );
																																			Matx44f	k2k_old		= runcl.ReadOutput_44f( runcl.k2kbuf);		PRINT_MATX44F( k2k_old,	);
																																		}
			pose_update_cpu			=	factor *  pose_update_cpu.mul( deltas_matx[layer] );/*(-2.0f)*/ /*  * 0.5f; */  				//NB matx.mul(  matx ) => elementwise multiplication.
																																		if( verbosity>local_verbosity_threshold-3 ){PRINT_MATX16F( pose_update_cpu, ); }
			newPose					=	LieToP_Matx( pose_update_cpu )  *  pose;
			newK2K					=	K  *  newPose  * invK ;

			runcl.update_k2k_buf(		newK2K,		newPose);
																																		if( verbosity>local_verbosity_threshold ){
																																			cout << "\nDynamic_slam::estimate_tracking() chk_5: ,  layer = "<<layer<<"##########"<<flush;
																																			PRINT_MATX16F( deltas_matx[layer], );							PRINT_MATX16F( pose_update_cpu, );
																																			PRINT_MATX16F( PToLie( LieToP_Matx(pose_update_cpu).inv() ), );
																																			PRINT_MATX44F( newPose,	);										PRINT_MATX16F( PToLie( newPose ), );
																																			PRINT_MATX44F( newK2K,			);
																																			Matx44f	pose_now	= runcl.ReadOutput_44f( runcl.pose_buf );	PRINT_MATX44F( pose_now, );
																																			Matx44f	k2k_now		= runcl.ReadOutput_44f( runcl.k2kbuf);		PRINT_MATX44F( k2k_now,	);
																																		}
			if( SE_iter-(iter/10) < layer) {
				layer --;																											// Step down to lower layer of image pyramid
				cout << "\n( SE_iter-(iter/10) < layer),  layer = "	<<	layer <<endl<<flush;
				old_sum_rho_sq			=	FLT_MAX-1;
			}
		}auto step_1 = high_resolution_clock::now();																					if( verbosity>local_verbosity_threshold-3){
																																			cout << "\nDynamic_slam::estimate_tracking() loop finished  ###########################"\
																																			<<"Tracking loop time = "<<  duration_cast<microseconds>(step_1 - step_0).count()
																																			<<" microseconds,  layer="<<layer<<endl<<flush;
																																		}
	}
}


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
		if( layer>0){
			runcl.propagate_depth_next_layer(layer-1);
		}

		// NB this kernel would be faster if it used 1 thread per depth patch, ie 16 pixels. ... Maybe not. The existing method uses half as many threads, BUT benefits from contiguious reads of data.
	}
	// copy depth to tracking depth map ### TODO

}
