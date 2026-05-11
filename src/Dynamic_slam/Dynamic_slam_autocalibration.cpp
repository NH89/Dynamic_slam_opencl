#include "Dynamic_slam.hpp"
#include <fstream>
#include <iomanip>   // for std::setprecision, std::setw

using namespace cv;
using namespace std;

void Dynamic_slam::estimate_calibration(){
	string fname = "Dynamic_slam::estimate_calibration(()";
	int 	local_verbosity_threshold 		= V_DYNAMIC_SLAM_ESTIMATE_CALIBRATION;//verbosity_mp["Dynamic_slam::estimateSE3"];
																																		if(verbosity>local_verbosity_threshold) {
																																			cout << "\fDynamic_slam::estimate_calibration(() chk_0"
																																			<<"  ##############################################################"<< flush;
																																		}
	precompute_cam_matrix_and_lens_distortion_buffers();
	estimate_camera_matrix();
	estimate_lens_distortion();
}

void Dynamic_slam::precompute_cam_matrix_and_lens_distortion_buffers(){			// needs to be run _after_ computing the SE3 transform,  if the aim is to find the actual values of K, rather than the change in K between frames.
	const int local_verbosity_threshold = V_DYNAMIC_SLAM_PRECOMPUTE_BUFFERS;
																																			if (verbosity>local_verbosity_threshold) { cout << "\nprecompute_cam_matrix_and_lens_distortion_buffers_chk 0:" <<flush;
																																				cout<<"\n frame_data.size() = "<<frame_data.size()<<flush;
																																			}
	// Camera intrinsic matrix
	float camera_matrix_k2k[  max_mipmap_layers* num_camera_matrix_DoF *16  ];
	generate_camera_matrix_k2k_vec( camera_matrix_k2k );
	runcl.precomp_param_maps ( camera_matrix_k2k,	runcl.camera_matrix_map_mem,	num_camera_matrix_DoF );

	// Lens distortion parameters



}

void Dynamic_slam::generate_camera_matrix_k2k_vec( float _camera_matrix_k2k[  max_mipmap_layers* num_camera_matrix_DoF *16  ] ) {			// Generates a set of 5 "k2k" to be used to compute the camera_matrix maps for the current camera frame_to_frame transpose.
	int local_verbosity_threshold = V_DYNAMIC_SLAM_GENERATE_SE3_K2K;//verbosity_mp["Dynamic_slam::generate_SE3_k2k"];// -2;
																																			if(verbosity>local_verbosity_threshold) cout << "\nDynamic_slam::generate_SE3_k2k_vec( float _SE3_k2k[6*16] ) chk_0" << endl << flush;
	cv::Matx44f d_k[	num_camera_matrix_DoF];
	cv::Matx44f d_inv_k[num_camera_matrix_DoF];
	cv::Matx44f cam2cam[num_camera_matrix_DoF];

	for(int layer=0; layer<max_mipmap_layers; layer++){

		for (int i=0; i<num_camera_matrix_DoF; i++) {			d_k[i] = initial_K;	}	// start by copying initial estimate of camera matrix.  TODO ? Could change to current camera matrix - chk if pixel motion maps are different.

		d_k[0].operator()(0,0)	+=	1;	// change of focal length
		d_k[0].operator()(1,1)	+=	1;
		d_k[1].operator()(0,0)	+=	1;	// change of fx:fy ratio
		d_k[1].operator()(1,1)	-=	1;
		d_k[2].operator()(0,2)	+=	1;	// change of image centre Cx in pixel column
		d_k[3].operator()(1,2)	+=	1;	// change of image centre Cy in pixel row
		d_k[4].operator()(0,1)	+=	1;	// change of image skew

		for (int i=0; i<num_camera_matrix_DoF; i++){			d_inv_k[i]		= generate_invK_( d_k[i] ); }

		for (int i=0; i<num_camera_matrix_DoF; i++) {

			cam2cam[i] 			= d_k[i]  * frame_data.back().frame_data.pose  *  d_inv_k[i];

			for (uint row=0; row<4; row++) {
				for (uint col=0; col<4; col++){
					_camera_matrix_k2k[ ((layer * num_camera_matrix_DoF) + i)*16 + row*4 + col]		= cam2cam[i].operator()(row,col);
				}
			}
		}
	}
																																			if(verbosity>local_verbosity_threshold) {
																																				cout << "\n\nDynamic_slam::generate_SE3_k2k_vec( float _SE3_k2k[6*16] )   finished" << endl << flush;
																																			}
}

void Dynamic_slam::estimate_camera_matrix(){
	string fname = "Dynamic_slam::_camera_matrix()";
	int 	local_verbosity_threshold 		= V_DYNAMIC_SLAM_ESTIMATE_TRACKING;//verbosity_mp["Dynamic_slam::estimateSE3"];
																																		if(verbosity>local_verbosity_threshold) {
																																			cout << "\fDynamic_slam::_camera_matrix() chk_0"
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
																																			cout << "\nDynamic_slam::_camera_matrix() chk_1: layer="<<layer
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
																																			cout << "\nDynamic_slam::_camera_matrix() chk_4: ,  ###########################"<<
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
																																			cout << "\nDynamic_slam::_camera_matrix() chk_5: ,  layer = "<<layer<<"##########"<<flush;
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
																																			cout << "\nDynamic_slam::_camera_matrix() loop finished  ###########################"\
																																			<<"Tracking loop time = "<<  duration_cast<microseconds>(step_1 - step_0).count()
																																			<<" microseconds,  layer="<<layer<<endl<<flush;
																																		}
	}

	Matx44f pose_temp 					= runcl.update_pose_bufs_cur_frames( );											// NB these two lines are req because nextFrame() calls  runcl.set_cam_bufs(..), using frame_data.
	frame_data.back().frame_data.pose 	= pose_temp ;

	float16arry_To_Matx44f(	 &runcl.current_frames[	runcl.current_frames_idx[0]	].k2k_0to1_est[0]	, frame_data.back().frame_data.K2K );
																																		if(verbosity>local_verbosity_threshold) {
																																			cout << "\fDynamic_slam::_camera_matrix() finished"<< flush;
																																			PRINT_MATX44F(frame_data.back().frame_data.pose,);
																																			PRINT_MATX44F(frame_data.back().frame_data.K2K, );
																																		}
}


void Dynamic_slam::estimate_lens_distortion(){



}
