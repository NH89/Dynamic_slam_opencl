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

	uint layer =  4 ;
	runcl.patch_cam_and_lens_Hessian(			layer, runcl.camera_matrix_map_mem,				runcl.camera_matrix_grad_map_mem,			runcl.camera_matrix_hessian_map_mem );
	runcl.patch_cam_and_lens__hessian_reduce(	layer, runcl.camera_matrix_hessian_map_mem,		runcl.current_frames[ runcl.current_frames_idx[0] ].inv_camera_matrix_Hessian[layer] );
																																			PRINT_MATX55D( runcl.current_frames[ runcl.current_frames_idx[0] ].inv_camera_matrix_Hessian[layer], );
	//////////
	estimate_camera_matrix();


//	estimate_lens_distortion();
}

void Dynamic_slam::precompute_cam_matrix_and_lens_distortion_buffers(){			// needs to be run _after_ computing the SE3 transform,  if the aim is to find the actual values of K, rather than the change in K between frames.
	string fname = "precompute_cam_lens_buffers";
	const int local_verbosity_threshold = V_DYNAMIC_SLAM_PRECOMPUTE_CAM_LENS_BUFFERS;
																																			if (verbosity>local_verbosity_threshold) { cout << "\nprecompute_cam_matrix_and_lens_distortion_buffers_chk 0:" <<flush;
																																				cout<<"\n frame_data.size() = "<<frame_data.size()<<flush;
																																			}
	// Camera intrinsic matrix

	cl_float16 camera_matrix_k2k[  max_mipmap_layers* (num_camera_matrix_DoF +1)  ];
	generate_camera_matrix_k2k_vec( camera_matrix_k2k );
	runcl.precomp_cam_and_lens_maps ( camera_matrix_k2k,	runcl.camera_matrix_map_mem,	num_camera_matrix_DoF, fname );

	// Lens distortion parameters	### TODO
	//generate_lens_distortion_vec(..);
	//runcl.precomp_lens_distortion_maps(  lens_distortion_map_mem... k2k..);	// use existing k2k, with lens distortion increments.
}

void Dynamic_slam::generate_camera_matrix_k2k_vec( cl_float16 _camera_matrix_k2k[  max_mipmap_layers* (num_camera_matrix_DoF +1)  ] ) {			// Generates a set of 5 "k2k" to be used to compute the camera_matrix maps for the current camera frame_to_frame transpose.
	int local_verbosity_threshold = V_DYNAMIC_SLAM_GENERATE_CAMERA_MATRIX_K2K;//verbosity_mp["Dynamic_slam::generate_SE3_k2k"];// -2;
																																			if(verbosity>local_verbosity_threshold) cout << "\nDynamic_slam::generate_camera_matrix_k2k_vec( float _SE3_k2k[6*16] ) chk_0" << endl << flush;
	cv::Matx44f d_k[	num_camera_matrix_DoF+1];
	cv::Matx44f d_inv_k[num_camera_matrix_DoF+1];
	cv::Matx44f cam2cam[num_camera_matrix_DoF+1];

	for(int layer=0; layer<max_mipmap_layers; layer++){

		//cam2cam[num_camera_matrix_DoF] 			= initial_K  * frame_data.back().frame_data.pose  *  inv_initial_K;							// Store current k2k in last Matx44f of the array. // ### TODO use the true current k & inv_k.

		for (int i=0; i<=num_camera_matrix_DoF; i++) {			d_k[i] = initial_K;	}	// start by copying initial estimate of camera matrix.  TODO ? Could change to current camera matrix - chk if pixel motion maps are different.

		d_k[0].operator()(0,0)	+=	1;	// change of focal length
		d_k[0].operator()(1,1)	+=	1;
		d_k[1].operator()(0,0)	+=	1;	// change of fx:fy ratio
		d_k[1].operator()(1,1)	-=	1;
		d_k[2].operator()(0,2)	+=	1;	// change of image centre Cx in pixel column
		d_k[3].operator()(1,2)	+=	1;	// change of image centre Cy in pixel row
		d_k[4].operator()(0,1)	+=	1;	// change of image skew
		//d_k[5]						// no change, reference k2k.

		for (int i=0; i<=num_camera_matrix_DoF; i++){			d_inv_k[i]		= generate_invK_( d_k[i] ); }

		for (int i=0; i<=num_camera_matrix_DoF; i++){	// NB include  am2cam[num_camera_matrix_DoF] = ref_k2k

			cam2cam[i] 			= d_k[i]  * frame_data.back().frame_data.pose  *  d_inv_k[i];

			Matx44f_To_cl_float16( cam2cam[i],		_camera_matrix_k2k[ (layer * (num_camera_matrix_DoF+1)) + i] );
																																			if(verbosity>local_verbosity_threshold){
																																				cout<<"\n########################### i="<<i<<flush;
																																				PRINT_MATX44F( d_k[i], );
																																				PRINT_MATX44F( d_inv_k[i], );
																																				PRINT_MATX44F( frame_data.back().frame_data.pose, );
																																				PRINT_MATX44F( cam2cam[i], );
																																			}
		}
	}
																																			if(verbosity>local_verbosity_threshold) {
																																				cout << "\n\nDynamic_slam::generate_SE3_k2k_vec( float _SE3_k2k[6*16] )   finished" << endl << flush;
																																			}
}

void Dynamic_slam::estimate_camera_matrix(){
	string fname = "Dynamic_slam::_camera_matrix()";
	string fname_short = "est_cam";
	int 	local_verbosity_threshold 		= V_DYNAMIC_SLAM_ESTIMATE_CAMERA_MATRIX;//verbosity_mp["Dynamic_slam::estimateSE3"];
																																		if(verbosity>local_verbosity_threshold) {
																																			cout << "\n\nDynamic_slam::_camera_matrix() chk_0"
																																			<<"  ##############################################################"<< flush;
																																		}
	int		layer 				= 4;//SE3_start_layer;
	uint 	frame_idx			= 1;
	uint 	out_block_size 		= 4;
	float	old_sum_rho_sq		= FLT_MAX-1;
	float	factor				= -2.0f;
	Matx44f	old_k				= Matx44f::eye();
	Matx44f	old_inv_k			= Matx44f::eye();
	Matx44f	old_k2k				= Matx44f::eye();

	RunCL::frame 	this_frame	= runcl.current_frames[ runcl.current_frames_idx[0] ];

	Matx44f	pose				= this_frame.pose_from_0;				// runcl.ReadOutput_44f( runcl.pose_buf );
	Matx44f new_k				= this_frame.K;							// runcl.ReadOutput_44f( runcl.K_buf );
	Matx44f new_inv_k			= this_frame.inv_K;						// runcl.ReadOutput_44f( runcl.inv_K_buf );
	Matx44f newK2K				= this_frame.k2k_from_0;				// runcl.ReadOutput_44f( runcl.k2kbuf, cl_flt16_size ); 		// offset = current_frames_idx * cl_flt16_size
																																		//for (uint out_block_size = 4/*32*/; out_block_size > 2; out_block_size /=2){
	for (uint iter = 0; iter<SE_iter; iter++){
		auto step_0 = high_resolution_clock::now();
																																		if(verbosity>local_verbosity_threshold) {
																																			cout << "\nDynamic_slam::_camera_matrix() chk_1: layer="<<layer
																																			<<", out_block_size="<<out_block_size<<",  iter="<<iter<<",  ###########################"<<flush;
																																			uint	out_block_size		= 2;
																																			uint	layer				= 0;
																																			runcl.rho_sq_from_0( out_block_size, iter, frame_idx, layer, num_camera_matrix_DoF, fname_short );// For debugging, get a larger, finer Rho map
																																			PRINT_MATX44F( old_k2k, ); PRINT_MATX44F( old_k, );
																																		}
		runcl.rho_sq_from_0( 		out_block_size, iter, frame_idx,	(uint)layer,  				num_camera_matrix_DoF, fname_short );
		runcl.reduce_patch_Rho( 	out_block_size, iter, 				(uint)layer,				num_camera_matrix_DoF );
		runcl.get_rho_result(		runcl.camera_matrix_result,			(uint)layer,				num_camera_matrix_DoF );

		float		sum_rho			=	runcl.camera_matrix_result.Rho.x;																// currently .x colour channel only.
		float		sum_rho_sq		=	runcl.camera_matrix_result.Rho.y;
		if( isnan(sum_rho_sq) ){											cout << "\nisnan(sum_rho_sq)" <<flush;
			break;
		}else if(sum_rho_sq > old_sum_rho_sq){																							// Rho, photometric error, got worse not better
			if(layer<=0) {break;}																										// Reached bottom of image pyramid.
			else {															cout << "\nsum_rho_sq > old_sum_rho_sq = "<< old_sum_rho_sq;
				if(factor<-1.0f){																										// End amplified steps
					factor 			=	-1.0f;								cout << ",  factor -2.0f -> -1.0f";
				}else {
					layer--;												cout << "\nlayer = "	<<	layer;
					old_sum_rho_sq	=	FLT_MAX-1;																						// Re-set old_sum_rho_sq for new layer
				}																														PRINT_MATX44F( old_k2k, ); PRINT_MATX44F( old_k, ); PRINT_MATX44F( old_inv_k, );
				runcl.update_44f_buf(	old_k2k,	this_frame.k2k_buf_from_0,	fname );												// Re-set to previous k, camera matrix
				this_frame.K		= old_k;
				this_frame.inv_K	= old_inv_k;
																			cout << endl << flush;
			}
		}else{
			old_sum_rho_sq			=	sum_rho_sq;
			old_k					=	new_k;
			old_inv_k				=	new_inv_k;
			old_k2k					=	newK2K;

			float		num_pixels	=	runcl.camera_matrix_result.param_incr_arry[1];													// TO DO move numpixels to SE3_incr.w   & reduce SE3_incr_map_mem from float8 tro float4
			Matx15d		camera_matrix_incr;	for (int i=0;	i<num_camera_matrix_DoF; i++){ camera_matrix_incr.operator()(i)	=	runcl.camera_matrix_result.param_incr_arry[i*2];  };
																																		if( verbosity>local_verbosity_threshold ){
																																			cout << "\nDynamic_slam::_camera_matrix() chk_4: ,  ###########################"<<
																																			"\n sum_rho = "			<< sum_rho		<<
																																			",	sum_rho_sq	= "		<< sum_rho_sq	<<
																																			",	num_pixels = "		<< num_pixels	<< endl<<flush;
																																			PRINT_MATX15D( camera_matrix_incr, );
																																			cout<<endl<<flush;
																																		}
			Matx55d	invH			=	runcl.current_frames[ runcl.current_frames_idx[0] ].inv_camera_matrix_Hessian[layer];
			Matx15d param_update	=	camera_matrix_incr * invH;																		// Double precision is required
																																		if( verbosity>local_verbosity_threshold ){
																																			cout << "\nlayer = "					<<layer					<<endl<<flush;
																																			cout << "\ncamera_matrix_incr="			<<camera_matrix_incr	<<endl<<flush;
																																			cout << "\ninv_camera_matrix_Hessian="	<<invH					<<endl<<flush;
																																			cout << "\nparam_update="				<<param_update			<<endl<<flush;
																																			PRINT_MATX55D( invH, );
																																			PRINT_MATX15D( param_update, );
																																			//Matx44f	k_old		= runcl.ReadOutput_44f( runcl.K_buf );		PRINT_MATX44F( k_old, );
																																			//Matx44f	k2k_old		= runcl.ReadOutput_44f( runcl.k2kbuf);		PRINT_MATX44F( k2k_old,	);
																																		}
			param_update			=	factor *  param_update;	//.mul( deltas_matx[layer] );/*(-2.0f)*/ /*  * 0.5f; */  				//NB matx.mul(  matx ) => elementwise multiplication.
																																		if( verbosity>local_verbosity_threshold-3 ){PRINT_MATX15D( param_update, ); }
			new_k(0,0)				+= param_update(0,0);	// f
			new_k(1,1)				+= param_update(0,0);

			new_k(0,0)				+= param_update(0,1);	// fx:fy
			new_k(1,1)				-= param_update(0,1);

			new_k(0,2)				+= param_update(0,2);	// cx
			new_k(1,2)				+= param_update(0,3);	// cy

			new_k(0,1)				+= param_update(0,4);	// skew

			new_inv_k				= generate_invK_( new_k );
			newK2K					= new_k  *  pose  * new_inv_k ;

			//runcl.update_k_buf(		newK2K,		new_k, new_inv_k);// may need to include   offset = frame_idx*cl_flt16_size

			runcl.update_44f_buf(	newK2K,		this_frame.k2k_buf_from_0,  fname );
			this_frame.K			= new_k;
			this_frame.inv_K		= new_inv_k;
																																		if( verbosity>local_verbosity_threshold ){
																																			cout << "\nDynamic_slam::_camera_matrix() chk_5: ,  layer = "<<layer<<"##########"<<flush;
																																			PRINT_MATX15D( param_update, );
																																			PRINT_MATX44F( new_k,		 );
																																			PRINT_MATX44F( newK2K,		 );
																																			//Matx44f	pose_now	= runcl.ReadOutput_44f( runcl.pose_buf );							PRINT_MATX44F( pose_now, );
																																			//Matx44f	k2k_now		= runcl.ReadOutput_44f( runcl.k2kbuf, frame_idx*cl_flt16_size);		PRINT_MATX44F( k2k_now,	);
																																		}
			if( SE_iter-(iter/10) < layer) {
				layer --;																											// Step down to lower layer of image pyramid
				cout << "\n( SE_iter-(iter/10) < layer),  layer = "	<<	layer <<endl<<flush;
				old_sum_rho_sq			=	FLT_MAX-1;
			}
		}auto step_1 = high_resolution_clock::now();																					if( verbosity>local_verbosity_threshold-3){
																																			cout << "\nDynamic_slam::_camera_matrix() loop finished  ###########################"\
																																			<<"K Calibration loop time = "<<  duration_cast<microseconds>(step_1 - step_0).count()
																																			<<" microseconds,  layer="<<layer<<endl<<flush;
																																		}
	}

	// Matx44f pose_temp 					= runcl.update_pose_bufs_cur_frames( pose );											// NB these two lines are req because nextFrame() calls  runcl.set_cam_bufs(..), using frame_data.
	// frame_data.back().frame_data.pose 	= pose_temp ;
	//
	// float16arry_To_Matx44f(	 &runcl.current_frames[	runcl.current_frames_idx[0]	].k2k_0to1_est[0]	, frame_data.back().frame_data.K2K );
																																		if(verbosity>local_verbosity_threshold) {
																																			cout << "\fDynamic_slam::_camera_matrix() finished"<< flush;
																																			PRINT_MATX44F(frame_data.back().frame_data.pose,);
																																			PRINT_MATX44F(frame_data.back().frame_data.K2K, );
																																		}
}


void Dynamic_slam::estimate_lens_distortion(){



}
