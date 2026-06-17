#include "Dynamic_slam.hpp"
#include <fstream>
#include <iomanip>   // for std::setprecision, std::setw

using namespace cv;
using namespace std;

void Dynamic_slam::estimate_calibration(){
	string fname = "Dynamic_slam::estimate_calibration()";
	int 	local_verbosity_threshold 		= V_DYNAMIC_SLAM_ESTIMATE_CALIBRATION;//verbosity_mp["Dynamic_slam::estimateSE3"];
																																if(verbosity>local_verbosity_threshold) {
																																	cout << "\nDynamic_slam::estimate_calibration() chk_0"
																																	<<"  ##############################################################"<< flush;
																																	for (int idx=0; idx<num_current_frames; idx++){
																																		cout<<"\nidx="<<idx<<flush;
																																		RunCL::frame	*this_frame	= &runcl.current_frames[runcl.current_frames_idx[idx]	];
																																		PRINT_MATX44F(	this_frame->pose_from_0,	);
																																		PRINT_MATX44F(	this_frame->pose_to_0,		);
																																		PRINT_MATX44F(	this_frame->K,				);
																																		PRINT_MATX44F(	this_frame->inv_K,			);
																																		PRINT_MATX44F(	this_frame->k2k_from_0,		);
																																		PRINT_MATX44F(	this_frame->k2k_to_0,		);
																																	}
																																}
	uint			layer						= 0;
	uint			frame_idx					= 4;
	RunCL::frame	*this_frame					= &runcl.current_frames[runcl.current_frames_idx[frame_idx]	];

	cl_float16		camera_matrix_k2k[			num_camera_matrix_DoF +1 ];
	generate_camera_matrix_k2k_vec(				this_frame->K, this_frame->pose_from_0, camera_matrix_k2k );
	runcl.precomp_cam_calib_maps (				layer, camera_matrix_k2k,	runcl.camera_matrix_map_mem,	num_camera_matrix_DoF, fname );
/*
	cl_float16		cam_param_weights			= {{0}};
	generate_cam_param_weights(					this_frame->pose_from_0,  cam_param_weights );
*/
	runcl.patch_cam_calib_Hessian (				layer );
	runcl.patch_cam_calib__Hessian_reduce (		layer,	this_frame->inv_camera_matrix_Hessian[layer],  this_frame->sum_elemwise_sq_J[layer] );
																																if(verbosity>local_verbosity_threshold) {
																																	cout	<<"\nDynamic_slam::estimate_calibration() chk_1"
																																			<<"\nframe_ixd = "								<<frame_idx
																																			<<"\nlayer = "									<<layer
																																			<<"\nruncl.current_frames_idx[frame_idx] = "	<<runcl.current_frames_idx[frame_idx]
																																			<<endl<<flush;
																																	PRINT_MATX55D( this_frame->inv_camera_matrix_Hessian[layer], );
																																}
	//////////
	estimate_camera_matrix( layer, frame_idx);

	// update SE_param_maps.... after K update.
}


void Dynamic_slam::generate_camera_matrix_k2k_vec( cv::Matx44f K, cv::Matx44f pose, cl_float16 _camera_matrix_k2k[ num_camera_matrix_DoF+1 ] ) {			// Generates a set of 5 "k2k" to be used to compute the camera_matrix maps for the current camera frame_to_frame transpose.
	int local_verbosity_threshold = V_DYNAMIC_SLAM_GENERATE_CAMERA_MATRIX_K2K;//verbosity_mp["Dynamic_slam::generate_SE3_k2k"];// -2;
																																			if(verbosity>local_verbosity_threshold) cout << "\nDynamic_slam::generate_camera_matrix_k2k_vec( float _SE3_k2k[6*16] ) chk_0" << endl << flush;
	cv::Matx44f d_k[	num_camera_matrix_DoF+1];
	cv::Matx44f d_inv_k[num_camera_matrix_DoF+1];
	cv::Matx44f cam_param_k2k[num_camera_matrix_DoF+1];


		for (int i=0; i<=num_camera_matrix_DoF; i++) {	d_k[i] = K;	}

		float step = 1.0f;
		d_k[0](0,0)	+=	step;	// change of focal length
		d_k[0](1,1)	+=	step;

		d_k[1](0,0)	+=	step;	// change of fx:fy ratio
		d_k[1](1,1)	-=	step;

		d_k[2](0,2)	+=	step;	// change of image centre Cx in pixel column
		d_k[3](1,2)	+=	step;	// change of image centre Cy in pixel row
		d_k[4](0,1)	+=	step;	// change of image skew
		//d_k[5]							// no change, reference k2k.

		for (int i=0; i<=num_camera_matrix_DoF; i++){			d_inv_k[i]		= generate_invK_( d_k[i] ); }

		for (int param=0; param<=num_camera_matrix_DoF; param++){	// NB include  am2cam[num_camera_matrix_DoF] = ref_k2k

			cam_param_k2k[param]			= d_k[param]  * pose  *  d_inv_k[param];

			Matx44f_To_cl_float16( cam_param_k2k[param],		_camera_matrix_k2k[  param] );
																																			if(verbosity>local_verbosity_threshold){
																																				cout<<"\n########################### i="<<param<<flush;
																																				PRINT_MATX44F( d_k[param],				);
																																				PRINT_MATX44F( d_inv_k[param],			);
																																				PRINT_MATX44F( pose,					);
																																				PRINT_MATX44F( cam_param_k2k[param],	);
																																			}
		}
																																			if(verbosity>local_verbosity_threshold) {
																																				cout << "\n\nDynamic_slam::generate_SE3_k2k_vec( float _SE3_k2k[6*16] )   finished" << endl << flush;
																																			}
}


void Dynamic_slam::generate_cam_param_weights( cv::Matx44f pose,  cl_float16 &cam_param_weights ){

	Matx16f		pose_algebra	=  PToLie( pose );

	float		rotx			= pose_algebra(0,0);
	float		roty			= pose_algebra(0,1);
	float		rotz			= pose_algebra(0,2);

	float		tx				= pose_algebra(0,3);
	float		ty				= pose_algebra(0,4);
	float		tz				= pose_algebra(0,5);

	//focal length
	cam_param_weights.s0		= sqrt( rotx*rotx 	+ roty*roty + rotz*rotz );
	cam_param_weights.s1		= sqrt( tx*tx 		+ ty*ty 	);

	// fx:fy
	cam_param_weights.s2		= sqrt( fabs(rotx*roty)	+ rotz*rotz	);
	cam_param_weights.s3		= sqrt( fabs(tx*ty) );

	// cx
	cam_param_weights.s4		= sqrt( rotx*rotx + rotz*rotz );
	cam_param_weights.s5		= fabs( tz);

	// cy
	cam_param_weights.s6		= sqrt( roty*roty + rotz*rotz );
	cam_param_weights.s7		= cam_param_weights.s5;

	// skew
	cam_param_weights.s8		= cam_param_weights.s0;
	cam_param_weights.s9		= fabs( ty);
																																			cout<< "\n\nDynamic_slam::generate_cam_param_weights() "<<flush;
																																			PRINT_MATX16F( pose_algebra, );
																																			cout<<"\n\ncam_param_weights = "
																																			<< cam_param_weights.s0 << ", "
																																			<< cam_param_weights.s1 << ",   "
																																			<< cam_param_weights.s2 << ", "
																																			<< cam_param_weights.s3 << ",   "
																																			<< cam_param_weights.s4 << ", "
																																			<< cam_param_weights.s5 << ",   "
																																			<< cam_param_weights.s6 << ", "
																																			<< cam_param_weights.s7 << ",   "
																																			<< cam_param_weights.s8 << ", "
																																			<< cam_param_weights.s9 <<endl<<flush;
}

void Dynamic_slam::estimate_camera_matrix( uint	layer, uint	frame_idx){
	string fname = "Dynamic_slam::_camera_matrix()";
	string fname_short = "est_cam";
	int 	local_verbosity_threshold 		= V_DYNAMIC_SLAM_ESTIMATE_CAMERA_MATRIX;//verbosity_mp["Dynamic_slam::estimateSE3"];
																																		if(verbosity>local_verbosity_threshold) {
																																			cout << "\n\nDynamic_slam::estimate_camera_matrix() chk_0"
																																			<<"  ##############################################################"<< flush;
																																			}
	//int		layer				= 4;//4;//SE3_start_layer;
	//int		frame_idx			= 1;
	uint	out_block_size		= 2;//4;
	bool	stop_loop			= false;
	float	old_sum_rho_sq		= FLT_MAX-1;
	float	factor				= -1.0f;

	Matx44f	old_k				= Matx44f::eye();
	Matx44f	old_inv_k			= Matx44f::eye();
	Matx44f	old_k2k				= Matx44f::eye();

	RunCL::frame 	*this_frame	= &runcl.current_frames[ runcl.current_frames_idx[frame_idx] ];

	Matx44f new_k				= this_frame->K;						// runcl.ReadOutput_44f( runcl.K_buf );
	Matx44f new_inv_k			= this_frame->inv_K;					// runcl.ReadOutput_44f( runcl.inv_K_buf );
	Matx44f newK2K				= this_frame->k2k_from_0;				// runcl.ReadOutput_44f( runcl.k2kbuf, cl_flt16_size );			// offset = current_frames_idx * cl_flt16_size
																																		//for (uint out_block_size = 4/*32*/; out_block_size > 2; out_block_size /=2){
	const uint	max_frames		= min(this_frame->frame_count+1, num_current_frames);
																																		if(verbosity>local_verbosity_threshold) {
																																			cout << "\n\nDynamic_slam::estimate_camera_matrix() chk_0.5"
																																			<<"  ##############################################################"<< flush;
																																		}
	for (uint iter = 0; iter<6; iter++){
		auto step_0									= high_resolution_clock::now();

		Matx15d param_update[num_current_frames]	= { Matx15d::zeros() };
		float	sum_rho[num_current_frames]			= {0.0f};
		float	sum_rho_sq[num_current_frames]		= {0.0f};
		float	num_pixels[num_current_frames]		= {0.0f};
		Matx15d sum_param_update					= Matx15d::zeros();
		float	sum_sum_rho_sq						= 0.0f;

		//for (int frame_idx = max_frames-1; frame_idx < max_frames; frame_idx++){
			RunCL::frame	*this_frame	= &runcl.current_frames[ runcl.current_frames_idx[frame_idx] ];
			Matx55d	invH				= this_frame->inv_camera_matrix_Hessian[layer];
																																		if(verbosity>local_verbosity_threshold) {
																																			cout << "\nDynamic_slam::estimate_camera_matrix() chk_1:   ###########################"
																																			<<"\nframe_ixd = "								<<frame_idx
																																			<<"\nruncl.current_frames_idx[frame_idx] = "	<<runcl.current_frames_idx[frame_idx]
																																			<<"\nlayer="									<< layer
																																			<<"\nout_block_size="							<< out_block_size
																																			<<"\niter="										<< iter
																																			<<"\nmax_frames="								<< max_frames
																																			<< flush;
																																			Matx44f	pose									= this_frame->pose_from_0;
																																			PRINT_MATX44F( pose,		);
																																			PRINT_MATX16F( PToLie(pose),);
																																			PRINT_MATX55D( invH, );
																																		}
			runcl.rho_sq_from_0(		out_block_size, iter, frame_idx,	(uint)layer,	num_camera_matrix_DoF, fname_short );
			runcl.reduce_patch_Rho(		out_block_size, iter, 				(uint)layer,	num_camera_matrix_DoF );
			runcl.get_rho_result(		runcl.camera_matrix_result,			(uint)layer,	num_camera_matrix_DoF );

			sum_rho[ 	frame_idx]		=	runcl.camera_matrix_result.Rho.x;															// currently .x colour channel only.
			sum_rho_sq[ frame_idx]		=	runcl.camera_matrix_result.Rho.y;

			num_pixels[ frame_idx]		=	runcl.camera_matrix_result.param_incr_arry[1];												// TO DO move numpixels to SE3_incr.w   & reduce SE3_incr_map_mem from float8 tro float4
			Matx15d	camera_matrix_incr;	for (int i=0; i<num_camera_matrix_DoF; i++){		camera_matrix_incr(i)	= runcl.camera_matrix_result.param_incr_arry[i*2];  };

			param_update[ frame_idx]	=	camera_matrix_incr * invH;																	// Double precision is required
																																		if( verbosity>local_verbosity_threshold ){
																																			cout << "\nDynamic_slam::estimate_camera_matrix() chk_3: ,  ###########################"<<endl<<flush;
																																			PRINT_MATX15D( param_update[ frame_idx], );
																																		}
			param_update[ frame_idx]	=	factor *  param_update[ frame_idx];
																																		if( verbosity>local_verbosity_threshold ){
																																			cout << "\nDynamic_slam::estimate_camera_matrix() chk_4: ,  ###########################"<<
																																			"\n sum_rho = "			<< sum_rho		<<
																																			",	sum_rho_sq	= "		<< sum_rho_sq	<<
																																			",	num_pixels = "		<< num_pixels	<<
																																			",  layer = "			<< layer		<< endl<<endl<<flush;
																																			PRINT_MATX15D( camera_matrix_incr, );
																																			PRINT_MATX55D( invH, );
																																			PRINT_MATX15D( param_update[ frame_idx], );
																																			PRINT_MATX15D( this_frame->sum_elemwise_sq_J[layer], );
																																		}
			sum_sum_rho_sq				+= sum_rho_sq[ frame_idx];
			sum_param_update			+= param_update[ frame_idx].mul(	this_frame->sum_elemwise_sq_J[layer]	);
		//}

		float len						= sqrt( powf(sum_param_update(0) ,2) + powf(sum_param_update(1) ,2) + powf(sum_param_update(2) ,2) + powf(sum_param_update(3) ,2) + powf(sum_param_update(4) ,2)  );
		//if(len>1.0f) sum_param_update	/= len;
																																		if( verbosity>local_verbosity_threshold ){
																																			cout << "\n\nDynamic_slam::estimate_camera_matrix() chk_5: ,  ###########################"<<
																																			"\niter="				<< iter				<<
																																			"\nsum_sum_rho_sq = "	<< sum_sum_rho_sq	<<
																																			"\nlen uppdate = "		<< len				<<endl<<flush;
																																			PRINT_MATX15D( (sum_param_update ), );								/*/(float)(num_current_frames-1)*/
																																		}
		if( isnan(sum_sum_rho_sq) ){											cout << "\nisnan(sum_rho_sq)" <<flush;
			stop_loop					= true;
			new_k						= old_k;
			new_inv_k					= old_inv_k;
		}else if(sum_sum_rho_sq > old_sum_rho_sq){
																				cout << "\n"<<sum_sum_rho_sq<<" = sum_sum_rho_sq > old_sum_rho_sq = "<< old_sum_rho_sq;
			stop_loop					= true;
			new_k						= old_k;
			new_inv_k					= old_inv_k;
		}else{
		old_sum_rho_sq					= sum_sum_rho_sq ;
		//sum_param_update				/= (float)max_frames;

		old_k							=	new_k;
		old_inv_k						=	new_inv_k;
		old_k2k							=	newK2K;

		new_k(0,0)						+= sum_param_update(0,0);	// f
		new_k(1,1)						+= sum_param_update(0,0);

		new_k(0,0)						+= sum_param_update(0,1);	// fx:fy
		new_k(1,1)						-= sum_param_update(0,1);

		new_k(0,2)						+= sum_param_update(0,2);	// cx
		new_k(1,2)						+= sum_param_update(0,3);	// cy

		new_k(0,1)						+= sum_param_update(0,4);	// skew

	//	new_k = frame_data.back().frame_data_GT.K;		/////   Ignores update and sets K to GT.  ##################################

		new_inv_k						= generate_invK_( new_k );
		}
		for (int frame_idx_ = 0; frame_idx_ < max_frames; frame_idx_++){
			RunCL::frame *this_frame	= &runcl.current_frames[ runcl.current_frames_idx[frame_idx_] ];

			Matx44f	pose				= this_frame->pose_from_0;
			Matx44f	old_k2k				= this_frame->k2k_from_0;
			newK2K						= new_k  *  pose  * new_inv_k ;

			runcl.update_44f_buf(		newK2K,		this_frame->k2k_buf_from_0,  fname );
			this_frame->K				= new_k;
			this_frame->inv_K			= new_inv_k;
																																		if( verbosity>local_verbosity_threshold && frame_idx_==frame_idx){
																																			cout 	<< "\nDynamic_slam::estimate_camera_matrix() chk_6: ,  layer = "<<layer
																																					<<",  frame_idx = "<<frame_idx_
																																					<<",  runcl.current_frames_idx["<<frame_idx_<<"] = "<<runcl.current_frames_idx[frame_idx_]
																																					<<",  ##########"<<flush;
																																			PRINT_MATX15D( sum_param_update, );
																																			PRINT_MATX44F( old_k,		 );
																																			PRINT_MATX44F( new_k,		 );
																																			PRINT_MATX44F( pose,		 );
																																			PRINT_MATX44F( old_k2k,		.);
																																			PRINT_MATX44F( newK2K,		 );
																																		}
		}
		auto step_1 = high_resolution_clock::now();																						if( verbosity>local_verbosity_threshold){
																																			cout << "\nDynamic_slam::estimate_camera_matrix() loop finished  ###########################"\
																																			<<"K Calibration loop time = "<<  duration_cast<microseconds>(step_1 - step_0).count()
																																			<<" microseconds,  layer="<<layer<<endl<<flush;
																																		}
		if (stop_loop == true) break;
	}
	frame_data.back().frame_data.K		=	new_k;																						// Needed for Dynamic_slam::nextFrame() -> runcl.set_cam_bufs(...)
	frame_data.back().frame_data.inv_K	=	new_inv_k;
	frame_data.back().frame_data.K2K	=	newK2K;
	/*
	// // Matx44f pose_temp 					= runcl.update_pose_bufs_cur_frames( pose );											// NB these two lines are req because nextFrame() calls  runcl.set_cam_bufs(..), using frame_data.
	// // frame_data.back().frame_data.pose 	= pose_temp ;
	// //
	// // float16arry_To_Matx44f(	 &runcl.current_frames[	runcl.current_frames_idx[0]	].k2k_0to1_est[0]	, frame_data.back().frame_data.K2K );
	// 																																	if(verbosity>local_verbosity_threshold) {
	// 																																		cout << "\n\nDynamic_slam::_camera_matrix() finished"<< flush;
	// 																																		PRINT_MATX44F(frame_data.back().frame_data.K,);
	// 																																		PRINT_MATX44F(frame_data.back().frame_data.pose,);
	// 																																		PRINT_MATX44F(frame_data.back().frame_data.K2K, );
	// 																																	}
	*/
}

void Dynamic_slam::focal_length_grad_descent(){
	string fname = "Dynamic_slam::focal_length_grad_descent()";
	string fname_short = "est_focal_len";
	int		local_verbosity_threshold 		= V_DYNAMIC_SLAM_FOCAL_LENGTH_GRAD_DESC;//verbosity_mp["Dynamic_slam::estimateSE3"];
																																if(verbosity>local_verbosity_threshold) {
																																	cout << "\nDynamic_slam::focal_length_grad_descent() chk_0"
																																	<<"  ##############################################################"<< flush;
																																}
	uint	out_block_size		= 2;
	uint	layer				= 4;
	bool	stop_loop			= false;
	float	old_sum_rho_sq		= FLT_MAX-1;
	float	factor				= -0.05f;

	Matx44f	old_k				= Matx44f::eye();
	Matx44f	old_inv_k			= Matx44f::eye();
	Matx44f	old_k2k				= Matx44f::eye();

	RunCL::frame 	*this_frame	= &runcl.current_frames[ runcl.current_frames_idx[0] ];

	//Matx55d	invH				= this_frame->inv_camera_matrix_Hessian[layer];															PRINT_MATX55D( invH, );
	Matx44f	pose				= this_frame->pose_from_0;				// runcl.ReadOutput_44f( runcl.pose_buf );
	Matx44f new_k				= this_frame->K;						// runcl.ReadOutput_44f( runcl.K_buf );
	Matx44f new_inv_k			= this_frame->inv_K;					// runcl.ReadOutput_44f( runcl.inv_K_buf );
	Matx44f newK2K				= this_frame->k2k_from_0;				// runcl.ReadOutput_44f( runcl.k2kbuf, cl_flt16_size );			// offset = current_frames_idx * cl_flt16_size
																																		//for (uint out_block_size = 4/*32*/; out_block_size > 2; out_block_size /=2){
	const uint	max_frames		= min(this_frame->frame_count+1, num_current_frames);


	for (uint iter = 0; iter<3; iter++){

		Matx15d param_update[num_current_frames]	= { Matx15d::zeros() };
		float	sum_rho[num_current_frames]			= {0.0f};
		float	sum_rho_sq[num_current_frames]		= {0.0f};
		float	num_pixels[num_current_frames]		= {0.0f};
		Matx15d sum_param_update					= Matx15d::zeros();
		float	sum_sum_rho_sq						= 0.0f;

		for (int frame_idx = max_frames-1; frame_idx < max_frames; frame_idx++){
			RunCL::frame	*this_frame	= &runcl.current_frames[ runcl.current_frames_idx[frame_idx] ];

			runcl.rho_sq_from_0(		out_block_size, iter, frame_idx,	(uint)layer,	num_camera_matrix_DoF, fname_short );
			runcl.reduce_patch_Rho(		out_block_size, iter,				(uint)layer,	num_camera_matrix_DoF );
			runcl.get_rho_result(		runcl.camera_matrix_result,			(uint)layer,	num_camera_matrix_DoF );

			sum_rho[ 	frame_idx]		=	runcl.camera_matrix_result.Rho.x;															// currently .x colour channel only.
			sum_rho_sq[ frame_idx]		=	runcl.camera_matrix_result.Rho.y;

			sum_sum_rho_sq				+= sum_rho_sq[ frame_idx];
			sum_param_update			+= param_update[ frame_idx];
		}

		if( isnan(sum_sum_rho_sq) ){											cout << "\nisnan(sum_rho_sq)" <<flush;
			stop_loop					= true;
			new_k						= old_k;
			new_inv_k					= old_inv_k;
		}else if(sum_sum_rho_sq > old_sum_rho_sq){
																				cout << "\n"<<sum_sum_rho_sq<<" = sum_sum_rho_sq > old_sum_rho_sq = "<< old_sum_rho_sq;
			stop_loop					= true;
			new_k						= old_k;
			new_inv_k					= old_inv_k;
		}else{
			old_sum_rho_sq				= sum_sum_rho_sq ;

			old_k						=	new_k;
			old_inv_k					=	new_inv_k;
			old_k2k						=	newK2K;

			new_k(0,0)					+= sum_param_update(0,0);	// f
			new_k(1,1)					+= sum_param_update(0,0);

			new_inv_k					= generate_invK_( new_k );
		}

		for (int frame_idx = 0; frame_idx < max_frames; frame_idx++){
			RunCL::frame *this_frame	= &runcl.current_frames[ runcl.current_frames_idx[frame_idx] ];

			Matx44f	pose				= this_frame->pose_from_0;
			Matx44f	old_k2k				= this_frame->k2k_from_0;
			newK2K						= new_k  *  pose  * new_inv_k ;

			runcl.update_44f_buf(		newK2K,		this_frame->k2k_buf_from_0,  fname );
			this_frame->K				= new_k;
			this_frame->inv_K			= new_inv_k;
		}

		if (stop_loop == true) break;
	}
	frame_data.back().frame_data.K		=	new_k;																						// Needed for Dynamic_slam::nextFrame() -> runcl.set_cam_bufs(...)
	frame_data.back().frame_data.inv_K	=	new_inv_k;
	frame_data.back().frame_data.K2K	=	newK2K;

}




void Dynamic_slam::estimate_lens_distortion(){



}
