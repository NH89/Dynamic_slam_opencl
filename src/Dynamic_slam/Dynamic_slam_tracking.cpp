#include "Dynamic_slam.hpp"
#include <fstream>
#include <iomanip>   // for std::setprecision, std::setw

using namespace cv;
using namespace std;

void Dynamic_slam::precompute_SE3_buffers(){	// For Lucas-Kanade inverse compositional optimization of (1) tracking (2) camera calibration, (3) lens distortion.
	string fname = "precompute_SE3_buffers()";
	const int local_verbosity_threshold = V_DYNAMIC_SLAM_PRECOMPUTE_BUFFERS;
																																			if (verbosity>local_verbosity_threshold) { cout << "\nprecompute_SE3_buffers_chk 0:" <<flush;
																																				cout<<"\n frame_data.size() = "<<frame_data.size()<<flush;
																																			}
	// SE3 tracking
	generate_SE3_deltas();																													// depends on f &=> camera_intrinsic_matrix
	generate_SE3_k2k_vec( SE3_k2k );																										// fills float[96] ie 6xfloat[16] from conf.json intrinsic camera matrix + SE3 increments.
/*
	// cout<<"\nprecompute_SE3_buffers_chk 1, SE3_k2k = \n"
	// 					<<SE3_k2k[0]<<", "<<SE3_k2k[1]<<", "<<SE3_k2k[2]<<", "<<SE3_k2k[3]<<", \n"
	// 					<<SE3_k2k[4]<<", "<<SE3_k2k[5]<<", "<<SE3_k2k[6]<<", "<<SE3_k2k[7]<<", \n"
	// 					<<SE3_k2k[8]<<", "<<SE3_k2k[9]<<", "<<SE3_k2k[10]<<", "<<SE3_k2k[11]<<flush;
*/
	runcl.precomp_SE3_param_maps ( SE3_k2k, 	runcl.SE3_map_mem,	num_SE3_DoF, fname );														// GPU computes J(u,v/SE3) Jacobian of optical flow wrt SE3.
}

void Dynamic_slam::generate_SE3_deltas(){	// Principle : delta for each parameter causes maximum 1 pixel of warp in the full size image.
										// i.e. when computing J = (d_warp/d_param) * img_grad, only the difference betwen neigbouring pixels counts.
										// NB images should be blurred to eliminate noise and bilinear interpolation artefacts.
	int local_verbosity_threshold = V_DYNAMIC_GENERATE_DELTAS;
																																			if (verbosity>local_verbosity_threshold) { cout << "\nDynamic_slam::generate_deltas()_chk 0:"<<flush;}
	f								= fmaxf(	frame_data.back().frame_data.K(0,0),	frame_data.back().frame_data.K(1,1)	);	//(initial_K.operator()(0,0),	initial_K.operator()(1,1)	);							// NB [0]&[4] are u,v focal length
	float min_depth					= obj["min_depth"].asFloat();

	for (int layer=0; layer<max_mipmap_layers; layer++){
		float factor				= 1.0f; //pow(2,layer)
		delta[layer]				= factor ;
		delta_theta[layer]			= factor * 1/f;
		cos_theta[layer]			= cos(delta_theta[layer]);
		sin_theta[layer]			= sin(delta_theta[layer]);
		delta_depth[layer]			= factor * 	f * 2.0f 	/ ( min_depth * 	fmaxf( 	frame_data.back().frame_data.K(0,2),	frame_data.back().frame_data.K(1,2)		)	);//(obj["cameraMatrix"][2].asFloat(),	obj["cameraMatrix"][5].asFloat() )  );	// NB [2]&[5] are image sensor size

		deltas_matx[layer]			= { delta_theta[layer], delta_theta[layer], delta_theta[layer], delta[layer], delta[layer], delta[layer] };

																																			if (verbosity>local_verbosity_threshold) { cout<<endl
																																				<<"\nlayer			= "<<layer
																																				<<"\nf				= "<<f
																																				<<"\nmin_depth		= "<<min_depth
																																				<<"\ndelta			= "<<delta[layer]
																																				<<"\ndelta_theta	= "<<delta_theta[layer]
																																				<<"\ncos_theta		= "<<cos_theta[layer]
																																				<<"\nsin_theta		= "<<sin_theta[layer]
																																				<<"\ndelta_depth	= "<<delta_depth[layer]
																																				<<"\ndeltas_matx	= "<<deltas_matx[layer]
																																				<<"\nDynamic_slam::generate_deltas()_finished"<<flush;
																																			}
	}
}

void Dynamic_slam::generate_SE3_k2k_vec( float _SE3_k2k[  max_mipmap_layers* num_SE3_DoF *16  ] ) {																			// Generates a set of 6 k2k to be used to compute the SE3 maps for the current camera intrinsic matrix.
	int local_verbosity_threshold = V_DYNAMIC_SLAM_GENERATE_SE3_K2K;//verbosity_mp["Dynamic_slam::generate_SE3_k2k"];// -2;
																																			if(verbosity>local_verbosity_threshold) cout << "\nDynamic_slam::generate_SE3_k2k_vec( float _SE3_k2k[6*16] ) chk_0" << endl << flush;
	// SE3
	//const float res			= ( obj["cameraMatrix"][2].asFloat() + obj["cameraMatrix"][5].asFloat() ) /2.0;
	// const float f			= ( obj["cameraMatrix"][0].asFloat() + obj["cameraMatrix"][4].asFloat() ) /2.0;									// focal length in pixels.
	// const float delta 	  	= obj["ST3_delta"].asFloat() * obj["min_depth"].asFloat()  / f ;												// ST3_delta * (Translation to cause 1 pixel of parallax at min_depth)  	//1.0;//0.01; //0.001;  //  * obj["min_depth"].asFloat()
	// const float delta_theta = obj["SO3_delta_theta"].asFloat() / f;																			// SO3_delta_theta * (Rotation to cause 1 pixel of rotation flow) //0.01; //0.001;
	// const float cos_theta   = cos(delta_theta);
	// const float sin_theta   = sin(delta_theta);
																																			// Old :  Rotate 0.01 radians i.e 0.573  degrees.  Translate 0.001 'units' of distance
																																			if(verbosity>local_verbosity_threshold){ cout << "\nDynamic_slam::generate_SE3_k2k_vec( ) chk_1,"<<endl << flush;
																																				print_json_float_9(obj, "cameraMatrix");										cout	<<endl << flush;
																																				cout << "  delta_theta = "	<<delta_theta	<< " radians," 								<<endl << flush;
																																				cout << "  delta = "		<<delta			<< " units distance," 						<<endl << flush;
																																				cout << "  f = "			<<f				<< " pixels," 								<<endl << flush;
																																				//cout << "  obj[\"ST3_delta\"] =  "            <<obj["ST3_delta"].asFloat()				<<endl << flush;
																																				cout << "  obj[\"min_depth\"] =  "            <<obj["min_depth"].asFloat()				<<endl << flush;
																																				//cout << "  obj[\"SO3_delta_theta\"] =  "      <<obj["SO3_delta_theta"].asFloat()		<<endl << flush;
																																			}
	//Identity =				(1,			0,			0,			0,  			0,			1,			0,			0,  			0,			0,			1,			0,  			0,	0,	0,	1);
	cv::Matx44f transform[6];

	cv::Matx44f cam2cam[6];
																																			if(verbosity>local_verbosity_threshold){
																																				if(GT_available==true) {
																																				PRINT_MATX44F(frame_data.back().frame_data_GT.K,);
																																				PRINT_MATX44F(frame_data.back().frame_data_GT.inv_K,);
																																				}
																																				PRINT_MATX44F(frame_data.back().frame_data.K,);
																																				PRINT_MATX44F(frame_data.back().frame_data.inv_K,);
																																				cout<<"\nuse_conf_camera_matx = "<<use_conf_camera_matx<<flush;
																																			}
	for(int layer=0; layer<max_mipmap_layers; layer++){
		float _cos_theta	= cos_theta[layer];
		float _sin_theta	= sin_theta[layer];
		float _delta		= delta[layer];
																																			if(verbosity>local_verbosity_threshold){  cout<<"\nDynamic_slam::generate_SE3_k2k_vec(..)"
																																				<<",  layer  =	"		<<layer
																																				<<",  _cos_theta  = "	<<_cos_theta
																																				<<",  _sin_theta  = "	<<_sin_theta
																																				<<",  _delta  = "	 	<<_delta
																																				<<",  ############"<<endl<<flush;
																																			}
		transform[Rx] = cv::Matx44f(1,				0,				0,				0,\
									0,				_cos_theta,		-_sin_theta,	0,\
									0,				_sin_theta,		_cos_theta,		0,\
									0,				0,				0,				1);

		transform[Ry] = cv::Matx44f(_cos_theta,		0,				_sin_theta,		0,\
									0,				1,				0,				0,\
									-_sin_theta,	0,				_cos_theta,		0,\
									0,				0,				0,				1);

		transform[Rz] = cv::Matx44f(_cos_theta,		-_sin_theta,	0,				0,\
									_sin_theta,		_cos_theta,		0,				0,\
									0,				0,				1,				0,\
									0,				0,				0,				1);

		transform[Tx] = cv::Matx44f(1,0,0,_delta, 	0,1,0,0,		0,0,1,0,		0,0,0,1);
		transform[Ty] = cv::Matx44f(1,0,0,0, 		0,1,0,_delta,	0,0,1,0,		0,0,0,1);
		transform[Tz] = cv::Matx44f(1,0,0,0, 		0,1,0,0,		0,0,1,_delta,	0,0,0,1);


		for (int i=0; i<num_SE3_DoF; i++) {
			Matx16f incr = Matx16f::zeros();
			if (i<3){	incr(i) = 1/f;}
			else{		incr(i) = 1.0f; }		// same as above.

			Matx44f transform_	=  LieToP_Matx( incr /*deltas_matx[layer]*/ );
			cam2cam[i] 			= frame_data.back().frame_data.K  * transform_ /*transform[i]*/ *  frame_data.back().frame_data.inv_K;
																																			if(verbosity>local_verbosity_threshold) {
																																				PRINT_MATX16F( incr, );						/*deltas_matx[layer]*/
																																				PRINT_MATX44F( transform_,  SE3_incr );
																																				cout << "\ni=" << i << endl;
																																				PRINT_MATX44F(transform[i],);
																																				PRINT_MATX44F(cam2cam[i], _SE3_k2k);
																																			}
			for (uint row=0; row<4; row++) {
				for (uint col=0; col<4; col++){
					_SE3_k2k[ ((layer * num_SE3_DoF) + i)*16 + row*4 + col]		= cam2cam[i].operator()(row,col);
				}
			}
		}
	}
																																			if(verbosity>local_verbosity_threshold) {
																																				cout<<"check SE3_k2k #############  "<<flush;
																																				for(int layer=0; layer<max_mipmap_layers; layer++){
																																					cout<<"\n\nlayer = "<< layer <<"  ###############"<<flush;
																																					for (int i=0; i<num_SE3_DoF; i++) {
																																						cout<<"\nSE3dof = "<<i<<"  SE3_k2k = "<<flush;
																																						for (uint row=0; row<4; row++) {
																																							cout<<"\n(";
																																							for (uint col=0; col<4; col++){
																																								cout<<", "<<SE3_k2k[((layer * num_SE3_DoF) + i)*16 + row*4 + col];
																																							}
																																							cout<<"),"<<flush;
																																						}
																																					}
																																				}
																																				/*
																																				cout << endl << setprecision(9);
																																				// for (int i=0; i<6; i++) {
																																				// 	cout << "\n _SE3_k2k ["<<i<<"*16 + row*4 + col]=\n";
																																				// 	for (int row=0; row<4; row++) {
																																				// 		for (int col=0; col<4; col++){
																																				// 			cout << setw(6) << _SE3_k2k[i*16 + row*4 + col] <<"\t  ";
																																				// 		}cout<<endl;
																																				// 	}cout<<endl;
																																				// }

																																				// Chk k2k makes sense for each SE3 DoF
																																				// Image size 640 * 480, from K as loaded from .json file.
																																				// Homogeneous image coords : (col, row, depth, w), so ater transformation divide by w to find new (col, row, depth, 1).
																																				// NB if there are points at infinite distance, then w=0.
																																				// For this reason we store inverse depth, and use a minimum depth cut off, to avoid points at the optical centre of the camera.
																																				// See __kernel void compute_param_maps(..) , or hand calculate the elements of the matrix multiplication.
																																				cv::Matx14f topleft = {10,10,1,1},  topright = {630,10,1,1}, centre = {320,240,1,1}, bottomleft = {10,470,1,1}, bottomright = {630,470,1,1} , result;	// (column, row, w, 1/depth)
																																				cv::Matx44f identity = {1,0,0,0,  0,1,0,0,  0,0,1,0,  0,0,0,1};

																																				cout << "\n topleft * 	identity = " << topleft * 		identity << flush;
																																				cout << "\n topright * 	identity = " << topright * 		identity << flush;
																																				cout << "\n bottomleft *  identity = " << bottomleft * 	identity << flush;
																																				cout << "\n bottomright * identity = " << bottomright * identity << flush;
																																				*/
																																				cout << "\n\nDynamic_slam::generate_SE3_k2k_vec( float _SE3_k2k[6*16] )   finished" << endl << flush;
																																			}
}

void Dynamic_slam::estimate_tracking(){
	string fname = "Dynamic_slam::estimate_tracking()";
	string fname_short = "est_tracking()";
	int 	local_verbosity_threshold 		= V_DYNAMIC_SLAM_ESTIMATE_TRACKING;//verbosity_mp["Dynamic_slam::estimateSE3"];
																																		if(verbosity>local_verbosity_threshold) {
																																			cout << "\nDynamic_slam::estimate_tracking() chk_0"
																																			<<"  ##############################################################"<< flush;
																																		}
	uint			out_block_size		= 2;


	Matx44f			old_pose			= Matx44f::eye();
	Matx44f			old_k2k				= Matx44f::eye();
	RunCL::frame	*frame1				= &runcl.current_frames[ runcl.current_frames_idx[1] ];
	RunCL::frame	*frame2				= &runcl.current_frames[ runcl.current_frames_idx[2] ];
	Matx44f			invK				= frame1->inv_K;
	Matx44f			K					= frame1->K;
	int				max_frames			= min( (1+runcl.current_frames[ runcl.current_frames_idx[0] ].frame_count),		num_current_frames);
																																		if(verbosity>local_verbosity_threshold) {
																																			cout << "\nDynamic_slam::estimate_tracking() chk_1, max_frames="<<max_frames<<flush;
																																			// PRINT_MATX44F( invK,	);
																																			// PRINT_MATX44F( K,		);
																																			// PRINT_MATX44F( K * invK,		);
																																			// PRINT_MATX44F( invK * K,		);
																																			// cout<<"runcl.current_frames[ runcl.current_frames_idx[0-4] ].frame_count="
																																			// <<runcl.current_frames[ runcl.current_frames_idx[0] ].frame_count<<", "
																																			// <<runcl.current_frames[ runcl.current_frames_idx[1] ].frame_count<<", "
																																			// <<runcl.current_frames[ runcl.current_frames_idx[2] ].frame_count<<", "
																																			// <<runcl.current_frames[ runcl.current_frames_idx[3] ].frame_count<<", "
																																			// <<runcl.current_frames[ runcl.current_frames_idx[4] ].frame_count<<", "<<flush;

																																			for (int frame=0; frame<num_current_frames; frame++){
																																				cout<<"\ncurrent_frames_idx["<<frame<<"] = "<<runcl.current_frames_idx[frame]
																																				<<",    current_frames[	current_frames_idx["<<frame<<"] ].img_buf = "
																																				<<runcl.current_frames[	runcl.current_frames_idx[frame] ].img_buf
																																				<<",    .frame_count = "
																																				<<runcl.current_frames[ runcl.current_frames_idx[frame] ].frame_count
																																				<<",     .pose_to_0 = "
																																				<<PToLie (runcl.current_frames[ runcl.current_frames_idx[frame] ].pose_to_0)
																																				<<flush;
																																			}
																																		}
	for (int frame_idx =1; frame_idx<max_frames; frame_idx++){
		RunCL::frame *this_frame = &runcl.current_frames[ runcl.current_frames_idx[frame_idx] ];
																																		if(verbosity>local_verbosity_threshold) {
																																			cout << "\nDynamic_slam::estimate_tracking() chk_2: frame_idx="<<frame_idx<<",  ###########################"<<flush;

																																			PRINT_MATX16F( PToLie(this_frame->pose_to_0), "before initial estimate" );
																																			PRINT_MATX16F( PToLie (frame1->pose_to_0), "before initial estimate" );
																																			PRINT_MATX16F( PToLie (frame2->pose_to_0), "before initial estimate" );
																																		}
		int			layer 				= SE3_start_layer;
		float		old_sum_rho_sq		= FLT_MAX-1;
		float		old_sum_rho_sq_l0	= FLT_MAX-1;
		float		factor				= -1.0f;

		Matx44f		pose;
		Matx44f		newK2K;
		if (frame_idx>1){																													// Initial SE3 estimate is same as previous time step.
			this_frame->pose_to_0		= this_frame->pose_to_0 * frame1->pose_to_0;														// Subsequent past frames start from frame1_to_0 estimate, as update to this_frame to frame1 from previous time step.
		}
		pose							= this_frame->pose_to_0;
		newK2K							= K  *  pose  * invK ;
		runcl.update_44f_buf(	newK2K,	this_frame->k2k_buf_to_0,	fname);
																																		if(verbosity>local_verbosity_threshold) {
																																			cout << "\nDynamic_slam::estimate_tracking() chk_2.5: frame_idx="<<frame_idx<<",  layer="<<layer
																																			<<",  this_frame->frame_count="<<this_frame->frame_count<<flush;
																																			PRINT_MATX16F( PToLie(this_frame->pose_to_0), "initial estimate" );
																																		}
		for (uint iter = 0; iter<SE_iter; iter++){
			auto step_0 = high_resolution_clock::now();
																																		if(verbosity>local_verbosity_threshold) {
																																			cout << "\nDynamic_slam::estimate_tracking() chk_3: frame_idx="<<frame_idx<<",  layer="<<layer
																																			<<", out_block_size="<<out_block_size<<",  iter="<<iter
																																			<<",  this_frame->dataset_frame_num="<<this_frame->dataset_frame_num<<"###########################"<<flush;
																																			PRINT_MATX16F( PToLie(pose), "current estimate" );
																																			//PRINT_MATX44F( old_k2k, ); PRINT_MATX16F( PToLie(old_pose), );

																																			uint	out_block_size		= 2;
																																			uint	layer				= 0;
																																			string fname_ = fname_short + to_string(frame_idx) +"_"+ to_string(this_frame->dataset_frame_num)+"_"+to_string(old_sum_rho_sq_l0)+"_#";
																																			runcl.rho_sq_to_0( out_block_size, iter, frame_idx, layer, this_frame->k2k_buf_to_0, num_SE3_DoF, fname_	);

																																			// For debugging, get a larger, finer Rho map
																																			runcl.reduce_patch_Rho(		out_block_size, iter,				(uint)layer,							num_SE3_DoF);
																																			runcl.get_rho_result(		runcl.se3_rho_result,				(uint)layer,							num_SE3_DoF);

																																			old_sum_rho_sq_l0	=	runcl.se3_rho_result.Rho.y;		cout << "\nold_sum_rho_sq_l0	= "<< old_sum_rho_sq_l0 <<flush;
																																		}
			string fname_ = fname_short + to_string(frame_idx) +"_"+ to_string(this_frame->dataset_frame_num)+"_"+to_string(old_sum_rho_sq);

			runcl.rho_sq_to_0(			out_block_size, iter,	frame_idx,	(uint)layer,  this_frame->k2k_buf_to_0,	num_SE3_DoF, fname_);
			runcl.reduce_patch_Rho(		out_block_size, iter,				(uint)layer,							num_SE3_DoF);
			runcl.get_rho_result(		runcl.se3_rho_result,				(uint)layer,							num_SE3_DoF);

			float		sum_rho			=	runcl.se3_rho_result.Rho.x;																		// currently .x colour channel only.
			float		sum_rho_sq		=	runcl.se3_rho_result.Rho.y;		cout << "\nsum_rho_sq	= "<< sum_rho_sq <<flush;
			if( isnan(sum_rho_sq) ){
																		//	cout << "\nisnan(sum_rho_sq)" <<flush;
				break;
			}else if(sum_rho_sq > old_sum_rho_sq     ){																						// Rho, photometric error, got worse not better
				if(layer<=0) {break;}																										// Reached bottom of image pyramid.
				else {
																		//	cout << "\nsum_rho_sq > old_sum_rho_sq = "<< old_sum_rho_sq;
					if(factor<-0.1f){																										// End amplified steps
						factor *= 0.5f;
																			cout << ",  factor -2.0f -> -1.0f";
					}else {
						layer --;																											// Step down to lower layer of image pyramid
																			cout << "\nlayer = "	<<	layer;
						old_sum_rho_sq	=	FLT_MAX-1;																						// Re-set old_sum_rho_sq for new layer
					}
																																		//	PRINT_MATX44F( old_k2k, ); PRINT_MATX16F( PToLie(old_pose), );
					runcl.update_44f_buf(	old_k2k,	this_frame->k2k_buf_to_0,	fname);
					pose				= old_pose;
																			cout << endl << flush;
				}
			}else{
				old_sum_rho_sq			=	sum_rho_sq;
				old_pose				=	pose;
				old_k2k					=	newK2K;

				float		num_pixels	=	runcl.se3_rho_result.param_incr_arry[1];													// TO DO move numpixels to SE3_incr.w   & reduce SE3_incr_map_mem from float8 tro float4
				Matx16d		SE3_incr;	for (int i=0;	i<6; i++){	SE3_incr.operator()(i)	=	runcl.se3_rho_result.param_incr_arry[i*2];  };
																																		// if( verbosity>local_verbosity_threshold ){
																																		// 	cout << "\nDynamic_slam::estimate_tracking() chk_4: ,  ###########################"<<
																																		// 	"\n sum_rho = "			<< sum_rho		<<
																																		// 	",	sum_rho_sq	= "		<< sum_rho_sq	<<
																																		// 	",	num_pixels = "		<< num_pixels	<< endl<<flush;
																																		// 	PRINT_MATX16F( SE3_incr, );
																																		// 	/*PRINT_MATX44F( pose, "before update" );*/	PRINT_MATX16F( PToLie(pose), "before update");
																																		// }
				Matx66d	invH			=	runcl.current_frames[ runcl.current_frames_idx[frame_idx] ].inv_SE3_Hessian[layer];
				Matx16d pose_update		=	SE3_incr * invH;																			// Double precision is required
																																		// if( verbosity>local_verbosity_threshold ){
																																		// 	cout << "\nSE3_incr="			<<SE3_incr		<<endl<<flush;
																																		// 	cout << "\ninvH="				<<invH			<<endl<<flush;
																																		// 	cout << "\npose_update_cpu="	<<pose_update	<<endl<<flush;
																																		// 	PRINT_MATX66F( invH, );
																																		// 	PRINT_MATX16F( pose_update, "before factor and deltas_matx");
																																		// }
				pose_update				=	factor *  pose_update.mul( deltas_matx[layer] );											//NB matx.mul(  matx ) => elementwise multiplication.
																																	//	if( verbosity>local_verbosity_threshold-3 ){PRINT_MATX16F( pose_update, ); }
				pose					=	LieToP_Matx( pose_update )  *  pose;
				newK2K					=	K  *  pose  * invK ;
				runcl.update_44f_buf(	newK2K,	this_frame->k2k_buf_to_0,	fname);
																																		if( verbosity>local_verbosity_threshold ){
																																			cout << "\nDynamic_slam::estimate_tracking() chk_5: ,  layer = "<<layer<<"##########"<<flush;
																																		/*	PRINT_MATX16F( deltas_matx[layer], );	*/						PRINT_MATX16F( pose_update, );
																																		//	PRINT_MATX16F( PToLie( LieToP_Matx( pose_update ).inv() ), );
																																			/*PRINT_MATX44F( pose, "after update"	);*/				//	PRINT_MATX16F( PToLie( pose ), "after update" );
																																		//	PRINT_MATX44F( newK2K,			);
																																		}
				if( SE_iter-(iter/10) < layer) {
					layer --;																											// Step down to lower layer of image pyramid
					cout << "\n( SE_iter-(iter/10) < layer),  layer = "	<<	layer <<endl<<flush;
					old_sum_rho_sq		=	FLT_MAX-1;
				}
			}
			if( ( SE3_start_layer - layer) < (iter/3) ) layer--;
			auto step_1 = high_resolution_clock::now();																					/*if( verbosity>local_verbosity_threshold-3){
																																			cout << "\nDynamic_slam::estimate_tracking() loop finished  ###########################"\
																																			<<"Tracking loop time = "<<  duration_cast<microseconds>(step_1 - step_0).count()
																																			<<" microseconds,  layer="<<layer<<endl<<flush;
																																		}*/
		}
		this_frame->pose_to_0			= pose;
		this_frame->k2k_to_0			= newK2K;
		Matx44f new_pose_from_0			= getInvPose( pose );																			// set this_frame.k2k_buf for mapping, and
		Matx44f new_k2k_from_0			= K  *  new_pose_from_0	  * invK;
		runcl.update_44f_buf(	new_k2k_from_0,		this_frame->k2k_buf_from_0,	fname);

	}
																																		if(verbosity>local_verbosity_threshold) {
																																			cout << "\fDynamic_slam::tracking() finished"<< flush;
																																			PRINT_MATX44F(frame_data.back().frame_data.pose,);
																																			PRINT_MATX44F(frame_data.back().frame_data.K2K, );
																																		}
}





