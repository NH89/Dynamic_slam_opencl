#include "Dynamic_slam.hpp"
#include <fstream>
#include <iomanip>   // for std::setprecision, std::setw

using namespace cv;
using namespace std;

void Dynamic_slam::patch_slam(){																										// Adaptive step size LM tracking and halting
	int 	local_verbosity_threshold 			= V_DYNAMIC_SLAM_ESTIMATESE3;//verbosity_mp["Dynamic_slam::estimateSE3"];
																																		if(verbosity>local_verbosity_threshold) {	cout << "\nDynamic_slam::patch_slam() chk_0"
																																			<<"  ##############################################################"<< flush;
																																		}
	const uint max_iter							=12;	// must equal SE_iter = 10
	Matx16f update[max_iter]					= {{0,0,0, 0,0,0}};																			// SE3 Lie Algebra holding the DoF of SE3.

	uint  	layer 								= SE3_start_layer;
	uint  	channel  							= 2;

	Matx44f K 									= frame_data.back().frame_data.K;															// load function local variables fot the current frame.
	Matx44f inv_K 								= frame_data.back().frame_data.inv_K;
	Matx44f keyframe2pose[3] 					= {frame_data.back().frame_data.prev_pose2pose};
	Matx16f	keyframe2pose_SE3[3]				={{0}};
	keyframe2pose_SE3[0] 						= PToLie( keyframe2pose[0] );

	Matx44f gt_keyframe2pose;
	Matx16f gt_Pose;
	if(GT_available==true){
		gt_keyframe2pose		 				= frame_data.back().frame_data_GT.prev_pose2pose;	// TODO if(GT_available==true){}else{}
		gt_Pose									= PToLie( gt_keyframe2pose);
	}else{
		gt_keyframe2pose		 				= Matx44f::eye();
		gt_Pose									= {0};
	}
	const Matx44f keyframe2pose_GT				= gt_keyframe2pose;
	const Matx16f Pose_GT						= gt_Pose;

	Matx44f keyframe_k2k						= K*keyframe2pose[0]*inv_K;
	float 	k2k_4_16[tracking_tot_samples][16]	= {{0}};
	Matx44f_To_float16arry( keyframe_k2k,		k2k_4_16[0] );																				// NB float float 	k2k_4_16[..][16]  is passed by RunCL to kernels.

	float 	result_[max_iter][num_SE3_DoF]		={{0}};
	float 	update_[max_iter][num_SE3_DoF]		={{0}};

	Matx16f Pose_estimate[max_iter]				={{0}};
	Matx16f Pose_error[max_iter]				={{0}};
	float	Rho_valid_pixels[max_iter]			={0};
																																		if(verbosity>local_verbosity_threshold) {
																																			cout << "\n\nDynamic_slam::patch_slam() chk_1 ########################"<<flush;
																																			PRINT_MATX44F(K,);
																																			PRINT_MATX44F(inv_K,);
																																			PRINT_MATX44F(keyframe2pose[0],);
																																			PRINT_MATX44F(keyframe_k2k,);
																																			PRINT_FLOAT_16(k2k_4_16[0],);
																																		}
	for (uint iter = 0; iter<SE_iter; iter++){																							// The pose optimization loop.##################################################
																																		if(verbosity>local_verbosity_threshold) {
																																			cout << "\n\nDynamic_slam::patch_slam() iter="<< iter
																																			<<"  ##############################################################"<< flush;
																																		}

		float 	SE3_weights[max_mipmap_layers][num_SE3_DoF][tracking_num_colour_channels]	 				= {{{0}}};
		float 	SE3_results[max_mipmap_layers][num_SE3_DoF][tracking_num_colour_channels]	 				= {{{0}}};
		float 	Rho_sq_results[tracking_tot_samples][max_mipmap_layers][tracking_num_colour_channels]		= {{{FLT_MAX*0.99}}};
/*
		// float count[4];
		// count[0]  = iter;
		// count[1]  = layer;
		// count[2]  = factor;
		// count[3]  = 0;
		// float prediction, optimum1;
*/
																																		if(verbosity>local_verbosity_threshold) {
																																			cout << "\n Dynamic_slam::patch_slam(): chk_2   launching runcl.estimateSE3_LK(..),"
																																			<<"   iter="<<iter
																																			<<"   layer="<<layer
																																			<<"   SE3_stop_layer="<<SE3_stop_layer
																																			<<"   SE3_start_layer="<<SE3_start_layer
																																			<<"   obj['SE3_start_layer'].asUInt()="<<obj["SE3_start_layer"].asUInt()
																																			<<endl<<flush;
																																			//PRINT_FLOAT_16(runcl.fp32_k2keyframe,);
																																			if (layer>6) runcl.exit_(1);
																																		}
		/////////////////////////////////////////////////// testing kernel baased tracking
		uint out_block_size = 32;
		runcl.rho_sq( out_block_size, iter, layer/*, delta_theta, delta*/   );
		//runcl.update_SE3( layer, delta_theta, delta );
		///////////////////////////////////////////////////

																																		//TODO NB currently runcl.img_stats[..] only for layer"0"
		for (int SE3=0; SE3<num_SE3_DoF; SE3++) {	result_[iter][SE3] = SE3_results[layer][SE3][channel]  / (SE3_weights[layer][SE3][channel] * runcl.img_stats[/*layer*8 +*/ IMG_VAR*4 + channel] ) ;  }  // NB divide by total edge weighting, and image variance.
																																		if(verbosity>local_verbosity_threshold) {
																																			cout << "\n\nDynamic_slam::patch_slam() :chk_3"<<flush;
																																			Rho_valid_pixels[iter]	= Rho_sq_results[iter][layer][channel] / Rho_sq_results[iter][layer][3];
																																			cout << "\nRho_sq_results[iter="<<iter<<"][layer="<<layer<<"][channel="<<channel<<"] = "
																																				 << Rho_sq_results[iter][layer][channel]
																																				 <<",\t Rho/valid_pixels ="	<< Rho_valid_pixels[iter]
																																				 << flush;
																																		}
		for (int SE3=0; SE3<6; SE3++) {																									// Exit if tracking fails #####################################################
			if ( isfinite( update[iter].operator()(SE3) ) ) continue;
			else {
																																		cout << "\n\n\nDynamic_slam::patch_slam() : Tracking failed,  isfinite( update.operator()("<<SE3<<") ) = "
																																		<<  isfinite( update[0].operator()(SE3) ) << endl<<endl<<flush;
				runcl.exit_(1);
			}
		}

		// Need to adjust stepsize depending on angle between present and previous update.
		float mag_SO3 = 	sqrt( result_[iter][0]*result_[iter][0] 	+ result_[iter][1]*result_[iter][1] 	+ result_[iter][2]*result_[iter][2] );		// magnitude of the SO3 (rotation) update
		float mag_ST3 = 	sqrt( result_[iter][3]*result_[iter][3] 	+ result_[iter][4]*result_[iter][4] 	+ result_[iter][5]*result_[iter][5] );		// magnitude of the ST3 (translation) update
		if (iter==0){
			for (uint i=0; i<3; i++){
				update_[iter][i]						= result_[iter][i]		* delta_theta	/ mag_SO3;									// delta_theta is the minimal step used to compute the partial gradient wrt SO3.
				update_[iter][i+3]						= result_[iter][i+3]	* delta 		/ mag_ST3;									// delta is the minimal step used to compute the partial gradient wrt ST3. NB this depend on the the scale and range of the depthmap.
			}
			for (int SE3=0; SE3<6; SE3++) { update[iter].operator()(SE3) = update_[iter][SE3]; }											// For 1st iter take a 1 pixel step, in the direction of the gradient of Rho.
		}else if(iter>=1){
			float delta_SO3								= result_[iter][0]-result_[iter-1][0] 	+ result_[iter][1]-result_[iter-1][1] 	+ result_[iter][2]-result_[iter-1][2];
			float delta_ST3								= result_[iter][3]-result_[iter-1][3] 	+ result_[iter][4]-result_[iter-1][4] 	+ result_[iter][5]-result_[iter-1][5];
			for (uint i=0; i<3; i++){
				update_[iter][i]						= result_[iter][i]		* delta_SO3/mag_SO3; 						//( mag_SO3	/ delta_SO3 * 2 );
				update_[iter][i+3]						= result_[iter][i+3]	* delta_ST3/mag_ST3; 						//( mag_ST3	/ delta_ST3 * 2 );
				if ( update_[iter][i] 	< -delta_theta	|| update_[iter][i] 	<	delta_theta	){		cout << "\n update_["<<i<<"]="		<<update_[i]	<<",  delta_theta="<<delta_theta<<flush; }
				if ( update_[iter][i+3] < -delta		|| update_[iter][i+3] 	<	delta		){		cout << "\n update_["<<i+3<<"]="	<<update_[i+3]	<<",  delta="<<delta<<flush; }

				// TODO break out of layer loop if gradient nears zero....  OR change technique.  e.g. use optimim from 3rd iter.
			}
		}

		for (uint i=0; i<3; i++){
			update[iter].operator()(i)					= std::clamp(update_[iter][i],		-delta_theta,	delta_theta	);
			update[iter].operator()(i+3)				= std::clamp(update_[iter][i+3],	-delta,			delta		);					// For iter>=1, scale update to reach zero gradient, i.e. optimum. Clamp to prevent giant steps at low gradient.
		}

		keyframe2pose[ iter + 1 ]						=  keyframe2pose[iter]  *  LieToP_Matx( update[iter] );
		keyframe_k2k									=  K * 	keyframe2pose[ iter+1 ]	* inv_K	;
		Matx44f_To_float16arry( keyframe_k2k, k2k_4_16[ iter+1 ] );
		//update_k2k( iter_1, k2k_4_16);
																																		if(verbosity>local_verbosity_threshold) {
																																			cout << "\n\nDynamic_slam::patch_slam() iter="<<iter<<" :  ";
																																			cout << "mag_SO3="<< mag_SO3 << ",  mag_ST3=" << mag_ST3 << \
																																			",  delta_theta="<<delta_theta<< ",  delta="<<delta<<flush;

																																			cout <<"\nresult:\t";
																																			for(uint i=0; i<num_SE3_DoF; i++){ cout <<result_[iter][i] << ",\t\t"; }
																																			cout << flush;

																																			cout <<"\nupdate:\t";
																																			for(uint i=0; i<num_SE3_DoF; i++){ cout <<update_[iter][i] << ",\t\t"; }
																																			cout << flush;

																																			PRINT_MATX16F(update[iter], );
																																			PRINT_MATX44F(keyframe2pose[ iter+1 ], );

																																			Pose_estimate[iter] = PToLie( keyframe2pose[ iter+1 ] );
																																			PRINT_MATX16F( Pose_estimate[iter], )

																																			if(GT_available==true){
																																				PRINT_MATX16F( Pose_GT, )

																																				Matx44f error 		= keyframe2pose_GT * keyframe2pose[ iter+1 ].inv();
																																				Pose_error[iter] 	= PToLie( (error) );
																																				PRINT_MATX16F( Pose_error[iter], );
																																			}
																																			PRINT_MATX44F(  keyframe_k2k, );
																																			PRINT_FLOAT_16( k2k_4_16[ iter+1 ], );
																																		}
		if(iter%3 == 2){	layer--; }	// TODO 1) change layers 2) use patch kernel

		// TODO Problem, need to undo previous update.									// Generate two sample steps
		// TODO also need to update relative to the other reference frames.
	}
																																		if (verbosity>local_verbosity_threshold){
																																			cout << "\nDynamic_slam::patch_slam() End of loop :"<<flush;

																																			cout <<"\n\nresult:";
																																			for(uint iter=0; iter<SE_iter; iter++){
																																				cout<<"\n";
																																				for(uint i=0; i<num_SE3_DoF; i++){ cout <<result_[iter][i] << ",\t\t"; }
																																				cout << flush;
																																				if(iter%3 == 2) cout << endl<<flush;
																																			}
																																			cout <<"\n\nupdate:";
																																			for(uint iter=0; iter<SE_iter; iter++){
																																				cout<<"\n";
																																				for(uint i=0; i<num_SE3_DoF; i++){ cout <<update_[iter][i] << ",\t\t"; }
																																				cout << flush;
																																				if(iter%3 == 2) cout << endl<<flush;
																																			}
																																			cout<<"\n\nPose_estimate:";
																																			for(uint iter=0; iter<SE_iter; iter++){
																																				print_matx16f(Pose_estimate[iter]);
																																				if(iter%3 == 2) cout << endl<<flush;
																																			}

																																			if(GT_available==true){
																																				Matx16f Pose_GT	= PToLie( keyframe2pose_GT);
																																				PRINT_MATX16F( Pose_GT, )

																																				cout<<"\n\nPose_error:";
																																				for(uint iter=0; iter<SE_iter; iter++){
																																					print_matx16f(Pose_error[iter]);
																																					if(iter%3 == 2) cout << endl<<flush;
																																				}
																																			}
																																			cout <<"\n\nRho/valid_pixels[iter]:" << flush;
																																			for(uint iter=0; iter<SE_iter; iter++){
																																				cout << "\n" << Rho_valid_pixels[iter] << flush;
																																				if(iter%3 == 2) cout << endl<<flush;
																																			}
																																		}
}


void Dynamic_slam::estimateSLAM(){																										// Adaptive step size LM tracking and halting
	string fname = "Dynamic_slam::estimateSLAM()";
	int 	local_verbosity_threshold 		= V_DYNAMIC_SLAM_ESTIMATESE3;//verbosity_mp["Dynamic_slam::estimateSE3"];
																																		if(verbosity>local_verbosity_threshold) {
																																			cout << "\fDynamic_slam::estimate_SLAM() chk_0"
																																			<<"  ##############################################################"<< flush;
																																		}
	float zero		= 0;
	float count[4];
	count[2]		= obj["SE_factor"].asFloat();
	count[3]		= 0;

	float k_buf_arr[16];
	void * ptr		= k_buf_arr;
	runcl.ReadOutput( (uchar*)ptr, runcl.K_buf, sizeof(float)*16, 0 );
	PRINT_FLOAT_16(k_buf_arr, )
////debug
	/*
		cout<<"\n\n##Dynamic_slam::estimateSLAM() debug#########################################################################################\n\n"<<flush;
	for (int layer=0; layer<runcl.mm_stop; layer++){
		cout<<"\n## debug ######  layer="<<layer<<",  inv_H =\n"<<runcl.current_frames[ runcl.current_frames_idx[0] ].invHessian[layer] <<flush;
	}

	for (int layer=0; layer<runcl.mm_stop; layer++){
		uint iter=0;
		runcl.rho_sq( out_block_size, iter, (uint)layer	);
		runcl.reduce_patch_Rho ( out_block_size, iter, (uint)layer );

		cl_float2	Rho =	{{0}};	runcl.ReadOutput( 	(uchar*)&Rho,			runcl.SE3_rho_map_mem, 	sizeof(cl_float2),		32*sizeof(cl_float2)	);

		float	SE3_incr_arry[6*2];	runcl.ReadOutput(	(uchar*)SE3_incr_arry,	runcl.SE3_incr_map_mem,	6*sizeof(cl_float2),	32*sizeof(cl_float2)	);

		cout<<"\n## debug ######  layer="<<layer<<flush;
		cout<<"\nRho = "<<Rho.x <<",  "<<Rho.y <<flush;
		cout<<",   SE3_incr_arry[]= (";
		for(int i=0; i<6*2; i++) cout << ", "<< SE3_incr_arry[i];
		cout<<" ) "<<endl<<flush;
	}
	cout<<"\n\n##Dynamic_slam::estimateSLAM() end debug#########################################################################################\n\n"<<flush;
	*/
////end debug
																																		//cout <<"\nDynamic_slam::estimate_SLAM() chk_0.5  SE3_start_layer="<<SE3_start_layer<<",  SE3_stop_layer="<<SE3_stop_layer<<flush;
																																			// "SE3_start_layer":4,
																																			// "SE3_stop_layer":1,
	for (int	layer = 5 /*runcl.mm_stop-2*//*SE3_start_layer*/; layer>=5 /*runcl.mm_stop-3*/ /*SE3_stop_layer*/;	layer--){														// NB when uint passes zero it becomes UINT_MAX
																																		cout << "\nDynamic_slam::estimate_SLAM() chk_0.6  layer="<<layer<<flush;
		count[1]  = layer;
		runcl._clEnqueueFillBuffer(  runcl.uload_queue,  runcl.pose_update_buf,  &zero,  sizeof( float),  0,  6*sizeof(float),  fname  );// pose_update_buf zeroed for new layer, because old Rho not valid for comparison.

		//for (uint out_block_size = 4/*32*/; out_block_size > 2; out_block_size /=2){
		uint out_block_size = 4;
																																		//cout << "\nDynamic_slam::estimate_SLAM() chk_0.7  out_block_size="<<out_block_size <<flush;
			for (uint iter = 0; iter<SE_iter; iter++){
				count[0]  = iter;
																																		cout << "\nDynamic_slam::estimate_SLAM() chk_1: layer="<<layer<<", out_block_size="<<out_block_size<<",  iter="<<iter<<",  ###########################"<<flush;
				{
					uint	out_block_size		= 2;
					uint	layer				= 0;
					runcl.rho_sq( out_block_size, iter, layer	);																		// For debugging, get a larger, finer Rho map
				}
																																		cout << "\nDynamic_slam::estimate_SLAM() chk_2: ,  ###########################"<<flush;
				runcl.rho_sq( out_block_size, iter, (uint)layer	);
				//runcl.ReadOutput( (uchar*)ptr, runcl.K_buf, sizeof(float)*16, 0 );
				//PRINT_FLOAT_16(k_buf_arr, )
																																		cout << "\nDynamic_slam::estimate_SLAM() chk_3: ,  ###########################"<<flush;
				runcl.reduce_patch_Rho ( out_block_size, iter, (uint)layer );
				//runcl.ReadOutput( (uchar*)ptr, runcl.K_buf, sizeof(float)*16, 0 );
				//PRINT_FLOAT_16(k_buf_arr, )
				runcl.update_k2k_cpu( 	(uint)layer,	deltas_matx,	frame_data.back().frame_data_GT.prev_pose2pose );				// frame_data_GT.keyframe2pose for comparision only.
																																		cout << "\nDynamic_slam::estimate_SLAM() chk_4: ,  ###########################"<<flush;
				//runcl.update_k2k(  		(uint)layer, delta_theta, delta, frame_data.back().frame_data_GT.keyframe2pose );
																																		cout << "\nDynamic_slam::estimate_SLAM() chk_5: ,  ##############################################################"<<endl<<flush;
			}
		//}
	}
}
