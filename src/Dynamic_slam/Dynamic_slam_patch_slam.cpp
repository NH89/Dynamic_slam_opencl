#include "Dynamic_slam.hpp"
#include <fstream>
#include <iomanip>   // for std::setprecision, std::setw

using namespace cv;
using namespace std;

float cos_sq_updates(Matx16f old_, Matx16f new_   ){		// tests cos^2 angle between updates of rotation SO3 & translation ST3. Returns the smaller of the two. NB -ve value -> overshoot.
	float dot_sq_SO3 = old_.operator()(0)*new_.operator()(0) + old_.operator()(1)*new_.operator()(1) + old_.operator()(2)*new_.operator()(2) ;
	float dot_sq_ST3 = old_.operator()(3)*new_.operator()(3) + old_.operator()(4)*new_.operator()(4) + old_.operator()(5)*new_.operator()(5) ;
	dot_sq_SO3	*= dot_sq_SO3;
	dot_sq_ST3	*= dot_sq_ST3;

	float mag_sq_SO3 = old_.operator()(0)*new_.operator()(0) * old_.operator()(1)*new_.operator()(1) * old_.operator()(2)*new_.operator()(2) ;
	float mag_sq_ST3 = old_.operator()(3)*new_.operator()(3) * old_.operator()(4)*new_.operator()(4) * old_.operator()(5)*new_.operator()(5) ;
	mag_sq_SO3	*= mag_sq_SO3;
	mag_sq_ST3	*= mag_sq_ST3;

	float cos_sq_SO3	= dot_sq_SO3 / mag_sq_SO3;
	float cos_sq_ST3	= dot_sq_ST3 / mag_sq_ST3;

	float cos_sq = cos_sq_ST3;
	if (cos_sq_SO3 < cos_sq_ST3) cos_sq = cos_sq_SO3;

	return cos_sq;
}

void Dynamic_slam::patch_slam(){																										// Adaptive step size LM tracking and halting
	int 	local_verbosity_threshold 		= V_DYNAMIC_SLAM_ESTIMATESE3;//verbosity_mp["Dynamic_slam::estimateSE3"];
																																		if(verbosity>local_verbosity_threshold) {
																																			cout << "\nDynamic_slam::patch_slam() chk_0"
																																			<<"  ##############################################################"<< flush;
																																		}

	const uint max_iter						=12;	// must equal SE_iter = 10
	Matx16f update[max_iter]				= {{0,0,0, 0,0,0}};																			// SE3 Lie Algebra holding the DoF of SE3.
	//Matx16f old_update						= {0,0,0, 0,0,0};																			// SE3 Lie Algebra holding the DoF of SE3.
	//float 	old_Rho_sq_results				= FLT_MAX /2.0f;

	uint  	layer 							= SE3_start_layer;
	//float 	factor 							= obj["SE_factor"].asFloat();
	uint  	channel  						= 2;
	// float 	steps[3] 						= {0, 1, 3};
	// float 	stepsize 						= 1.0;

	Matx44f K 								= frame_data.back().frame_data.K;															// load function local variables fot the current frame.
	Matx44f inv_K 							= frame_data.back().frame_data.inv_K;
	Matx44f keyframe2pose[3] 				= {frame_data.back().frame_data.keyframe2pose};
	Matx16f	keyframe2pose_SE3[3]			={{0}};
	keyframe2pose_SE3[0] 					= PToLie( keyframe2pose[0] );

	const Matx44f keyframe2pose_GT 			= frame_data.back().frame_data_GT.keyframe2pose;
	const Matx16f Pose_GT					= PToLie( keyframe2pose_GT);

	//Matx44f old_keyframe2pose 				= keyframe2pose[0];
																																		// old_keyframe2pose.operator()(1,1)=1.234567f;
																																		// cout << "/n chk old_keyframe2pose vs keyframe2pose.operator()(1,1) = "<< keyframe2pose[0].operator()(1,1)<< flush;
	Matx44f keyframe_k2k					= K*keyframe2pose[0]*inv_K;
	float 	k2k_4_16[tracking_tot_samples][16] 		= {{0}};
	Matx44f_To_float16arry( keyframe_k2k,  k2k_4_16[0] );																				// NB float float 	k2k_4_16[..][16]  is passed by RunCL to kernels.
	// uint 	local_num_samples, start_sample_idx;

	// const float f			= ( obj["cameraMatrix"][0].asFloat() + obj["cameraMatrix"][4].asFloat() ) /2.0;									// focal length in pixels.
	// const float delta 	  	= obj["ST3_delta"].asFloat() * obj["min_depth"].asFloat()  / f ;												// ST3_delta * (Translation to cause 1 pixel of parallax at min_depth)  	//1.0;//0.01; //0.001;  //  * obj["min_depth"].asFloat()
	// const float delta_theta = obj["SO3_delta_theta"].asFloat() / f;																			// SO3_delta_theta * (Rotation to cause 1 pixel of rotation flow) //0.01; //0.001;

	//float	old_update_[num_SE3_DoF]		={0};
	float 	result_[max_iter][num_SE3_DoF]	={{0}};
	float 	update_[max_iter][num_SE3_DoF]	={{0}};

	Matx16f Pose_estimate[max_iter]			={{0}};
	Matx16f Pose_error[max_iter]			={{0}};
	float	Rho_valid_pixels[max_iter]		={0};
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
		//uint 	mod3_iter 		= iter;// % 3;
		//uint 	mod3_iter_1 	= (iter+1);// % 3;

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
		runcl.estimateSE3_LK( k2k_4_16[ iter ], SE3_results, SE3_weights, Rho_sq_results[ iter ], iter, layer, layer );		// NB processes largest layer first.		// Find the gradient "update" wrt SE3

		/////////////////////////////////////////////////// testing kernel baased tracking
		uint out_block_size = 32;
		runcl.rho_sq( out_block_size, iter, layer  );
		runcl.update_SE3( layer, delta_theta, delta );
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

																																			PRINT_MATX16F( Pose_GT, )

																																			Matx44f error = keyframe2pose_GT * keyframe2pose[ iter+1 ].inv();
																																			Pose_error[iter] = PToLie( (error) ) ;
																																			PRINT_MATX16F( Pose_error[iter], )

																																			PRINT_MATX44F(  keyframe_k2k, );
																																			PRINT_FLOAT_16( k2k_4_16[ iter+1 ], );
																																		}
		if(iter%3 == 2){	layer--; }	// TODO 1) change layers 2) use patch kernel
		//swap( update_, old_update_ );	// float [6] arrays.
/*
		old_mag_SO3 = mag_SO3;
		old_mag_ST3 = mag_ST3;
		for (int SE3=0; SE3<num_SE3_DoF; SE3++) {	old_update_[SE3] = update_[SE3]; }

		// if (Rho_sq_results[0][layer][channel] > old_Rho_sq_results  && iter>0 ){
		// 	stepsize /=2.0;
		// 	keyframe2pose = old_keyframe2pose * LieToP_Matx( old_update * stepsize );
		// }else{																			cout << "\n!(Rho_sq_results[0][layer][channel] > old_Rho_sq_results  )" << flush;
		// 	old_Rho_sq_results	= Rho_sq_results[0][layer][channel] ;					// TODO (i) use all channels, (ii) handle layer change.
  //
		// 	if (iter>0){
		// 		float cos_sq 	= cos_sq_updates( old_update, update );					// 1) find cos^2 angle between old and new SO3 & ST3. NB -1 <= cos^2 <=1 , -ve => overshoot.
		// 																				// Returns the smaller of cos^2 SO3 & cos^2 ST3.      NB elements not comparable between SO3 & ST3.
		// 		stepsize 		*= powf(2.0f, cos_sq); 									// 2) adjust stepsize, min *0.5, max *2.0
		// 	}
		// 	keyframe2pose 		= keyframe2pose * LieToP_Matx( update * stepsize );
		// 	old_update 			= update;
		// }																				cout << "\nstepsize = "<<stepsize<<flush;
		// 																				PRINT_MATX44F(keyframe2pose,);
  //
		// Matx44f sample_1_k2k 	= K * keyframe2pose * inv_K;							// Generate next sample step
		// Matx44f_To_float16arry( sample_1_k2k,  k2k_4_16[0] ); 							// TODO also need to update relative to the other reference frames.
		// 																				PRINT_FLOAT_16(k2k_4_16[0],);
*/
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

																																			Matx16f Pose_GT	= PToLie( keyframe2pose_GT);
																																			PRINT_MATX16F( Pose_GT, )

																																			cout<<"\n\nPose_error:";
																																			for(uint iter=0; iter<SE_iter; iter++){
																																				print_matx16f(Pose_error[iter]);
																																				if(iter%3 == 2) cout << endl<<flush;
																																			}
																																			cout <<"\n\nRho/valid_pixels[iter]:" << flush;
																																			for(uint iter=0; iter<SE_iter; iter++){
																																				cout << "\n" << Rho_valid_pixels[iter] << flush;
																																				if(iter%3 == 2) cout << endl<<flush;
																																			}
																																		}
}


void Dynamic_slam::estimateSLAM(){																										// Adaptive step size LM tracking and halting
	int 	local_verbosity_threshold 		= V_DYNAMIC_SLAM_ESTIMATESE3;//verbosity_mp["Dynamic_slam::estimateSE3"];
																																		if(verbosity>local_verbosity_threshold) {
																																			cout << "\fDynamic_slam::estimate_SLAM() chk_0"
																																			<<"  ##############################################################"<< flush;
																																		}
	float count[4];
	count[2]  = obj["SE_factor"].asFloat();;
	count[3]  = 0;

	for (uint  	layer = SE3_start_layer; layer>= SE3_stop_layer;	layer--){ 															// NB when uint passes zero it becomes UINT_MAX
		count[1]  = layer;
		for (uint out_block_size = 32; out_block_size > 2; out_block_size /=2){
			for (uint iter = 0; iter</*SE_iter/2*/1; iter++){
				count[0]  = iter;
																																		cout << "\nDynamic_slam::estimate_SLAM() chk_1: layer="<<layer<<", out_block_size="<<out_block_size<<",  iter="<<iter<<",  ###########################"<<flush;
				runcl.rho_sq( out_block_size, iter, layer  );		// uint out_block_size, const float count[4], uint start, uint stop

				runcl.update_SE3( layer, delta_theta, delta );


			}
		}
	}

}


