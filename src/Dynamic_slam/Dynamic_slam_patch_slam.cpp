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
				update_[iter][i]						= result_[iter][i]		* delta_theta[layer]	/ mag_SO3;									// delta_theta is the minimal step used to compute the partial gradient wrt SO3.
				update_[iter][i+3]						= result_[iter][i+3]	* delta[layer] 		/ mag_ST3;									// delta is the minimal step used to compute the partial gradient wrt ST3. NB this depend on the the scale and range of the depthmap.
			}
			for (int SE3=0; SE3<6; SE3++) { update[iter].operator()(SE3) = update_[iter][SE3]; }											// For 1st iter take a 1 pixel step, in the direction of the gradient of Rho.
		}else if(iter>=1){
			float delta_SO3								= result_[iter][0]-result_[iter-1][0] 	+ result_[iter][1]-result_[iter-1][1] 	+ result_[iter][2]-result_[iter-1][2];
			float delta_ST3								= result_[iter][3]-result_[iter-1][3] 	+ result_[iter][4]-result_[iter-1][4] 	+ result_[iter][5]-result_[iter-1][5];
			for (uint i=0; i<3; i++){
				update_[iter][i]						= result_[iter][i]		* delta_SO3/mag_SO3; 						//( mag_SO3	/ delta_SO3 * 2 );
				update_[iter][i+3]						= result_[iter][i+3]	* delta_ST3/mag_ST3; 						//( mag_ST3	/ delta_ST3 * 2 );
				if ( update_[iter][i] 	< -delta_theta[layer]	|| update_[iter][i] 	<	delta_theta[layer]	){		cout << "\n update_["<<i<<"]="		<<update_[i]	<<",  delta_theta[layer]="<<delta_theta[layer]<<flush; }
				if ( update_[iter][i+3] < -delta[layer]		|| update_[iter][i+3] 	<	delta[layer]		){		cout << "\n update_["<<i+3<<"]="	<<update_[i+3]	<<",  delta[layer]="<<delta[layer]<<flush; }

				// TODO break out of layer loop if gradient nears zero....  OR change technique.  e.g. use optimim from 3rd iter.
			}
		}

		for (uint i=0; i<3; i++){
			update[iter].operator()(i)					= std::clamp(update_[iter][i],		-delta_theta[layer],	delta_theta[layer]	);
			update[iter].operator()(i+3)				= std::clamp(update_[iter][i+3],	-delta[layer],			delta[layer]		);					// For iter>=1, scale update to reach zero gradient, i.e. optimum. Clamp to prevent giant steps at low gradient.
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
	constexpr float zero		= 0;
	int		layer 				= SE3_start_layer;																							cout << "\nDynamic_slam::estimate_SLAM() chk_0.6  layer="<<layer<<flush;
	float 	count[4]			= {0};
	count[1]					= layer;

	float 	k_buf_arr[16];
	runcl.ReadOutput( (uchar*)k_buf_arr, runcl.K_buf, sizeof(float)*16, 0 );																PRINT_FLOAT_16(k_buf_arr, )
	runcl._clEnqueueFillBuffer(  runcl.uload_queue,  runcl.pose_update_buf,  &zero,  sizeof( float),  0,  6*sizeof(float),  fname  );		// pose_update_buf zeroed for new layer, because old Rho not valid for comparison.

	uint out_block_size = 4;
	float	old_sum_rho_sq		= FLT_MAX-1;
	Matx44f	old_pose			= runcl.ReadOutput_44f( runcl.pose_buf );
	Matx44f	old_k2k				= runcl.ReadOutput_44f( runcl.k2kbuf );
	Matx44f newPose;
	Matx44f newK2K;
																																			//for (uint out_block_size = 4/*32*/; out_block_size > 2; out_block_size /=2){
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
																																		cout << "\nDynamic_slam::estimate_SLAM() chk_3: ,  ###########################"<<flush;
		runcl.reduce_patch_Rho ( out_block_size, iter, (uint)layer );

		runcl.update_k2k_cpu( 	(uint)layer );				// frame_data_GT.keyframe2pose for comparision only.
																																		cout<<"\nSE3_incr_arry[]= (";
																																		for(int i=0; i<6*2; i++) cout << ", "<< runcl.se3_rho_result.SE3_incr_arry[i];
																																		cout<<" ) "<<endl<<flush;
		float		sum_rho		=	runcl.se3_rho_result.Rho.x;		// currently .x colour channel only.
		float		sum_rho_sq	=	runcl.se3_rho_result.Rho.y;
		if(sum_rho_sq > old_sum_rho_sq || isnan(sum_rho_sq)    ){
			/* python
					// 	if (SSD > old_SSD or np.isnan(SSD) ):
					// 		if (SSD > old_SSD ):
					// 			print("\n SSD > old_SSD = {0:.4} #___________________ ".format( old_SSD) )
					// 		else:
					// 			print("\n isnan(SSD) ")
					// 			break
					// 		if (pyr_level <1) :
					// 			break
					//	#factor                  = factor * 0.9
					//	pyr_level              = pyr_level -1
					//	print(" factor = {0} ".format(factor) )
					//	print(" pyr_level = {0} ".format(pyr_level) )
					//	current_pose            = old_pose
					//	SSD                     = np.finfo('float32').max -10
			*/
			if(sum_rho_sq > old_sum_rho_sq){
				cout << "\nsum_rho_sq > old_sum_rho_sq = "<< old_sum_rho_sq <<flush;
				if (layer<=0) break;
				layer --;
			}else{
				cout << "\nisnan(sum_rho_sq)" <<flush;
				break;
			}
			cout << "\nlayer = "	<<	layer <<flush;
			runcl.update_k2k_buf(		old_k2k,		old_pose);
			old_sum_rho_sq			=	FLT_MAX-1;
		}else{
			old_sum_rho_sq			=	sum_rho_sq;
			old_pose				=	newPose;
			old_k2k					=	newK2K;

			float		num_pixels	=	runcl.se3_rho_result.SE3_incr_arry[1];																		// TODO move numpixels to SE3_incr.w   & reduce SE3_incr_map_mem from float8 tro float4
			Matx16f		SE3_incr;	for (int i=0;	i<6; i++){	SE3_incr.operator()(i)	=	runcl.se3_rho_result.SE3_incr_arry[i*2];  };
																																		if( verbosity>local_verbosity_threshold ){
																																			cout << "\nDynamic_slam::estimate_SLAM() chk_4: ,  ###########################"<<
																																			"\n sum_rho = "			<< sum_rho		<<
																																			",	sum_rho_sq	= "		<< sum_rho_sq	<<
																																			",	num_pixels = "		<< num_pixels	<< endl<<flush;
																																			PRINT_MATX16F( SE3_incr, );
																																		}
			Matx44f		pose		=	runcl.ReadOutput_44f( 	runcl.pose_buf );															PRINT_MATX44F( pose,	from pose_buf );	PRINT_MATX16F( PToLie(pose),);
			Matx44f		invK		=	runcl.ReadOutput_44f(	runcl.inv_K_buf);															PRINT_MATX44F( invK,	);
			Matx44f		K			=	runcl.ReadOutput_44f(	runcl.K_buf	 );																PRINT_MATX44F( K,		);
																																			PRINT_MATX44F( K * invK,		);
																																			PRINT_MATX44F( invK * K,		);

			Matx66f	invH			=	runcl.current_frames[ runcl.current_frames_idx[0] ].invHessian[layer];								PRINT_MATX66F( invH, );

			Matx16f pose_update_cpu	=	SE3_incr * invH;																					PRINT_MATX16F( pose_update_cpu, );

			float lim				= 1.0f + layer/2.0f;
			for(int i=0; i<num_SE3_DoF; i++){
				pose_update_cpu(i) = clamp(	pose_update_cpu(i),	-lim,	lim );																// clamp update wrt elta & blur for level, to avoid excessive steps. ? should it scale the whole vector instead ?
			}
			pose_update_cpu			=	(-1.0f) *  pose_update_cpu.mul( deltas_matx[layer]);/*  * 0.5f; */											PRINT_MATX16F( deltas_matx[layer], );				PRINT_MATX16F( pose_update_cpu, );
																																			PRINT_MATX16F( PToLie( LieToP_Matx(pose_update_cpu).inv() ), );
			newPose					=	LieToP_Matx( pose_update_cpu )  *  pose;															PRINT_MATX44F( newPose,	);							PRINT_MATX16F( PToLie( newPose ), );
			newK2K					=	K  *  newPose  * invK ;																			PRINT_MATX44F( newK2K,			);
																																			Matx44f	pose_old	= runcl.ReadOutput_44f( runcl.pose_buf );	PRINT_MATX44F( pose_old, );
																																			Matx44f	k2k_old		= runcl.ReadOutput_44f( runcl.k2kbuf);		PRINT_MATX44F( k2k_old,	);
			runcl.update_k2k_buf(		newK2K,		newPose);
																																			Matx44f	pose_now	= runcl.ReadOutput_44f( runcl.pose_buf );	PRINT_MATX44F( pose_now, );
																																			Matx44f	k2k_now		= runcl.ReadOutput_44f( runcl.k2kbuf);		PRINT_MATX44F( k2k_now,	);
			count[1]  = layer;
		}																																cout << "\nDynamic_slam::estimate_SLAM() loop finished  ###########################"<<flush;
	}																																	cout << "\nDynamic_slam::estimate_SLAM() finished  ###########################"<<flush;
}
