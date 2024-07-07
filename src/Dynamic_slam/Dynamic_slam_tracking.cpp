#include "Dynamic_slam.hpp"
#include <fstream>
#include <iomanip>   // for std::setprecision, std::setw

using namespace cv;
using namespace std;

void Dynamic_slam::report_GT_pose_error(){ 																									// TODO this could be done in Max44f, without back and forth coversion.
	int local_verbosity_threshold = verbosity_mp["Dynamic_slam::report_GT_pose_error"];// -2;
	pose2pose_accumulated_error_algebra = LieSub(PToLie(pose2pose_accumulated), PToLie(pose2pose_accumulated_GT) ); // TODO wrong fornula	//pose2pose_accumulated_error_algebra 	= pose2pose_accumulated_algebra - pose2pose_accumulated_GT_algebra;
	keyframe_pose2pose_error_algebra 	= LieSub(PToLie(keyframe_pose2pose), PToLie(keyframe_pose2pose_GT)); 								//pose2pose_error_algebra 				= pose2pose_algebra - pose2pose_GT_algebra;
																																			if(verbosity>local_verbosity_threshold){ cout << "\n Dynamic_slam::report_GT_error_chk 0\n" << flush;
																																				PRINT_MATX44F(pose2pose_accumulated_GT,);
																																				PRINT_MATX44F(pose2pose_accumulated,);
																																				PRINT_MATX16F(pose2pose_accumulated_error_algebra,);
																																				//
																																				PRINT_MATX44F(keyframe_pose2pose_GT,);
																																				PRINT_MATX44F(keyframe_pose2pose,);
																																				keyframe_pose2pose_GT_algebra 	= PToLie(keyframe_pose2pose_GT);
																																				keyframe_pose2pose_algebra		= PToLie(keyframe_pose2pose);
																																				PRINT_MATX16F(keyframe_pose2pose_GT_algebra,);
																																				PRINT_MATX16F(keyframe_pose2pose_algebra,);
																																				PRINT_MATX16F(keyframe_pose2pose_error_algebra,);
																																			}
}

void Dynamic_slam::display_frame_resluts(){
	int local_verbosity_threshold = verbosity_mp["Dynamic_slam::display_frame_resluts"];// 1;																										if(verbosity>local_verbosity_threshold){ cout << "\n Dynamic_slam::display_frame_resluts_chk 0\n" << flush;}
	stringstream ss;
	ss << "_p2p_error_" ;				for (int i=0; i<6; i++) ss << "," << pose2pose_error_algebra.operator()(i);
	ss << "_cumulative_error_" ;		for (int i=0; i<6; i++) ss << "," << pose2pose_accumulated_error_algebra.operator()(i); // TODO wrong formula
	cout << "\nDynamic_slam::display_frame_resluts "<< ss.str() << flush;
	runcl.tracking_result( "Dynamic_slam::display_frame_resluts" );
}

void Dynamic_slam::artificial_pose_error(){
	int local_verbosity_threshold = verbosity_mp["Dynamic_slam::artificial_pose_error"];//-2;																										if(verbosity>local_verbosity_threshold){ cout << "\n\nDynamic_slam::artificial_pose_error() chk_0"<<flush; }
	Matx16f pose_step_algebra;
	for (int SE3=0; SE3<6; SE3++)  pose_step_algebra.operator()(0,SE3) = obj["Artif_pose_err_algebra"][SE3].asFloat();
	Matx44f poseStep 	= LieToP_Matx(pose_step_algebra);																					if(verbosity>local_verbosity_threshold){ PRINT_MATX44F(poseStep,);	PRINT_MATX16F(pose_step_algebra,);	PRINT_MATX16F(PToLie(keyframe_pose2pose),True);  }
	keyframe_pose2pose 	= keyframe_pose2pose * poseStep;																					if(verbosity>local_verbosity_threshold){ PRINT_MATX16F(PToLie(keyframe_pose2pose),Start); }
	K2K 				= old_K * keyframe_pose2pose * inv_K;																				if(verbosity>local_verbosity_threshold){ PRINT_MATX44F(K2K,New);	PRINT_FLOAT_16(runcl.fp32_k2keyframe,Old); }// Add error of one step in the 2nd SE3 DoF.
	for (int i=0; i<16; i++){ runcl.fp32_k2keyframe[i] = K2K.operator()(i/4, i%4);  }														if(verbosity>local_verbosity_threshold){ PRINT_FLOAT_16(runcl.fp32_k2keyframe,New); cout << "\nDynamic_slam::artificial_pose_error()_finish ##############################################" << flush;}
}

void Dynamic_slam::predictFrame(){
	int local_verbosity_threshold = verbosity_mp["Dynamic_slam::predictFrame"];/* -2;*/														if(verbosity>local_verbosity_threshold){ cout << "\n Dynamic_slam::predictFrame_chk 0.  runcl.dataset_frame_num = "<< runcl.dataset_frame_num  << flush;
																																				PRINT_MATX44F(K2K,Old);
																																				PRINT_MATX44F(pose2pose,Old);
																																				cout << "\nruncl.dataset_frame_num  = " << runcl.dataset_frame_num;
																																				cout << "\nruncl.costvol_frame_num  = " << runcl.costvol_frame_num;
																																				PRINT_MATX44F(keyframe_inv_old_pose,Old);
																																				PRINT_MATX44F(pose2pose,Old);
																																				PRINT_MATX44F(old_pose2pose,Old);
																																				PRINT_MATX44F(keyframe_old_pose2pose,Old);
																																				PRINT_MATX44F(keyframe_pose2pose,Old);
																																				PRINT_MATX44F(keyframe_old_K2K,Old);
																																				PRINT_MATX44F(K2K,Old);
																																				PRINT_MATX44F(keyframe_K2K,Old);
																																			}
	keyframe_inv_old_pose 			= getInvPose(keyframe_pose2pose);
	pose2pose 						= keyframe_inv_old_pose * keyframe_pose2pose;
	if (runcl.costvol_frame_num>0)	{
		pose2pose 					= pose2pose * getInvPose(old_pose2pose) * pose2pose;													// Only use accel if there are enough previous frames.
	}
	old_pose2pose					= pose2pose;
	keyframe_old_pose2pose			= keyframe_pose2pose;
	keyframe_pose2pose 				= keyframe_pose2pose * pose2pose;
	keyframe_old_K2K 				= keyframe_K2K;
	keyframe_K2K 					= K * keyframe_pose2pose * inv_old_K;
																																			if(verbosity>local_verbosity_threshold){ cout << "\n Dynamic_slam::predictFrame_chk 1\n" << flush;
																																				cout << "\nruncl.dataset_frame_num  = " << runcl.dataset_frame_num;
																																				cout << "\nruncl.costvol_frame_num  = " << runcl.costvol_frame_num;
																																				PRINT_MATX44F(keyframe_inv_old_pose,New);
																																				PRINT_MATX44F(pose2pose,New);
																																				PRINT_MATX44F(old_pose2pose,New);
																																				PRINT_MATX44F(keyframe_old_pose2pose,New);
																																				PRINT_MATX44F(keyframe_pose2pose,New);
																																				PRINT_MATX44F(keyframe_old_K2K,New);
																																				PRINT_MATX44F(K2K,New);
																																				PRINT_MATX44F(keyframe_K2K,New);
																																			}

	////////////////////////////////////////////////////////////////////////////
	// keyframe_old_K 					= keyframe_K;			//= K;
	// keyframe_inv_old_K				= keyframe_inv_K;		//= inv_K;
	// keyframe_old_pose				= keyframe_pose;		//= pose;
	// keyframe_inv_old_pose			= keyframe_inv_pose;	//= inv_pose;


	//keyframe_pose2pose		= pose2pose;
	/*pose2pose_algebra_2		= pose2pose_algebra_1;
	pose2pose_algebra_1		= PToLie(pose2pose);
	Matx16f 	p2p_alg 	= LieSub(pose2pose_algebra_1, pose2pose_algebra_2);

	if (runcl.costvol_frame_num==0){ pose2pose_algebra_0	= 								(runcl.dataset_frame_num > 2)*(p2p_alg );}		// Only use accel if there are enough previous frames.
	else{ 							 pose2pose_algebra_0	= LieAdd(pose2pose_algebra_1, 	(runcl.dataset_frame_num > 2)*(p2p_alg));}

	pose2pose 				= LieToP_Matx(pose2pose_algebra_0 );
	K2K 					= old_K * pose2pose * inv_K;
	keyframe_K2K 			= K * pose * keyframe_inv_pose * inv_old_K;   					// TODO fix this ?  Pose should reflect intertia, i.e. expect the same step again.

	for (int i=0; i<16; i++){ runcl.fp32_k2k[i] = K2K.operator()(i/4, i%4); }
																																			if(verbosity>local_verbosity_threshold){ cout << "\n Dynamic_slam::predictFrame_chk 1\n" << flush;
																																				cout << "\nruncl.dataset_frame_num  = " << runcl.dataset_frame_num;
																																				cout << "\nruncl.costvol_frame_num  = " << runcl.costvol_frame_num;
																																				PRINT_MATX16F(pose2pose_algebra_2,);
																																				PRINT_MATX16F(pose2pose_algebra_1,);
																																				PRINT_MATX16F(pose2pose_algebra_0,);
																																				PRINT_MATX44F(pose2pose,);
																																				PRINT_MATX44F(K2K,);
																																				PRINT_MATX44F(keyframe_K2K,);
																																			}
	*/// kernel update DepthMap with RelVelMap

	// kernel predict new frame
};
///
cv::Matx44f Dynamic_slam::generate_invK_(cv::Matx44f K_){
	int local_verbosity_threshold = verbosity_mp["Dynamic_slam::generate_invK_"];// 0;
	cv::Matx44f inv_K_;

	float fx   =  K_.operator()(0,0);
	float fy   =  K_.operator()(1,1);
	float skew =  K_.operator()(0,1);
	float cx   =  K_.operator()(0,2);
	float cy   =  K_.operator()(1,2);
																																			if(verbosity>local_verbosity_threshold) {
																																				cout << "\nDynamic_slam::generate_invK_chk 1\n";
																																				cout<<"\nfx="<<fx <<"\nfy="<<fy <<"\nskew="<<skew <<"\ncx="<<cx <<"\ncy= "<<cy;
																																				cout << flush;
																																			}
	///////////////////////////////////////////////////////////////////// Inverse camera intrinsic matrix, see:
	// https://www.imatest.com/support/docs/pre-5-2/geometric-calibration-deprecated/projective-camera/#:~:text=Inverse,lines%20from%20the%20camera%20center.
	inv_K_ = inv_K_.zeros();
	inv_K_.operator()(0,0)  = 1.0/fx;  																										if(verbosity>local_verbosity_threshold) cout<<"\n1.0/fx="<<1.0/fx;
	inv_K_.operator()(1,1)  = 1.0/fy;  																										if(verbosity>local_verbosity_threshold) cout<<"\n1.0/fy="<<1.0/fy;
	inv_K_.operator()(2,2)  = 1.0;
	inv_K_.operator()(3,3)  = 1.0;

	inv_K_.operator()(0,1)  = -skew/(fx*fy);
	inv_K_.operator()(0,2)  = (cy*skew - cx*fy)/(fx*fy);
	inv_K_.operator()(1,2)  = -cy/fy;
																																			if(verbosity>local_verbosity_threshold) {
																																				cv::Matx44f test_K_ = inv_K_ * K_;
																																				PRINT_MATX44F(test_K_,test_camera_intrinsic_matrix inversion);
																																				//PRINT_MATX44F(pose,);
																																				//PRINT_MATX44F(inv_old_pose,);
																																				PRINT_MATX44F(K_,);
																																				PRINT_MATX44F(inv_K,);
																																			}
	return inv_K_;
}

void Dynamic_slam::generate_invK(){ inv_K = generate_invK_(K);  }


void Dynamic_slam::generate_SE3_k2k( float _SE3_k2k[6*16] ) {																				// Generates a set of 6 k2k to be used to compute the SE3 maps for the current camera intrinsic matrix.
	int local_verbosity_threshold = verbosity_mp["Dynamic_slam::generate_SE3_k2k"];// -2;
																																			if(verbosity>local_verbosity_threshold) cout << "\nDynamic_slam::generate_SE3_k2k( float _SE3_k2k[6*16] ) chk_0" << endl << flush;
	// SE3
	//const float res			= ( obj["cameraMatrix"][2].asFloat() + obj["cameraMatrix"][5].asFloat() ) /2.0;
	const float f			= ( obj["cameraMatrix"][0].asFloat() + obj["cameraMatrix"][4].asFloat() ) /2.0;									// focal length in pixels.
	const float delta 	  	= obj["ST3_delta"].asFloat() * obj["min_depth"].asFloat()  / f ;												// ST3_delta * (Translation to cause 1 pixel of parallax at min_depth)  	//1.0;//0.01; //0.001;  //  * obj["min_depth"].asFloat()
	const float delta_theta = obj["SO3_delta_theta"].asFloat() / f;																			// SO3_delta_theta * (Rotation to cause 1 pixel of rotation flow) //0.01; //0.001;
	const float cos_theta   = cos(delta_theta);
	const float sin_theta   = sin(delta_theta);
																																			// Old :  Rotate 0.01 radians i.e 0.573  degrees.  Translate 0.001 'units' of distance
																																			if(verbosity>local_verbosity_threshold){ cout << "\nDynamic_slam::generate_SE3_k2k( ) chk_1,"<<endl << flush;
																																				print_json_float_9(obj, "cameraMatrix");
																																				cout << "  delta_theta = "	<<delta_theta	<< " radians," 								<<endl << flush;
																																				cout << "  delta = "		<<delta			<< " units distance," 						<<endl << flush;
																																				cout << "  f = "			<<f				<< " pixels," 								<<endl << flush;
																																				cout << "  obj[\"ST3_delta\"] =  "            <<obj["ST3_delta"].asFloat()				<<endl << flush;
																																				cout << "  obj[\"min_depth\"] =  "            <<obj["min_depth"].asFloat()				<<endl << flush;
																																				cout << "  obj[\"SO3_delta_theta\"] =  "      <<obj["SO3_delta_theta"].asFloat()		<<endl << flush;
																																			}
	//Identity =				(1,			0,			0,			0,  			0,			1,			0,			0,  			0,			0,			1,			0,  			0,	0,	0,	1);
	cv::Matx44f transform[6];
	transform[Rx] = cv::Matx44f(1,         0,          0,          0,\
								0,         cos_theta, -sin_theta,  0,\
								0,         sin_theta,  cos_theta,  0,\
								0,         0,          0,          1);

	transform[Ry] = cv::Matx44f(cos_theta,   0,         sin_theta,  0,\
								0,           1,         0,          0,\
								-sin_theta,  0,         cos_theta,  0,\
								0,           0,         0,          1);

	transform[Rz] = cv::Matx44f(cos_theta, -sin_theta,  0,          0,\
								sin_theta,  cos_theta,  0,          0,\
								0,           0,         1,          0,\
								0,           0,         0,          1);

	transform[Tx] = cv::Matx44f(1,0,0,delta, 	0,1,0,0,		0,0,1,0,		0,0,0,1);
	transform[Ty] = cv::Matx44f(1,0,0,0, 		0,1,0,delta,	0,0,1,0,		0,0,0,1);
	transform[Tz] = cv::Matx44f(1,0,0,0, 		0,1,0,0,		0,0,1,delta,	0,0,0,1);

	cv::Matx44f cam2cam[6];
	for (int i=0; i<6; i++) {  cam2cam[i] = K*transform[i]*inv_K; 																			if(verbosity>local_verbosity_threshold) PRINT_MATX44F(transform[i],);	}

	for (int i=0; i<6; i++) {
		cam2cam[i] = K * transform[i] *  inv_K;
																																			if(verbosity>local_verbosity_threshold) {
																																				cout << "\ncam2cam["<<i<<"]=";
																																				for (int j=0; j<16; j++) cout << ", "<<cam2cam[i].operator()(j/4,j%4);
																																				cout << flush;
																																			}
		for (uint row=0; row<4; row++) {
			for (uint col=0; col<4; col++){
				_SE3_k2k[i*16 + row*4 + col] 	= cam2cam[i].operator()(row,col);
			}
		}
	}
																																			if(verbosity>local_verbosity_threshold) {
																																				cout << setprecision(9);
																																				for (int i=0; i<6; i++) {
																																					cout << "\n _SE3_k2k ["<<i<<"*16 + row*4 + col]=\n";
																																					for (int row=0; row<4; row++) {
																																						for (int col=0; col<4; col++){
																																							cout << _SE3_k2k[i*16 + row*4 + col] <<"\t  ";
																																						}cout<<endl;
																																					}cout<<endl;
																																				}

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
																																				cout << "\n bottomright * identity = " << bottomright * 	identity << flush;

																																				cout << "\n\nDynamic_slam::generate_SE3_k2k( float _SE3_k2k[6*16] )   finished" << endl << flush;
																																			}
}

void Dynamic_slam::update_k2k(Matx16f update_){
	int local_verbosity_threshold = verbosity_mp["Dynamic_slam::update_k2k"];// -3;
																																			if(verbosity>local_verbosity_threshold) { cout << "\n\n Dynamic_slam::update_k2k()_chk 1, compute idealSE3Incr_algebra :" << flush;
																																				PRINT_MATX44F(keyframe_pose2pose,);		PRINT_MATX44F(keyframe_pose2pose_GT,);
																																				Matx16f keyframe_pose2pose_alg_ 		= PToLie(pose2pose);												PRINT_MATX16F(keyframe_pose2pose_alg_	,update_k2k()_chk 1);
																																				Matx16f keyframe_pose2pose_GT_alg_ 		= PToLie(pose2pose_GT);												PRINT_MATX16F(keyframe_pose2pose_GT_alg_,update_k2k()_chk 1);
																																				Matx16f idealSE3Incr_alg_ 				= LieSub(keyframe_pose2pose_alg_, keyframe_pose2pose_GT_alg_);		PRINT_MATX16F(idealSE3Incr_alg_			,update_k2k()_chk 1);
																																				Matx44f idealSE3Incr_ 					= LieToP_Matx(idealSE3Incr_alg_);									PRINT_MATX44F(idealSE3Incr_				,update_k2k()_chk 1);

																																				Matx44f keyframe_pose2pose_inv 			= keyframe_pose2pose.inv(); 										PRINT_MATX44F(keyframe_pose2pose_inv	,update_k2k()_chk 1);
																																				Matx44f idealSE3Incr 					= keyframe_pose2pose_GT * keyframe_pose2pose_inv;					PRINT_MATX44F(idealSE3Incr				,update_k2k()_chk 1);
																																				Matx16f idealSE3Incr_algebra 			= PToLie(idealSE3Incr); 											PRINT_MATX16F(idealSE3Incr_algebra		,update_k2k()_chk 1);
																																			}
	cv::Matx44f SE3Incr_matx = LieToP_Matx(update_); 																						// 						= SE3_Matx44f(update_);
	keyframe_pose2pose 									= keyframe_pose2pose *  SE3Incr_matx;
	keyframe_K2K 										= old_K * keyframe_pose2pose * inv_K;
	for (int i=0; i<16; i++){ runcl.fp32_k2keyframe[i] 	= keyframe_K2K.operator()(i/4, i%4);   }
																																			if(verbosity>local_verbosity_threshold) { cout << "\n\n Dynamic_slam::update_k2k()_chk 2, compute K2K :" << flush;
																																				PRINT_MATX16F(update_			,update_k2k()_chk 2);
																																				PRINT_MATX44F(SE3Incr_matx		,update_k2k()_chk 2);
																																				PRINT_MATX44F(K2K				,update_k2k()_chk 2);
																																				PRINT_MATX44F(keyframe_pose2pose,update_k2k()_chk 2);
																																				PRINT_MATX44F(old_K				,update_k2k()_chk 2);
																																				PRINT_MATX44F(inv_K				,update_k2k()_chk 2);

																																				cout << "\n####################################### finished Dynamic_slam::update_k2k(Matx16f update_)"<<flush;
																																			}
}

void Dynamic_slam::update_k2k(Matx16f update_,  Matx44f local_keyframe_pose2pose){
	int local_verbosity_threshold = verbosity_mp["Dynamic_slam::update_k2k"];// -3;
																																			if(verbosity>local_verbosity_threshold) { cout << "\n\n Dynamic_slam::update_k2k()_chk 1, compute idealSE3Incr_algebra :" << flush;
																																				PRINT_MATX44F(keyframe_pose2pose,);		PRINT_MATX44F(keyframe_pose2pose_GT,);
																																				Matx16f keyframe_pose2pose_alg_ 		= PToLie(pose2pose);												PRINT_MATX16F(keyframe_pose2pose_alg_	,update_k2k()_chk 1);
																																				Matx16f keyframe_pose2pose_GT_alg_ 		= PToLie(pose2pose_GT);												PRINT_MATX16F(keyframe_pose2pose_GT_alg_,update_k2k()_chk 1);
																																				Matx16f idealSE3Incr_alg_ 				= LieSub(keyframe_pose2pose_alg_, keyframe_pose2pose_GT_alg_);		PRINT_MATX16F(idealSE3Incr_alg_			,update_k2k()_chk 1);
																																				Matx44f idealSE3Incr_ 					= LieToP_Matx(idealSE3Incr_alg_);									PRINT_MATX44F(idealSE3Incr_				,update_k2k()_chk 1);

																																				Matx44f keyframe_pose2pose_inv 			= keyframe_pose2pose.inv(); 										PRINT_MATX44F(keyframe_pose2pose_inv	,update_k2k()_chk 1);
																																				Matx44f idealSE3Incr 					= keyframe_pose2pose_GT * keyframe_pose2pose_inv;					PRINT_MATX44F(idealSE3Incr				,update_k2k()_chk 1);
																																				Matx16f idealSE3Incr_algebra 			= PToLie(idealSE3Incr); 											PRINT_MATX16F(idealSE3Incr_algebra		,update_k2k()_chk 1);
																																			}
	cv::Matx44f SE3Incr_matx = LieToP_Matx(update_); 																						// 						= SE3_Matx44f(update_);
	keyframe_pose2pose 									= local_keyframe_pose2pose *  SE3Incr_matx;
	keyframe_K2K 										= old_K * keyframe_pose2pose * inv_K;
	for (int i=0; i<16; i++){ runcl.fp32_k2keyframe[i] 	= keyframe_K2K.operator()(i/4, i%4);   }
																																			if(verbosity>local_verbosity_threshold) { cout << "\n\n Dynamic_slam::update_k2k()_chk 2, compute K2K :" << flush;
																																				PRINT_MATX16F(update_			,update_k2k()_chk 2);
																																				PRINT_MATX44F(SE3Incr_matx		,update_k2k()_chk 2);
																																				PRINT_MATX44F(K2K				,update_k2k()_chk 2);
																																				PRINT_MATX44F(keyframe_pose2pose,update_k2k()_chk 2);
																																				PRINT_MATX44F(old_K				,update_k2k()_chk 2);
																																				PRINT_MATX44F(inv_K				,update_k2k()_chk 2);

																																				cout << "\n####################################### finished Dynamic_slam::update_k2k(Matx16f update_)"<<flush;
																																			}
}

void Dynamic_slam::update_k2k_3( float steps[3], Matx16f update_, float k2k_3_16[tracking_num_samples][16] ){												// Generates a set of 3 k2k to be used to compute the optimal SE3 step.
	int local_verbosity_threshold = verbosity_mp["Dynamic_slam::update_k2k_3"];

	Matx44f keyframe_pose2pose_3[tracking_num_samples];
	Matx44f k2k_3[tracking_num_samples];
	cv::Matx44f SE3Incr_matx = LieToP_Matx(update_);
																																			if(verbosity>local_verbosity_threshold) {
																																				cout << "\n Dynamic_slam::update_k2k_3: " << endl << flush;
																																				PRINT_MATX16F(update_,				update_);
																																				PRINT_MATX44F(SE3Incr_matx, 		SE3Incr_matx);
																																				//PRINT_MATX44F(SE3Incr_matx*SE3Incr_matx, 		SE3Incr_matx*SE3Incr_matx);

																																				PRINT_MATX44F(keyframe_pose2pose,	keyframe_pose2pose);
																																				PRINT_MATX44F(old_K, 				old_K)
																																				PRINT_MATX44F(inv_K,				inv_K);
																																				PRINT_MATX44F(keyframe_pose2pose,	keyframe_pose2pose);
																																			}
	for (int sample = 0; sample<tracking_num_samples; sample++){
		float update_multiplier = steps[1+sample];
		cv::Matx44f SE3Incr_matx = LieToP_Matx( update_*steps[1+sample] );
																																			if(verbosity>local_verbosity_threshold) {
																																				cout << "\n Dynamic_slam::update_k2k_3:  update_multiplier="<< update_multiplier << endl << flush;
																																				PRINT_MATX44F(SE3Incr_matx, 		SE3Incr_matx);
																																			}
		keyframe_pose2pose_3[sample] 					= keyframe_pose2pose *  SE3Incr_matx;
		k2k_3[sample] 									= old_K * keyframe_pose2pose_3[sample] * inv_K;
		for (int j=0; j<16; j++) { k2k_3_16[sample][j] 	= k2k_3[sample].operator()(j/4, j%4); }

		Matx44f tempMatx 		= SE3Incr_matx * SE3Incr_matx;
		SE3Incr_matx			= tempMatx;
																																			if(verbosity>local_verbosity_threshold) {
																																				cout << "\n Dynamic_slam::update_k2k_3: sample="<< sample << endl << flush;
																																				PRINT_FLOAT_16( k2k_3_16[sample], 	k2k_3_16[sample]);
																																			}
	}
}

void Dynamic_slam::compute_optimum( float steps[3], float Rho_sq_results_[tracking_num_samples+1][max_mipmap_layers][tracking_num_colour_channels], int layer, int channel, float *prediction, float *optimum, float *stepsize  ){
	int local_verbosity_threshold = verbosity_mp["Dynamic_slam::compute_tracking_increment"];

	float a, b, c,   d, e,   f, g,   h, i,   j, k, d2, f2, h2;																				// compute x value of the optimum of parabola, y= a*x*x + b*x + c
																																			// given samples at x=1,2,4
	d = steps[0];	e = Rho_sq_results_[0][layer][channel] ;
	f = steps[1];	g = Rho_sq_results_[1][layer][channel] ;
	h = steps[2];	i = Rho_sq_results_[2][layer][channel] ;

	d2 = d * d;
	f2 = f * f;
	h2 = h * h;

	j = (f2 - d2)*(f-h) - (h2 - f2)*(d-f) ; // (1*1 - 0*0)*(1-3)  - (3*3 - 1*1)*(0-1)  = 1*-2  - (9-1)*-1 = -2  -8*-1 = -2 +8 =6   //// d=0, f=1, h=3  // d2=0, 	f2=1, 	h2=9, 	j=6, k=0.72331,
	k = (g-i)*(d-f) - (e-g)*(f-h);   		// (0.456964 - 0.4644)*(0-1) - (0.814901 - 0.456964)*(1-3) = −0,007436*-1  - 0,357937*-2 = 0,72331   //// (d,e)=(0,0.814901), 	(f,g)=(1,0.456964), 	(h,i)=(3,0.4644),
	a = k/j;						 		// 0,72331 / 6 = 0,120551667
	b = (e-g +a*(f2-d2))  /  (d-f);  		// (0,814901 - 0,456964  +  0,120551667 * (1*1 - 0*0))  / (0-1)  =  -0,478488667
	c = e - a*d2 - b*d;				 		// 0.814901

	if (a>0){																																// IF concavity leads to a minimum, use it.
		float x 		= -b /(2*a);
		*prediction 	= a*(x*x) + b*x + c;
		*optimum 		= x;																												// dy/dx = 0 = 2*a*x + b   =>  x = -b /(2*a)
/*
																																			if(verbosity>local_verbosity_threshold) {
																																				cout << "\n Dynamic_slam::compute_optimum: if (a>0)"
																																				<< ", \ta=" << a
																																				<< ", \tb=" << b
																																				<< ", \tc=" << c

																																				<< ", \td2=" << d2
																																				<< ", \tf2=" << f2
																																				<< ", \th2=" << h2

																																				<< ", \tj=" << j
																																				<< ", \tk=" << k

																																				<< ", \t(d,e)=("<<d<<","<<e<<")"
																																				<< ", \t(f,g)=("<<f<<","<<g<<")"
																																				<< ", \t(h,i)=("<<h<<","<<i<<")"
																																				<< ", \tprediction="<<*prediction
																																				<< ", \toptimum="<<*optimum
																																				<< ", \tstepsize="<<*stepsize
																																				<< endl << flush;
																																			}
*/
	}else{																																	// IF concavity leads to a maximum, pick the best sample so far.
		if (e>=i){
			*prediction = i;
			*optimum 	= h;
			//*stepsize 	*=2;
		}else{
			*prediction = e;
			*optimum 	= d;
			//*stepsize 	/=2;
		}
	}
																																			if(verbosity>local_verbosity_threshold) {
																																				cout << "\n Dynamic_slam::compute_optimum: "
																																				<< ", \ta=" << a
																																				<< ", \tb=" << b
																																				<< ", \tc=" << c
																																				<< ", \t(d,e)=("<<d<<","<<e<<")"
																																				<< ", \t(f,g)=("<<f<<","<<g<<")"
																																				<< ", \t(h,i)=("<<h<<","<<i<<")"
																																				<< ", \tprediction="<<*prediction
																																				<< ", \toptimum="<<*optimum
																																				<< ", \tstepsize="<<*stepsize
																																				<< endl << flush;
																																			}
																																		if( e<*prediction || g<*prediction || i<*prediction ) {cout <<"\n logic error: prediction > sample." << flush; runcl.exit_(1); }
}

void Dynamic_slam::estimateSE3(){																										// Adaptive step size LM tracking and halting
	int 	local_verbosity_threshold 		= verbosity_mp["Dynamic_slam::estimateSE3"];
	Matx16f update 							= {0,0,0, 0,0,0};																			// SE3 Lie Algebra holding the DoF of SE3.
	uint  	layer 							= SE3_start_layer;
	float 	factor 							= obj["SE_factor"].asFloat();
	uint  	channel  						= 2;
	float 	steps[3] 						= {0, 1, 3};
	float 	stepsize 						= 1.0;

	for (int iter = 0; iter<SE_iter; iter++){
																																		if(verbosity>local_verbosity_threshold) {
																																			cout << "\n\nDynamic_slam::estimateSE3() iter="<< iter
																																			<<"  ##############################################################"<< flush;
																																		}
		float SE3_weights[max_mipmap_layers][num_SE3_DoF][tracking_num_colour_channels]	 		= {{{0}}};
		float SE3_results[max_mipmap_layers][num_SE3_DoF][tracking_num_colour_channels]	 		= {{{0}}};
		float Rho_sq_results[4][max_mipmap_layers][tracking_num_colour_channels]	 			= {{{FLT_MAX*0.99}}};
		float k2k_3_16[4][16] = {{0}};
		float kf_k2k[16];

		float count[4];
		count[0]  = iter;
		count[1]  = layer;
		count[2]  = factor;
		count[3]  = 0;
		float prediction, optimum1;

		runcl.estimateSE3_LK(SE3_results, SE3_weights, Rho_sq_results[0], iter, layer, layer+1);

		for (int SE3=0; SE3<6; SE3++) {	update.operator()(SE3) = 	SE3_update_dof_weights[SE3] * SE3_update_layer_weights[layer] * factor * SE3_results[layer][SE3][channel] 	/ (SE3_weights[layer][SE3][channel] * runcl.img_stats[IMG_VAR+channel] ) ;  }
		for (int SE3=0; SE3<6; SE3++) {																										// Exit if tracking fails #############################################################################
			if ( isfinite( update.operator()(SE3) ) ) continue;
			else {
				cout << "\n\nTracking failed,  isfinite( update.operator()("<<SE3<<") ) = " <<  isfinite( update.operator()(SE3) ) << endl<<endl<<flush;
				runcl.exit_(1);
			}
		}
		update_k2k_3( steps, update*stepsize, k2k_3_16 );

		cout << endl << endl;
		cout << "\nk2k_3_16[0] = ";		print_float_16(k2k_3_16[0]);
		cout << "\nk2k_3_16[1] = ";		print_float_16(k2k_3_16[1]);
		cout << "\nk2k_3_16[2] = ";		print_float_16(k2k_3_16[2]);
		cout << "\nk2k_3_16[3] = ";		print_float_16(k2k_3_16[3]);

		runcl.se3_rho_sq( Rho_sq_results, count, layer, layer+1, k2k_3_16 );

		compute_optimum( steps, Rho_sq_results, layer, channel, &prediction, &optimum1, &stepsize );

		Matx44f sample_k2k =  K  *	 keyframe_pose2pose *  LieToP_Matx(optimum1 * update ) * inv_K;

		Matx44f_To_float16arry(sample_k2k, kf_k2k);
/*
		cout << flush;
		PRINT_MATX44F( K , );
		PRINT_MATX44F( keyframe_pose2pose ,  );
		PRINT_MATX44F( LieToP_Matx(optimum1 * update ) ,  );
		PRINT_MATX44F( inv_K ,  );
*/
		PRINT_MATX44F(sample_k2k, );
		print_float_16(kf_k2k);

		cout << flush;

		runcl.se3_rho_sq( Rho_sq_results[3], count, layer, layer+1, kf_k2k );

		float 	lowest = Rho_sq_results[3][layer][channel];																				// choose best of the 4 samples
		int 	index  = 3;

		for (int i=0; i<3; i++)  {
			if (Rho_sq_results[i][layer][channel] < lowest ){
				lowest = Rho_sq_results[i][layer][channel];
				index = i;
			}
			cout << "\n Rho_sq_results["<<i<<"][layer][channel]="<<Rho_sq_results[i][layer][channel]<<", "<< Rho_sq_results[i][layer][3];
		}
		cout << "\n Rho_sq_results[i][layer][channel]="<<Rho_sq_results[3][layer][channel]<<", "<< Rho_sq_results[3][layer][3];

		switch (index){
			case 0:{														// The original sample is best, reduce step and repeat, or change level.
				if ( stepsize > 1 / pow(2,(5-layer))  ){
					stepsize  /=2;
					cout <<"\ncostvol_frame_num="<<runcl.costvol_frame_num<<",  case 0.0,  stepsize="<<stepsize <<flush;
				}
				else if ( layer > 1 ){
					layer--;
					cout <<"\ncostvol_frame_num="<<runcl.costvol_frame_num<<",  case 0.1,  layer="<<layer <<flush;
				}
				break;
			}
			case 1:{														// Use sample 1
				update_k2k( steps[1] * update *stepsize);
				stepsize  /=2;
				cout <<"\ncostvol_frame_num="<<runcl.costvol_frame_num<<",  case 1,  stepsize="<<stepsize <<flush;
				break;
			}
			case 2:{														// Use sample 2
				update_k2k( steps[2] * update *stepsize);
				stepsize  *=2;
				cout <<"\ncostvol_frame_num="<<runcl.costvol_frame_num<<",  case 2,  stepsize="<<stepsize <<flush;
				break;
			}
			case 3:{														// The predicted optimum is best
				update_k2k( optimum1 * update );
				cout <<"\ncostvol_frame_num="<<runcl.costvol_frame_num<<",  case 3" <<flush;
				break;
			}
			default: {
				cerr << "\n\nDynamic_slam::estimateSE3()error, invalid index " << index << flush;
				runcl.exit_(1);
			}
		}
	}
}



// void Dynamic_slam::estimateSE3(){																											// Adaptive step size LM tracking and halting
// 	int local_verbosity_threshold = verbosity_mp["Dynamic_slam::estimateSE3"];// -1;
// 																																			/*
// 																																			* 1) find update direction: 	runcl.estSE3()
// 																																			* 2) generate three poses along the direction "update"
// 																																			* 3) find update magnitude: 	runcl.se3_rho_sq()
// 																																			* 4) compute ideal increment, i.e. damped
// 																																			* 5) halt condition, per layer and global
// 																																			*/
// 	Matx16f update = {0,0,0, 0,0,0};		// old_update = {0,0,0, 0,0,0}, 																// SE3 Lie Algebra holding the DoF of SE3.
// 	//Matx44f SE3Incr_matx;																													// SE3 transformation matrix.
// 																											if (obj["sample_se3_incr"]==true){
// 																												initialize_resultsMat();
// 																											}
// 																																			if(verbosity>local_verbosity_threshold){ cout << "\n ###  Dynamic_slam::estimateSE3()_chk 0 : \t SE_factor = "<<SE_factor<<
// 																																				",\t obj[\"SE_factor\"].asFloat() = "<<obj["SE_factor"].asFloat() <<
// 																																				",\t SE_iter = " << SE_iter <<
// 																																				flush;}
// 																																			// # Get 1st & 2nd order gradients of SE3 wrt updated pose. (Translation requires depth map, middle depth initally.)
// 	float Rho_sq_result	=FLT_MAX*0.99,   next_layer_Rho_sq_result=FLT_MAX*0.99;				// old_Rho_sq_result=FLT_MAX*0.99 ,
// 	uint  layer 						= SE3_start_layer;
// 	float factor 						= obj["SE_factor"].asFloat();						//0.04
// /*
// 	// float factor_layer_multiplier 		= obj["SE_factor_layer_multiplier"].asFloat();  //0.75
// 	// float factor_iter_multiplier 		= obj["SE_factor_iter_multiplier"].asFloat();	//0.9
// 	// int   iter_per_layer 				= obj["SE_iter_per_layer"].asInt();				//1
// */
// 	const uint num_samples 	= tracking_num_samples+1;
// 	uint  channel  			= 2;
// 	float steps[3] 			= {0, 1, 3};					//  NB see  compute_optimum(..) constraints on d,f,h.   NB steps[0] must be 0. Otherwise edit  tracking_num_samples, update_k2k_3(...),  compute_optimum(...)
// 	float stepsize 			= 1.0;																					// TODO store and update stepsize multipler wrt predicted optmum. i.e. LM damping.
// 	//float old_Rho			= FLT_MAX;
// 	int	  old_layer			= -1;
// 	cv::Matx44f local_old_pose2pose = keyframe_pose2pose;
// 	float old_Rho_sq_results[3] = {FLT_MAX*0.99};
// 																																			if(verbosity>local_verbosity_threshold) {
// 																																				cout <<  "\n### Dynamic_slam::estimateSE3(): (Rho_sq_result < SE3_Rho_sq_threshold[layer][channel])=("<<Rho_sq_result<<
// 																																				" < "<<SE3_Rho_sq_threshold[layer][channel]<<")  ";
// 																																				for(int i=0;i<5;i++){cout<<" ( "; for(int j=0;j<3;j++) {cout<<", ["<<i<<"]["<<j<<"]"<<SE3_Rho_sq_threshold[i][j];}	cout << " ) "; }
// 																																				cout << ",\t layer = "<<layer<< ",\t factor = "<<factor << endl << flush;
// 																																			}
// 	for (int iter = 0; iter<SE_iter; iter++){ 																// TODO step down layers if fits well enough, and out if fits before iteration limit. Set iteration limit param in config.json file.
// 																																			if(verbosity>local_verbosity_threshold) {cout << "\n###  Dynamic_slam::estimateSE3()_chk 1.0" << "\t  iter = " << iter <<
// 																																				",\t layer = "<<layer<< ",\t factor = "<<factor<<flush;
// 																																			}
// 		float SE3_weights[max_mipmap_layers][num_SE3_DoF][tracking_num_colour_channels]	 		= {{{0}}};
// 		float SE3_results[max_mipmap_layers][num_SE3_DoF][tracking_num_colour_channels]	 		= {{{0}}};
// 		float Rho_sq_results[num_samples][max_mipmap_layers][tracking_num_colour_channels]	 	= {{{FLT_MAX*0.99}}};
// 		float Rho_sq_results__[max_mipmap_layers][tracking_num_colour_channels]	 				= {{FLT_MAX*0.99}};
//
// 		runcl.estimateSE3_LK(SE3_results, SE3_weights, Rho_sq_results[0], iter, layer, layer+1);	//runcl.mm_start, runcl.mm_stop);		// ### 1) find update direction: 	runcl.estSE3()
// 																																			if(verbosity>local_verbosity_threshold) {cout 	<< "\n###  Dynamic_slam::estimateSE3_chk 2.0:" << flush;
// 																																				cout << endl;
// 																																				for (int i=runcl.mm_start; i<=runcl.mm_stop; i++){ 													// SE3_results / (num_valid_px * img_variance)
// 																																					cout 									<< "\n\n###  Dynamic_slam::estimateSE3_chk 1.6.1:"<<
// 																																					", Layer "<<i<<" SE3_results = (";   															//   /SE3_weights
// 																																					for (int k=0; k<6; k++){
// 																																						cout << "\n(";
// 																																						for (int l=0; l<3; l++){ cout << ", \t" << SE3_results[i][k][l]   ; }  						//   / SE3_weights [i][k][l]
// 																																						cout << ", \t" << SE3_results[i][k][3] << ")";
// 																																					}cout << ")";
// 																																					cout << "\t\t IMG_VAR = ";
// 																																					for (int l=0; l<3; l++) cout << " ,\t " << runcl.img_stats[IMG_VAR+l] ;
// 																																					cout << endl << flush;
// 																																				}
// 																																				cout 										<< "\n###  Dynamic_slam::estimateSE3_chk 1.6.2"<<
// 																																				", \titer="<<iter<<
// 																																				", \tlayer="<<layer<<
// 																																				", \tnext_layer_Rho_sq_result="<< next_layer_Rho_sq_result <<
// 																																				", \tSE3_results["<<layer<<"][SE3]["<<channel<<"]=(\t"<< flush;
//
// 																																				if (isnormal(SE3_results[layer][0][channel])) cout << SE3_results[layer][0][channel]<<",\t"<< flush; else cout << "not_normal"<<flush;
// 																																				if (isnormal(SE3_results[layer][1][channel])) cout << SE3_results[layer][1][channel]<<",\t"<< flush; else cout << "not_normal"<<flush;
// 																																				if (isnormal(SE3_results[layer][2][channel])) cout << SE3_results[layer][2][channel]<<",\t"<< flush; else cout << "not_normal"<<flush;
// 																																				if (isnormal(SE3_results[layer][3][channel])) cout << SE3_results[layer][3][channel]<<",\t"<< flush; else cout << "not_normal"<<flush;
// 																																				if (isnormal(SE3_results[layer][4][channel])) cout << SE3_results[layer][4][channel]<<",\t"<< flush; else cout << "not_normal"<<flush;
// 																																				if (isnormal(SE3_results[layer][5][channel])) cout << SE3_results[layer][5][channel]<<",\t"<< flush; else cout << "not_normal"<<flush;
// 																																				cout << "), \tfactor="<<factor<<
// 																																				flush;
//
// 																																				cout << "\n";
// 																																				cout << ",\tRho_sq_results["<<layer<<"]["<<channel<<"] = ";
// 																																				if( isfinite(Rho_sq_results[0][layer][channel]) ) cout << Rho_sq_results[layer][channel] ;
// 																																				cout << flush;
//
// 																																				cout << "\n#### update = ";
// 																																			cout << "\niter="<<iter <<"  layer="<<layer <<"   stepsize="<<stepsize <<" ";
// 																																			cout << "\tSwitch: Rho_sq_results[0][layer][channel]="<<Rho_sq_results[0][layer][channel];
// 																																			}
// 		float kf_k2k[16];
// 		Matx44f_To_float16arry(keyframe_K2K, kf_k2k);
// 		float count_[4];  count_[0] = iter;  count_[1] = layer;  count_[2] = factor;  count_[3] = 0;
//
// 		runcl.se3_rho_sq( Rho_sq_results__, count_, layer, layer+1, kf_k2k );																cout << "\tRho_sq_results__[layer][channel]="<<Rho_sq_results__[layer][channel];
//
//
// 		for(int sample=0; sample<3; sample++){cout << ", \t old_Rho_sq_results["<<sample<<"]="<< old_Rho_sq_results[sample];}
//
// 		if (old_layer==layer){
// 			float lowest_Rho = Rho_sq_results__[layer][channel];//Rho_sq_results[0][layer][channel];		// set "lowest_Rho" to the current result, AND "lowest" index to -1
// 			int lowest = -1;
// 			for(int sample=0; sample<3; sample++){						// compare current result to the 3 samples from the previous iteration.
// 				if (lowest_Rho > old_Rho_sq_results[sample]){
// 					lowest = sample;
// 					lowest_Rho =  old_Rho_sq_results[sample];
// 				}
// 			}
// 			switch(lowest){												// choose the lowest of the 4 samples.
// 				case 0:{																						// previous sample 0 was lowest
// 					cout<<"\tcase 0, ";
// 					Matx16f no_update_ = {0};
// 					update_k2k( no_update_			, local_old_pose2pose  );									// reset keyframe_pose2pose and keyframe_K2K
// 					if ( stepsize > 1/8) {																		// reduce step size
// 						stepsize /= 2; cout<<"\tcase0.0"<<flush;
// 						continue;
// 					}
// 					else if (layer > 1) {																		// change layer
// 						layer--; cout<<"\tcase 0.1"<<flush; continue;
// 					}
// 					else{
// 						cout<<"\tcase 0.2"<<flush;
// 						return;																					// terminate
// 						}
// 					}
// 				case 1:{
// 					cout<<"\tcase 1"<<flush;
// 					//stepsize /= 2;
// 					update_k2k( stepsize * steps[1] * update 	, local_old_pose2pose  );						// repeat at sample 1
// 					continue;
// 				}
// 				case 2:{
// 					cout<<"\tcase 2"<<flush;
// 					stepsize *= 2;
// 					update_k2k( stepsize * steps[2] * update 	, local_old_pose2pose  );						// repeat at sample 2
// 					continue;
// 				}
// 				default:																						// case -1, complete below.
// 					cout<<"\tcase -1"<<flush;
// 					break;
// 			}
// 		}
//
// 		for (int SE3=0; SE3<6; SE3++) { //6
// 																																			if(verbosity>local_verbosity_threshold) {
// 																																				cout << ", \nupdate se3 dof "<<SE3<<", layer "<<layer
// 																																				<<" = ("<< SE3_update_dof_weights[SE3]<<" * "<<SE3_update_layer_weights[layer]<<" * "<<factor<<" * "<<SE3_results[layer][SE3][channel]
// 																																				<<" / ( "<<SE3_weights[layer][SE3][channel]<<" * "<<runcl.img_stats[IMG_VAR+channel] ;
// 																																			}
// 			update.operator()(SE3) = 	SE3_update_dof_weights[SE3] * SE3_update_layer_weights[layer] * factor * SE3_results[layer][SE3][channel] 	/ (SE3_weights[layer][SE3][channel] * runcl.img_stats[IMG_VAR+channel] ) ;							// apply se3_dim weights and global factor.
//
// 																																			if(verbosity>local_verbosity_threshold) {		cout << " ) ) = \t "<< update.operator()(SE3) << flush;}
// 		}
// 		for (int SE3=0; SE3<6; SE3++) {																										// Exit if tracking fails #############################################################################
// 			if ( isfinite( update.operator()(SE3) ) ) continue;												// TODO handle exception and recover.
// 			else {
// 				cout << "\n\nTracking failed,  isfinite( update.operator()("<<SE3<<") ) = " <<  isfinite( update.operator()(SE3) ) << endl<<endl<<flush;
// 				exit(1);
// 			}
// 		}
// 																																			if(verbosity>local_verbosity_threshold) {cout << "\n\n###  Dynamic_slam::estimateSE3_chk 3.0" << flush;
// 																																				stringstream ss;
// 																																				for (int sample=0; sample<num_samples;  sample++){
// 																																					ss << "\nSample="<<sample<<", \tRho_sq_results = ";  // Rho_sq_results[8][tracking_num_colour_channels]
// 																																					for (int layer_=0; layer_<8  ; layer_++){
// 																																						ss  << "\nlayer = " << layer_ << " : ";
// 																																						for (int channel=0; channel< 4 ; channel++)  {ss  << Rho_sq_results[sample][layer_][channel] << ",  \t"; }
// 																																					}
// 																																				}
// 																																				ss << "\n\nSE3_results[layer][se3][chan=2 'value'] :";
// 																																				for (int se3 = 0; se3<6; se3++) {
// 																																					ss<< "\nse3 dof = "<< se3 << " : ";
// 																																					for (int layer_ = 0; layer_<obj["num_reductions"].asInt(); layer_ ++){
// 																																						ss << SE3_results[layer_][se3][2] << "  \t";
// 																																					}ss << "\t";
// 																																				}cout << ss.str() << endl << flush;
// 																																			}
// 		float k2k_3_16[tracking_num_samples][16] = {{0}};
// 		int num = TRACKING_NUM_SAMPLES;
//
//
// 		update_k2k_3( steps, update*stepsize, k2k_3_16 ); 																					// ### 2) generate three poses along the direction "update"
// 																																			if(verbosity>local_verbosity_threshold) {
// 																																				cout << "\n\nDynamic_slam::estimateSE3_chk 3.1:";
// 																																				for (int sample=0; sample<tracking_num_samples; sample++){
// 																																					cout << "\nsample="<< sample << endl << flush;
// 																																					print_float_16( k2k_3_16[sample] );
// 																																				}
// 																																			}
//
//
// 		float count[4];  count[0] = iter;  count[1] = layer;  count[2] = factor;  count[3] = 0;												// ### 3) find update magnitude: 	runcl.se3_rho_sq()
// 																																			if(verbosity>local_verbosity_threshold) {
// 																																				cout << "\n\nDynamic_slam::estimateSE3_chk 3.1:  layer="<<layer<<endl<<flush;
// 																																			}
// // void RunCL::se3_rho_sq( float Rho_sq_results[3][8][4], const float count[4], uint start, uint stop,  float k2k_3_16_[3][16]  )
// 		runcl.se3_rho_sq( Rho_sq_results, count, layer, layer+1, k2k_3_16 );																// should get back  Rho_sq_results for the three sample steps float k2k_3_16[3][16]. Steps are 0.5, 1.0, 2.0 times previous step size.
// 																																			// void RunCL::se3_rho_sq(float Rho_sq_results[8][4], const float count[4], uint start, uint stop,  float k2k_3_16_[3][16]  )
// 																																			if(verbosity>local_verbosity_threshold) {cout << "\n\n###  Dynamic_slam::estimateSE3_chk 4.0" << flush;
// 																																				stringstream ss;
// 																																				ss << "\tRho_sq_results = "  ;
// 																																				for (int sample = 0; sample<3; sample++ ){
// 																																					ss << "\n\n sample="<< sample <<"\n";
// 																																					for (int layer=0; layer<8; layer++){
// 																																						ss << "\nlayer="<< layer << "\t";
// 																																						for (int channel=0; channel<4; channel++){
// 																																							ss << Rho_sq_results[sample][layer][channel] <<",  \t";
// 																																						}
// 																																					}
// 																																				}ss << "\n\n ";
// 																																				cout << ss.str() << endl << flush;
// 																																			}
//
//
//
// 		old_layer 				= layer;
// 		//old_Rho					= Rho_sq_results[0][layer][channel];
// 		for(int sample=0; sample<3; sample++){ old_Rho_sq_results[sample]  = Rho_sq_results[sample][layer][channel]; }
//
// 		local_old_pose2pose 	= keyframe_pose2pose;
//
// 		float prediction, optimum1;//, optimum2;																							// ### 4) compute ideal increment, (i.e. damped)  given three Rho_sq_results and their step sizes .
// 		compute_optimum( steps, Rho_sq_results, layer, channel, &prediction, &optimum1, &stepsize );															// 1st estimate of optimum step from local curvature.
// 																																			if(verbosity>local_verbosity_threshold) {cout << "\n\n###  Dynamic_slam::estimateSE3_chk 5.0:"
// 																																				<<", \titer="<<iter
// 																																				<<", \tlayer="<<layer
// 																																				<<", \toptimum1="<<optimum1
// 																																				<<", \tprediction="<<prediction
// 																																				<<", \tRho_sq_results[0]["<<layer<<"]["<<channel<<"]="<<Rho_sq_results[0][layer][channel]
// 																																				<<", \tRho_sq_results[1]["<<layer<<"]["<<channel<<"]="<<Rho_sq_results[1][layer][channel]
// 																																				<<", \tRho_sq_results[2]["<<layer<<"]["<<channel<<"]="<<Rho_sq_results[2][layer][channel]
// 																																				<<", \tstepsize="<<stepsize
// 																																				<<", \tupdate=";
// 																																				print_matx16f(update);
// 																																				cout <<", \terror=";
// 																																				Matx16f error = LieSub(keyframe_pose2pose_GT_algebra, update);
// 																																				print_matx16f(error);
// 																																				cout <<", \tkeyframe_pose2pose=";
// 																																				print_matx16f(PToLie(keyframe_pose2pose));
// 																																				cout <<", \tkeyframe_pose2pose_GT=";
// 																																				print_matx16f(keyframe_pose2pose_GT_algebra);
// 																																			}
// 		// if (prediction > 0.98 * Rho_sq_results[0][layer][channel] ){ 																		// ### 5) halt condition, per layer and global ##########################
// 		// 	if (layer > SE3_stop_layer) {																									if(verbosity>local_verbosity_threshold) {cout << "\n\n###  Dynamic_slam::estimateSE3_chk 5.1:"
// 		// 																																		<< ", \thalt, move to next layer"
// 		// 																																		<< flush;
// 		// 																																	}
// 		// 		layer --;
// 		// 		continue;
// 		// 	} else { break; }
// 		// }
//   //
// 		// if (optimum1 >= stepsize*steps[2] ){
// 		// 	optimum1 = stepsize*steps[2];
// 		// 	stepsize *=2;
// 		// }else if (optimum1 <= stepsize*steps[1]){
// 		// 	stepsize /=2;
// 		// }
//
// 		update_k2k( optimum1 * update );
//
// /*
// 																																			//compute_tracking_increment( Rho_sq_results, Rho_sq_results, update, k2k_3_16, stepsize, optimum, increment );
// 																																			// sample Rho at optimum : replace "update" with stepsize*optimum
// 		update_k2k_3( steps, update*stepsize*optimum1, k2k_3_16 );																					// generates 3 samples spread around the expected optimum. NB stapsize = 0.5 will give 0.25. 0.5, 1.0 times optimum.
// 																																			if(verbosity>local_verbosity_threshold) {
// 																																				cout << "\n\nDynamic_slam::estimateSE3_chk 5.2:";
// 																																				for (int sample=0; sample<tracking_num_samples; sample++){
// 																																					cout << "\tsample="<< sample << endl << flush;
// 																																					print_float_16( k2k_3_16[sample] );
// 																																				}
// 																																			}
//
// 		runcl.se3_rho_sq( Rho_sq_results, count, layer, layer+1, k2k_3_16 );
//
// 		compute_optimum( steps, Rho_sq_results, layer, channel, &prediction, &optimum2, &stepsize );													// 2nd estimate of optimum, on the basis of curvature around expected step size.
// 		// If optimum2 is beyond the sample range, clip it.
//
//
//
//
// 																																			// On the basis of 3 samples around the 1st predicted optimum, predict a new optimum.
// 																																			// NB these samples are likely to be closer to the true optimum, so the new predction will be better.
// 																																			// Any original oveshoot may be corrected.
// 																																			// 1st estimate of optimum step from local curvature.
// 																																			if(verbosity>local_verbosity_threshold) {cout << "\n\n###  Dynamic_slam::estimateSE3_chk 5.3:"
// 																																				<<", \titer="<<iter
// 																																				<<", \tlayer="<<layer
// 																																				<<", \toptimum2="<<optimum2
// 																																				<<", \tprediction="<<prediction
// 																																				<<", \tRho_sq_results[0]["<<layer<<"]["<<channel<<"]="<<Rho_sq_results[0][layer][channel]
// 																																				<<", \tRho_sq_results[1]["<<layer<<"]["<<channel<<"]="<<Rho_sq_results[1][layer][channel]
// 																																				<<", \tRho_sq_results[2]["<<layer<<"]["<<channel<<"]="<<Rho_sq_results[2][layer][channel]
// 																																				<<", \tstepsize="<<stepsize
// 																																				<< flush;
// 																																				PRINT_MATX16F(update, update);
// 																																			}
// 		if ( optimum2 > 2*stepsize*optimum1  )  { update_k2k( 2*optimum1 * update ); 		// anti instability
// 		}else if (optimum2 <=0) 				{ update_k2k( 0.5*optimum1 * update );		// use 0.5 * optimum1   // TODO think through this condition.
// 		}else  									{ update_k2k( optimum2 * update ); }
// */
// 	}// end for (int iter = 0; iter<SE_iter; iter++).
//
//
// 	///////////////////////////////////////////////
// 	// old_update 				= update;
// 	// old_Rho_sq_result 		= Rho_sq_result;
// 																																			/*
// 																																					// # TODO maybe ...
// 																																					// # Predict dammped least squares step of SE3 for whole image + residual of translation for relative velocity map.
// 																																					// # Pass prediction to lower layers. Does it fit better ?
// 																																					// # Repeat SE3 fitting n-times. ? Damping factor adjustment ?
// 																																			*/
// 																											if(obj["sample_se3_incr"].asBool()==true) { cout << "\n###  Dynamic_slam::estimateSE3_chk 6.3, display and save ResultsMat\n" << flush;
// 																												if(obj["sample_se3_incr::display"].asBool()==true){
// 																													cv::namedWindow( "Dynamic_slam::estimateSE3_LK()_chk 6: writeToResultsMat" , 0 );									// show runcl.resultsMat
// 																													cv::imshow("Dynamic_slam::estimateSE3_LK()_chk 6: writeToResultsMat" , runcl.resultsMat);
// 																													cv::waitKey(-1);
// 																													destroyWindow( "Dynamic_slam::estimateSE3_LK()_chk 6: writeToResultsMat" );
// 																												}
// 																												stringstream ss;																										// save runcl.resultsMat
// 																												ss <<  runcl.paths.at("SE3_rho_map_mem").string() << "resultsMat_"<<runcl.dataset_frame_num<<".png";
// 																												cv::imwrite( ss.str(), runcl.resultsMat );
// 																											}
// 																																			if(verbosity>local_verbosity_threshold) { cout << "\n###  Dynamic_slam::estimateSE3_chk 7\n" << flush;
// 																																				cout << "\nruncl.frame_num = "<<runcl.dataset_frame_num;
// 																																				PRINT_MATX44F(pose2pose_accumulated,);
// 																																				PRINT_MATX44F(pose2pose,);
// 																																				PRINT_MATX44F(keyframe_pose2pose,);
// 																																			}
// 	if (runcl.dataset_frame_num > 0 ) pose2pose_accumulated = pose2pose_accumulated * pose2pose; // TODO wrong formula.
// 																																			if(verbosity>local_verbosity_threshold){ cout << "\n###  Dynamic_slam::estimateSE3_chk 8  Finished ####################################\n" << flush;}
//}


void Dynamic_slam::estimateSE3_LK(){
	int local_verbosity_threshold = verbosity_mp["Dynamic_slam::estimateSE3_LK"];// -1;

	Matx16f old_update = {0,0,0, 0,0,0}, update = {0,0,0, 0,0,0};																			// SE3 Lie Algebra holding the DoF of SE3.
	cv::Matx44f SE3Incr_matx;																												// SE3 transformation matrix.
	if (obj["sample_se3_incr"]==true){
		initialize_resultsMat();
	}
																																			if(verbosity>local_verbosity_threshold){ cout << "\n ###  Dynamic_slam::estimateSE3_LK_chk 0 : \t SE_factor = "<<SE_factor<<
																																				",\t obj[\"SE_factor\"].asFloat() = "<<obj["SE_factor"].asFloat() <<
																																				",\t SE_iter = " << SE_iter <<
																																				",\t dataset_frame_num = " << runcl.dataset_frame_num <<
																																				",\t runcl.costvol_frame_num = " << runcl.costvol_frame_num <<
																																				flush;}
																																			// # Get 1st & 2nd order gradients of SE3 wrt updated pose. (Translation requires depth map, middle depth initally.)
	float Rho_sq_result	=FLT_MAX*0.99,   old_Rho_sq_result=FLT_MAX*0.99 ,   next_layer_Rho_sq_result=FLT_MAX*0.99;
	uint  layer 						= SE3_start_layer;
	float factor 						= obj["SE_factor"].asFloat();					//0.04
	float factor_layer_multiplier 		= obj["SE_factor_layer_multiplier"].asFloat();  //0.75
	float factor_iter_multiplier 		= obj["SE_factor_iter_multiplier"].asFloat();	//0.9
	int   iter_per_layer 				= obj["SE_iter_per_layer"].asInt();				//1
	uint channel  						= 2; 																								// TODO combine Rho HSV channels
																																			if(verbosity>local_verbosity_threshold) {
																																				cout <<  "\n### Dynamic_slam::estimateSE3_LK(): (Rho_sq_result < SE3_Rho_sq_threshold[layer][channel])=("<<Rho_sq_result<<
																																				" < "<<SE3_Rho_sq_threshold[layer][channel]<<")  ";
																																				for(int i=0;i<5;i++){cout<<" ( "; for(int j=0;j<3;j++) {cout<<", ["<<i<<"]["<<j<<"]"<<SE3_Rho_sq_threshold[i][j];}	cout << " ) "; }
																																				cout << ",\t layer = "<<layer<< ",\t factor = "<<factor << endl << flush;
																																			}
	//if( runcl.costvol_frame_num  >0 ) runcl.update_tracking_depthmap(); // TODO later put this at the end of updateDepthCostVol(); OR just use amem for tracking, and therefore update amem in RunCL::transform_depthmap(..).

	for (int iter = 0; iter<SE_iter; iter++){ 																								// TODO step down layers if fits well enough, and out if fits before iteration limit. Set iteration limit param in config.json file.
																																			if(verbosity>local_verbosity_threshold) {cout << "\n###  Dynamic_slam::estimateSE3_LK()_chk 1.0" << "\t  iter = " << iter <<
																																				",\t layer = "<<layer<< ",\t factor = "<<factor<<flush;
																																			}
		//////////////////////////////////////
		if (iter%iter_per_layer==0 && iter>0 ) {if (layer>0) layer --; factor *= factor_layer_multiplier;}

		float SE3_weights[8][6][tracking_num_colour_channels] = {{{0}}};
		float SE3_results[8][6][tracking_num_colour_channels] = {{{0}}};
		float Rho_sq_results[8][tracking_num_colour_channels] = {{0}};

		runcl.estimateSE3_LK(SE3_results, SE3_weights, Rho_sq_results, iter, layer, layer+1);//runcl.mm_start, runcl.mm_stop);
																																			if(verbosity>local_verbosity_threshold) {cout 	<< "\n###  Dynamic_slam::estimateSE3_LK()_chk 1.6.0:" << flush;
																																				cout << endl;
																																				for (int i=runcl.mm_start; i<=runcl.mm_stop; i++){ 							// SE3_results / (num_valid_px * img_variance)
																																					cout 									<< "\n\n###  Dynamic_slam::estimateSE3_LK()_chk 1.6.1:"<<
																																					", Layer "<<i<<" SE3_results = (";   //   /SE3_weights
																																					for (int k=0; k<6; k++){
																																						cout << "\n(";
																																						for (int l=0; l<3; l++){ cout << ", \t" << SE3_results[i][k][l]   ; }  //   / SE3_weights [i][k][l]
																																						cout << ", \t" << SE3_results[i][k][3] << ")";
																																					}cout << ")";
																																					cout << "\n\n IMG_VAR = ";
																																					for (int l=0; l<3; l++) cout << " ,\t " << runcl.img_stats[IMG_VAR+l] ;
																																					cout << endl << flush;
																																				}
																																				cout 										<< "\n###  Dynamic_slam::estimateSE3_LK()_chk 1.6.2"<<
																																				"  \titer="<<iter<<
																																				", \tlayer="<<layer<<
																																				", \tnext_layer_Rho_sq_result="<< next_layer_Rho_sq_result <<
																																				", \tSE3_results["<<layer<<"][SE3]["<<channel<<"]=(\t"<< flush;

																																				if (isnormal(SE3_results[layer][0][channel])) cout << SE3_results[layer][0][channel]<<",\t"<< flush; else cout << "not_normal"<<flush;
																																				if (isnormal(SE3_results[layer][1][channel])) cout << SE3_results[layer][1][channel]<<",\t"<< flush; else cout << "not_normal"<<flush;
																																				if (isnormal(SE3_results[layer][2][channel])) cout << SE3_results[layer][2][channel]<<",\t"<< flush; else cout << "not_normal"<<flush;
																																				if (isnormal(SE3_results[layer][3][channel])) cout << SE3_results[layer][3][channel]<<",\t"<< flush; else cout << "not_normal"<<flush;
																																				if (isnormal(SE3_results[layer][4][channel])) cout << SE3_results[layer][4][channel]<<",\t"<< flush; else cout << "not_normal"<<flush;
																																				if (isnormal(SE3_results[layer][5][channel])) cout << SE3_results[layer][5][channel]<<",\t"<< flush; else cout << "not_normal"<<flush;
																																				cout << "), \tfactor="<<factor<<
																																				flush;

																																				cout << "\n";
																																				cout << ",\tRho_sq_results["<<layer<<"]["<<channel<<"] = ";
																																				if( isfinite(Rho_sq_results[layer][channel]) ) cout << Rho_sq_results[layer][channel] ;
																																				cout << flush;
																																			}
																																			if(verbosity>local_verbosity_threshold) {		cout << "\n#### update = ";
																																				cout << "\n ( SE3_update_dof_weights[SE3]  *  SE3_update_layer_weights[layer]  *  factor  *  SE3_results[layer][SE3][channel] ) "
																																				<<"\n/  ( SE3_weights[layer][SE3][channel]  *  runcl.img_stats[IMG_VAR+channel]  )  =  update.operator()(SE3)" << flush;
																																			}
		for (int SE3=0; SE3<6; SE3++) { //6
																																			if(verbosity>local_verbosity_threshold) {
																																				cout << ", \nupdate se3 dof "<<SE3<<", layer "<<layer
																																				<<" = ("<< SE3_update_dof_weights[SE3]<<" * "<<SE3_update_layer_weights[layer]<<" * "<<factor<<" * "<<SE3_results[layer][SE3][channel]
																																				<<" / ( "<<SE3_weights[layer][SE3][channel]<<" * "<<runcl.img_stats[IMG_VAR+channel] ;
																																			}
			update.operator()(SE3) = SE3_update_dof_weights[SE3] * SE3_update_layer_weights[layer] * factor * SE3_results[layer][SE3][channel] / (SE3_weights[layer][SE3][channel] * runcl.img_stats[IMG_VAR+channel] ) ;							// apply se3_dim weights and global factor.

																																			if(verbosity>local_verbosity_threshold) {		cout << " ) ) = \t "<< update.operator()(SE3) << flush;}
		}
		for (int SE3=0; SE3<6; SE3++) {																										// Exit if tracking fails #############################################################################
			if ( isfinite( update.operator()(SE3) ) ) continue;
			else {
				cout << "\n\nTracking failed,  isfinite( update.operator()("<<SE3<<") ) = " <<  isfinite( update.operator()(SE3) ) << endl<<endl<<flush;
				runcl.exit_(1);
			}
		}
		update_k2k( update );																												if(verbosity>local_verbosity_threshold) {cout << "\n\n###  Dynamic_slam::estimateSE3_LK()_chk 6: (iter>0 && Rho_sq_result > old_Rho_sq_result)" << flush;}
		old_update 				= update;
		old_Rho_sq_result 		= Rho_sq_result;
																																			if(verbosity>local_verbosity_threshold) {cout << "\n\n###  Dynamic_slam::estimateSE3_LK()_chk 6.1" << flush;
																																				stringstream ss;
																																				ss << "\nRho_sq_results = ";  // Rho_sq_results[8][tracking_num_colour_channels]
																																				for (int layer=0; layer<8  ; layer++){
																																					ss  << "\nlayer = " << layer << " : ";
																																					for (int channel=0; channel< 4 ; channel++)  {ss  << Rho_sq_results[layer][channel] << ",  \t"; }
																																				}
																																				ss << "\nSE3_results[layer][se3][chan=2 'value'] :";
																																				for (int se3 = 0; se3<6; se3++) { ss<< "\nse3 dof = "<< se3 << " : ";
																																					for (int layer = 0; layer<obj["num_reductions"].asInt(); layer ++){
																																						ss << SE3_results[layer][se3][2] << "  \t";
																																					}ss << "\t";
																																				}cout << ss.str() << endl << flush;
																																			}
		factor *= factor_iter_multiplier;
		// # TODO maybe ...
		// # Predict dammped least squares step of SE3 for whole image + residual of translation for relative velocity map.
		// # Pass prediction to lower layers. Does it fit better ?
		// # Repeat SE3 fitting n-times. ? Damping factor adjustment ?
	}
																																			if(obj["sample_se3_incr"].asBool()==true) { cout << "\n###  Dynamic_slam::estimateSE3_LK()_chk 6.3, display and save ResultsMat\n" << flush;
																																				if(obj["sample_se3_incr::display"].asBool()==true){
																																					cv::namedWindow( "Dynamic_slam::estimateSE3_LK()_chk 6: writeToResultsMat" , 0 );														// show runcl.resultsMat
																																					cv::imshow("Dynamic_slam::estimateSE3_LK()_chk 6: writeToResultsMat" , runcl.resultsMat);
																																					cv::waitKey(-1);
																																					destroyWindow( "Dynamic_slam::estimateSE3_LK()_chk 6: writeToResultsMat" );
																																				}
																																				stringstream ss;																														// save runcl.resultsMat
																																				ss <<  runcl.paths.at("SE3_rho_map_mem").string() << "resultsMat_"<<runcl.dataset_frame_num<<".png";
																																				cv::imwrite( ss.str(), runcl.resultsMat );
																																			}
																																			if(verbosity>local_verbosity_threshold) { cout << "\n###  Dynamic_slam::estimateSE3_LK()_chk 7,   runcl.dataset_frame_num="<<runcl.dataset_frame_num<<"\n" << flush;
																																				cout << "\nruncl.frame_num = "<<runcl.dataset_frame_num;
																																				PRINT_MATX44F(pose2pose_accumulated,);
																																				PRINT_MATX44F(pose2pose,);
																																				PRINT_MATX44F(keyframe_pose2pose,);
																																				Matx16f keyframe_pose2pose_algebra = PToLie(keyframe_pose2pose);
																																				PRINT_MATX16F (keyframe_pose2pose_algebra   ,  );
																																			}
	if (runcl.dataset_frame_num > 0 ) pose2pose_accumulated = pose2pose_accumulated * pose2pose; // TODO wrong formula.
																																			if(verbosity>local_verbosity_threshold){ cout << "\n###  Dynamic_slam::estimateSE3_LK()_chk 8  Finished ####################################\n" << flush;}
}
