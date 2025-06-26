#include "Dynamic_slam.hpp"
#include <fstream>
#include <iomanip>   // for std::setprecision, std::setw

using namespace cv;
using namespace std;

void Dynamic_slam::predictFrame_vec(){
	int local_verbosity_threshold = V_DYNAMIC_SLAM_PREDICTFRAME;//verbosity_mp["Dynamic_slam::predictFrame"];
																																			if(verbosity>local_verbosity_threshold){ cout << "\n\n Dynamic_slam::predictFrame_vec_chk 0. "<<flush; }
	vector<frame_datum>::iterator frame_minus_one 		= 	frame_data.end();
	//frame_minus_one--;																													// current frame
	frame_minus_one -=2;																													// previous frame
	if (frame_minus_one < frame_data.begin() ) frame_minus_one = frame_data.begin();														// Prevent out of range read.
	vector<frame_datum>::iterator frame_minus_two 		= 	frame_minus_one;																// if (frame_data.size() <= 2) frame_minus_one is the first frame,  therefore duplicate data for zero motion prediction.
	if (frame_data.size() > 2)	{frame_minus_two--;																							cout << "\n Dynamic_slam::predictFrame_vec_chk 1.1 (frame_data.size() > 2),   frame_data.size()="<< frame_data.size() <<flush;
	}
																																			// verify the previous two frames have valid data
																																			if(verbosity>local_verbosity_threshold){ cout << "\n\n Dynamic_slam::predictFrame_vec_chk 2. "<<flush;
																																				PRINT_MATX44F( frame_minus_one->frame_data.K , )
																																				PRINT_MATX44F( frame_minus_one->frame_data_GT.K , )
																																				cout << endl;
																																				PRINT_MATX44F( frame_minus_one->frame_data.pose , )
																																				PRINT_MATX44F( frame_minus_two->frame_data.pose , )
																																			}
	frame_data.back().frame_data.K 						= 	frame_minus_one->frame_data.K;
	frame_data.back().frame_data.inv_K 					= 	frame_minus_one->frame_data.inv_K;
																																			// Assume linear velocity in SE3. NB Using poses from start to avoid conflict when starting new keyframe.
	frame_data.back().frame_data.pose					=	frame_minus_one->frame_data.pose  *  frame_minus_one->frame_data.pose  *  frame_minus_two->frame_data.inv_pose;			//  TODO problem on 1st iteration when inv pose is not correctly set yet
																																				PRINT_MATX44F( frame_data.back().frame_data.pose ,  			"predicted pose of the current frame"  );
	frame_data.back().frame_data.inv_pose				=	getInvPose(frame_data.back().frame_data.pose, verbosity);
																																				PRINT_MATX44F( frame_data.back().frame_data.inv_pose , );
																																				print_frame_data_vector(	0, 10, frame_data,		"frame_data  vector"	);
																																				print_keyframe_data_vector( 0, 10, keyframe_data,	"keyframe_data  vector" );
	uint 				keyframe_index					=	frame_data.back().keyframe_index;												// frame_data.key_frame_index;
																																				cout << "\n\n keyframe_index="<<keyframe_index<<", frame_data.size()="<<frame_data.size()<<flush;
	cv::Matx44f			keyframe_inv_pose 				=	keyframe_data[keyframe_index].frame_data.frame_data.inv_pose ;
																																				PRINT_MATX44F( keyframe_inv_pose ,  							"inverse pose of the current key_frame"  ); // FIXME uninitialized data
	frame_data.back().frame_data.keyframe2pose    		= 	keyframe_inv_pose 	*  frame_data.back().frame_data.pose;
																																				PRINT_MATX44F( frame_data.back().frame_data.keyframe2pose ,  	"predicted transform from key_frame to current frame"  );
	frame_data.back().frame_data.keyframe2pose_algebra	=	PToLie(frame_data.back().frame_data.keyframe2pose );

	frame_data.back().frame_data.K2K					=	frame_data.back().frame_data.K  *  frame_data.back().frame_data.keyframe2pose  	*  frame_data.back().frame_data.inv_K;
																																			if(verbosity>local_verbosity_threshold){ cout << "\n\n Dynamic_slam::predictFrame_vec_chk 3. "<<flush;
																																				report_GT_pose_error();
																																				cv::Matx44f K_invK_check  =  frame_data.back().frame_data.K  *  frame_data.back().frame_data.inv_K;
																																				PRINT_MATX44F( K_invK_check , )
																																			}
																																			if(verbosity>local_verbosity_threshold){ cout << "\n\n Dynamic_slam::predictFrame_vec Finished ################. "<<flush; }
}


void Dynamic_slam::generate_SE3_k2k_vec( float _SE3_k2k[6*16] ) {																			// Generates a set of 6 k2k to be used to compute the SE3 maps for the current camera intrinsic matrix.
	int local_verbosity_threshold = V_DYNAMIC_SLAM_GENERATE_SE3_K2K;//verbosity_mp["Dynamic_slam::generate_SE3_k2k"];// -2;
																																			if(verbosity>local_verbosity_threshold) cout << "\nDynamic_slam::generate_SE3_k2k( float _SE3_k2k[6*16] ) chk_0" << endl << flush;
	// SE3
	//const float res			= ( obj["cameraMatrix"][2].asFloat() + obj["cameraMatrix"][5].asFloat() ) /2.0;
	// const float f			= ( obj["cameraMatrix"][0].asFloat() + obj["cameraMatrix"][4].asFloat() ) /2.0;									// focal length in pixels.
	// const float delta 	  	= obj["ST3_delta"].asFloat() * obj["min_depth"].asFloat()  / f ;												// ST3_delta * (Translation to cause 1 pixel of parallax at min_depth)  	//1.0;//0.01; //0.001;  //  * obj["min_depth"].asFloat()
	// const float delta_theta = obj["SO3_delta_theta"].asFloat() / f;																			// SO3_delta_theta * (Rotation to cause 1 pixel of rotation flow) //0.01; //0.001;
	// const float cos_theta   = cos(delta_theta);
	// const float sin_theta   = sin(delta_theta);
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
																																			if(verbosity>local_verbosity_threshold) {
																																				PRINT_MATX44F(frame_data.back().frame_data_GT.K,);
																																				PRINT_MATX44F(frame_data.back().frame_data_GT.inv_K,);
																																			}

	for (int i=0; i<6; i++) {  cam2cam[i] = frame_data.back().frame_data_GT.K  *  transform[i]  *  frame_data.back().frame_data_GT.inv_K;
																																			if(verbosity>local_verbosity_threshold) {
																																				cout << "\ni=" << i << endl;
																																				PRINT_MATX44F(transform[i],);
																																				PRINT_MATX44F(cam2cam[i],);
																																			}
	}

	for (int i=0; i<6; i++) {
		cam2cam[i] = frame_data.back().frame_data_GT.K  *  transform[i] *  frame_data.back().frame_data_GT.inv_K;
																																			if(verbosity>local_verbosity_threshold) {
																																				cout << "\ncam2cam["<<i<<"]=";
																																				for (int j=0; j<16; j++) cout << setw(12) << ", "<<cam2cam[i].operator()(j/4,j%4);
																																				cout << flush;
																																			}
		for (uint row=0; row<4; row++) {
			for (uint col=0; col<4; col++){
				_SE3_k2k[i*16 + row*4 + col] 	= cam2cam[i].operator()(row,col);
			}
		}
	}
																																			if(verbosity>local_verbosity_threshold) {
																																				cout << endl << setprecision(9);
																																				for (int i=0; i<6; i++) {
																																					cout << "\n _SE3_k2k ["<<i<<"*16 + row*4 + col]=\n";
																																					for (int row=0; row<4; row++) {
																																						for (int col=0; col<4; col++){
																																							cout << setw(6) << _SE3_k2k[i*16 + row*4 + col] <<"\t  ";
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


void Dynamic_slam::update_k2k(  int case_idx,  float k2k_4_16[tracking_tot_samples][16]  ){
	int local_verbosity_threshold = V_DYNAMIC_SLAM_UPDATE_K2K;//verbosity_mp["Dynamic_slam::update_k2k"];// -3;
																																			if(verbosity>local_verbosity_threshold) { cout << "\n\n Dynamic_slam::update_k2k()_chk 1, compute idealSE3Incr_algebra :" << flush; }
	for (int i=0; i<16; i++) {
		k2k_4_16[0][i]	 =	k2k_4_16[case_idx][i];
		k2k_4_16[1][i]	 =	0;
		k2k_4_16[2][i]	 =	0;
		k2k_4_16[3][i]	 =	0;
	}
}


void Dynamic_slam::update_k2k(Matx16f update_){
	int local_verbosity_threshold = V_DYNAMIC_SLAM_UPDATE_K2K;//verbosity_mp["Dynamic_slam::update_k2k"];// -3;
																																			if(verbosity>local_verbosity_threshold) { cout << "\n\n Dynamic_slam::update_k2k()_chk 1, compute idealSE3Incr_algebra :" << flush;
																																				PRINT_MATX44F(frame_data.back().frame_data.keyframe2pose,);
																																				PRINT_MATX44F(frame_data.back().frame_data_GT.keyframe2pose,);
																																				PRINT_MATX16F(frame_data.back().frame_data.keyframe2pose_algebra	,update_k2k()_chk 1);
																																				PRINT_MATX16F(frame_data.back().frame_data_GT.keyframe2pose_algebra ,update_k2k()_chk 1);

																																				Matx16f idealSE3Incr_alg_ 						= LieSub(frame_data.back().frame_data.keyframe2pose_algebra, frame_data.back().frame_data_GT.keyframe2pose_algebra);
																																				PRINT_MATX16F(idealSE3Incr_alg_					,update_k2k()_chk 1);
																																				PRINT_MATX44F(LieToP_Matx(idealSE3Incr_alg_)	,update_k2k()_chk 1);

																																				PRINT_MATX44F(frame_data.back().frame_data.keyframe2pose.inv()	,update_k2k()_chk 1);

																																				Matx44f idealSE3Incr 							= frame_data.back().frame_data_GT.keyframe2pose * frame_data.back().frame_data.keyframe2pose.inv();
																																				PRINT_MATX44F(idealSE3Incr						,update_k2k()_chk 1);
																																				PRINT_MATX16F(PToLie(idealSE3Incr)				,update_k2k()_chk 1);
																																			}
	cv::Matx44f SE3Incr_matx = LieToP_Matx(update_); 																						// 						= SE3_Matx44f(update_);
	frame_data.back().frame_data.keyframe2pose 			= frame_data.back().frame_data.keyframe2pose *  SE3Incr_matx;
	frame_data.back().frame_data.K2K 					= frame_data.back().frame_data.K * frame_data.back().frame_data.keyframe2pose * frame_data.back().frame_data.inv_K;
	for (int i=0; i<16; i++){ runcl.fp32_k2keyframe[i] 	= frame_data.back().frame_data.K2K.operator()(i/4, i%4);   }
																																			if(verbosity>local_verbosity_threshold) { cout << "\n\n Dynamic_slam::update_k2k()_chk 2, compute K2K :" << flush;
																																				PRINT_MATX16F(update_										,update_k2k()_chk 2);
																																				PRINT_MATX44F(SE3Incr_matx									,update_k2k()_chk 2);
																																				PRINT_MATX44F(frame_data.back().frame_data.K2K				,update_k2k()_chk 2);
																																				PRINT_MATX44F(frame_data.back().frame_data.keyframe2pose	,update_k2k()_chk 2);
																																				PRINT_MATX44F(frame_data.back().frame_data.K				,update_k2k()_chk 2);
																																				PRINT_MATX44F(frame_data.back().frame_data.inv_K			,update_k2k()_chk 2);

																																				cout << "\n####################################### finished Dynamic_slam::update_k2k(Matx16f update_)"<<flush;
																																			}
}


void Dynamic_slam::compute_optimum( float steps[3], float Rho_sq_results_[tracking_num_samples][max_mipmap_layers][tracking_num_colour_channels], int layer, int channel, float *prediction, float *optimum, float *stepsize  ){
	int local_verbosity_threshold = V_DYNAMIC_SLAM_COMPUTE_OPTIMUM;//verbosity_mp["Dynamic_slam::compute_tracking_increment"];

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
																																				<< ", \na=" << a
																																				<< ", \tb=" << b
																																				<< ", \tc=" << c
																																				<< ", \n(d,e)=("<<d<<","<<e<<")"<<a*d*d + b*d + c
																																				<< ", \n(f,g)=("<<f<<","<<g<<")"<<a*f*f + b*f + c
																																				<< ", \n(h,i)=("<<h<<","<<i<<")"<<a*h*h + b*h + c
																																				<< ", \nprediction="<<*prediction
																																				<< ", \toptimum="<<*optimum
																																				<< ", \tstepsize="<<*stepsize
																																				<< ", \tlayer="<<layer
																																				<< endl << flush;
																																			}
																																		float prediction_ = *prediction * 0.999f; // prevents rounding error from triggering error.
																																		if( e<prediction_ || g<prediction_ || i<prediction_ ) {
																																			cout <<"\n logic error: prediction > sample." << flush; runcl.exit_(1); }
}

void Dynamic_slam::estimateSE3(){																										// Adaptive step size LM tracking and halting
	int 	local_verbosity_threshold 		= V_DYNAMIC_SLAM_ESTIMATESE3;//verbosity_mp["Dynamic_slam::estimateSE3"];
																																		// cout <<"\nDynamic_slam::estimateSE3(),  verbosity="<<verbosity\
																																		// <<",   local_verbosity_threshold="<<local_verbosity_threshold\
																																		// <<",   (verbosity>local_verbosity_threshold)="<<(verbosity>local_verbosity_threshold) << flush;
																																		if(verbosity>local_verbosity_threshold) {
																																			cout << "\nDynamic_slam::estimateSE3() chk_0,"
																																			<<"  ##############################################################"<< flush;
																																		}
	Matx16f update 							= {0,0,0, 0,0,0};																			// SE3 Lie Algebra holding the DoF of SE3.
	uint  	layer 							= SE3_start_layer;
	float 	factor 							= obj["SE_factor"].asFloat();
	uint  	channel  						= 2;
	float 	steps[3] 						= {0, 1, 3};
	float 	stepsize 						= 1.0;

	Matx44f K 								= frame_data.back().frame_data.K;															// load function local variables fot the current frame.
	Matx44f inv_K 							= frame_data.back().frame_data.inv_K;
	Matx44f keyframe2pose 					= frame_data.back().frame_data.keyframe2pose;
	Matx44f keyframe_k2k					= K*keyframe2pose*inv_K;
	float 	k2k_4_16[tracking_tot_samples][16] 		= {{0}};
	Matx44f_To_float16arry(keyframe_k2k, k2k_4_16[0]);																					// NB float float 	k2k_4_16[..][16]  is passed by RunCL to kernels.
	uint 	local_num_samples, start_sample_idx;
																																		if(verbosity>local_verbosity_threshold) {
																																			cout << "\n\nDynamic_slam::estimateSE3() chk_0.5 ########################"<<flush;
																																			//print_pose_vectors(0,10);
																																			cout << "\n\nDynamic_slam::estimateSE3() chk_0.6 ########################"<<flush;
																																			PRINT_MATX44F(K,);
																																			PRINT_MATX44F(inv_K,);
																																			PRINT_MATX44F(keyframe2pose,);
																																			PRINT_MATX44F(keyframe_k2k,);
																																			PRINT_FLOAT_16(k2k_4_16[0],);
																																		}
	for (uint iter = 0; iter<SE_iter; iter++){																							// The pose optimization loop.##################################################
																																		if(verbosity>local_verbosity_threshold) {
																																			cout << "\n\nDynamic_slam::estimateSE3() iter="<< iter
																																			<<"  ##############################################################"<< flush;
																																		}
		float SE3_weights[max_mipmap_layers][num_SE3_DoF][tracking_num_colour_channels]	 				= {{{0}}};
		float SE3_results[max_mipmap_layers][num_SE3_DoF][tracking_num_colour_channels]	 				= {{{0}}};
		float Rho_sq_results[tracking_tot_samples][max_mipmap_layers][tracking_num_colour_channels]	= {{{FLT_MAX*0.99}}};

		float count[4];
		count[0]  = iter;
		count[1]  = layer;
		count[2]  = factor;
		count[3]  = 0;
		float prediction, optimum1;
																																		if(verbosity>local_verbosity_threshold) {
																																			PRINT_FLOAT_16(runcl.fp32_k2keyframe,);
																																			cout << "\n Dynamic_slam::estimateSE3(): launching runcl.estimateSE3_LK(..),"
																																			<<"   iter="<<iter
																																			<<"   layer="<<layer
																																			<<"   SE3_stop_layer="<<SE3_stop_layer
																																			<<"   SE3_start_layer="<<SE3_start_layer
																																			<<"   obj['SE3_start_layer'].asUInt()="<<obj["SE3_start_layer"].asUInt()
																																			<<endl<<flush;
																																			//layer=obj["SE3_start_layer"].asUInt();
																																			if (layer>6) runcl.exit_(1);
																																		}
		runcl.estimateSE3_LK( k2k_4_16[0], SE3_results, SE3_weights, Rho_sq_results[0], iter, layer-2, layer );		// NB processes largest layer first.		// Find the gradient "update" wrt SE3

		for (int SE3=0; SE3<6; SE3++) {	update.operator()(SE3) = 	SE3_update_dof_weights[SE3] * SE3_update_layer_weights[layer] * factor * SE3_results[layer][SE3][channel] 	/ (SE3_weights[layer][SE3][channel] * runcl.img_stats[IMG_VAR+channel] ) ;  }

																																		if(verbosity>local_verbosity_threshold) {
																																			cout << "\nDynamic_slam::estimateSE3() : update.operator()(SE3) = 	SE3_update_dof_weights[SE3] * SE3_update_layer_weights[layer] * factor * SE3_results[layer][SE3][channel] "
																																			<< " / (SE3_weights[layer][SE3][channel] * runcl.img_stats[layer*8 + IMG_VAR*4 +channel] )";
																																			for (int SE3=0; SE3<6; SE3++) {
																																				cout << "\n update.operator()("<<SE3<<") = " << update.operator()(SE3)
																																				<< "  =  "   <<  SE3_update_dof_weights[SE3]
																																				<< " * " << SE3_update_layer_weights[layer]
																																				<< " * " << factor
																																				<< " * " << SE3_results[layer][SE3][channel]
																																				<< " /  () " << SE3_weights[layer][SE3][channel]
																																				<< " * " << runcl.img_stats[layer*8 + IMG_VAR*4 +channel]
																																				<< ")" << flush;
																																			}
																																		}
		for (int SE3=0; SE3<6; SE3++) {																									// Exit if tracking fails #####################################################
			if ( isfinite( update.operator()(SE3) ) ) continue;
			else {
																																		cout << "\n\n\nDynamic_slam::estimateSE3() : Tracking failed,  isfinite( update.operator()("<<SE3<<") ) = "
																																		<<  isfinite( update.operator()(SE3) ) << endl<<endl<<flush;
				runcl.exit_(1);
			}
		}
		Matx44f sample_1_k2k = K * keyframe2pose * LieToP_Matx( update * stepsize ) * inv_K;											// Generate two sample steps
		Matx44f_To_float16arry( sample_1_k2k,  k2k_4_16[1] );

		Matx44f sample_2_k2k = K * keyframe2pose * LieToP_Matx( update * stepsize * 3 ) * inv_K;
		Matx44f_To_float16arry( sample_2_k2k,  k2k_4_16[2] );
																																		if(verbosity>local_verbosity_threshold) {
																																			for (int pose_idx=0; pose_idx<4; pose_idx++){
																																				PRINT_FLOAT_16(k2k_4_16[pose_idx],);
																																			}
																																			PRINT_MATX44F(keyframe2pose , );
																																			PRINT_MATX44F(keyframe2pose * LieToP_Matx( update * stepsize ) , );
																																			PRINT_MATX44F(keyframe2pose * LieToP_Matx( update * stepsize * 3 )  , );
																																		}
		local_num_samples	= 2;
		start_sample_idx	= 1;
		runcl.se3_rho_sq( local_num_samples, start_sample_idx, Rho_sq_results, count, layer, layer+1, k2k_4_16 );						// Find Rho for the two sample steps
																																		if(verbosity>local_verbosity_threshold) {
																																			cout << "\n\nDynamic_slam::estimateSE3(): k2k_4_16[tracking_tot_samples][16] : ";
																																			for (int a=0; a<tracking_tot_samples; a++){
																																				cout << "\n\n tracking_sample k2k_4_16[a], a = " << a << " : ";
																																				PRINT_FLOAT_16( k2k_4_16[a] , );
																																			}

																																			cout << "\n\nDynamic_slam::estimateSE3():  Rho_sq_results[tracking_tot_samples][max_mipmap_layers][tracking_num_colour_channels] : ";
																																			for (int a=0; a<tracking_tot_samples; a++){
																																				cout << "\n\n tracking_sample = " << a << " : ";
																																				for (int b=0; b<max_mipmap_layers; b++){
																																					cout << "\nlayer = " << b << " ( ";
																																					for (int c=0; c<tracking_num_colour_channels; c++){
																																						cout << Rho_sq_results[a][b][c] << ", ";
																																					}
																																					cout << " ), ";
																																				}
																																			}
																																		}
		compute_optimum( steps, Rho_sq_results, layer, channel, &prediction, &optimum1, &stepsize );									// Compute optimal step from the three samples above

		Matx44f sample_k2k   =   K  *  keyframe2pose  *  LieToP_Matx(optimum1*update*stepsize )  *  inv_K;
		Matx44f_To_float16arry( sample_k2k,  k2k_4_16[3] );
																																		if(verbosity>local_verbosity_threshold) {
																																			PRINT_MATX44F(keyframe2pose * LieToP_Matx(optimum1*update*stepsize ), );
																																			PRINT_MATX44F(sample_k2k, );
																																			PRINT_FLOAT_16(k2k_4_16[3],);
																																		}
		local_num_samples	=1;
		start_sample_idx	=3;
		runcl.se3_rho_sq( local_num_samples, start_sample_idx, Rho_sq_results, count, layer, layer+1, k2k_4_16 );						// Find Rho at the predicted optimal step

		float 	lowest = FLT_MAX;																										// Choose best of the 4 samples
		int 	index  = 0;
		for (int i=0; i<4; i++)  {																										// Loop through the 4 sets of Rho_sq_results[set][layer][channel]
			if ( (Rho_sq_results[i][layer][channel] < lowest)  &&  (Rho_sq_results[i][layer][3] > 0.8*Rho_sq_results[0][layer][3])  ){	// Exclude result if it lacks valid overlap, between keframe and transposed current frame.
				lowest = Rho_sq_results[i][layer][channel];
				index = i;
			}else if (Rho_sq_results[i][layer][3] <= 0.8*Rho_sq_results[0][layer][3]) {
																																		cout << "\n\n Rho_sq_results["<<i<<"][layer][channel]  excluded due to low overlap";
			}
																																		if(verbosity>local_verbosity_threshold) {
																																			cout << "\nDynamic_slam::estimateSE3() : "
																																			<<" \t Rho_sq_results["<<i<<"][layer][channel]="    << Rho_sq_results[i][layer][channel]
																																			<<",\t valid_pixels="								<< Rho_sq_results[i][layer][3]
																																			<<",\t Rho/valid_pixels ="							<< Rho_sq_results[i][layer][channel] / Rho_sq_results[i][layer][3]
																																			<<",\t iter="										<< iter
																																			<<",\t dataset_frame_num="							<< runcl.dataset_frame_num
																																			<<",\t costvol_frame_num="							<< runcl.costvol_frame_num
																																			<< flush;
																																		}
		}
																																		if(verbosity>local_verbosity_threshold) {
																																			cout << "\n\n\n  "<<index;
																																		}
		update_k2k( index, k2k_4_16);										// Sets k2k_4_16[0] = k2k_4_16[index],   zeros k2k_4_16[1-3].
		switch (index){														// Set the best sample as the step for the next iteration
			case 0:{														// The original sample is best, reduce step and repeat, or change level.
				if ( stepsize > 1 / pow(2,(5-layer))  ){					// Reduce stepsize, unless already minimum for layer
					stepsize  /=2;
																																		if(verbosity>local_verbosity_threshold) { cout <<"\ndataset_frame_num="<<runcl.dataset_frame_num<<",  costvol_frame_num="<<runcl.costvol_frame_num<<",  case 0.0,  stepsize="<<stepsize <<flush; }
				}
				else if ( layer > 1 ){										// Else step down a layer, unless already minimum layer
					layer--;
																																		if(verbosity>local_verbosity_threshold) { cout <<"\ncostvol_frame_num="<<runcl.costvol_frame_num<<",  case 0.1,  layer="<<layer <<flush;}
				}
				break;
			}
			case 1:{														// Use sample 1
																																		if(verbosity>local_verbosity_threshold) { cout <<"\ncostvol_frame_num="<<runcl.costvol_frame_num<<",  case 1,  stepsize="<<stepsize <<",  halve stepsize for next iteration."<<flush;}
				keyframe2pose = keyframe2pose * LieToP_Matx( update * stepsize );
				stepsize  /=2;
				break;
			}
			case 2:{														// Use sample 2
																																		if(verbosity>local_verbosity_threshold) { cout <<"\ncostvol_frame_num="<<runcl.costvol_frame_num<<",  case 2,  stepsize="<<stepsize <<",  double stepsize for netx iteration."<<flush;}
				keyframe2pose = keyframe2pose * LieToP_Matx( update * stepsize * 3 );
				stepsize  *=2;
				break;
			}
			case 3:{														// The predicted optimum is best
																																		if(verbosity>local_verbosity_threshold) { cout <<"\ncostvol_frame_num="<<runcl.costvol_frame_num<<",  case 3,  stepsize="<<stepsize <<",  optimum1="<<optimum1 <<flush; }
				keyframe2pose = keyframe2pose * LieToP_Matx( optimum1*update*stepsize );
				break;
			}
			default:{
																																cerr << "\n\nDynamic_slam::estimateSE3()error, invalid index " << index << flush;
				runcl.exit_(1);
			}
		}
																																		if(verbosity>local_verbosity_threshold) { PRINT_MATX44F(keyframe2pose, final); }
		if ( lowest/Rho_sq_results[0][layer][3] < 0.00005 ) layer--;
	}
																																		//  TODO Update all variables in frame_data.back()
	for (int i=0; i<16; i++){ runcl.fp32_k2keyframe[i] 		= k2k_4_16[0][i]; }
	runcl.update_k2k_buf( k2k_4_16[0] );																								// Sets values in k2k_buf, for depth stages of the algorithm.

	uint 				keyframe_index						= frame_data.back().keyframe_index;		//.frame_data.key_frame_index;		// compute and record "pose_from_start"
	cv::Matx44f			keyframe_pose 						= keyframe_data[keyframe_index].frame_data.frame_data.pose; 				// .pose_from_start

	frame_data.back().frame_data.pose						= keyframe_pose * keyframe2pose ;
	frame_data.back().frame_data.inv_pose					= getInvPose( frame_data.back().frame_data.pose, verbosity );
	frame_data.back().frame_data.keyframe2pose  			= keyframe2pose;
	frame_data.back().frame_data.keyframe2pose_algebra 		= PToLie(keyframe2pose);
	frame_data.back().frame_data.K2K						= K * keyframe2pose * inv_K;

																																		if(verbosity>local_verbosity_threshold) {
																																			cout << "\n\nDynamic_slam::estimateSE3() Results" << flush;
																																			print_pose_datum( frame_data.back().frame_data );
																																			cout << "\n\nDynamic_slam::estimateSE3() Finished  ##############################################################\f"<< flush;
																																		}
}



