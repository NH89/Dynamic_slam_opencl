#include "Dynamic_slam.hpp"
#include <fstream>
#include <iomanip>   // for std::setprecision, std::setw

using namespace cv;
using namespace std;

Dynamic_slam::~Dynamic_slam(){ runcl.~RunCL(); };

Dynamic_slam::Dynamic_slam( Json::Value obj_, int_map verbosity_mp_  ):   runcl( obj_ ,  verbosity_mp_ ) {   //    //     //// conf_params j_params
	obj = obj_;																																// NB save obj_ to class member obj, so that it persists within this Dynamic_slam object.
	verbosity_mp = verbosity_mp_;
	verbosity 					= verbosity_mp["verbosity"];						//	obj["verbosity"]["verbosity"].asInt();
	int local_verbosity_threshold = verbosity_mp["Dynamic_slam::Dynamic_slam"];		// verbosity_mp["Dynamic_slam"];

	runcl.dataset_frame_num 	= obj["data_file_offset"].asUInt();			//	j_params.int_mp["data_file_offset"];		//
	invert_GT_depth  			= obj["invert_GT_depth"].asBool();			//	j_params.bool_mp["invert_GT_depth"];		//

	SE3_start_layer 			= obj["SE3_start_layer"].asInt();			//	j_params.int_mp["SE3_start_layer"];			//
    SE3_stop_layer 				= obj["SE3_stop_layer"].asInt();			//	j_params.int_mp["SE3_stop_layer"];			//
	SE_iter_per_layer 			= obj["SE_iter_per_layer"].asInt();			//	j_params.int_mp["SE_iter_per_layer"];		//
    SE_iter 					= obj["SE_iter"].asInt();					//	j_params.int_mp["SE_iter"];					//
	SE_factor					= obj["SE_factor"].asFloat();				//	j_params.float_mp["SE_factor"];				//

	for (int layer=0; layer<MAX_LAYERS; layer++){for (int chan=0; chan<3; chan++)	SE3_Rho_sq_threshold[layer][chan]  	= obj["SE3_Rho_sq_threshold"][layer][chan].asFloat();  }	//j_params.float_vecvec_mp["SE3_Rho_sq_threshold"][layer][chan]; }		//
	for (int se3=0; se3<8; se3++)													SE3_update_dof_weights[se3] 		= obj["SE3_update_dof_weights"][se3].asFloat();				//j_params.float_vec_mp["SE3_update_dof_weights"][se3];					//
    for (int layer=0; layer<MAX_LAYERS; layer++) 									SE3_update_layer_weights[layer] 	= obj["SE3_float update_layer_weights"][layer].asFloat();	//j_params.float_vec_mp["SE3_float update_layer_weights"][layer];		//

																																			if(verbosity>local_verbosity_threshold) {cout << "\n Dynamic_slam::Dynamic_slam_chk 0,  SE3_Rho_sq_threshold[i][j] = ";
																																				for (int i=0; i<5; i++){cout << "( "; for (int j=0; j<3; j++) {cout << ", ["<<i<<"]["<<j<<"]" << SE3_Rho_sq_threshold[i][j]; }   cout << " )";}
																																				cout << ",\t SE_factor = "<<SE_factor;
																																				cout << endl << flush;
																																			}
	stringstream  ss0;
	ss0 << obj["data_path"].asString()  <<  obj["data_file"].asString();	// j_params.paths_mp["data_path"]   <<  j_params.paths_mp["data_file"];  //
	rootpath 	= ss0.str();
	root 		= rootpath;

	if ( exists(root)==false )		{ cout << "Data folder "<< ss0.str()  <<" does not exist.\n" <<flush; exit(0); }
	if ( is_directory(root)==false ){ cout << "Data folder "<< ss0.str()  <<" is not a folder.\n"<<flush; exit(0); }
	if ( empty(root)==true )		{ cout << "Data folder "<< ss0.str()  <<" is empty.\n"		 <<flush; exit(0); }
																																			if(verbosity>local_verbosity_threshold) cout << "\n Dynamic_slam::Dynamic_slam_chk 1\n" << flush;
	get_all(root, ".txt",   txt);																											// Get lists of files. Gathers all filepaths with each suffix, into c++ vectors.
	get_all(root, ".png",   png);
	get_all(root, ".depth", depth);
																																			if(verbosity>local_verbosity_threshold) cout << "\n Dynamic_slam::Dynamic_slam_chk 2\n" << flush;
																																			if(verbosity>local_verbosity_threshold) cout << "\nDynamic_slam::Dynamic_slam(): "<< png.size()  <<" .png images found in data folder.\t"<<"png[runcl.dataset_frame_num].string()="<< png[runcl.dataset_frame_num].string()  <<flush;
	runcl.baseImage 	= imread(png[runcl.dataset_frame_num].string());																	// Set image params, ref for dimensions and data type.
																																			if (verbosity>local_verbosity_threshold) cout << "\nDynamic_slam::Dynamic_slam_chk 3: runcl.baseImage.size() = "<< runcl.baseImage.size() \
																																				<<" runcl.baseImage.type() = " << runcl.baseImage.type() << "\t"<< checkCVtype(runcl.baseImage.type()) <<flush;
																																			if(verbosity>1) { imshow("runcl.baseImage",runcl.baseImage); cv::waitKey(-1); }
	runcl.initialize_RunCL();
	runcl.allocatemem();																													if (verbosity>local_verbosity_threshold) cout << "\nDynamic_slam::Dynamic_slam_chk 4: runcl.baseImage.size() = "<< runcl.baseImage.size() \
																																				<<" runcl.baseImage.type() = " << runcl.baseImage.type() << "\t"<< checkCVtype(runcl.baseImage.type()) <<flush;
	initialize_camera_vec();
	//initialize_camera();
																																			if(verbosity>local_verbosity_threshold) cout << "\n Dynamic_slam::Dynamic_slam_ finished\n" << flush;
};

void Dynamic_slam::initialize_resultsMat(){	// need to take layer 2, or read it from .json .
	int local_verbosity_threshold = verbosity_mp["Dynamic_slam::initialize_resultsMat"];// verbosity_mp[""];//-2;																										if(verbosity>local_verbosity_threshold) cout << "\n\n Dynamic_slam::initialize_resultsMat()_chk 1" << flush;

	uint reduction 		= obj["sample_layer"].asUInt();		//j_params.int_mp["sample_layer"]; 	//
	uint SE_iter 		= obj["SE_iter"].asUInt();			//j_params.int_mp["SE_iter"];  		//
	int rows 			= 7 * ( runcl.MipMap[reduction*8 + MiM_READ_ROWS] +  runcl.mm_margin );
	int cols 			= SE_iter * ( runcl.MipMap[reduction*8 + MiM_READ_COLS] +  runcl.mm_margin );										if(verbosity>local_verbosity_threshold) cout << "\n Dynamic_slam::initialize_resultsMat()_chk 3: rows = "	<<rows<<", cols = "<<cols<< flush;
	runcl.resultsMat 	= cv::Mat::zeros ( rows, cols , CV_8UC4);																			if(verbosity>local_verbosity_threshold) cout << ",  runcl.resultsMat.size() = "<< runcl.resultsMat.size() 	<< flush;
																																			if(verbosity>local_verbosity_threshold) cout << "\n Dynamic_slam::initialize_resultsMat()_chk  finished\n" 	<< flush;
}

void Dynamic_slam::initialize_camera_vec(){
	int local_verbosity_threshold = verbosity_mp["Dynamic_slam::initialize_camera"];// verbosity_mp[""];//2;
																																			if (verbosity>local_verbosity_threshold) { cout << "\nDynamic_slam::initialize_camera_vec_chk 0:" <<flush;}
	cv::Matx44f k ;//= frame_data.back()->frame_data.K;
																																			if (verbosity>local_verbosity_threshold) { cout << "\nDynamic_slam::initialize_camera_vec_chk 0.1:" <<flush;}
	k = k.zeros();																															// NB In DTAM_opencl, "cameraMatrix" found by convertAhandPovRay, called by fileLoader
	k.operator()(3,3) = 1;
	for (int i=0; i<9; i++){k.operator()(i/3,i%3) = obj["cameraMatrix"][i].asFloat(); }
																																			if (verbosity>local_verbosity_threshold) { cout << "\nDynamic_slam::initialize_camera_vec_chk 0.2:" <<flush;}
cout<<"\n frame_data.size() = "<<frame_data.size()<<flush;
	//frame_data.back()->frame_data.K 	= k;

	R 								= cv::Mat::eye(3,3 , CV_32FC1);																				// intialize ground truth extrinsic data, NB Mat (int rows, int cols, int type)
	T 								= cv::Mat::zeros(3,1 , CV_32FC1);
																																			if (verbosity>local_verbosity_threshold) { cout << "\nDynamic_slam::initialize_camera_vec_chk 1:" <<flush;}
	pose_datum data;
	data.inv_K 						= generate_invK_(k);
	data.pose						= Matx44f_eye;               					// pose in abs coords. (not pose2pose from prev_frame, nor from keyframe) ?
	data.inv_pose					= Matx44f_eye;
	data.keyframe2pose				= Matx44f_eye;
	data.keyframe2pose_algebra		= {0};
	data.pose_from_start			= Matx44f_eye;
	data.K2K						= Matx44f_eye;

	frame_datum datum;
	datum.frame_data	= data;
	datum.frame_data_GT	= data;
	datum.error_data	= data;
	frame_data.push_back(datum);
																																			if (verbosity>local_verbosity_threshold) { cout << "\nDynamic_slam::initialize_camera_vec_chk 2:" <<flush;}
	getFrameData_vec();
																																			if (verbosity>local_verbosity_threshold) { cout << "\nDynamic_slam::initialize_camera_vec_chk 3:" <<flush;}

																																			if(verbosity>local_verbosity_threshold) {
																																				PRINT_MATX44F(frame_data.back().frame_data_GT.K,);
																																				PRINT_MATX44F(frame_data.back().frame_data_GT.inv_K,);
																																			}
	generate_SE3_k2k_vec( SE3_k2k );																											// fills float[96] ie 6xfloat[16] from conf.json intrinsic camera matrix + SE3 increments.
	runcl.precom_param_maps( SE3_k2k );																										// GPU computes J(u,v/SE3) Jacobian of optical flow wrt SE3.
	getFrame();
																																			if (verbosity>local_verbosity_threshold){ cout << "\nDynamic_slam::initialize_camera_vec Finished:" <<flush; }
}

void Dynamic_slam::initialize_camera(){
	int local_verbosity_threshold = verbosity_mp["Dynamic_slam::initialize_camera"];// verbosity_mp[""];//2;
																																			if (verbosity>local_verbosity_threshold) { cout << "\nDynamic_slam::initialize_camera_chk 0:" <<flush;}
	K = K.zeros();																															// NB In DTAM_opencl, "cameraMatrix" found by convertAhandPovRay, called by fileLoader
	K.operator()(3,3) = 1;
	for (int i=0; i<9; i++){K.operator()(i/3,i%3) = obj["cameraMatrix"][i].asFloat(); }		//j_params.float_vec_mp["cameraMatrix"][i];   //
	old_K		= K;
	generate_invK();
																																			if(verbosity>local_verbosity_threshold) {
																																				PRINT_MATX44F(K,);
																																				print_json_float_9(obj, "cameraMatrix");
																																			}
	R 							= cv::Mat::eye(3,3 , CV_32FC1);																				// intialize ground truth extrinsic data, NB Mat (int rows, int cols, int type)
	T 							= cv::Mat::zeros(3,1 , CV_32FC1);

	keyframe_pose2pose 			= Matx44f_eye;
	pose2pose					= Matx44f_eye;
	K2K							= Matx44f_eye;
																																			if (verbosity>local_verbosity_threshold) { cout << "\nDynamic_slam::initialize_camera_chk 1:" <<flush;
																																				PRINT_MAT33F(R,);
																																			}
	// TODO Also initialize any lens distorsion, vignetting. etc
	getFrameData();																															// Loads GT depth of the new frame. NB depends on image.size from getFrame().

	K_start 					= K_GT; 																									// NB The same frame will be loaded to the opposite imgmem, on the first iteration of Dynamic_slam::nextFrame()
	inv_K_start 				= inv_K_GT;
	pose_start 					= pose_GT;
	inv_pose_start 				= inv_pose_GT;

	K2K_GT 						= Matx44f_eye; 																								//  = cv::Matx44f::eye();
	K2K_start 					= Matx44f_eye;
	pose2pose_GT 				= Matx44f_eye;
	pose2pose_start 			= Matx44f_eye;
	pose2pose_accumulated 		= Matx44f_eye;
	pose2pose_accumulated_GT 	= Matx44f_eye;
																																			if (verbosity>local_verbosity_threshold){ cout << "\nDynamic_slam::initialize_camera_chk 5:" <<flush; PRINT_MATX44F(pose2pose_accumulated,); }
	generate_SE3_k2k( SE3_k2k );
	runcl.precom_param_maps( SE3_k2k );
	getFrame();																																// Causes the first frame to be loaded into first imgmem, and prepared.
}

int Dynamic_slam::nextFrame() {
	int local_verbosity_threshold = verbosity_mp["Dynamic_slam::nextFrame"];// -2;
																																			if(verbosity>local_verbosity_threshold) cout << "\n Dynamic_slam::nextFrame_chk 0,  runcl.dataset_frame_num="<<runcl.dataset_frame_num<<" \n" << flush; //  runcl.frame_bool_idx="<<runcl.frame_bool_idx<<"
																						auto step_0 = high_resolution_clock::now();
	frame_datum new_frame;
	new_frame.keyframe_index		= keyframe_data.size();
	frame_data.push_back(new_frame); //////////////////////////////////////////

	predictFrame_vec();																	auto step_1 = high_resolution_clock::now();			// updates pose2pose for next frame in cost volume.
	//predictFrame();
	getFrameData_vec();																	auto step_2 = high_resolution_clock::now();			// Loads GT depth of the new frame. NB depends on image.size from getFrame().
	//getFrameData();

	if(obj["use_GT_pose"].asBool() == true )		{	use_GT_pose_vec();	}			auto step_3 = high_resolution_clock::now();			// use_GT_pose();
	getFrame();																			auto step_4 = high_resolution_clock::now();
	if(obj["Artif_pose_err_bool"].asBool() == true ){ 	artificial_pose_error_vec();}	auto step_5 = high_resolution_clock::now();
	estimateSE3();
	//estimateSE3_LK();
																						auto step_6 = high_resolution_clock::now();			// own thread ? num iter ?
	//estimateCalibration(); 																												// own thread, one iter.
	if(verbosity>local_verbosity_threshold){
		report_GT_pose_error();
		//display_frame_resluts();
	}
	////////////////////////////////// Test kernels
	//runcl.atomic_test1();
	//runcl.atomic_test2();
	////////////////////////////////// Parallax depth mapping
																						auto step_7 = high_resolution_clock::now();
	updateDepthCostVol();																auto step_8 = high_resolution_clock::now();			// Update cost vol with the new frame, and repeat optimization of the depth map.
																																			// NB Cost vol needs to be initialized on a particular keyframe.
	getNextFrameProfile(step_0, step_1, step_2, step_3, step_4, step_5, step_6, step_7, step_8);											// A previous depth map can be transfered, and the updated depth map after each frame, can be used to track the next frame.
	runcl.costvol_frame_num++;
	runcl.dataset_frame_num++;
	return(0);																																// NB option to return an error that stops the main loop.
};

void Dynamic_slam::getNextFrameProfile(time_pt step_0, time_pt step_1, time_pt step_2, time_pt step_3, time_pt step_4, time_pt step_5, time_pt step_6, time_pt step_7, time_pt step_8){
	stringstream ss;
	ss 	<<"\nExecution times:(microseconds)####################################################"
		<<"\npredictFrame()                                    "	<<  duration_cast<microseconds>(step_1 - step_0).count()
		<<"\ngetFrameData()...................................."	<<  duration_cast<microseconds>(step_2 - step_1).count()
		<<"\nuse_GT_pose()                                     "	<<  duration_cast<microseconds>(step_3 - step_2).count()
		<<"\ngetFrame()........................................"	<<  duration_cast<microseconds>(step_4 - step_3).count()
		<<"\nartificial_pose_error()                           "	<<  duration_cast<microseconds>(step_5 - step_4).count()
		<<"\nestimateSE3_LK()                                  "	<<  duration_cast<microseconds>(step_6 - step_5).count()
		<<"\nreport_GT_pose_error() &  display_frame_resluts() "	<<  duration_cast<microseconds>(step_7 - step_6).count()
		<<"\nupdateDepthCostVol()                              "	<<  duration_cast<microseconds>(step_8 - step_7).count()
		<<"\nTotal                                             "	<<  duration_cast<microseconds>(step_8 - step_0).count()
		<<"\n##################################################################################";
	cout << ss.str() << flush;
}

void Dynamic_slam::getFrame() { // can load use separate CPU thread(s) ?  // NB also need to change type CV_8UC3 -> CV_16FC3
	int local_verbosity_threshold = verbosity_mp["Dynamic_slam::getFrame"];// -1;
																																			if(verbosity>local_verbosity_threshold){ cout << "\n Dynamic_slam::getFrame_chk 0.  runcl.dataset_frame_num = "<< runcl.dataset_frame_num  << flush;
																																				// # load next image to buffer NB load at position [log_2 index]
																																				// See CostVol::updateCost(..) & RunCL::calcCostVol(..)
																																				cout << "\n\nDynamic_slam::getFrame()";
																																				cout << "\nruncl.baseImage.size() =" 	<< runcl.baseImage.size();
																																				cout << "\nruncl.baseImage_size =" 		<< runcl.baseImage_size;
																																				cout << "\nruncl.baseImage_type =" 		<< runcl.baseImage_type ;
																																				cout << "\nruncl.image_size_bytes =" 	<< runcl.image_size_bytes ;
																																				cout << endl;
																																				cout << "\nruncl.mm_Image_type =" 		<< runcl.mm_Image_type ;
																																				cout << "\nruncl.mm_size_bytes_C3 =" 	<< runcl.mm_size_bytes_C3 ;
																																				cout << "\nruncl.mm_size_bytes_C1 =" 	<< runcl.mm_size_bytes_C1 ;
																																				cout << "\nruncl.mm_vol_size_bytes =" 	<< runcl.mm_vol_size_bytes ;
																																				cout << "\nruncl.mm_Image_size =" 		<< runcl.mm_Image_size ;
																																				cout << "\n" << flush ;
																																			}
	image = imread(png[runcl.dataset_frame_num].string());																					if(verbosity>local_verbosity_threshold){
																																				cout << "\n Dynamic_slam::getFrame_chk 0.5, Image file = " << png[runcl.dataset_frame_num].string() << "\t" << flush;
																																			}
																																			if (image.type()!= runcl.baseImage.type() || image.size()!=runcl.baseImage.size() ) {
																																				cout<< "\n\nError: Dynamic_slam::getFrame(), runcl.dataset_frame_num = " << runcl.dataset_frame_num << " : missmatched. runcl.baseImage.size()="<<runcl.baseImage.size()<<\
																																				", image.size()="<<image.size()<<", runcl.baseImage.type()="<<runcl.baseImage.type()<<", image.type()="<<image.type()<<"\n\n"<<flush;
																																				exit(0);
																																			}
																																			//image.convertTo(image, CV_16FC3, 1.0/256, 0.0); // NB cv_16FC3 is preferable, for faster half precision processing on AMD, Intel & ARM GPUs.
	runcl.loadFrame( image );																												// NB Nvidia GeForce have 'Tensor Compute" FP16, accessible by PTX. AMD have RDNA and CDNA. These need PTX/assembly code and may use BF16 instead of FP16.
																																			// load a basic image in CV_8UC3, then convert on GPU to 'half'
	runcl.cvt_color_space( );
	runcl.blur_image();
	runcl.mipmap_linear();
	runcl.img_variance();
	runcl.img_gradients();
																																			// # Get 1st & 2nd order image gradients of MipMap
																																			// see CostVol::cacheGValues(), RunCL::cacheGValue2 & __kernel void CacheG3
																																			if(verbosity>local_verbosity_threshold){ cout << "\n Dynamic_slam::getFrame_chk 1  Finished\n" << flush;}
}

cv::Matx44f Dynamic_slam::getPose(Mat R, Mat T){																							// Mat R, Mat T, Matx44f& pose  // NB Matx::operator()() does not copy, but creates a submatrix. => would be updated when R & T are updated.
	int local_verbosity_threshold = verbosity_mp["Dynamic_slam::getPose"];// 1;
	cv::Matx44f pose;
	for (int i=0; i<9; i++) pose.operator()(i/3,i%3) = 1 * R.at<float>(i/3,i%3);															if(verbosity>local_verbosity_threshold) { PRINT_MAT33F(R,);  PRINT_MATX44F(pose,); }
	for (int i=0; i<3; i++) pose.operator()(i,3)      = T.at<float>(i);
	for (int i=0; i<3; i++) pose.operator()(3,i)      = 0.0f;
	pose.operator()(3,3) = 1.0f;
	return pose;
}

cv::Matx44f Dynamic_slam::getInvPose(cv::Matx44f pose) {	// Matx44f pose, Matx44f& inv_pose
	int local_verbosity_threshold = verbosity_mp["Dynamic_slam::getInvPose"];

	cv::Matx44f local_inv_pose;
	cv::Matx33f local_rotation;
	cv::Matx31f local_translation;
	cv::Matx31f inv_local_translation;

	for (int i=0; i<3; i++) { for (int j=0; j<3; j++)	{    local_inv_pose.operator()(i,j) = pose.operator()(j,i); } }
	for (int i=0; i<3; i++) { for (int j=0; j<3; j++)	{    local_rotation.operator()(i,j) = pose.operator()(i,j); } }
	for (int i=0; i<3; i++) 							{ local_translation.operator()(i,0) = pose.operator()(i,3); }

	inv_local_translation = - local_rotation.t() * local_translation;
	for (int i=0; i<3; i++) local_inv_pose.operator()(i,3) = inv_local_translation.operator()(i,0);
	for (int i=0; i<4; i++) local_inv_pose.operator()(3,i) =                  pose.operator()(3,i);
																																			if(verbosity>local_verbosity_threshold){
																																				PRINT_MATX31F(local_translation,"Dynamic_slam::getInvPose(cv::Matx44f pose)" );
																																				PRINT_MATX31F(inv_local_translation,"Dynamic_slam::getInvPose(cv::Matx44f pose)" );
																																				PRINT_MATX44F(local_inv_pose, "Dynamic_slam::getInvPose(cv::Matx44f pose)" );
																																			}
	return local_inv_pose;
																																			/*
																																			*  Inverse of a transformation matrix:
																																			*  http://www.info.hiroshima-cu.ac.jp/~miyazaki/knowledge/teche0053
																																			*
																																			*   {     |   }-1       {       |        }
																																			*   {  R  | t }     =   {  R^T  |-R^T .t }
																																			*   {_____|___}         {_______|________}
																																			*   {0 0 0| 1 }         {0  0  0|    1   }
																																			*
																																			*/
}

void Dynamic_slam::getFrameData_vec(){
	int local_verbosity_threshold = verbosity_mp["Dynamic_slam::getFrameData"];
																																			if(verbosity>local_verbosity_threshold) cout << "\n Dynamic_slam::getFrameData_vec_chk 0.  runcl.dataset_frame_num = "<< runcl.dataset_frame_num <<flush;
	std::string str = txt[runcl.dataset_frame_num].c_str();																					// grab .txt file from array of files (e.g. "scene_00_0000.txt")
    char        *ch = new char [str.length()+1];
    std::strcpy (ch, str.c_str());
	cv::Mat T_alt;
    convertAhandaPovRayToStandard(verbosity_mp, ch,R,T,cameraMatrix);
	free(ch);
																																			if(verbosity>local_verbosity_threshold) {
																																				cout << "\n Dynamic_slam::getFrameData_vec_chk 1";
																																				cout << "\n R = " << R;
																																				cout << "\n T = " << T;
																																				cout << "\n cameraMatrix = " << cameraMatrix;
																																			}

	cv::Matx44f K_GT = cv::Matx44f::zeros();
	for (int i=0; i<3; i++){
		for (int j=0; j<3; j++){
			K_GT.operator()(i,j) = cameraMatrix.at<float>(i,j);
			cout << ", " <<  cameraMatrix.at<float>(i,j);
		}
	}K_GT.operator()(3,3) = 1;

	cv::Matx44f		K2K_GT;
	cv::Matx44f		pose2pose_GT;
	cv::Matx44f		keyframe_pose2pose_GT;
	cv::Matx44f		keyframe_K2K_GT;
	cv::Matx44f		pose2pose_accumulated_GT;

	if ( runcl.dataset_frame_num > 0 ){
		K2K_GT 						= old_K_GT * old_pose_GT * inv_pose_GT * inv_K_GT;  // ?
		pose2pose_GT 				= old_pose_GT * inv_pose_GT;
		keyframe_pose2pose_GT 		= pose_GT * keyframe_inv_pose_GT;
		keyframe_K2K_GT 			= K_GT * pose_GT * keyframe_inv_pose_GT * keyframe_inv_K_GT;
		pose2pose_accumulated_GT 	= frame_data.back().frame_data_GT.pose_from_start * pose2pose_GT;	// .back()
	}else{
		K2K_GT 						= cv::Matx44f::eye();												// If first frame, set to identity matrix.
		pose2pose_GT 				= cv::Matx44f::eye();
		keyframe_pose2pose_GT 		= cv::Matx44f::eye();
		keyframe_K2K_GT 			= cv::Matx44f::eye();
		pose2pose_accumulated_GT 	= cv::Matx44f::eye();
	}


	if ( runcl.baseImage.empty() ) {cerr << "\nDynamic_slam::getFrameData_vec():   Error runcl.baseImage.empty() "<<flush;  exit(1); }
	int r = runcl.baseImage.rows;  //image.rows;
    int c = runcl.baseImage.cols;  //image.cols;
	depth_GT = loadDepthAhanda(verbosity_mp, depth[runcl.dataset_frame_num].string(), r,c,cameraMatrix);
	runcl.load_GT_depth(depth_GT, invert_GT_depth);

	frame_datum new_frame;
	new_frame.keyframe_index						=	runcl.dataset_frame_num;						// uint
	new_frame.frame_data_GT.K 						=   K_GT;											// cv::Matx44f
	new_frame.frame_data_GT.inv_K 					=   generate_invK_(new_frame.frame_data_GT.K);
	new_frame.frame_data_GT.pose 					=   getPose(R, T);
	new_frame.frame_data_GT.inv_pose 				=   getInvPose(new_frame.frame_data_GT.pose);
	new_frame.frame_data_GT.keyframe2pose_algebra 	=   PToLie(pose2pose_GT);							// cv::Matx16f
	new_frame.frame_data_GT.pose_from_start			=   pose2pose_accumulated_GT;
	new_frame.frame_data_GT.K2K 					=   K2K_GT;

	frame_data.push_back( new_frame );

																																			if(verbosity>local_verbosity_threshold) {
																																				PRINT_MATX44F(K_GT,);
																																				PRINT_MATX44F(new_frame.frame_data_GT.K,);
																																				PRINT_MATX44F(new_frame.frame_data_GT.inv_K,);
																																				PRINT_MATX44F(new_frame.frame_data_GT.pose,);
																																				PRINT_MATX44F(new_frame.frame_data_GT.inv_pose,);

																																				PRINT_MATX44F(frame_data.back().frame_data_GT.pose ,);
																																				PRINT_MATX44F(frame_data.back().frame_data_GT.K,);
																																				PRINT_MATX44F(frame_data.back().frame_data_GT.inv_K,);

																																				cout << "\nframe_data.size()="<<frame_data.size()<<flush;
																																				vector<frame_datum>::iterator ptr;
																																				int iteration=0;
																																				for (ptr = frame_data.begin(); ptr < frame_data.end(); ptr++, iteration++) {
																																					cout<<"\n\n### iteration="<< iteration  << flush;
																																					PRINT_MATX44F(frame_data[iteration].frame_data_GT.pose ,);
																																					PRINT_MATX44F(ptr->frame_data_GT.pose ,);
																																					PRINT_MATX44F(ptr->frame_data_GT.K,);
																																					PRINT_MATX44F(ptr->frame_data_GT.inv_K,);
																																				}
																																			}
																																			if(verbosity>local_verbosity_threshold) cout << "\n Dynamic_slam::getFrameData_vec_chk Finished "<<flush;
}


void Dynamic_slam::getFrameData(){  // can load use separate CPU thread(s) ?
	int local_verbosity_threshold = verbosity_mp["Dynamic_slam::getFrameData"];// -2;
																																			if(verbosity>local_verbosity_threshold) cout << "\n Dynamic_slam::getFrameData_chk 0.  runcl.dataset_frame_num = "<< runcl.dataset_frame_num <<flush;
	R.copyTo(old_R);																														// get ground truth frame to frame pose transform
	T.copyTo(old_T);
	old_pose_GT 		= pose_GT;
	inv_old_pose_GT 	= inv_pose_GT;																										// NB Confirmed this copies the data  NOT just the pointer.
	old_K_GT			= K_GT;
	inv_old_K_GT		= inv_K_GT;
																																			if(verbosity>local_verbosity_threshold){
																																				cout << "\n Dynamic_slam::getFrameData_chk 0.5, data file = " << txt[runcl.dataset_frame_num].c_str() << "\t" << flush;
																																			}
	std::string str = txt[runcl.dataset_frame_num].c_str();																					// grab .txt file from array of files (e.g. "scene_00_0000.txt")
    char        *ch = new char [str.length()+1];
    std::strcpy (ch, str.c_str());
	cv::Mat T_alt;
    convertAhandaPovRayToStandard(verbosity_mp, ch,R,T,cameraMatrix);
	free(ch);
																																			if(verbosity>local_verbosity_threshold) {
																																				cout << "\nR=";
																																				for (int i=0; i<3; i++){
																																					cout <<"\n(";
																																					for (int j=0; j<3; j++){
																																						cout << ", " << R.at<float>(i,j);
																																					}cout << ")\n";
																																				}cout<<endl<<flush;

																																				cout << "\nT=(";
																																				for (int i=0; i<3; i++) cout << ", " << T.at<float>(i);
																																				cout << ")\n"<<endl<<flush;
																																			}
	pose_GT 		= getPose(R, T);
	inv_pose_GT 	= getInvPose(pose_GT);
																																			if(verbosity>local_verbosity_threshold) { cout << "\n Dynamic_slam::getFrameData_chk 0.1.2"<<flush;
																																				PRINT_MATX44F(pose_GT,);
																																				PRINT_MATX44F(inv_pose_GT,);
																																			}
																																			if(verbosity>local_verbosity_threshold) cout << "\n Dynamic_slam::getFrameData_chk 0.2"<<flush;
	K_GT.zeros();
	for (int i=0; i<3; i++){
		for (int j=0; j<3; j++){
			K_GT.operator()(i,j) = cameraMatrix.at<float>(i,j);
		}
	}K_GT.operator()(3,3) = 1;
	K = K_GT;
	generate_invK();
	inv_K_GT = inv_K; 																														// TODO change this when we autocalibrate K.
																																			if(verbosity>local_verbosity_threshold) cout << "\n Dynamic_slam::getFrameData_chk 0.4"<<flush; // K2K
	K2K_GT 					= old_K_GT * old_pose_GT * inv_pose_GT * inv_K_GT;
	//   wrong way around :   old_K_GT * old_pose_GT * inv_pose_GT * inv_K_GT;																// TODO  Issue, not valid for first frame, pose  should be identty, Also what would estimate SE3 do ?

	pose2pose_GT 			= old_pose_GT * inv_pose_GT;
	keyframe_pose2pose_GT 	= pose_GT * keyframe_inv_pose_GT;
	keyframe_K2K_GT 		= K_GT * pose_GT * keyframe_inv_pose_GT * keyframe_inv_K_GT;
																																			if(verbosity>local_verbosity_threshold) { cout << "\n Dynamic_slam::getFrameData_chk 0.1.2"<<flush;
																																				cout << "\truncl.costvol_frame_num = " << runcl.costvol_frame_num << flush;
																																				PRINT_MATX44F(K_GT,);
																																				PRINT_MATX44F(pose_GT,);
																																				PRINT_MATX44F(keyframe_pose_GT,);
																																				PRINT_MATX44F(keyframe_inv_pose_GT,);
																																				PRINT_MATX44F(keyframe_inv_K_GT,);
																																				PRINT_MATX44F(keyframe_K2K_GT,);
																																			}

	if (runcl.dataset_frame_num > 0 ) {  pose2pose_accumulated_GT = pose2pose_accumulated_GT * pose2pose_GT;	}							// Tracks pose tranform from first frame.
																																			if(verbosity>local_verbosity_threshold){ cout << "\n Dynamic_slam::getFrameData_chk 1,"<<flush;
																																				PRINT_MATX44F(old_K_GT,);
																																				PRINT_MATX44F(old_pose_GT,);
																																				PRINT_MATX44F(inv_pose_GT,);
																																				PRINT_MATX44F(inv_K_GT,);
																																				PRINT_MATX44F(K_GT,);
																																				PRINT_MATX44F(K2K_GT,);
																																				PRINT_MATX44F(pose2pose_GT,);
																																				PRINT_MAT33F(cameraMatrix,);
																																			}
																																			// get ground truth depth map
																																			if(verbosity>local_verbosity_threshold){ cout << "\n Dynamic_slam::getFrameData_chk 2,"<<flush;}
	int r = runcl.baseImage.rows;  //image.rows;
    int c = runcl.baseImage.cols;  //image.cols;
	depth_GT = loadDepthAhanda(verbosity_mp, depth[runcl.dataset_frame_num].string(), r,c,cameraMatrix);
																																			if(verbosity>local_verbosity_threshold){ cout << "\n Dynamic_slam::getFrameData_chk 3,"<<flush;//}
																																				stringstream ss;
																																				stringstream png_ss;
																																				boost::filesystem::path folder_tiff = runcl.paths.at("depth_GT");
																																				string type_string = checkCVtype(depth_GT.type() );
																																				ss << "/" << folder_tiff.filename().string() << "_original_" << runcl.dataset_frame_num <<"type_"<<type_string;
																																				png_ss << "/" << folder_tiff.filename().string() << "_original_" << runcl.dataset_frame_num;
																																				boost::filesystem::path folder_png = folder_tiff;
																																				folder_tiff += "/tiff/";
																																				folder_tiff += ss.str();
																																				folder_tiff += ".tiff";
																																				//folder_png  += "/png/";

																																				folder_png  += png_ss.str();
																																				folder_png  += ".png";

																																			//if(verbosity>local_verbosity_threshold){ cout << "\n Dynamic_slam::getFrameData_chk 4,"<<flush;
																																				cout << "\n runcl.frame_num = " << runcl.dataset_frame_num << ",  depth[runcl.frame_num].string() = " << depth[runcl.dataset_frame_num].string() << flush;
																																				cout << "\n depth_GT.size() = " << depth_GT.size() << ",  depth_GT.type() = "<< type_string << ",  depth_GT.empty() = " <<  depth_GT.empty()   << flush;
																																				cout << "\n " << folder_png.string() << flush;

																																				cv::imwrite(folder_png.string(), depth_GT );
																																				cv::imwrite(folder_tiff.string(), depth_GT );
																																			}

	//runcl.loadFrameData(depth_GT, K2K, pose2pose);
	runcl.load_GT_depth(depth_GT, invert_GT_depth);																							// loads to depth_mem_GT buffer.
																																			if(verbosity>local_verbosity_threshold) cout << "\n Dynamic_slam::getFrameData finished,#################################################################"<<flush;
}

void Dynamic_slam::use_GT_pose_vec(){
	frame_data.back().frame_data = frame_data.back().frame_data_GT;
}

void Dynamic_slam::use_GT_pose(){
	int local_verbosity_threshold = verbosity_mp["Dynamic_slam::use_GT_pose"];// -1;
																																			if(verbosity>local_verbosity_threshold) cout << "\n Dynamic_slam::use_GT_pose_chk_0,"<<flush;
	old_K				= old_K_GT;
	inv_K				= inv_K_GT;
	old_pose			= old_pose_GT;
	inv_pose			= inv_pose_GT;

	pose 				= pose_GT;
	inv_pose			= inv_pose_GT;
	keyframe_K2K 		= keyframe_K2K_GT;

	keyframe_pose2pose 	= keyframe_pose2pose_GT;
	// pose2pose = keyframe_pose2pose_GT;

	for (int i=0; i<16; i++){ runcl.fp32_k2keyframe[i] = keyframe_K2K.operator()(i/4, i%4);}
																																			if(verbosity>local_verbosity_threshold){
																																				PRINT_MATX44F(K,);
																																				PRINT_MATX44F(inv_K,);
																																				PRINT_MATX44F(keyframe_pose2pose,);
																																				PRINT_FLOAT_16(runcl.fp32_k2keyframe,);
																																			}
																																			if(verbosity>local_verbosity_threshold) cout << "\n Dynamic_slam::use_GT_pose finished,"<<flush;
}


//////

void Dynamic_slam::estimateCalibration(){
	int local_verbosity_threshold = verbosity_mp["Dynamic_slam::estimateCalibration"];
// # Get 1st & 2nd order gradients wrt calibration parameters.
//


// # Take one dammped least squares step of calibration.
//

}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// ## Regularize Maps : AbsDepth, GradDepth, SurfNormal, RelVel,
void Dynamic_slam::SpatialCostFns(){
	int local_verbosity_threshold = verbosity_mp["Dynamic_slam::SpatialCostFns"];
// # Spatial cost functions
// see CostVol::updateQD(..), RunCL::updateQD(..) & __kernel void UpdateQD(..)

}

void Dynamic_slam::ParsimonyCostFns(){
	int local_verbosity_threshold = verbosity_mp["Dynamic_slam::ParsimonyCostFns"];
// # Parsimony cost functions : NB Bin sort pixels to find non-spatial neighbours
// see SIFS for priors & Morphogenesis for BinSort

}

void Dynamic_slam::ExhaustiveSearch(){
	int local_verbosity_threshold = verbosity_mp["Dynamic_slam::ExhaustiveSearch"];
// # Update A : exhaustive search on cost vol with cost fns -> update maps.
// see CostVol::updateA(..), RunCL::updateA(..) & __kernel void UpdateA2(..)

}

void Dynamic_slam::getResult(){
	int local_verbosity_threshold = verbosity_mp["Dynamic_slam::getResult"];
	if(verbosity>local_verbosity_threshold){
		stringstream ss;
		ss << "getResult"<< runcl.save_index << "_QD_count_" << runcl.QD_count <<"_epsilon_"<<epsilon<<"_sigmaQ_"<<sigma_q<<"_D_"<<sigma_d<<"_theta_"<<theta;
																																				//int this_count = save_index * 1000 + QD_count;
																																				//ss << save_index << "_QD_count_" << QD_count;
		cv::Size q_size( runcl.mm_Image_size.width, 2* runcl.mm_Image_size.height ); 			// 2x sized for qx and qy.
		runcl.DownloadAndSave(runcl.dmem,   ss.str(), runcl.paths.at("dmem"),    runcl.mm_size_bytes_C1 , runcl.mm_Image_size , CV_32FC1, false ,    runcl.fp32_params[MAX_INV_DEPTH]  );
		runcl.DownloadAndSave(runcl.amem,   ss.str(), runcl.paths.at("amem"),    runcl.mm_size_bytes_C1 , runcl.mm_Image_size , CV_32FC1, false ,    runcl.fp32_params[MAX_INV_DEPTH]  );

		//DownloadAndSave(qmem,   ss.str(), paths.at("qmem"),  2*mm_size_bytes_C1 , q_size        , CV_32FC1, false , -1*fp32_params[MAX_INV_DEPTH]  );
		//DownloadAndSave(qmem2,   ss.str(), paths.at("qmem2"),2*mm_size_bytes_C1 , q_size        , CV_32FC1, false , -1*fp32_params[MAX_INV_DEPTH]  );  // 1/uint_params[MM_PIXELS]
		cout<<"\nRunCL::updateQD_chk3_finished\n"<<flush;
	}
};
