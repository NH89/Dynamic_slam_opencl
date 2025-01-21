
#include "Dynamic_slam.hpp"

#include <fstream>
#include <iomanip>   // for std::setprecision, std::setw

using namespace cv;
using namespace std;


Dynamic_slam::~Dynamic_slam(){ runcl.~RunCL(); };


Dynamic_slam::Dynamic_slam( Json::Value obj_  ):   runcl( obj_  ) {  //, int_map verbosity_mp_
	obj = obj_;																																// NB save obj_ to class member obj, so that it persists within this Dynamic_slam object.
	verbosity 						= obj["verbosity"].asInt();
	int local_verbosity_threshold 	= V_DYNAMIC_SLAM_DYNAMIC_SLAM;//verbosity_mp["Dynamic_slam::Dynamic_slam"];
																																			if(verbosity>local_verbosity_threshold) cout << "\f Dynamic_slam::Dynamic_slam_chk -1\n" << flush;
	runcl.dataset_frame_num 		= obj["data_file_offset"].asUInt();
	invert_GT_depth  				= obj["invert_GT_depth"].asBool();

	SE3_start_layer 				= obj["SE3_start_layer"].asUInt();
    SE3_stop_layer 					= obj["SE3_stop_layer"].asUInt();
	SE_iter_per_layer 				= obj["SE_iter_per_layer"].asUInt();
    SE_iter 						= obj["SE_iter"].asUInt();
	SE_factor						= obj["SE_factor"].asFloat();
																																			if(verbosity>local_verbosity_threshold) cout << "\n  Dynamic_slam::Dynamic_slam_chk -0.5\n" << flush;

	for (int layer=0; layer<MAX_LAYERS; layer++){for (int chan=0; chan<3; chan++)	SE3_Rho_sq_threshold[layer][chan]  	= obj["SE3_Rho_sq_threshold"][layer][chan].asFloat();  }
																																			if(verbosity>local_verbosity_threshold) cout << "\n  Dynamic_slam::Dynamic_slam_chk -0.4\n" << flush;

	for (int se3=0; se3<6; se3++)													SE3_update_dof_weights[se3] 		= obj["SE3_update_dof_weights"][se3].asFloat();
																																			if(verbosity>local_verbosity_threshold) cout << "\n  Dynamic_slam::Dynamic_slam_chk -0.3\n" << flush;

    for (int layer2 = 0; layer2 < 5 ; layer2++) {
		SE3_update_layer_weights[layer2] 	= obj["SE3_float update_layer_weights"][layer2].asFloat();

		cout << "\nMAX_LAYERS = " << MAX_LAYERS << ",\t layer2 = " << layer2 << "\t  SE3_update_layer_weights[layer2] = " <<  SE3_update_layer_weights[layer2]  <<flush;

		if (layer2 > 6) {
			cout << "\t (layer2 > 5)" << flush;
			break;
		}
	}
																																			if(verbosity>local_verbosity_threshold) {cout << "\n Dynamic_slam::Dynamic_slam_chk 0,  SE3_Rho_sq_threshold[i][j] = ";
																																				for (int i=0; i<5; i++){cout << "( "; for (int j=0; j<3; j++) {

																																					std::cout << "dummy text" << 2 ;

																																					cout << ", [" << i <<"]["<<j<<"]" << SE3_Rho_sq_threshold[i][j]; }   cout << " )";

																																				}
																																				cout << ",\t SE_factor = "<<SE_factor;
																																				cout << endl << flush;
																																			}
	stringstream  ss0;
	ss0 << obj["data_path"].asString()  <<  obj["data_file"].asString();																	// Collect the filenames of all the input images, plus ground truth files for camera data and depth maps- #####################
	rootpath 	= ss0.str();
	root 		= rootpath;

	if ( exists(root)==false )		{ cout << "Data folder "<< ss0.str()  <<" does not exist.\n" <<flush; runcl.exit_(0); }
	if ( is_directory(root)==false ){ cout << "Data folder "<< ss0.str()  <<" is not a folder.\n"<<flush; runcl.exit_(0); }
	if ( empty(root)==true )		{ cout << "Data folder "<< ss0.str()  <<" is empty.\n"		 <<flush; runcl.exit_(0); }
																																			if(verbosity>local_verbosity_threshold) cout << "\n Dynamic_slam::Dynamic_slam_chk 1\n" << flush;
	get_all(root, ".txt",   txt);																											// Get lists of files. Gathers all filepaths with each suffix, into c++ vectors.
	get_all(root, ".png",   png);
	get_all(root, ".depth", depth);
																																			if(verbosity>local_verbosity_threshold){cout << "\n Dynamic_slam::Dynamic_slam_chk 2\n" << flush;
																																				cout << "\nDynamic_slam::Dynamic_slam(): "<< png.size()  <<" .png images found in data folder.\t"
																																				<<"png[runcl.dataset_frame_num].string()="<< png[runcl.dataset_frame_num].string()  <<flush;
																																			}
	runcl.initialize_RunCL( imread(png[runcl.dataset_frame_num].string() ) );																// Set image params, ref for dimensions and data type. ########################################################################
	runcl.allocatemem();																													// Allocate buffers on the GPU ######
	runcl.compute_lookup_table(runcl.mm_start, runcl.mm_stop);																				// Calls kernel. Used for disparity kernels now, maybe  more later...

	initialize_camera_vec();
	initialize_keyframe_vec();																												// First keyframe
	initialize_resultsMat();
																																			if(verbosity>local_verbosity_threshold) cout << "\n Dynamic_slam::Dynamic_slam_ finished "
																																				<< "#####################################################################################\f" << flush;
};

void Dynamic_slam::initialize_camera_vec(){
	int local_verbosity_threshold = V_DYNAMIC_SLAM_INITIALIZE_CAMERA;//verbosity_mp["Dynamic_slam::initialize_camera"];
																																			if (verbosity>local_verbosity_threshold) { cout << "\fDynamic_slam::initialize_camera_vec_chk 0:" <<flush;}
	cv::Matx44f k 					= Matx44f_eye;		//= frame_data.back()->frame_data.K;												// NB In DTAM_opencl, "cameraMatrix" found by convertAhandPovRay, called by fileLoader
	for (int i=0; i<9; i++){ k.operator()(i/3,i%3) = obj["cameraMatrix"][i].asFloat(); }
																																			if (verbosity>local_verbosity_threshold) { cout << "\nDynamic_slam::initialize_camera_vec_chk 0.1:" <<flush;
																																				cout<<"\n frame_data.size() = "<<frame_data.size()<<flush;
																																			}
	R 								= cv::Mat::eye(3,3 , CV_32FC1);																			// intialize ground truth extrinsic data, NB Mat (int rows, int cols, int type)
	T 								= cv::Mat::zeros(3,1 , CV_32FC1);
																																			if (verbosity>local_verbosity_threshold) { cout << "\nDynamic_slam::initialize_camera_vec_chk 1:" <<flush;}
	frame_datum 			datum 	= {};																									// default initialization, to values in header, or zero if not set in header.
	datum.keyframe_index			= 0 ;								// Expects that this frame will be used for new vector of keyframes.
																																			cout << "\n\n datum.keyframe_index = "<< datum.keyframe_index << flush;
	datum.frame_data.K 				= k;
	datum.frame_data.inv_K 			= generate_invK_( k , verbosity);																		// Current frame must be set as the new keyframe.

	frame_data.push_back( datum );																											// pushback a pose_datum, ready for getFrameData_vec() to write to.
																																			if (verbosity>local_verbosity_threshold) { cout << "\nDynamic_slam::initialize_camera_vec_chk 2:" <<flush;
																																				PRINT_MATX44F(frame_data.back().frame_data.keyframe2pose,);  // gets corrupted by getFrameData_vec()
																																			}
	getFrameData_vec();													// TODO if( GT available )
																																			if (verbosity>local_verbosity_threshold) { cout << "\nDynamic_slam::initialize_camera_vec_chk 3:" <<flush;
																																				PRINT_MATX44F(frame_data.back().frame_data_GT.keyframe2pose,);
																																				PRINT_MATX44F(frame_data.back().frame_data.keyframe2pose,);
																																			}
	frame_data.back().frame_data 	= frame_data.back().frame_data_GT;  // TODO if( use_GT )
	frame_data.push_back( frame_data.back() );																								// Propagate the initialization over the first three entries in the "frame_data" vector.
	frame_data.push_back( frame_data.back() );																								// Required because predictFrame_vec() samples previous pose and inverse pose.
																																			if(verbosity>local_verbosity_threshold) {
																																				PRINT_MATX44F(frame_data.back().frame_data_GT.K,);
																																				PRINT_MATX44F(frame_data.back().frame_data_GT.inv_K,);

																																				PRINT_MATX44F(frame_data.back().frame_data.K,);
																																				PRINT_MATX44F(frame_data.back().frame_data.inv_K,);

																																				PRINT_MATX44F(frame_data.back().frame_data_GT.keyframe2pose,);
																																				PRINT_MATX44F(frame_data.back().frame_data.keyframe2pose,);
																																				PRINT_MATX16F(frame_data.back().frame_data.keyframe2pose_algebra,);

																																				cout << "/n/nPrevious frames :  ############################################" << flush;
																																				vector<frame_datum>::iterator frame_minus_one			= 	frame_data.end();
																																				frame_minus_one 										-=	2;
																																				PRINT_MATX44F( frame_minus_one->frame_data.pose, );

																																				vector<frame_datum>::iterator frame_minus_two			=	frame_minus_one;
																																				frame_minus_two --;
																																				PRINT_MATX44F( frame_minus_two->frame_data.inv_pose, );

																																			}
	generate_SE3_k2k_vec( SE3_k2k );																										// fills float[96] ie 6xfloat[16] from conf.json intrinsic camera matrix + SE3 increments.
	runcl.precom_param_maps( SE3_k2k );																										// GPU computes J(u,v/SE3) Jacobian of optical flow wrt SE3.
	getFrame();
	runcl.costvol_frame_num++;
	runcl.dataset_frame_num++;
																																			if (verbosity>local_verbosity_threshold){ cout << "\nDynamic_slam::initialize_camera_vec Finished:"
																																				<<"##############################################################################\f" <<flush; }
}

int Dynamic_slam::nextFrame() {
	int local_verbosity_threshold = V_DYNAMIC_SLAM_NEXTFRAME;
																																			if(verbosity>local_verbosity_threshold) cout << "\f Dynamic_slam::nextFrame_chk 0,  runcl.dataset_frame_num="<<runcl.dataset_frame_num
																																				<<",\t depth = runcl.amem  \n" << flush; //  runcl.frame_bool_idx="<<runcl.frame_bool_idx<<"
																						auto step_0 = high_resolution_clock::now();
	frame_data.push_back( frame_data.back() ); //////////////////////////////////////////   new_frame										// duplicate last frame, as basis for new frame.

	if ( obj["initialize_tracking_from_GT_depth"].asBool() == false  ){ runcl.update_tracking_depthmap( runcl.amem   );	}					// copies buffer: amem to keyframe_depth_mem.  NB amem initialization will affect 1st tracking.
																																			// This would update keyframe_depth_mem ith the raw amem, every frame.

	getFrameData_vec();		/*Only IF GT available*/									auto step_1 = high_resolution_clock::now();			// updates pose2pose for next frame in cost volume.
	predictFrame_vec();																	auto step_2 = high_resolution_clock::now();			// Loads GT depth of the new frame. NB depends on image.size from getFrame().

	if(obj["use_GT_pose"].asBool() == true )		{	use_GT_pose_vec();	}			auto step_3 = high_resolution_clock::now();			// use_GT_pose();
	getFrame();																			auto step_4 = high_resolution_clock::now();
	if(obj["Artif_pose_err_bool"].asBool() == true ){ 	artificial_pose_error_vec();}	auto step_5 = high_resolution_clock::now();

	estimateSE3();
																						auto step_6 = high_resolution_clock::now();			// own thread ? num iter ?
	//disparity();
	//binocular_reference_frame();
	binocular_disparity();
	//estimateCalibration(); 																												// own thread, one iter.
																																			if(verbosity>local_verbosity_threshold){ cout << "\n  Dynamic_slam::nextFrame_chk 1, Pose error:" << flush;
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
																																			if(verbosity>local_verbosity_threshold){ cout << "\n  Dynamic_slam::nextFrame Finished "
																																				<<"##################################################################################\f" << flush; }
	return(0);																																// NB option to return an error that stops the main loop.
};

void Dynamic_slam::getFrame() { // can load use separate CPU thread(s) ?  // NB also need to change type CV_8UC3 -> CV_16FC3
	int local_verbosity_threshold = V_DYNAMIC_SLAM_GETFRAME;//verbosity_mp["Dynamic_slam::getFrame"];// -1;
																																			if(verbosity>local_verbosity_threshold){ cout << "\f Dynamic_slam::getFrame_chk 0.  runcl.dataset_frame_num = "<< runcl.dataset_frame_num  << flush;
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
																																				runcl.exit_(0);
																																			}
																																			//image.convertTo(image, CV_16FC3, 1.0/256, 0.0); // NB cv_16FC3 is preferable, for faster half precision processing on AMD, Intel & ARM GPUs.
	runcl.loadFrame( image );																												// NB Nvidia GeForce have 'Tensor Compute" FP16, accessible by PTX. AMD have RDNA and CDNA. These need PTX/assembly code and may use BF16 instead of FP16.
																																			// load a basic image in CV_8UC3, then convert on GPU to 'half'
	runcl.cvt_color_space( );
	runcl.blur_image();
	runcl.mipmap_linear();
	runcl.sum_image_variance();
	runcl.img_gradients();
																																			// # Get 1st & 2nd order image gradients of MipMap
																																			// see CostVol::cacheGValues(), RunCL::cacheGValue2 & __kernel void CacheG3
																																			if(verbosity>local_verbosity_threshold){ cout << "\n Dynamic_slam::getFrame_chk 1  Finished "
																																				<<"###########################################################################\f" << flush;}
}



//////

void Dynamic_slam::estimateCalibration(){
	int local_verbosity_threshold = V_DYNAMIC_SLAM_ESTIMATECALIBRATION;//verbosity_mp["Dynamic_slam::estimateCalibration"];
// # Get 1st & 2nd order gradients wrt calibration parameters.
//


// # Take one dammped least squares step of calibration.
//

}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// ## Regularize Maps : AbsDepth, GradDepth, SurfNormal, RelVel,
void Dynamic_slam::SpatialCostFns(){
	int local_verbosity_threshold = V_DYNAMIC_SLAM_SPATIALCOSTFNS;//verbosity_mp["Dynamic_slam::SpatialCostFns"];
// # Spatial cost functions
// see CostVol::updateQD(..), RunCL::updateQD(..) & __kernel void UpdateQD(..)

}

void Dynamic_slam::ParsimonyCostFns(){
	int local_verbosity_threshold = V_DYNAMIC_SLAM_PARSIMONYCOSTFNS;//verbosity_mp["Dynamic_slam::ParsimonyCostFns"];
// # Parsimony cost functions : NB Bin sort pixels to find non-spatial neighbours
// see SIFS for priors & Morphogenesis for BinSort

}

void Dynamic_slam::ExhaustiveSearch(){
	int local_verbosity_threshold = V_DYNAMIC_SLAM_EXHAUSTIVESEARCH;//verbosity_mp["Dynamic_slam::ExhaustiveSearch"];
// # Update A : exhaustive search on cost vol with cost fns -> update maps.
// see CostVol::updateA(..), RunCL::updateA(..) & __kernel void UpdateA2(..)

}


