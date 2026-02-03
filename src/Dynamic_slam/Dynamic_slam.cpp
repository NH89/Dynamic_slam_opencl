#include "Dynamic_slam.hpp"

#include <fstream>
#include <iomanip>   // for std::setprecision, std::setw

using namespace cv;
using namespace std;

Dynamic_slam::~Dynamic_slam(){ runcl.~RunCL(); };

Dynamic_slam::Dynamic_slam( Json::Value obj_  ):   runcl( obj_  ) {  //, int_map verbosity_mp_
	obj = obj_;																																// NB save obj_ to class member obj, so that it persists within this Dynamic_slam object.
	verbosity 							= obj["verbosity"].asInt();
	int local_verbosity_threshold 		= V_DYNAMIC_SLAM_DYNAMIC_SLAM;																		if(verbosity>local_verbosity_threshold) cout << "\f Dynamic_slam::Dynamic_slam_chk 0\n" << flush;

	runcl.dataset_frame_num 			= obj["data_file_offset"].asUInt();

	use_conf_camera_matx				= obj["use_conf_camera_matx"].asBool();
	GT_available						= obj["GT_available"].asBool();
	use_artif_pose_error				= obj["Artif_pose_err_bool"].asBool();
	invert_GT_depth						= obj["invert_GT_depth"].asBool();
	initialize_keyframe_from_GT  		= obj["initialize_keyframe_from_GT"].asBool();
	initialize_tracking_from_GT_depth	= obj["initialize_tracking_from_GT_depth"].asBool();

	SE3_start_layer 					= obj["SE3_start_layer"].asUInt();
	SE3_stop_layer 						= obj["SE3_stop_layer"].asUInt();
	//SE_iter_per_layer 				= obj["SE_iter_per_layer"].asUInt();
	SE_iter 							= obj["SE_iter"].asUInt();

	stringstream  ss0;
	ss0 << obj["data_path"].asString()  <<  obj["data_file"].asString();																	// Collect the filenames of all the input images, plus ground truth files for camera data and depth maps- #####################
	rootpath 	= ss0.str();
	root 		= rootpath;

	if ( exists(root)==false )		{ cout << "Data folder "<< ss0.str()  <<" does not exist.\n" <<flush; runcl.exit_(0); }
	if ( is_directory(root)==false ){ cout << "Data folder "<< ss0.str()  <<" is not a folder.\n"<<flush; runcl.exit_(0); }
	if ( empty(root)==true )		{ cout << "Data folder "<< ss0.str()  <<" is empty.\n"		 <<flush; runcl.exit_(0); }
																																			if(verbosity>local_verbosity_threshold) cout << "\n Dynamic_slam::Dynamic_slam_chk 2\n" << flush;
	get_all(root, ".txt",   txt);																											// Get lists of files. Gathers all filepaths with each suffix, into c++ vectors.
	get_all(root, ".png",   png);
	get_all(root, ".depth", depth);
	if (txt.size()<=0){	GT_available = false;		cout<<",  WARNING no gound truth .txt file."<<flush;}
	if (png.size()<=0){	GT_available = false;		cout<<",  WARNING no gound truth .depth file."<<flush;}

																																			if(verbosity>local_verbosity_threshold){cout << "\n Dynamic_slam::Dynamic_slam_chk 3\n" << flush;
																																				cout<<"\n txt.size() = "<<txt.size() <<flush;

																																				cout << "\nDynamic_slam::Dynamic_slam(): "<< png.size()  <<" .png images found in data folder.\t"
																																				<<"png[runcl.dataset_frame_num].string()="<< png[runcl.dataset_frame_num].string()  <<flush;
																																			}
	runcl.initialize_RunCL( imread( png[ runcl.dataset_frame_num ].string() ) );															// Set image params, ref for dimensions and data type. ########################################################################

	initialize_camera_vec();				// Calls runcl.precomp_param_maps, depends on deltas.
																																			if(verbosity>local_verbosity_threshold){ cout << "\n Dynamic_slam::Dynamic_slam_ finished "
																																				<< "#####################################################################################\f" << flush;
																																			}
}

void Dynamic_slam::initialize_camera_intrinsic_matrix(){
	int local_verbosity_threshold = V_DYNAMIC_SLAM_INITIALIZE_CAMERA;
																																			if (verbosity>local_verbosity_threshold) { cout << "\fDynamic_slam::initialize_camera_vec_chk 0:" <<flush;}
	cv::Matx44f k 					= Matx44f::eye();;												// NB In DTAM_opencl, "cameraMatrix" found by convertAhandPovRay, called by fileLoader

		if(use_conf_camera_matx==true){
			for (int i=0; i<9; i++){ k.operator()(i/3,i%3) = obj["cameraMatrix"][i].asFloat(); }												// 3x3 Camera matrix from conf file.
		}else{
			for (int i=0; i<9; i++){ k.operator()(i/3,i%3) = 0.0f;}																				// 3x3 Default naive camera matrix.
			f	= 	2*min( runcl.baseImage_height, runcl.baseImage_width);
			 k.operator()(0,0) = f;
			 k.operator()(1,1) = f;
			 k.operator()(0,2) = runcl.baseImage_width  / 2.0;
			 k.operator()(1,2) = runcl.baseImage_height / 2.0;
			 k.operator()(2,2) = 1.0;
		}
	initial_K = k;																															if (verbosity>local_verbosity_threshold){ PRINT_MATX44F( initial_K , ); }
}

void Dynamic_slam::generate_deltas(){	// Principle : delta for each parameter causes maximum 1 pixel of warp in the full size image.
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

void Dynamic_slam::initialize_camera_vec(){
	const int local_verbosity_threshold = V_DYNAMIC_SLAM_INITIALIZE_CAMERA;
																																			if (verbosity>local_verbosity_threshold) { cout << "\fDynamic_slam::initialize_camera_vec_chk 0:" <<flush;
																																				cout<<"\n frame_data.size() = "<<frame_data.size()<<flush;
																																			}
	initialize_camera_intrinsic_matrix();	// depends on runcl.baseImage
	R 								= cv::Mat::eye(  3,3 , CV_32FC1);																			// intialize ground truth extrinsic data, NB Mat (int rows, int cols, int type)
	T 								= cv::Mat::zeros(3,1 , CV_32FC1);
	frame_datum 			datum 	= {};																									// default initialization, to values in header, or zero if not set in header.
	datum.keyframe_index			= 0 ;								// Expects that this frame will be used for new vector of keyframes.
																																			if (verbosity>local_verbosity_threshold) { cout << "\nDynamic_slam::initialize_camera_vec_chk 1:" <<flush;
																																				cout << "\n\n datum.keyframe_index = "<< datum.keyframe_index << flush;
																																				PRINT_MATX44F( initial_K ,  );
																																			}
	datum.frame_data.K 				= initial_K;
	cv::Matx44f inv_k				= generate_invK_( initial_K );
	datum.frame_data.inv_K 			= inv_k;																								// Current frame must be set as the new keyframe.
	frame_data.push_back( 	datum );																											// pushback a pose_datum, ready for getFrameData_vec() to write to.
																																			if (verbosity>local_verbosity_threshold) { cout << "\nDynamic_slam::initialize_camera_vec_chk 2:" <<flush;
																																				PRINT_MATX44F( inv_k, initial_K ); PRINT_MATX44F( initial_K*inv_k, );
																																				PRINT_MATX44F(frame_data.back().frame_data.prev_pose2pose,);  // gets corrupted by getFrameData_vec()
																																			}
	if(GT_available==true){									getFrameData_vec();		//}														// Sets frame_data.back().frame_data_GT
																																			// We use orthographic matrix, then convert to perspectiveby dividing by depth.
																																			// See notes in convertTransforms.cpp
																																			if (verbosity>local_verbosity_threshold) { cout << "\nDynamic_slam::initialize_camera_vec_chk 3:" <<flush;
																																				if(GT_available==true){
																																					PRINT_MATX44F(frame_data.back().frame_data_GT.pose,);
																																					PRINT_MATX44F(frame_data.back().frame_data_GT.prev_pose2pose,);
																																				}
																																				PRINT_MATX44F(frame_data.back().frame_data.prev_pose2pose,);
																																			}
	//if(			 GT_available		 ==true){				getFrameData_vec();																// Sets frame_data.back().frame_data_GT
		if(		 use_artif_pose_error==true){				set_artif_pose_error();	}
		else if( use_GT_pose		 ==true){				use_GT_pose_vec();		}
	}
																																			if(verbosity>local_verbosity_threshold){
																																				cout<<"\nruncl.current_frames[ current_frames_idx[0] ].K = \n"
																																					<< runcl.current_frames[ runcl.current_frames_idx[0] ].K <<endl<<flush;
																																			}
	runcl.set_cam_bufs(  frame_data.back().frame_data.K ,  frame_data.back().frame_data.inv_K,  frame_data.back(). frame_data.pose,  frame_data.back().frame_data.K2K  );
																																			if(verbosity>local_verbosity_threshold){
																																				cout<<"\nruncl.current_frames[ current_frames_idx[0] ].K = \n"
																																					<< runcl.current_frames[ runcl.current_frames_idx[0] ].K <<endl<<flush;
																																			}
	frame_data.push_back( frame_data.back() );																								// Propagate the initialization over the first three entries in the "frame_data" vector.
	frame_data.push_back( frame_data.back() );																								// Required because predictFrame_vec() samples previous pose and inverse pose.
																																			if(verbosity>local_verbosity_threshold) {
																																				if(GT_available==true){
																																					PRINT_MATX44F(frame_data.back().frame_data_GT.K,);
																																					PRINT_MATX44F(frame_data.back().frame_data_GT.inv_K,);
																																					PRINT_MATX44F(frame_data.back().frame_data_GT.prev_pose2pose,);
																																				}
																																				PRINT_MATX44F(frame_data.back().frame_data.K,);
																																				PRINT_MATX44F(frame_data.back().frame_data.inv_K,);

																																				PRINT_MATX44F(frame_data.back().frame_data.prev_pose2pose,);
																																				PRINT_MATX16F(frame_data.back().frame_data.keyframe2pose_algebra,);

																																				cout << "\n\nPrevious frames :  ############################################" << flush;
																																				vector<frame_datum>::iterator frame_minus_one			= 	frame_data.end();
																																				frame_minus_one 										-=	2;
																																				PRINT_MATX44F( frame_minus_one->frame_data.pose, );

																																				vector<frame_datum>::iterator frame_minus_two			=	frame_minus_one;
																																				frame_minus_two --;
																																				PRINT_MATX44F( frame_minus_two->frame_data.inv_pose, );

																																			}
	generate_deltas();																														// depends on f &=> camera_intrinsic_matrix
	generate_SE3_k2k_vec( SE3_k2k );																										// fills float[96] ie 6xfloat[16] from conf.json intrinsic camera matrix + SE3 increments.
	runcl.precomp_param_maps ( SE3_k2k );																									// GPU computes J(u,v/SE3) Jacobian of optical flow wrt SE3.
	getFrame();
	runcl.dataset_frame_num++;
																																			if (verbosity>local_verbosity_threshold){ cout << "\nDynamic_slam::initialize_camera_vec Finished:"
																																				<<"##############################################################################\f" <<flush;
																																			}
}

int Dynamic_slam::nextFrame() {
	string fname ="Dynamic_slam::nextFrame()";
	int local_verbosity_threshold = V_DYNAMIC_SLAM_NEXTFRAME;
																																			if(verbosity>local_verbosity_threshold-3){ cout << "\n Dynamic_slam::nextFrame_chk 0,  runcl.dataset_frame_num="<<runcl.dataset_frame_num
																																				<<",\t depth = runcl.amem  \n" << flush; //  runcl.frame_bool_idx="<<runcl.frame_bool_idx<<"
																																			}
																						auto step_0 = high_resolution_clock::now();
	frame_data.push_back( frame_data.back() ); //////////////////////////////////////////   new_frame										// duplicate last frame, as basis for new frame.
																																			if(verbosity>local_verbosity_threshold){
																																				cout<<"\nruncl.current_frames[ current_frames_idx[0] ].K = \n"
																																					<< runcl.current_frames[ runcl.current_frames_idx[0] ].K <<endl<<flush;
																																			}
	runcl.update_current_frames_idx();																										// move to next img buffer and RunCL "frame" struct in the current_frames[] array. NB current_frames_idx[..] pointer swap.
																																			if(verbosity>local_verbosity_threshold){
																																				cout<<"\nruncl.current_frames[ current_frames_idx[0] ].K = \n"
																																					<< runcl.current_frames[ runcl.current_frames_idx[0] ].K <<endl<<flush;
																																			}
	if(			 GT_available		 ==true){				getFrameData_vec();																// Sets frame_data.back().frame_data_GT
		if(		 use_artif_pose_error==true){				set_artif_pose_error();	}
		else if( use_GT_pose		 ==true){				use_GT_pose_vec();		}
		if(	initialize_tracking_from_GT_depth == true){		runcl.update_tracking_depthmap(runcl.depth_mem_GT);}
	}																																		// else implies ( GT_available==false || (use_artif_pose_erro==false && use_GT_pose== flase) )
																																			// NB new frame_data element was a copy of the previous one, so identity for first step, and constant vel thereafter.
	runcl.set_cam_bufs(  frame_data.back().frame_data.K ,  frame_data.back().frame_data.inv_K,  frame_data.back(). frame_data.pose,  frame_data.back().frame_data.K2K  );

																						auto step_1 = high_resolution_clock::now();			// updates pose2pose for next frame in cost volume.
																																			// if(verbosity>local_verbosity_threshold){ cout << "\n  Dynamic_slam::nextFrame_chk 1, Pose error after getFrameData_vec():" << flush;
																																			// 	report_GT_pose_error();
																																			// 	//display_frame_resluts();
																																			// }
																						auto step_2 = high_resolution_clock::now();			// Loads GT depth of the new frame. NB depends on image.size from getFrame().

																						auto step_3 = high_resolution_clock::now();			// use_GT_pose();
																																			// if(verbosity>local_verbosity_threshold){ cout << "\n  Dynamic_slam::nextFrame_chk 2, Pose error after use_GT_pose:" << flush;
																																			// 	report_GT_pose_error();
																																			// 	//display_frame_resluts();
																																			// }
	getFrame();																			auto step_4 = high_resolution_clock::now();
/*	if(obj["Artif_pose_err_bool"].asBool() == true ){ 	artificial_pose_error_vec();} */	auto step_5 = high_resolution_clock::now();

	//estimateSE3(); // original tracking
																																			if(verbosity>local_verbosity_threshold){ cout << "\n  Dynamic_slam::nextFrame_chk 3, Pose error after Artif_pose_err:" << flush;
																																				report_GT_pose_error();
																																				//display_frame_resluts();
																																			}
//	patch_slam();		// new tracking prototype.
	estimateSLAM();		// kernel basedtracking - no data offload. nor CPU computing.
																						auto step_6 = high_resolution_clock::now();			// own thread ? num iter ?

	//estimateCalibration(); 																												// own thread, one iter.
																																			if(verbosity>local_verbosity_threshold){ cout << "\n  Dynamic_slam::nextFrame_chk 4, Pose error after tracking:" << flush;
																																				report_GT_pose_error();
																																				//display_frame_resluts();
																																			}
	////////////////////////////////// Parallax depth mapping
																						auto step_7 = high_resolution_clock::now();
// TODO replace with patch_slam and multi-frame depth+motion+accel maps,  together with vel, accel, jolt of camera,   and later reflectance & illum etc...

																						auto step_8 = high_resolution_clock::now();			// Update cost vol with the new frame, and repeat optimization of the depth map.
																																			// NB Cost vol needs to be initialized on a particular keyframe.
																																			// A previous depth map can be transfered, and the updated depth map after each frame, can be used to track the next frame.
																						if(verbosity>local_verbosity_threshold-1) {
																							getNextFrameProfile(step_0, step_1, step_2, step_3, step_4, step_5, step_6, step_7, step_8);
																						}
	runcl.dataset_frame_num++;
																																			if(verbosity>local_verbosity_threshold){ cout << "\n  Dynamic_slam::nextFrame Finished "
																																				<<"##################################################################################\f" << flush;
																																			}
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
	image = imread( png[runcl.dataset_frame_num].string() );																				if(verbosity>local_verbosity_threshold){
																																				cout << "\n Dynamic_slam::getFrame_chk 0.5, Image file = " << png[runcl.dataset_frame_num].string() << "\t" << flush;
																																			}
																												if (image.type()!= runcl.baseImage.type() || image.size()!=runcl.baseImage.size() ) {
																													cerr<< "\n\nError: Dynamic_slam::getFrame(), runcl.dataset_frame_num = " << runcl.dataset_frame_num << " : missmatched. runcl.baseImage.size()="<<runcl.baseImage.size()<<\
																															", image.size()="<<image.size()<<", runcl.baseImage.type()="<<runcl.baseImage.type()<<", image.type()="<<image.type()<<"\n\n"<<flush;
																													runcl.exit_(0);
																												}
																																			//image.convertTo(image, CV_16FC3, 1.0/256, 0.0); // NB cv_16FC3 is preferable, for faster half precision processing on AMD, Intel & ARM GPUs.
	runcl.loadFrame( image );																												// NB Nvidia GeForce have 'Tensor Compute" FP16, accessible by PTX. AMD have RDNA and CDNA. These need PTX/assembly code and may use BF16 instead of FP16.
																																			// load a basic image in CV_8UC3, then convert on GPU to 'half'
	runcl.cvt_color_space( );

	runcl.build_img_pyramid( "imgmem" );		// RunCL_patch_image_tracking.cpp  way to build pyramid, with additional blur layers at apex

	runcl.current_frames[	runcl.current_frames_idx[0] ].frame_num		=	runcl.dataset_frame_num;

	cl_int 			status;
	cl_event 		writeEvt;
	status = clEnqueueFillBuffer(runcl.uload_queue, runcl.SE3_hessian_pinv_map_mem, &zero, 	sizeof(float), 	0, 2*runcl.mm_size_bytes_C4, 	0, NULL, &writeEvt);
	clFlush(runcl.uload_queue); status = clFinish(runcl.uload_queue);
																												if (status != CL_SUCCESS)	{
																													cout << "\nstatus = " << runcl.checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.3\n" << endl;
																													runcl.exit_(status);
																												}
																																			if(verbosity>local_verbosity_threshold){
																																				cout<<"\nDynamic_slam::getFrame_chk 1:    runcl.mm_start = "<<runcl.mm_start
																																					<<								"      runcl.mm_stop = "<<runcl.mm_stop<<flush;
																																			}
	for(int layer=runcl.mm_stop-1; layer>=0; layer-- ){
																																			if(verbosity>local_verbosity_threshold){ cout<<"\nlayer = "<<layer<<flush; }
		runcl.patch_img_gradients(	layer);
		runcl.patch_hessian_reduce(	layer);
	}
	// Will need to decide which layers and ST3 patch sizes to compute Hessians for, then store them in a buffer on the GPU.
																																			// # Get 1st & 2nd order image gradients of MipMap
																																			// see CostVol::cacheGValues(), RunCL::cacheGValue2 & __kernel void CacheG3
																																			if(verbosity>local_verbosity_threshold){ cout << "\n Dynamic_slam::getFrame_chk 2  Finished "
																																				<<"###########################################################################\n" << flush;

																																				for(int layer=0; layer<max_mipmap_layers; layer++){
																																					Matx66f	invH	=	runcl.current_frames[ runcl.current_frames_idx[0] ].invHessian[layer];
																																					cout<<"\nlayer = "<<layer<<"  ";
																																					PRINT_MATX66F( invH, );
																																				}
																																			}
}


//////

void Dynamic_slam::estimateCalibration(){
	//int local_verbosity_threshold = V_DYNAMIC_SLAM_ESTIMATECALIBRATION;//verbosity_mp["Dynamic_slam::estimateCalibration"];
// # Get 1st & 2nd order gradients wrt calibration parameters.
//


// # Take one dammped least squares step of calibration.
//

}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// ## Regularize Maps : AbsDepth, GradDepth, SurfNormal, RelVel,
void Dynamic_slam::SpatialCostFns(){
	//int local_verbosity_threshold = V_DYNAMIC_SLAM_SPATIALCOSTFNS;//verbosity_mp["Dynamic_slam::SpatialCostFns"];
// # Spatial cost functions
// see CostVol::updateQD(..), RunCL::updateQD(..) & __kernel void UpdateQD(..)

}

void Dynamic_slam::ParsimonyCostFns(){
	//int local_verbosity_threshold = V_DYNAMIC_SLAM_PARSIMONYCOSTFNS;//verbosity_mp["Dynamic_slam::ParsimonyCostFns"];
// # Parsimony cost functions : NB Bin sort pixels to find non-spatial neighbours
// see SIFS for priors & Morphogenesis for BinSort

}

void Dynamic_slam::ExhaustiveSearch(){
	//int local_verbosity_threshold = V_DYNAMIC_SLAM_EXHAUSTIVESEARCH;//verbosity_mp["Dynamic_slam::ExhaustiveSearch"];
// # Update A : exhaustive search on cost vol with cost fns -> update maps.
// see CostVol::updateA(..), RunCL::updateA(..) & __kernel void UpdateA2(..)

}


