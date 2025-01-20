#include "Dynamic_slam.hpp"
#include <fstream>
#include <iomanip>   // for std::setprecision, std::setw

using namespace cv;
using namespace std;
/*
// void Dynamic_slam::binocular_reference_frame(){
// 	string fname = "RunCL::binocular_reference_frame( )";
// 	int local_verbosity_threshold = V_DYNAMIC_SLAM_BINOCULAR_REFERENCE_FRAME;
// 																												if(verbosity>local_verbosity_threshold){ cout << "\n Dynamic_slam::reference_frame() ######################################"<<flush;}
// 	// load image into buffers
// 	runcl._clEnqueueCopyBuffer( runcl.m_queue, runcl.keyframe_imgmem, 	runcl.curr_img_buf, 			0, 0, runcl.mm_size_bytes_C4, 	fname);
//
// 	// prepare reference frame
// 	uint iter = 0;
// 	for (int layer = runcl.mm_stop ; layer >=0 ; layer-- ){
// 		runcl.img_sq( 		layer, iter, runcl.curr_img_buf, 		runcl.curr_img_sq_buf,  "curr_img_sq_buf");	//RunCL::img_sq(	uint start, uint stop, uint layer , cl_mem img_buf,    cl_mem img_sq_buf);
// 		runcl.img_variance( layer, iter, runcl.curr_img_sq_buf, 	runcl.curr_img_var_buf, "curr_img_var_buf");
// 	}
// 	// predict warp ? from previous keyframe
//
//
// 																												if(verbosity>local_verbosity_threshold){ cout << "\n\nDynamic_slam::reference_frame() finished ############################\f"<<flush;}
// }
*/

void Dynamic_slam::binocular_reference_frame(){
	string fname = "RunCL::binocular_reference_frame( )";
	int local_verbosity_threshold = V_DYNAMIC_SLAM_BINOCULAR_REFERENCE_FRAME;
																												if(verbosity>local_verbosity_threshold){ cout << "\n Dynamic_slam::reference_frame() ######################################"<<flush;}
	// load image into buffers
	runcl._clEnqueueCopyBuffer( runcl.m_queue, runcl.keyframe_imgmem, 	runcl.ref_img_buf, 			0, 0, runcl.mm_size_bytes_C4, 	fname);
																												if( verbosity>local_verbosity_threshold ) {cout<<"\n\nRunCL::mean_sq_3rows( ..)_chk2 ."<<flush;								// Save buffers to file ###########
																														stringstream ss;
																														ss << "_binoc_" << runcl.save_index <<"_" ;
																														bool show 		= false;
																														bool old_tiff 	= runcl.tiff;
																														runcl.tiff 			= true;
																														float max_range = 1; 		// -1 -> gray = zero.
																														runcl._cl_flush_finish(runcl.m_queue, fname);
																														runcl.DownloadAndSave_3Channel(  runcl.ref_img_buf,  ss.str( ), runcl.paths.at( "ref_img_buf" ),  runcl.mm_size_bytes_C4,  runcl.mm_Image_size,  CV_32FC4,  show , max_range);
																														runcl.tiff 			= old_tiff;
																												}
	// prepare reference frame
	uint iter = 0;
	float reduction = 1.0f;
	for (int layer = runcl.mm_stop ; layer >=0 ; layer-- ){
		reduction = runcl.uint_params[MM_COLS] / runcl.MipMap[layer*8 + MiM_READ_COLS];		// uint reduction = mm_cols/read_cols_;  uint_params[MM_COLS];  mipmap_params_[MiM_READ_COLS];  uint8 mipmap_params_ = mipmap_params[layer];
		runcl.mean_sq_3rows(  		layer, iter, "ref_img_sq_mean_rows_buf",	runcl.ref_img_buf,					runcl.ref_img_sq_mean_rows_buf);
		runcl.mean_sq_cols(			layer, iter, "ref_img_sq_mean_buf", 		runcl.ref_img_sq_mean_rows_buf, 	runcl.ref_img_sq_mean_buf);
		runcl.set_warp_new_image(	layer, reduction);
	}
	// predict warp ? from previous keyframe

																												if(verbosity>local_verbosity_threshold){ cout << "\n\nDynamic_slam::reference_frame() finished ############################\f"<<flush;}
}

/*
void Dynamic_slam::binocular_disparity(){
	string fname = "RunCL::binocular_disparity( )";
	int local_verbosity_threshold = V_DYNAMIC_SLAM_BINOCULAR_DISPARITY;
																												if(verbosity>local_verbosity_threshold){ cout << "\n Dynamic_slam::disparity() ######################################"<<flush;}
	// load image into buffers
	runcl._clEnqueueCopyBuffer( runcl.m_queue, runcl.imgmem, 	runcl.new_img_buf, 			0, 0, runcl.mm_size_bytes_C4, 	fname);

	// predict warp from previous frame ?


	// compute warp for this frame  // can be until (layer > runcl.mm_start) but takes 3x longer.
	for (int layer = runcl.mm_stop ; layer > runcl.mm_start+1 ; layer-- ){										if(verbosity>local_verbosity_threshold){ cout << "\n Dynamic_slam::disparity() chk_1 layer="<<layer<<", runcl.mm_start="<<runcl.mm_start<<flush;}
		for (int iter = 0 ; iter < 5   ; iter++ ){																if(verbosity>local_verbosity_threshold){ cout << "\n Dynamic_slam::disparity() chk_2 layer="<<layer<<", iter="<<iter<<"_"<<flush;}
			runcl.warp_image(   layer, iter );
			runcl.img_sq( 		layer, iter, runcl.new_img_warped_buf, 	runcl.new_img_sq_buf,  "new_img_sq_buf");
			runcl.img_variance( layer, iter, runcl.new_img_sq_buf, 		runcl.new_img_var_buf, "new_img_var_buf");
			runcl.compute_warp( layer, iter );
		}
		if (layer>runcl.mm_start){ runcl.propagate_warp( layer ); }
	}
																												if(verbosity>local_verbosity_threshold){ cout << "\n\nDynamic_slam::disparity() finished ############################\f"<<flush;}
}
*/

void Dynamic_slam::binocular_disparity(){
	string fname = "RunCL::binocular_disparity( )";
	int local_verbosity_threshold = V_DYNAMIC_SLAM_BINOCULAR_DISPARITY;
																												if(verbosity>local_verbosity_threshold){ cout << "\n Dynamic_slam::disparity() ######################################"<<flush;}
	// load image into buffers
	runcl._clEnqueueCopyBuffer( runcl.m_queue, runcl.imgmem,	runcl.new_img_buf,			0, 0, runcl.mm_size_bytes_C4, 	fname);
																													if( verbosity>local_verbosity_threshold ) {cout<<"\n\nRunCL::mean_sq_3rows( ..)_chk2 ."<<flush;								// Save buffers to file ###########
																														stringstream ss;
																														ss << "_binoc_" << runcl.save_index <<"_" ;
																														bool show 		= false;
																														bool old_tiff 	= runcl.tiff;
																														runcl.tiff 		= true;
																														float max_range = 1; 		// -1 -> gray = zero.
																														runcl._cl_flush_finish(runcl.m_queue, fname);
																														runcl.DownloadAndSave_3Channel(  runcl.new_img_buf,  ss.str( ), runcl.paths.at( "new_img_buf" ),  runcl.mm_size_bytes_C4,  runcl.mm_Image_size,  CV_32FC4,  show , max_range);
																														runcl.tiff 			= old_tiff;
																												}
	const float zero  = 0.0f;
	runcl._clEnqueueFillBuffer( runcl.m_queue, runcl.confidence_buf,  &zero,    sizeof( float),  0,     runcl.mm_size_bytes_C4, 	fname);
	runcl._cl_flush_finish(runcl.m_queue, fname);
	// predict warp from previous frame ?


	// debug chk of initial warped image
	for (int layer = runcl.mm_stop ; layer >=0 ; layer-- ){										if(verbosity>local_verbosity_threshold){ cout << "\n Dynamic_slam::disparity() chk_1 layer="<<layer<<", runcl.mm_start+3="<<runcl.mm_start+3<<", runcl.mm_stop="<<runcl.mm_stop<<flush;}
		int iter = -1;
		runcl.warp_image(      layer, iter  );
	}


	// compute warp for this frame  // can be until (layer > runcl.mm_start) but takes 3x longer.
	for (int layer = runcl.mm_stop ; layer > runcl.mm_start+3 ; layer-- ){										if(verbosity>local_verbosity_threshold){ cout << "\n Dynamic_slam::disparity() chk_1 layer="<<layer<<", runcl.mm_start+3="<<runcl.mm_start+3<<", runcl.mm_stop="<<runcl.mm_stop<<flush;}
		for (int iter = 0 ; iter < 5   ; iter++ ){																if(verbosity>local_verbosity_threshold){ cout << "\n Dynamic_slam::disparity() chk_2 layer="<<layer<<", iter="<<iter<<"_"<<flush;}
			runcl.warp_image(      layer, iter  );
			runcl.mean_sq_3rows(   layer, iter, "warped_img_sq_mean_rows_buf",  runcl.warped_img_buf,				runcl.warped_img_sq_mean_rows_buf);
			runcl.mean_sq_cols(    layer, iter, "warped_img_sq_mean_buf", 	 	runcl.warped_img_sq_mean_rows_buf, 	runcl.warped_img_sq_mean_buf);
			runcl.co_mean_rows(    layer, iter  );
			runcl.covariance_cols( layer, iter  );
			runcl.regularize_warp( layer, iter  );
		}
		if (layer>runcl.mm_start){ runcl.propagate_warp( layer ); }
	}
																												if(verbosity>local_verbosity_threshold){ cout << "\n\nDynamic_slam::disparity() finished ############################\f"<<flush;}
}
