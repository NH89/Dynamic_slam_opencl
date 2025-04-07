#include "Dynamic_slam.hpp"
#include <fstream>
#include <iomanip>   // for std::setprecision, std::setw

using namespace cv;
using namespace std;

void Dynamic_slam::binocular_reference_frame(){
	string fname = "RunCL::binocular_reference_frame( )";
	int local_verbosity_threshold = V_DYNAMIC_SLAM_BINOCULAR_REFERENCE_FRAME;
																												if(verbosity>local_verbosity_threshold){ cout << "\n Dynamic_slam::binocular_reference_frame() ######################################"<<flush;}
	// load image into buffers
	runcl.disparity_load_frame( runcl.basemem, runcl.ref_img_buf, "ref_img_buf" );
	runcl.mipmap_3x3blur_linear( runcl.ref_img_buf, "ref_img_buf");
/*
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
*/
	// prepare reference frame
	// uint iter = 0;
	// for (int layer = runcl.mm_stop; layer >=0; layer-- ){
	// 	runcl.mean_3rows (  		layer, iter, "ref_img_mean_rows_buf",							runcl.ref_img_buf,				runcl.ref_img_mean_rows_buf );
	// 	runcl.mean_cols  (			layer, iter, "ref_img_mean_buf",			"ref_img_diff_buf",	runcl.ref_img_buf, 				runcl.ref_img_mean_rows_buf, runcl.ref_img_mean_buf, runcl.ref_img_diff_buf );
	// 	runcl.sigma_3rows( 			layer, iter, "ref_img_sigma_3rows", 							runcl.ref_img_diff_buf, 		runcl.ref_img_sigma_rows_buf);
	// 	runcl.sigma_3cols( 			layer, iter, "ref_img_sigma_3cols", 							runcl.ref_img_sigma_rows_buf, 	runcl.ref_img_mean_sigma_buf);
	// }
	// predict warp ? from previous keyframe

																												if(verbosity>local_verbosity_threshold){ cout << "\n\nDynamic_slam::binocular_reference_frame() finished ############################\f"<<flush;}
}

void Dynamic_slam::binocular_disparity(){
	string fname = "RunCL::binocular_disparity( )";
	int local_verbosity_threshold = V_DYNAMIC_SLAM_BINOCULAR_DISPARITY;
																												if(verbosity>local_verbosity_threshold){ cout << "\n Dynamic_slam::disparity() ######################################"<<flush;}
	// load image into buffers
	runcl.disparity_load_frame(  runcl.basemem, runcl.new_img_buf, "new_img_buf" );
	runcl.mipmap_3x3blur_linear( runcl.new_img_buf, "new_img_buf");
/*
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
*/
	const float zero  = 0.0f;
	runcl._clEnqueueFillBuffer( runcl.m_queue, runcl.confidence_buf,  &zero,    sizeof( float),  0,     runcl.mm_size_bytes_C4, 	fname);
	runcl._cl_flush_finish(runcl.m_queue, fname);

	// predict warp from previous frame ?  // Here we use only the SE3 tracking + keyframe_depthmap
	runcl.zero_warp_buffer();

	for (int layer = runcl.mm_stop ; layer >=0 ; layer-- ){
		float  reduction = runcl.uint_params[MM_COLS] / runcl.MipMap[layer*8 + MiM_READ_COLS];		// uint reduction = mm_cols/read_cols_;  uint_params[MM_COLS];  mipmap_params_[MiM_READ_COLS];  uint8 mipmap_params_ = mipmap_params[layer];
		runcl.set_warp_new_image(	layer, reduction);
	}

	// debug chk of initial warped image
	for (int layer = runcl.mm_stop ; layer >=0 ; layer-- ){										if(verbosity>local_verbosity_threshold){ cout << "\n Dynamic_slam::disparity() chk_1 layer="<<layer<<", runcl.mm_start+3="<<runcl.mm_start+3<<", runcl.mm_stop="<<runcl.mm_stop<<flush;}
		int iter = -1;
		runcl.warp_image(      layer, iter  );
	}

	// compute warp for this frame  // can be until (layer > runcl.mm_start) but takes 3x longer.		(NB "SE3_start_layer":4,  "SE3_stop_layer":1, )
	for (int layer = 3/*0*//*runcl.mm_stop*/ ; layer > -1/*runcl.mm_start+3*/ ; layer-- ){						if(verbosity>local_verbosity_threshold){ cout << "\n Dynamic_slam::disparity() chk_1 layer="<<layer<<", runcl.mm_start+3="<<runcl.mm_start+3<<", runcl.mm_stop="<<runcl.mm_stop<<flush;}
		for (int iter = 0 ; iter < 2   ; iter++ ){																if(verbosity>local_verbosity_threshold){ cout << "\n Dynamic_slam::disparity() chk_2 layer="<<layer<<", iter="<<iter<<"_"<<flush;}
			runcl.warp_image(			layer, iter );
			runcl.correlation_one_step( layer, iter );
			runcl.blur_volume( runcl.correlation_buf, runcl.correlation_blurred_buf , "correlation_buf",  5/*vol_layers*/, layer, iter );
			runcl.compute_warp( 		layer, iter );

			//runcl.cl_mem_swap_ptr(runcl.correlation_buf, runcl.correlation_blurred_buf );
			for (int reg=0; reg<(8-iter); reg++){
				runcl.regularize_warp(	layer, iter	);

				// // swap(runcl.warp_buf,			runcl.warp_buf_regularized);
				// // runcl.cl_mem_swap_ptr( runcl.warp_buf,			runcl.warp_buf_regularized			);			// swap pointers
				// // runcl.cl_mem_swap_ptr( runcl.confidence_buf,	runcl.confidence_buf_regularized	);
			}
		}
		if (layer>runcl.mm_start){ runcl.propagate_warp( layer ); }
	}
																												if(verbosity>local_verbosity_threshold){ cout << "\n\nDynamic_slam::disparity() finished ############################\f"<<flush;}
}
