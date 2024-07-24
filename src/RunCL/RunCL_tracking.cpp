#include "RunCL.hpp"

void RunCL::precom_param_maps(float SE3_k2k[6*16]){ //  Compute maps of pixel motion for each SE3 DoF, and camera params // Derived from RunCL::mipmap
	int local_verbosity_threshold = verbosity_mp["RunCL::precom_param_maps"];// -2;
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::precom_param_maps(float SE3_k2k[6*16])_chk_0 "<<flush;}
	cl_event 			writeEvt;
	cl_int 				res, status;
	cv::Mat depth		= cv::Mat::ones (mm_height, mm_width, CV_32FC1);																	// NB must recompute translation maps at run time. NB parallax motion is proportional to inv depth.
	float mid_depth 	= (fp32_params[MAX_INV_DEPTH] + fp32_params[MIN_INV_DEPTH])/2.0;                                                    // TODO fix : depthmap not used as a kernel arg. NB want to match scale of depth range, but ? parallax may vary.
	depth 				*= mid_depth;

	status = clEnqueueWriteBuffer(uload_queue, SE3_k2kbuf, 		CL_FALSE, 0, 6*16*sizeof(float), SE3_k2k,    0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.3\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	status = clEnqueueWriteBuffer(uload_queue, depth_mem, 		CL_FALSE, 0, mm_size_bytes_C1,	 depth.data, 0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.3\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	status = clFlush( uload_queue );																								if (status != CL_SUCCESS)	{ cout << "\nclEnqueueWriteBuffer clFlush( uload_queue ) status = " << checkerror(status) <<"\n"<<flush; exit_(status); }
	clFinish( uload_queue );

	//      __private	 uint layer, set in mipmap_call_kernel(..) below                                                                                                                              __private	 uint	    layer,		//0
    res = clSetKernelArg(comp_param_maps_kernel, 1, sizeof(cl_mem),     &mipmap_buf);				            if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__constant uint*	mipmap_params,	//1
	res = clSetKernelArg(comp_param_maps_kernel, 2, sizeof(cl_mem), 	&uint_param_buf);	                    if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 	uint*	uint_params		//2
	res = clSetKernelArg(comp_param_maps_kernel, 3, sizeof(cl_mem), 	&SE3_k2kbuf);		                    if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 	float* 	k2k,			//3
	res = clSetKernelArg(comp_param_maps_kernel, 4, sizeof(cl_mem), 	&SE3_map_mem);		                    if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 	float* 	SE3_map,		//4
																																			if(verbosity>local_verbosity_threshold) {cout<<"\nRunCL::precom_param_maps(float SO3_k2k[6*16])_chk_1 "<<flush;}
	// SE3_map_mem, k_map_mem, dist_map_mem;
	mipmap_call_kernel( comp_param_maps_kernel, m_queue );
																																			if(verbosity>local_verbosity_threshold) {cout<<"\nRunCL::precom_param_maps(float SO3_k2k[6*16])_chk_2 "<<flush;}
																																			if(verbosity>local_verbosity_threshold) {
																																																cout<<"\n\nRunCL::precom_param_maps(float SO3_k2k[6*16])_output "<<flush;
																																																for (int i=0; i<1; i++) { // TODO x & y for all 6 SE3 DoF
																																																	stringstream ss;	ss << dataset_frame_num << "_SE3_map";
																																																	DownloadAndSave_2Channel_volume(SE3_map_mem, ss.str(), paths.at("SE3_map_mem"), mm_size_bytes_C1*2, mm_Image_size, CV_32FC2, false, 1.0, 6 /*SE3, 6DoF */);
																																																}
																																			}
																																			if(verbosity>local_verbosity_threshold) {cout<<"\nRunCL::precom_param_maps(float SE3_k2k[6*16])_chk.. Finished "<<flush;}
}

void RunCL::update_tracking_depthmap(){
	int local_verbosity_threshold = verbosity_mp["RunCL::update_tracking_depthmap"];
	cl_event writeEvt;
    cl_int status;

	status = clEnqueueCopyBuffer( m_queue,  amem, keyframe_depth_mem, 0, 0, mm_size_bytes_C1, 0, NULL, &writeEvt);	if(status!= CL_SUCCESS){cout << " status = " << checkerror(status) <<", Error: RunCL::update_tracking_depthmap(..)_clEnqueueCopyBuffer\n" << flush;exit_(status);}

	clFlush(m_queue); status = clFinish(m_queue);																							if(status!= CL_SUCCESS){cout << " status = " << checkerror(status) <<", Error: RunCL::update_tracking_depthmap(..)_clfinish_clEnqueueCopyBuffer\n" << flush;exit_(status);}
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::update_tracking_depthmap(..)_finished ."<<flush;}
}


void RunCL::se3_rho_sq( float Rho_sq_results[8][4], const float count[4], uint start, uint stop,  float k2k_16[16]  ){
	int local_verbosity_threshold = verbosity_mp["RunCL::se3_rho_sq"];// -1;
	const int num_samples  = tracking_num_samples;	// Rho_sq_results[samples=3][layers=8][channels=4]
	const int num_channels = tracking_num_colour_channels;
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::se3_rho_sq(..)_chk0 .##################################################################"<<flush;}
	cl_event writeEvt;
	cl_int status;
																																			if(verbosity>local_verbosity_threshold) {
																																				cout << "\nRunCL::se3_rho_sq(..)__chk_0.3: K2K= ";
																																				for (int i=0; i<16; i++){ cout << ",  "<< fp32_k2keyframe[i];  }	cout << flush;

																																				cout << "\n\nk2k_16[16]= ";
																																				print_float_16( k2k_16 );
																																				cout << "\n\n" <<flush;
																																			}
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::se3_rho_sq(..)_chk0.4 ,  dataset_frame_num="<<dataset_frame_num<<",   count="<<count[0]<<flush;}

	status = clEnqueueWriteBuffer(uload_queue, k2kbuf,	CL_FALSE, 0, tracking_num_samples*16*sizeof(float), k2k_16, 	0, NULL, &writeEvt);			if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: RunCL::se3_rho_sq(..)_chk0.5\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	float zero  = 0;
	status = clEnqueueFillBuffer(uload_queue, SE3_rho_map_mem, 		&zero, sizeof(float), 0, num_samples*2*mm_size_bytes_C4,	0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: RunCL::se3_rho_sq(..)_chk0.8\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	status = clEnqueueFillBuffer(uload_queue, se3_sum_rho_sq_mem, 	&zero, sizeof(float), 0, num_samples*pix_sum_size_bytes, 	0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: RunCL::se3_rho_sq(..)_chk0.9\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);

	status = clFlush( uload_queue );																						if (status != CL_SUCCESS)	{ cout << "\nclEnqueueWriteBuffer clFlush( uload_queue ) status = " << checkerror(status) <<"\n"<<flush; exit_(status); }
	clFinish( uload_queue );
																															if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::se3_rho_sq(..)_chk0.7 "<<flush;}
																																			// NB GT_depth loaded to depth_mem by void RunCL::loadFrameData(..)
	cl_int 				res;
	const uint wg_divisor =2;  // 1,2,4,8  reduction in workgroup size for this kernel.
	//input
	//      __private	 uint layer, set in mipmap_call_kernel(..) below                                                                                                                              __private	    uint	    layer,		                    //0
    res = clSetKernelArg(se3_rho_sq_kernel, 1, sizeof(size_t), &se3_sum_size);										if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__private		uint		pix_sum_size_bytes,				//1
    res = clSetKernelArg(se3_rho_sq_kernel, 2, sizeof(cl_mem), &mipmap_buf);										if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__constant    uint*	    mipmap_params,	                //2
	res = clSetKernelArg(se3_rho_sq_kernel, 3, sizeof(cl_mem), &uint_param_buf);									if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__constant	uint*		uint_params,					//3
	res = clSetKernelArg(se3_rho_sq_kernel, 4, sizeof(cl_mem), &fp32_param_buf);									if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__constant	float*		fp32_params						//4
	res = clSetKernelArg(se3_rho_sq_kernel, 5, sizeof(cl_mem), &k2kbuf);											if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global		float* 		k2k,							//5		// TODO keyframe2K
	res = clSetKernelArg(se3_rho_sq_kernel, 6, sizeof(cl_mem), &keyframe_imgmem);									if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 		float4*		keyframe_imgmem,				//6		// TODO need keyframe mipmap   keyframe_imgmem , keyframe_depth_mem
	res = clSetKernelArg(se3_rho_sq_kernel, 7, sizeof(cl_mem), &imgmem);											if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 		float4*		imgmem,							//7
	res = clSetKernelArg(se3_rho_sq_kernel, 8, sizeof(cl_mem), &keyframe_depth_mem);								if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 	 	float*		keyframe_depth_mem				//8		// NB GT_depth, now stoed as inv_depth	// TODO need keyframe mipmap
	res = clSetKernelArg(se3_rho_sq_kernel, 9, sizeof(cl_mem), &keyframe_g1mem);									if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 	 	float8*		g1p								//9

	//output
	res = clSetKernelArg(se3_rho_sq_kernel,10, sizeof(cl_mem), &SE3_rho_map_mem);									if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global	    float4*     Rho_					        //10
	res = clSetKernelArg(se3_rho_sq_kernel,11, (num_samples*local_work_size*4/wg_divisor)*sizeof(float), NULL);		if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__local		float4*		local_sum_rho_sq				//11	// 1 DoF, float4 channels
	res = clSetKernelArg(se3_rho_sq_kernel,12, sizeof(cl_mem), &se3_sum_rho_sq_mem);		 						if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 		float4*		global_sum_rho_sq,				//12

																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::se3_rho_sq(..)_chk1 .  start="<<start<<",  stop="<<stop<<flush;}
	//mipmap_call_kernel( se3_rho_sq_kernel, m_queue, start, stop );

	mipmap_call_kernel( se3_rho_sq_kernel, m_queue, start, stop, false, local_work_size/wg_divisor);
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::se3_rho_sq(..)_chk3 ."<<flush;
																																				stringstream ss;	ss << dataset_frame_num <<"_iter_"<<count[0]<<"_layer"<<count[1]<<"_factor"<<count[2]<<"_se3_rho_sq_";
																																				stringstream ss_path;

																																				bool show 				= false;
																																				float max_range 		= -1; 					// i.e. gray = zero.
																																				uint vol_layers 		= num_samples;			// i.e. 3 samples.
																																				bool exception_tiff 	= false;
																																				bool display 			= false; 				//obj["sample_se3_incr"].asBool();
																																				cout << "\nRunCL::se3_rho_sq(..)_chk3.5   display="<< display<< endl << flush;

																																				DownloadAndSave_3Channel_volume(  SE3_rho_map_mem,  ss.str(), paths.at("SE3_rho_map_mem"),  mm_size_bytes_C4, mm_Image_size, CV_32FC4, show, max_range, vol_layers, exception_tiff, count[0], display );
																																			}
																																			// void RunCL::DownloadAndSave_3Channel_volume( cl_mem buffer,   std::string count,   boost::filesystem::path folder,   size_t image_size_bytes,   cv::Size size_mat,   int type_mat,   bool show,   float max_range,   uint vol_layers,    bool exception_tiff /*=false*/,   float iter,   bool display)
	read_Rho_sq(Rho_sq_results);
}



void RunCL::se3_rho_sq( float Rho_sq_results[3][8][4], const float count[4], uint start, uint stop,  float k2k_3_16_[3][16]  ){
	int local_verbosity_threshold = verbosity_mp["RunCL::se3_rho_sq"];// -1;
	const int num_samples  = tracking_num_samples;	// Rho_sq_results[samples=3][layers=8][channels=4]
	const int num_channels = tracking_num_colour_channels;
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::se3_rho_sq(..)_chk0 .##################################################################"<<flush;}
	cl_event writeEvt;
	cl_int status;
																																			if(verbosity>local_verbosity_threshold) {
																																				cout << "\nRunCL::se3_rho_sq(..)__chk_0.3: K2K= ";
																																				for (int i=0; i<16; i++){ cout << ",  "<< fp32_k2keyframe[i];  }	cout << flush;

																																				for (int sample=0; sample<tracking_num_samples; sample++){
																																					cout << "\n\nk2k_3_16_["<<sample<<"][16]= ";
																																					print_float_16( k2k_3_16_[sample] );
																																					cout << "\n\n" <<flush;
																																				}
																																			}
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::se3_rho_sq(..)_chk0.4 ,  dataset_frame_num="<<dataset_frame_num<<",   count="<<count[0]<<flush;}

	status = clEnqueueWriteBuffer(uload_queue, k2kbuf,	CL_FALSE, 0, tracking_num_samples*16*sizeof(float), k2k_3_16_, 	0, NULL, &writeEvt);			if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: RunCL::se3_rho_sq(..)_chk0.5\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	float zero  = 0;
	status = clEnqueueFillBuffer(uload_queue, SE3_rho_map_mem, 		&zero, sizeof(float), 0, num_samples*2*mm_size_bytes_C4,	0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: RunCL::se3_rho_sq(..)_chk0.8\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	status = clEnqueueFillBuffer(uload_queue, se3_sum_rho_sq_mem, 	&zero, sizeof(float), 0, num_samples*pix_sum_size_bytes, 	0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: RunCL::se3_rho_sq(..)_chk0.9\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);

	status = clFlush( uload_queue );																						if (status != CL_SUCCESS)	{ cout << "\nclEnqueueWriteBuffer clFlush( uload_queue ) status = " << checkerror(status) <<"\n"<<flush; exit_(status); }
	clFinish( uload_queue );
																															if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::se3_rho_sq(..)_chk0.7 "<<flush;}
																																			// NB GT_depth loaded to depth_mem by void RunCL::loadFrameData(..)
	cl_int 				res;
	const uint wg_divisor =2;  // 1,2,4,8  reduction in workgroup size for this kernel.
	//input
	//      __private	 uint layer, set in mipmap_call_kernel(..) below                                                                                                                              __private	    uint	    layer,		                    //0
    res = clSetKernelArg(se3_rho_sq_kernel, 1, sizeof(size_t), &se3_sum_size);										if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__private		uint		pix_sum_size_bytes,				//1
    res = clSetKernelArg(se3_rho_sq_kernel, 2, sizeof(cl_mem), &mipmap_buf);										if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__constant    uint*	    mipmap_params,	                //2
	res = clSetKernelArg(se3_rho_sq_kernel, 3, sizeof(cl_mem), &uint_param_buf);									if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__constant	uint*		uint_params,					//3
	res = clSetKernelArg(se3_rho_sq_kernel, 4, sizeof(cl_mem), &fp32_param_buf);									if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__constant	float*		fp32_params						//4
	res = clSetKernelArg(se3_rho_sq_kernel, 5, sizeof(cl_mem), &k2kbuf);											if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global		float* 		k2k,							//5		// TODO keyframe2K
	res = clSetKernelArg(se3_rho_sq_kernel, 6, sizeof(cl_mem), &keyframe_imgmem);									if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 		float4*		keyframe_imgmem,				//6		// TODO need keyframe mipmap   keyframe_imgmem , keyframe_depth_mem
	res = clSetKernelArg(se3_rho_sq_kernel, 7, sizeof(cl_mem), &imgmem);											if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 		float4*		imgmem,							//7
	res = clSetKernelArg(se3_rho_sq_kernel, 8, sizeof(cl_mem), &keyframe_depth_mem);								if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 	 	float*		keyframe_depth_mem				//8		// NB GT_depth, now stoed as inv_depth	// TODO need keyframe mipmap
	res = clSetKernelArg(se3_rho_sq_kernel, 9, sizeof(cl_mem), &keyframe_g1mem);									if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 	 	float8*		g1p								//9

	//output
	res = clSetKernelArg(se3_rho_sq_kernel,10, sizeof(cl_mem), &SE3_rho_map_mem);									if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global	    float4*     Rho_					        //10
	res = clSetKernelArg(se3_rho_sq_kernel,11, (num_samples*local_work_size*4/wg_divisor)*sizeof(float), NULL);		if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__local		float4*		local_sum_rho_sq				//11	// 1 DoF, float4 channels
	res = clSetKernelArg(se3_rho_sq_kernel,12, sizeof(cl_mem), &se3_sum_rho_sq_mem);		 						if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 		float4*		global_sum_rho_sq,				//12

																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::se3_rho_sq(..)_chk1 .  start="<<start<<",  stop="<<stop<<flush;}
	//mipmap_call_kernel( se3_rho_sq_kernel, m_queue, start, stop );

	mipmap_call_kernel( se3_rho_sq_kernel, m_queue, start, stop, false, local_work_size/wg_divisor);
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::se3_rho_sq(..)_chk3 ."<<flush;
																																				stringstream ss;	ss << dataset_frame_num <<"_iter_"<<count[0]<<"_layer"<<count[1]<<"_factor"<<count[2]<<"_se3_rho_sq_";
																																				stringstream ss_path;

																																				bool show 				= false;
																																				float max_range 		= -1; 					// i.e. gray = zero.
																																				uint vol_layers 		= num_samples;			// i.e. 3 samples.
																																				bool exception_tiff 	= false;
																																				bool display 			= false; 				//obj["sample_se3_incr"].asBool();
																																				cout << "\nRunCL::se3_rho_sq(..)_chk3.5   display="<< display<< endl << flush;

																																				DownloadAndSave_3Channel_volume(  SE3_rho_map_mem,  ss.str(), paths.at("SE3_rho_map_mem"),  mm_size_bytes_C4, mm_Image_size, CV_32FC4, show, max_range, vol_layers, exception_tiff, count[0], display );
																																			}
																																			// void RunCL::DownloadAndSave_3Channel_volume( cl_mem buffer,   std::string count,   boost::filesystem::path folder,   size_t image_size_bytes,   cv::Size size_mat,   int type_mat,   bool show,   float max_range,   uint vol_layers,    bool exception_tiff /*=false*/,   float iter,   bool display)
/*
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::se3_rho_sq(..)_chk4 ."<<flush;}
	cv::Mat rho_sq_sum_mat = cv::Mat::zeros (num_samples*se3_sum_size, num_channels, CV_32FC1); 		// cv::Mat::zeros (int rows, int cols, int type)			// NB the data returned is one float4 per group, holding HSV, plus entry[3]=pixel count.
	ReadOutput( rho_sq_sum_mat.data, se3_sum_rho_sq_mem, num_samples*pix_sum_size_bytes );																//float Rho_sq_reults[8][4] = {{0}};
																																			if(verbosity>local_verbosity_threshold+2) {
																																				cout << "\n\nRunCL::se3_rho_sq(..)_chk5 ."							<< flush;
																																				cout << "\nrho_sq_sum_mat.size()="	<< rho_sq_sum_mat.size()		<< flush;
																																				cout << "\nse3_sum_size="			<< se3_sum_size					<< flush;
																																				cout << "\n mm_num_reductions = " 	<< mm_num_reductions 			<< flush;
																																				cout << "\n num_samples = " 		<< num_samples 					<< flush;
																																				cout << "\n\nrho_sq_sum_mat.at<float> (i*num_DoFs + j,  k) ,  i=group,  j=SO3 DoF,  i=delta4 (H,S,V, (valid pixels/group_size) )";

																																				for (int sample = 0; sample<num_samples-1; sample++){
																																					cout <<"\n\nSample = " << sample << flush;
																																					for (int i=0; i< se3_sum_size ; i++){//&& i<30
																																						cout << "\ngroup ="<<i<<":   ";
																																						cout << ",     \t(";
																																						for (int k=0; k<4; k++){	cout << ", \t" << rho_sq_sum_mat.at<float>(i, k); }
																																						cout << ")";
																																					}cout << endl << endl;
																																				}
																																			}

	for (int sample = 0; sample<num_samples; sample++){
		cout <<"\n\nSample = " << sample << flush;

		for (int reduction=0; reduction<=mm_num_reductions+1; reduction++){
			uint read_offset_ 			= MipMap[reduction*8 +MiM_READ_OFFSET];																			// mipmap_params_[MiM_READ_OFFSET];
			uint global_sum_offset 		= read_offset_ / local_work_size ;
			uint groups_to_sum 			= rho_sq_sum_mat.at<float>(global_sum_offset, 0);
			uint start_group 			= global_sum_offset + 1;
			uint stop_group 			= start_group + groups_to_sum ;   																		if(verbosity>local_verbosity_threshold+1) {
																																					cout << "\nRunCL::se3_rho_sq(..)_chk6 layer = "	<<reduction<<
																																					", read_offset_="			<<read_offset_<<
																																					", global_sum_offset = "	<<global_sum_offset<<
																																					", groups_to_sum = "		<<groups_to_sum<<
																																					", start_group = "			<<start_group<<
																																					", stop_group = "			<<stop_group<< flush;
																																				}
			for (int group=start_group; group< stop_group; group++){
				for (int channel=0; channel<num_channels; channel++){ 					Rho_sq_results[sample][reduction][channel] += rho_sq_sum_mat.at<float>(group + sample * se3_sum_size , channel);		};						// sum j groups for this layer of the MipMap.
			}
		}
																																				if(verbosity>local_verbosity_threshold+1) {
																																					cout << endl;
																																					for (int reduction_=0; reduction_<=mm_num_reductions+1; reduction_++){ 																// results / (num_valid_px * img_variance)
																																						cout << "\nLayer "<<reduction_<<" mm_num_reductions = "<< mm_num_reductions <<",  Rho_sq_results/num_groups = (";
																																						if (Rho_sq_results[sample][reduction_][3] > 0){																					// channel = 3 i.e. alpha channel is not zero.
																																							for (int channel_=0; channel_<3; channel_++){	cout << ", \t" << Rho_sq_results[sample][reduction_][channel_] / ( Rho_sq_results[sample][reduction_][3]  *  img_stats[IMG_VAR+channel_]  );
																																							}
																																							cout << ", \t" << Rho_sq_results[sample][reduction_][3] << ")";
																																						}
																																						else{	for (int l=0; l<3; l++){	cout << ", \t" << 0.0f;		}
																																							cout << ", \t" << Rho_sq_results[sample][reduction_][3] << ")";
																																						}
																																					}cout << "\n\nRunCL::se3_rho_sq(..)_finish . ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"<<flush;
																																				}
	}
	*/
	
	for (int sample = 1; sample<=num_samples; sample++){
		read_Rho_sq(Rho_sq_results[sample], sample-1);
	}





////
/*
	cv::Mat rho_sq_sum_mat = cv::Mat::zeros (2*se3_sum_size, 4, CV_32FC1); // cv::Mat::zeros (int rows, int cols, int type)					// NB the data returned is one float4 per group, holding HSV, plus entry[3]=pixel count.
	ReadOutput( rho_sq_sum_mat.data, se3_sum_rho_sq_mem, 2*pix_sum_size_bytes, 0 );


																																			if(verbosity>local_verbosity_threshold+1) {cout<<"\nTest RunCL::read_Rho_sq(..)_chk4.1, "<<flush;}

																																			if(verbosity>local_verbosity_threshold+2) {
																																				cout << "\n\nTest RunCL::read_Rho_sq(..)_chk5,  "<<flush;
																																				cout << "\nrho_sq_sum_mat.size()="<<rho_sq_sum_mat.size()<<flush;
																																				cout << "\nse3_sum_size="<<se3_sum_size<<flush;
																																				cout << "\n mm_num_reductions = " << mm_num_reductions << endl << flush;
																																				cout << "\n\nrho_sq_sum_mat.at<float> (i*num_DoFs + j,  k) ,  i=group,  j=SO3 DoF,  i=delta4 (H,S,V, (valid pixels/group_size) )";
																																				for (int i=0; i< 2*se3_sum_size ; i++){//&& i<30
																																					cout << "\ngroup ="<<i<<":   ";
																																					cout << ",     \t(";
																																					for (int k=0; k<4; k++){	cout << ", \t" << rho_sq_sum_mat.at<float>(i, k); }
																																					cout << ")";
																																				}cout << endl << endl;
																																			}

																																			cout << "\n\nmm_start="<<mm_start<<",   mm_stop="<<mm_stop \
																																				<<",   se3_sum_size="<<se3_sum_size<<flush;

	for (int sample=0; sample<2; sample++){
		int sample_step = sample * pix_sum_size_bytes;
		for (int layer=mm_start; layer<=mm_stop; layer++){
			uint groups_to_sum 			= rho_sq_sum_mat.at<float>(layer+sample_step, 0);
			uint start_group 			= rho_sq_sum_mat.at<float>(layer+sample_step, 2);  //global_sum_offset;
			uint stop_group 			= start_group + groups_to_sum ;   																		// -1
																																			if(verbosity>local_verbosity_threshold+2) {
																																				cout << "\nTest RunCL::read_Rho_sq(..)_chk6,    layer = "<<layer<<
																																				", groups_to_sum = "<<groups_to_sum<<
																																				", start_group = "<<start_group<<
																																				", stop_group = "<<stop_group<< flush;
																																			}
			int group_ = start_group;
			for (int sample=0; sample<2; sample++){
				for ( int group=start_group; group< stop_group; group++, group_++ ){
					for (int chan=0; chan<4; chan++){
						Rho_sq_results[sample][layer][chan] += rho_sq_sum_mat.at<float>(group_, chan);
					};
				}																									// sum j groups for this layer of the MipMap.
		//}
																																			if(verbosity>local_verbosity_threshold+1) {

																																				cout << "\n\nTest RunCL::read_Rho_sq(..)_chk7,"  <<flush;
																																				//for (int sample=0; sample<2; sample++){
																																					for (int layer=0; layer<=mm_num_reductions+1; layer++){ 														// results / (num_valid_px * img_variance)
																																						cout << "\nSample="<<sample<<", Layer "<<layer<<" mm_num_reductions = "<< mm_num_reductions <<",  Rho_sq_results/num_groups = (";
																																						if (Rho_sq_results[sample][layer][3] > 0){
																																							for (int chan=0; chan<3; chan++){	cout << ",   \t" << Rho_sq_results[sample][layer][chan] / ( Rho_sq_results[sample][layer][3]  *  img_stats[IMG_VAR+chan]  );
																																							}
																																							cout << ", \t" << Rho_sq_results[sample][layer][3] << ")";
																																						}
																																						else{	for (int chan=0; chan<3; chan++){	cout << ", \t" << 0.0f;		}
																																							cout << ", \t" << Rho_sq_results[sample][layer][3] << ")";
																																						}
																																					}cout << "\nTest RunCL::read_Rho_sq(..)_finish . ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"<<flush;
																																				//}
																																			}
		}
	}
	}
	*/
}


void RunCL::estimateSE3_LK(float SE3_results[8][6][tracking_num_colour_channels], float SE3_weights_results[8][6][tracking_num_colour_channels], float Rho_sq_results[8][4], int count, uint start, uint stop){ //estimateSE3_LK(); 	(uint start=0, uint stop=8)			// TODO replace arbitrary fixed constant with a const uint variable in the header...
	int local_verbosity_threshold = verbosity_mp["RunCL::estimateSE3_LK"];																	if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::estimateSE3_LK(..)_chk0 .##################################################################"<<flush;}
    cl_event writeEvt;
    cl_int status;
																																			if(verbosity>local_verbosity_threshold) {cout << "\nRunCL::estimateSE3_LK(..)__chk_1: K2K= ";
																																				for (int i=0; i<16; i++){ cout << ",  "<< fp32_k2keyframe[i];  }	cout << flush;
																																			}
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::estimateSE3_LK(..)_chk_2,  dataset_frame_num="<<dataset_frame_num<<",   count="<<count<<flush;}
	status = clEnqueueWriteBuffer(uload_queue, k2kbuf,	CL_FALSE, 0, 16 * sizeof(float), fp32_k2keyframe, 	0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: RunCL::estimateSE3_LK(..)_chk0.5\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	float zero  = 0;
	status = clEnqueueFillBuffer(uload_queue, SE3_rho_map_mem, 		&zero, sizeof(float), 0, 2*mm_size_bytes_C4,	0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: RunCL::se3_rho_sq(..)_chk0.8\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	status = clEnqueueFillBuffer(uload_queue, se3_sum_rho_sq_mem, 	&zero, sizeof(float), 0, pix_sum_size_bytes, 	0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: RunCL::se3_rho_sq(..)_chk0.9\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	status = clEnqueueFillBuffer(uload_queue, se3_weight_sum_mem, 	&zero, sizeof(float), 0, se3_sum_size_bytes, 	0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: RunCL::estimateSE3_LK(..)_chk0.6\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	status = clEnqueueFillBuffer(uload_queue, se3_sum_mem, 			&zero, sizeof(float), 0, se3_sum_size_bytes, 	0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: RunCL::estimateSE3_LK(..)_chk0.6\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	status = clEnqueueFillBuffer(uload_queue, SE3_incr_map_mem, 	&zero, sizeof(float), 0, mm_size_bytes_C1*24, 	0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: RunCL::estimateSE3_LK(..)_chk0.7\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	status = clFlush( uload_queue );																										if (status != CL_SUCCESS)	{ cout << "\nclEnqueueWriteBuffer clFlush( uload_queue ) status = " << checkerror(status) <<"\n"<<flush; exit_(status); }
	clFinish( uload_queue );
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::estimateSE3_LK(..)_chk_3 "<<flush;}
																																			// NB GT_depth loaded to depth_mem by void RunCL::loadFrameData(..)
	cl_int 				res;
																																																	//input
	const uint wg_divisor =2;  // 1,2,4,8  reduction in workgroup size for this kernel.
	// inputs
	//      __private	 uint layer, set in mipmap_call_kernel(..) below																																__private		uint		layer,							//0
	res = clSetKernelArg(se3_lk_grad_kernel, 1, sizeof(cl_mem), &mipmap_buf);										if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__constant	uint*		mipmap_params,					//1
	res = clSetKernelArg(se3_lk_grad_kernel, 2, sizeof(cl_mem), &uint_param_buf);									if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__constant	uint*		uint_params,					//2
	res = clSetKernelArg(se3_lk_grad_kernel, 3, sizeof(cl_mem), &fp32_param_buf);									if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__constant	float*		fp32_params						//3
	res = clSetKernelArg(se3_lk_grad_kernel, 4, sizeof(cl_mem), &k2kbuf);											if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global		float* 		k2k,							//4		// TODO keyframe2K
	res = clSetKernelArg(se3_lk_grad_kernel, 5, sizeof(cl_mem), &keyframe_imgmem);									if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 		float4*		keyframe_imgmem,				//5		// TODO need keyframe mipmap   keyframe_imgmem , keyframe_depth_mem
	res = clSetKernelArg(se3_lk_grad_kernel, 6, sizeof(cl_mem), &imgmem);											if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 		float4*		imgmem,							//6
	res = clSetKernelArg(se3_lk_grad_kernel, 7, sizeof(cl_mem), &keyframe_SE3_grad_map_mem);						if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 	 	float4*		keyframe_SE3_grad_map_mem		//7
	res = clSetKernelArg(se3_lk_grad_kernel, 8, sizeof(cl_mem), &SE3_grad_map_mem);									if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 	 	float4*		SE3_grad_map					//8
	if( costvol_frame_num  >1 ){
		res = clSetKernelArg(se3_lk_grad_kernel, 9, sizeof(cl_mem), &amem);											if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 	 	float*		amem							//9		// NB GT_depth, now stoed as inv_depth	// TODO need keyframe mipmap
	}else{
		res = clSetKernelArg(se3_lk_grad_kernel, 9, sizeof(cl_mem), &keyframe_depth_mem);							if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 	 	float*		keyframe_depth_mem				//9		// NB GT_depth, now stoed as inv_depth	// TODO need keyframe mipmap
	}
	//outputs
	res = clSetKernelArg(se3_lk_grad_kernel,10, sizeof(cl_mem), &SE3_rho_map_mem);									if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global	    float4*     Rho_							//10

	res = clSetKernelArg(se3_lk_grad_kernel,11, (local_work_size*1*4/wg_divisor)*sizeof(float), NULL);				if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__local		float4*		local_sum_rho_sq				//11	1 DoF, float4 channels
	res = clSetKernelArg(se3_lk_grad_kernel,12, sizeof(cl_mem), &se3_sum_rho_sq_mem);		 						if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 		float4*		global_sum_rho_sq,				//12
	res = clSetKernelArg(se3_lk_grad_kernel,13, sizeof(cl_mem), &SE3_weight_map_mem);								if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global		float4* 	weights_map,					//13

	res = clSetKernelArg(se3_lk_grad_kernel,14, (local_work_size*6*4/wg_divisor)*sizeof(float), NULL);				if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__local		float4*		local_sum_weight,				//14	6 DoF, float4 channels
	res = clSetKernelArg(se3_lk_grad_kernel,15, sizeof(cl_mem), &se3_weight_sum_mem);		 						if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 		float4*		global_sum_weight,				//15
	res = clSetKernelArg(se3_lk_grad_kernel,16, sizeof(cl_mem), &SE3_incr_map_mem);									if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 	 	float4*		SE3_incr_map_					//16

	res = clSetKernelArg(se3_lk_grad_kernel,17, (local_work_size*6*4/wg_divisor)*sizeof(float), NULL);				if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__local		float4*		local_sum_grads					//17	6 DoF, float4 channels
	res = clSetKernelArg(se3_lk_grad_kernel,18, sizeof(cl_mem), &se3_sum_mem);		 								if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 		float4*		global_sum_grads,				//18

	res = clSetKernelArg(se3_lk_grad_kernel,19, sizeof(cl_mem), &keyframe_g1mem);									if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 	 	float8*		g1p								//19
	//res = clSetKernelArg(se3_lk_grad_kernel,19, sizeof(int), &wg_divisor);											if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__private		uint 		wg_divisor						//19

																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::estimateSE3_LK(..)_chk_4 ."<<flush;}
	mipmap_call_kernel( se3_lk_grad_kernel, m_queue, start, stop, false, local_work_size/wg_divisor); 										// reduced worksize to allow for local memory limit 4kb on rtx 3030
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::estimateSE3_LK(..)_chk_5 ."<<flush;
																																				stringstream ss;	ss << dataset_frame_num << "_iter_"<< count << "_estimateSE3_LK_";
                                                                                                                                                stringstream ss_path;
																																				bool show 				= false;
																																				bool display 			= false;
																																				bool exception_tiff 	= false;
																																				uint vol_layers 		= 1;
																																				float max_range 		= -1; 		// i.e. gray = zero.

																																				DownloadAndSave_3Channel_volume(  SE3_rho_map_mem,  	ss.str(), paths.at("SE3_rho_map_mem"),  	mm_size_bytes_C4, mm_Image_size, CV_32FC4, show, max_range, vol_layers, exception_tiff, count, display );
																																				DownloadAndSave_3Channel_volume(  SE3_weight_map_mem, 	ss.str(), paths.at("SE3_weight_map_mem"), 	mm_size_bytes_C4, mm_Image_size, CV_32FC4, show, max_range, 6, 			exception_tiff, count, display );
																																				DownloadAndSave_3Channel_volume(  SE3_incr_map_mem, 	ss.str(), paths.at("SE3_incr_map_mem"), 	mm_size_bytes_C4, mm_Image_size, CV_32FC4, show, max_range, 6, 			exception_tiff, count, display );
																																			}
																																			if(obj["sample_se3_incr"].asBool()==true) {
																																				PrepareResults_3Channel_volume(  SE3_rho_map_mem,  	mm_size_bytes_C4, mm_Image_size, CV_32FC4, -1, 1,  count );
																																				PrepareResults_3Channel_volume(  SE3_incr_map_mem, 	mm_size_bytes_C4, mm_Image_size, CV_32FC4, -1, 6,  count );
																																			}
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::estimateSE3_LK(..)_chk_6 ."<<flush;}
	read_Rho_sq(Rho_sq_results);
	read_se3_weights(SE3_weights_results);
	read_se3_incr(SE3_results);																												if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::estimateSE3_LK(..)_finished ."<<flush;}
}

void RunCL::read_Rho_sq( float Rho_sq_results[8][4],  int offset/*=0*/ ){
	int local_verbosity_threshold = verbosity_mp["RunCL::read_Rho_sq"];// -1;
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::read_Rho_sq(..)_chk1,  offset="<<offset
																																				<<",   offset*pix_sum_size_bytes="<<offset*pix_sum_size_bytes<<flush;}
	cv::Mat rho_sq_sum_mat = cv::Mat::zeros (se3_sum_size, 4, CV_32FC1); // cv::Mat::zeros (int rows, int cols, int type)					// NB the data returned is one float4 per group, holding HSV, plus entry[3]=pixel count.
	ReadOutput( rho_sq_sum_mat.data, se3_sum_rho_sq_mem, pix_sum_size_bytes, offset*pix_sum_size_bytes );																//float Rho_sq_reults[8][4] = {{0}};
	/*
	 * void RunCL::ReadOutput( uchar* outmat,   cl_mem buf_mem,   size_t data_size,   size_t offset/ *=0* / ) {...
	 * 		status = clEnqueueReadBuffer( dload_queue,		// command_queue
											buf_mem,		// buffer
											CL_FALSE,		// blocking_read
											offset,			// offset
											data_size,		// size
											outmat,			// pointer
											0,				// num_events_in_wait_list
											NULL,			// event_waitlist				needs to know about preceeding events:
											&readEvt);		// event
	 * ...}
	 */

																																			if(verbosity>local_verbosity_threshold+1) {cout<<"\nRunCL::read_Rho_sq(..)_chk2,  offset="<<offset<<flush;}

																																			if(verbosity>local_verbosity_threshold+2) {
																																				cout << "\n\nRunCL::read_Rho_sq(..)_chk3,  offset="<<offset<<flush;
																																				cout << "\nrho_sq_sum_mat.size()="<<rho_sq_sum_mat.size()<<flush;
																																				cout << "\nse3_sum_size="<<se3_sum_size<<flush;
																																				cout << "\n mm_num_reductions = " << mm_num_reductions << endl << flush;
																																				cout << "\n\nrho_sq_sum_mat.at<float> (i*num_DoFs + j,  k) ,  i=group,  j=SO3 DoF,  i=delta4 (H,S,V, (valid pixels/group_size) )";
																																				for (int i=0; i< se3_sum_size ; i++){//&& i<30
																																					cout << "\ngroup ="<<i<<":   ";
																																					cout << ",     \t(";
																																					for (int k=0; k<4; k++){	cout << ", \t" << rho_sq_sum_mat.at<float>(i, k); }
																																					cout << ")";
																																				}cout << endl << endl;
																																			}

	for (int layer=mm_start; layer<=mm_stop; layer++){
		uint groups_to_sum 			= rho_sq_sum_mat.at<float>(layer, 0);
		uint start_group 			= rho_sq_sum_mat.at<float>(layer, 2);  //global_sum_offset;
		uint stop_group 			= start_group + groups_to_sum ;   																		// -1
																																			if(verbosity>local_verbosity_threshold+2) {
																																				cout << "\nRunCL::read_Rho_sq(..)_chk4,  offset="<<offset<<",  layer = "<<layer<<
																																				", groups_to_sum = "<<groups_to_sum<<
																																				", start_group = "<<start_group<<
																																				", stop_group = "<<stop_group<< flush;
																																			}
		for (int group=start_group; group< stop_group; group++){	for (int chan=0; chan<4; chan++){ 		Rho_sq_results[layer][chan] 	+= rho_sq_sum_mat.at<float>(group, chan);		};
		}																									// sum j groups for this layer of the MipMap.
	}
																																			if(verbosity>local_verbosity_threshold+1) {
																																				cout << "\n\nRunCL::read_Rho_sq(..)_chk5,  offset="<<offset<<flush;
																																				for (int layer=0; layer<=mm_num_reductions+1; layer++){ 														// results / (num_valid_px * img_variance)
																																					cout << "\nLayer "<<layer<<" mm_num_reductions = "<< mm_num_reductions <<",  Rho_sq_results/num_groups = (";
																																					if (Rho_sq_results[layer][3] > 0){
																																						for (int chan=0; chan<3; chan++){	cout << ",   \t" << Rho_sq_results[layer][chan] / ( Rho_sq_results[layer][3]  *  img_stats[IMG_VAR+chan]  );
																																						}
																																						cout << ", \t" << Rho_sq_results[layer][3] << ")";
																																					}
																																					else{	for (int chan=0; chan<3; chan++){	cout << ", \t" << 0.0f;		}
																																						cout << ", \t" << Rho_sq_results[layer][3] << ")";
																																					}
																																				}cout << "\nRunCL::read_Rho_sq(..)_finish . ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"<<flush;
																																			}
}

void RunCL::read_se3_weights(float SE3_weights_results[8][6][tracking_num_colour_channels]){
	int local_verbosity_threshold = verbosity_mp["RunCL::read_se3_weights"];// -1;
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::read_se3_weights(..)_chk1 ."<<flush;}
                                                                                                                                            // directly read higher layers
	uint num_DoFs = 6;
    cv::Mat se3_sum_mat = cv::Mat::zeros (se3_sum_size, num_DoFs*4, CV_32FC1); // cv::Mat::zeros (int rows, int cols, int type)				// NB the data returned is one float8 per group, holding one float per 6DoF of SE3, plus entry[7]=pixel count.
	ReadOutput( se3_sum_mat.data, se3_weight_sum_mem, se3_sum_size_bytes );                                                                        // se3_sum_size_bytes
																																			if(verbosity>local_verbosity_threshold+2) {cout << "\n\nRunCL::read_se3_weights(..)_chk2 ."<<flush;
																																				cout << "\nse3_sum_mat.size()="<<se3_sum_mat.size()<<flush;
																																				cout << "\nse3_sum_size="<<se3_sum_size<<flush;
																																				cout << "\n mm_num_reductions = " << mm_num_reductions << endl << flush;
																																				cout << "\n\nse3_sum_mat.at<float> (i*num_DoFs + j,  k) ,  i=group,  j=SO3 DoF,  i=delta4 (H,S,V, (valid pixels/group_size) )";
																																				for (int i=0; i< se3_sum_size ; i++){//&& i<30
																																					cout << "\ngroup ="<<i<<":   ";
																																					for (int j=0; j<num_DoFs; j++){
																																						cout << ",     \t(";	for (int k=0; k<4; k++){	cout << ", \t" << se3_sum_mat.at<float>(i, j*4 + k); }	cout << ")";
																																					}cout << flush;
																																				}cout << endl << endl;
																																			}
	for (int layer=mm_start; layer<=mm_stop; layer++){
        /*
		uint read_offset_ 		= MipMap[i*8 + MiM_READ_OFFSET];																			// mipmap_params_[MiM_READ_OFFSET];
        uint global_sum_offset 	= read_offset_ / local_work_size ;
        uint groups_to_sum 		= se3_sum_mat.at<float>(i, 0);
        uint start_group 		= global_sum_offset + 1;
        uint stop_group 		= start_group + groups_to_sum ;   // -1																		// skip the last group due to odd 7th value.
        */
        uint groups_to_sum 			= se3_sum_mat.at<float>(layer, 0);
		uint start_group 			= se3_sum_mat.at<float>(layer, 2);
		uint stop_group 			= start_group + groups_to_sum;
																																			if(verbosity>local_verbosity_threshold+1) {
																																				cout << "\ni="<<layer<<
																																				",  groups_to_sum="<<groups_to_sum<<
																																				",  start_group="<<start_group<<
																																				",  stop_group="<<stop_group;
																																			}
		for (int group=start_group; group< stop_group  ; group++){	for (int dof=0; dof<num_DoFs; dof++){ 	for (int chan=0; chan<4; chan++){	SE3_weights_results[layer][dof][chan] += se3_sum_mat.at<float>(group, dof*4 + chan);	} }	}	//l =4 =num channels	// sum j groups for this layer of the MipMap. // se3_sum_mat.at<float>(j, k);
    }
																																			if(verbosity>local_verbosity_threshold+1) {
																																				cout << endl << " SE3_weights_results/num_groups = (H, S, V, alpha=num_groups) ";
																																				for (int layer=mm_start; layer<=mm_stop; layer++){ 																// results / (num_valid_px * img_variance)
																																					cout << "\nLayer "<<layer<<" SE3_weights_results = (";												// raw results
																																					for (int dof=0; dof<num_DoFs; dof++){
																																						cout << "\nse3 dof="<<dof<<" : (";  for (int chan=0; chan<4; chan++){	cout << ",   \t" << SE3_weights_results[layer][dof][chan] ;	}cout << ")";
																																					}cout << ")";
																																					///
																																					/*
																																					cout << "\nLayer "<<i<<" SE3_weights_results/num_groups = (";
																																					for (int k=0; k<num_DoFs; k++){
																																						cout << "\nDoF="<<k<<" ("; for (int l=0; l<3; l++){	cout << ", \t" << SE3_weights_results[i][k][l] / ( SE3_weights_results[i][k][3]  *  img_stats[IMG_VAR+l]  ); } cout << ", " << SE3_weights_results[i][k][3] << ")";
																																					}cout << ")";																					// << "{"<< img_stats[i*4 +IMG_VAR+l] <<"}"
																																					*/

																																				}cout << "\nRunCL::read_se3_weights(..)_finish . ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"<<flush;
																																			}
	}

void RunCL::read_se3_incr(float SE3_results[8][6][tracking_num_colour_channels]){
	int local_verbosity_threshold = verbosity_mp["RunCL::read_se3_incr"];// -1;
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::read_se3_incr(..)_chk1 ."<<flush;}
                                                                                                                                            // directly read higher layers
	uint num_DoFs = 6;
    cv::Mat se3_sum_mat = cv::Mat::zeros (se3_sum_size, num_DoFs*4, CV_32FC1); // cv::Mat::zeros (int rows, int cols, int type)				// NB the data returned is one float8 per group, holding one float per 6DoF of SE3, plus entry[7]=pixel count.
	ReadOutput( se3_sum_mat.data, se3_sum_mem, se3_sum_size_bytes );                                                                        // se3_sum_size_bytes
																																			if(verbosity>local_verbosity_threshold+2) {cout << "\nRunCL::read_se3_incr(..)_chk2 ."<<flush;
																																				cout << "\nse3_sum_mat.size()="<<se3_sum_mat.size()<<flush;
																																				cout << "\nse3_sum_size="<<se3_sum_size<<flush;
																																				cout << "\n mm_num_reductions = " << mm_num_reductions << endl << flush;
																																				cout << "\n\nse3_sum_mat.at<float> (i*num_DoFs + j,  k) ,  i=group,  j=SO3 DoF,  i=delta4 (H,S,V, (valid pixels/group_size) )";
																																				for (int i=0; i< se3_sum_size ; i++){//&& i<30
																																					cout << "\ngroup ="<<i<<":   ";
																																					for (int j=0; j<num_DoFs; j++){
																																						cout << ",     \t(";	for (int k=0; k<4; k++){	cout << ", \t" << se3_sum_mat.at<float>(i, j*4 + k); }	cout << ")";
																																					}cout << flush;
																																				}cout << endl << endl;
																																			}
	for (int layer=mm_start; layer<=mm_stop; layer++){
		uint groups_to_sum 			= se3_sum_mat.at<float>(layer, 0);
		uint start_group 			= se3_sum_mat.at<float>(layer, 2);
        uint stop_group 			= start_group + groups_to_sum;
																																			if(verbosity>local_verbosity_threshold+1) {
																																				cout << "\ni="<<layer<<
																																				",  groups_to_sum="<<groups_to_sum<<
																																				",  start_group="<<start_group<<
																																				",  stop_group="<<stop_group;
																																			}
		for (int group=start_group; group< stop_group  ; group++){	for (int dof=0; dof<num_DoFs; dof++){ 	for (int chan=0; chan<4; chan++){	SE3_results[layer][dof][chan] += se3_sum_mat.at<float>(group, dof*4 + chan);	} }	}	//l =4 =num channels	// sum j groups for this layer of the MipMap. // se3_sum_mat.at<float>(j, k);
    }
																																			if(verbosity>local_verbosity_threshold+1) {
																																				cout << endl << " SE3_results/num_groups = (H, S, V, alpha=num_groups) ";
																																				for (int layer=mm_start; layer<=mm_stop; layer++){ 																// results / (num_valid_px * img_variance)
																																					cout << "\nLayer "<<layer<<" SE3_results = (";														// raw results
																																					for (int dof=0; dof<num_DoFs; dof++){
																																						cout << "\nse3 dof="<<dof<<" : (";  for (int chan=0; chan<4; chan++){	cout << ", \t" << SE3_results[layer][dof][chan] ;	}cout << ")";
																																					}cout << ")";
																																					///
																																					/*
																																					cout << "\nLayer "<<i<<" SE3_results/num_groups = (";
																																					for (int k=0; k<num_DoFs; k++){
																																						cout << "\nDoF="<<k<<" ("; for (int l=0; l<3; l++){	cout << ", \t" << SE3_results[i][k][l] / ( SE3_results[i][k][3]  *  img_stats[IMG_VAR+l]  ); } cout << ", " << SE3_results[i][k][3] << ")";
																																					}cout << ")";																					// << "{"<< img_stats[i*4 +IMG_VAR+l] <<"}"
																																					*/
																																				}cout << "\nRunCL::read_se3_incr(..)_finish . ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"<<flush;
																																			}
}

void RunCL::tracking_result(string result){
	if(verbosity>verbosity_mp["RunCL::tracking_result"]) {
		cout<<"\n\nRunCL::tracking_result(..)_chk0"<<flush;
		stringstream ss;			ss << dataset_frame_num <<  "_img_grad_" << result;				// "_iter_"<< count <<
		stringstream ss_path_rho;	ss_path_rho << "SE3_rho_map_mem";
		cout << " , " << ss_path_rho.str() << " , " <<  paths.at(ss_path_rho.str()) << " , " << ss.str()  <<flush;
		DownloadAndSave_3Channel_volume(  SE3_rho_map_mem,  ss.str(), paths.at(ss_path_rho.str()), mm_size_bytes_C4, mm_Image_size, CV_32FC4, false, -1, /*display*/true );
	}
}

/*
* 	atomic_test1_buf	= clCreateBuffer(m_context, CL_MEM_READ_WRITE 						, 4*local_work_size*sizeof(int),	0, &res);	if(res!=CL_SUCCESS){cout<<"\nres 42= "<<checkerror(res)<<"\n"<<flush;exit_(res);}
*/

void RunCL::atomic_test1(){
	int local_verbosity_threshold = verbosity_mp["RunCL::atomic_test1"];

	cl_int res, status;
	cl_event ev, writeEvt;
	const int data_size = 4*local_work_size;
	const size_t num_threads = 2 * local_work_size;
	int num_threads_int = 1.5*local_work_size;

	int zero  = 0;
	status = clEnqueueFillBuffer(uload_queue, atomic_test1_buf, 	&zero, sizeof(int), 0, data_size*sizeof(int), 	0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.3\n" << endl;exit_(status);}

	clFlush(uload_queue); status = clFinish(uload_queue); 																					if (status != CL_SUCCESS)	{ cout << "\nclFinish(uload_queue)=" << status << checkerror(status) <<"\n"  << flush; exit_(status);}

	res = clSetKernelArg(atomic_test1_kernel, 0, sizeof(int), 		&num_threads_int);														if (res    !=CL_SUCCESS)	{ cout <<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	res = clSetKernelArg(atomic_test1_kernel, 1, sizeof(cl_mem), 	&atomic_test1_buf);														if (res    !=CL_SUCCESS)	{ cout <<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	/*
	cl_int clEnqueueNDRangeKernel(
									cl_command_queue 		command_queue,
									cl_kernel 				kernel,
									cl_uint 				work_dim,
									const size_t* 			global_work_offset,
									const size_t* 			global_work_size,
									const size_t* 			local_work_size,
									cl_uint 				num_events_in_wait_list,
									const cl_event* 		event_wait_list,
									cl_event* 				event
									);
	*/
	res 	= clEnqueueNDRangeKernel(m_queue, atomic_test1_kernel, 1, 0, &num_threads, &local_work_size, 0, NULL, &ev);						if (res    != CL_SUCCESS)	{ cout << "\nres = " << checkerror(res) <<"\n"<<flush; exit_(res);}
	status 	= clFlush(m_queue);																												if (status != CL_SUCCESS)	{ cout << "\nRunCL::atomic_test1(),  clFlush(queue_to_call) status  = "<<status<<" "<< checkerror(status) <<"\n"<<flush; exit_(status);}

	int atomic_test_output[ data_size ];
	for (int i = 0; i<data_size; i++) atomic_test_output[i] = -1;
	cl_event readEvt;
	status = clEnqueueReadBuffer(	dload_queue,			// command_queue
									atomic_test1_buf,		// buffer
									CL_FALSE,				// blocking_read
									0,						// offset
									data_size*sizeof(int),	// size
									atomic_test_output,		// pointer
									0,						// num_events_in_wait_list
									NULL,					// event_waitlist				needs to know about preceeding events:
									&readEvt				// event
								 );
													if (status != CL_SUCCESS) { cout << "\nclEnqueueReadBuffer(..) status=" << checkerror(status) <<"\n"<<flush; exit_(status);}
	status = clFlush(dload_queue);					if (status != CL_SUCCESS) { cout << "\nclFlush(m_queue) status = " 		<< checkerror(status) <<"\n"<<flush; exit_(status);}
	status = clWaitForEvents(1, &readEvt); 			if (status != CL_SUCCESS) { cout << "\nclWaitForEvents status="			<< checkerror(status) <<"\n"<<flush; exit_(status);}

	cout << "\n\n void RunCL::atomic_test1(): \t  local_work_size = "<<local_work_size<<", \t (buffer size) data_size = "<<data_size<<", \t (num threards launched)  num_threads="<<num_threads<<", \t (num threards run) num_threads_int = "<<num_threads_int<<", \t  (";
	int i=0;
	for (; i< data_size; i++) cout << ", " << atomic_test_output[i];
	cout << ") \t i="<< i<< "\n\n" << flush;
}


void RunCL::atomic_test2(){
	int local_verbosity_threshold = verbosity_mp["RunCL::atomic_test2"];

	cl_int res, status;
	cl_event ev, writeEvt;
	const 	int 	data_size 		= 4   *local_work_size;
	const 	size_t 	num_threads 	= 2   *local_work_size;
			int 	num_threads_int = 1.5 *local_work_size;




	float number  = 0.674;
	status = clEnqueueFillBuffer(uload_queue, atomic_test2_buf, &number, sizeof(int), 0, data_size*sizeof(float), 	0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.3\n" << endl;exit_(status);}

	clFlush(uload_queue); status = clFinish(uload_queue); 																					if (status != CL_SUCCESS)	{ cout << "\nclFinish(uload_queue)=" << status << checkerror(status) <<"\n"  << flush; exit_(status);}

	res = clSetKernelArg(atomic_test2_kernel, 0, sizeof(int), 		&num_threads_int);														if (res    !=CL_SUCCESS)	{ cout <<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	res = clSetKernelArg(atomic_test2_kernel, 1, sizeof(cl_mem), 	&atomic_test2_buf);														if (res    !=CL_SUCCESS)	{ cout <<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	/*
	cl_int clEnqueueNDRangeKernel(
									cl_command_queue 		command_queue,
									cl_kernel 				kernel,
									cl_uint 				work_dim,
									const size_t* 			global_work_offset,
									const size_t* 			global_work_size,
									const size_t* 			local_work_size,
									cl_uint 				num_events_in_wait_list,
									const cl_event* 		event_wait_list,
									cl_event* 				event
									);
	*/
	res 	= clEnqueueNDRangeKernel(m_queue, atomic_test2_kernel, 1, 0, &num_threads, &local_work_size, 0, NULL, &ev);						if (res    != CL_SUCCESS)	{ cout << "\nres = " << checkerror(res) <<"\n"<<flush; exit_(res);}
	status 	= clFlush(m_queue);																												if (status != CL_SUCCESS)	{ cout << "\nRunCL::atomic_test1(),  clFlush(queue_to_call) status  = "<<status<<" "<< checkerror(status) <<"\n"<<flush; exit_(status);}

	float atomic_test_output[ data_size ];
	for (int i = 0; i<data_size; i++) atomic_test_output[i] = -1;
	cl_event readEvt;
	status = clEnqueueReadBuffer(	dload_queue,			// command_queue
									atomic_test2_buf,		// buffer
									CL_FALSE,				// blocking_read
									0,						// offset
									data_size*sizeof(float),	// size
									atomic_test_output,		// pointer
									0,						// num_events_in_wait_list
									NULL,					// event_waitlist				needs to know about preceeding events:
									&readEvt				// event
								 );
													if (status != CL_SUCCESS) { cout << "\nclEnqueueReadBuffer(..) status=" << checkerror(status) <<"\n"<<flush; exit_(status);}
	status = clFlush(dload_queue);					if (status != CL_SUCCESS) { cout << "\nclFlush(m_queue) status = " 		<< checkerror(status) <<"\n"<<flush; exit_(status);}
	status = clWaitForEvents(1, &readEvt); 			if (status != CL_SUCCESS) { cout << "\nclWaitForEvents status="			<< checkerror(status) <<"\n"<<flush; exit_(status);}

	cout << "\n\n void RunCL::atomic_test2(): \t  local_work_size = "<<local_work_size<<", \t (buffer size) data_size = "<<data_size<<", \t (num threards launched)  num_threads="<<num_threads<<", \t (num threards run) num_threads_int = "<<num_threads_int<<", \t  (";
	int i=0;
	for (; i< data_size; i++) cout << ", " << atomic_test_output[i];
	cout << ") \t i="<< i<< "\n\n" << flush;



}
