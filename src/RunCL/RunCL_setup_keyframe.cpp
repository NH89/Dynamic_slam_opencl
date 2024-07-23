#include "RunCL.hpp"

/*
void RunCL::predictFrame(){ //predictFrame();
	int local_verbosity_threshold = verbosity_mp["RunCL::predictFrame"];

}
*/

void RunCL::estimateCalibration(){ //estimateCalibration(); 		// own thread, one iter.
	int local_verbosity_threshold = verbosity_mp["RunCL::estimateCalibration"];

}

void RunCL::transform_depthmap( cv::Matx44f K2K_ , cl_mem depthmap_ ){																		// NB must be used _before_ initializing the new cost_volume, because it uses keyframe_imgmem.
	int local_verbosity_threshold = verbosity_mp["RunCL::transform_depthmap"];// 0;
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::transform_depthmap(..)_chk0 .   runcl.dataset_frame_num="<< dataset_frame_num<<flush;}
	cl_event writeEvt;
	cl_int status;
	float K2K_arry[16]; for (int i=0; i<16;i++){ K2K_arry[i] = K2K_.operator()(i/4,i%4); }

																																			if(verbosity>local_verbosity_threshold) {
																																				cout<<"\n\nRunCL::transform_depthmap(..)_  K2K_arry[16] = "<<flush;
																																				for (int i=0; i<16;i++){
																																					if (i%4==0) cout << "\n";
																																					cout << ",  " << K2K_arry[i] ;
																																				}cout << "\n";
																																			}

	const float zero  = 0;
	status = clEnqueueFillBuffer (uload_queue, 	depth_mem, &zero,    sizeof(float),  0,     mm_size_bytes_C1, 0, NULL, &writeEvt);			if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: RunCL::transform_depthmap(..)_chk0.2\n" << endl;exit_(status);}
	status = clEnqueueWriteBuffer(uload_queue, 	k2kbuf,	   CL_FALSE, 0, 16 * sizeof(float), K2K_arry,         0, NULL, &writeEvt);			if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: RunCL::transform_depthmap(..)_chk0.3\n" << endl;exit_(status);}
	clFlush(uload_queue); status = clFinish(uload_queue);

	stringstream ss1;
	ss1 << "_transform_depthmap_1_";
	ss1 << save_index;
	DownloadAndSave(	keyframe_depth_mem,   	ss1.str(), 	paths.at("keyframe_depth_mem"), mm_size_bytes_C1,   mm_Image_size,   CV_32FC1, 	false , fp32_params[MAX_INV_DEPTH]); 	cout<<"\n\nRunCL::transform_depthmap(..)_chk0.1 ."	<<flush;
	DownloadAndSave(	depth_mem,   			ss1.str(), 	paths.at("depth_mem"),   		mm_size_bytes_C1,   mm_Image_size,   CV_32FC1, 	false , fp32_params[MAX_INV_DEPTH]); 	cout<<"\n\nRunCL::transform_depthmap(..)_chk4 ."	<<flush;   // ### corrupted !

	cl_int res;
	// inputs
	// __private	 uint layer, set in mipmap_call_kernel(..) below																																	//__private	    uint	    layer,							//0
	res = clSetKernelArg(transform_depthmap_kernel,  1, sizeof(cl_mem), &mipmap_buf);				if(res!=CL_SUCCESS){cout<<"\nmipmap_buf = "			<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__constant    uint*	    mipmap_params,					//1
	res = clSetKernelArg(transform_depthmap_kernel,  2, sizeof(cl_mem), &uint_param_buf);			if(res!=CL_SUCCESS){cout<<"\nuint_param_buf = "		<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__constant	uint*		uint_params,					//2
	res = clSetKernelArg(transform_depthmap_kernel,  3, sizeof(cl_mem), &k2kbuf);					if(res!=CL_SUCCESS){cout<<"\nk2kbuf = "				<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global		float* 		k2k,							//3
	res = clSetKernelArg(transform_depthmap_kernel,  4, sizeof(cl_mem), &keyframe_imgmem);			if(res!=CL_SUCCESS){cout<<"\nkeyframe_basemem = "	<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global		float4* 	keyframe_imgmem,				//4		// uses alpha channel to check bounds
	res = clSetKernelArg(transform_depthmap_kernel,  5, sizeof(cl_mem), &depthmap_/*keyframe_depth_mem*/);		if(res!=CL_SUCCESS){cout<<"\nkeyframe_depth_mem = "	<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global		float* 		keyframe_depth_mem,				//5
	// output
	res = clSetKernelArg(transform_depthmap_kernel,  6, sizeof(cl_mem), &depth_mem);				if(res!=CL_SUCCESS){cout<<"\ndepth_mem = "			<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global		float* 		depth_mem,						//6

																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::transform_depthmap(..)_chk1 ."<<flush;}
	mipmap_call_kernel( transform_depthmap_kernel, m_queue );
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::transform_depthmap(..)_chk3 ."<<flush;}
	clFlush(m_queue); status = clFinish(m_queue);																							if(status!= CL_SUCCESS){cout << " status = " << checkerror(status) <<", Error: RunCL::transform_depthmap(..)_transform_depthmap_kernel\n" << flush;exit_(status);}

	stringstream ss;
	ss << "_transform_depthmap_";
	ss << save_index;
	DownloadAndSave(	keyframe_depth_mem,   	ss.str(), 	paths.at("keyframe_depth_mem"), mm_size_bytes_C1,   mm_Image_size,   CV_32FC1, 	false , fp32_params[MAX_INV_DEPTH]); 	cout<<"\n\nRunCL::transform_depthmap(..)_chk0.1 ."	<<flush;
	DownloadAndSave(	depth_mem,   			ss.str(), 	paths.at("depth_mem"),   		mm_size_bytes_C1,   mm_Image_size,   CV_32FC1, 	false , fp32_params[MAX_INV_DEPTH]); 	cout<<"\n\nRunCL::transform_depthmap(..)_chk4 ."	<<flush;   // ### corrupted !

	cl_mem_swap_ptr(keyframe_depth_mem, depth_mem);
	clFlush(m_queue); status = clFinish(m_queue);																							if(status!= CL_SUCCESS){cout << " status = " << checkerror(status) <<", Error: RunCL::transform_depthmap(..)_clfinish_clEnqueueCopyBuffer\n" << flush;exit_(status);}
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::transform_depthmap(..)_finished ."<<flush;}
}


void RunCL::swap_costvol_pointers()
{
	int local_verbosity_threshold = verbosity_mp["RunCL::swap_costvol_pointers"];
																																			if(verbosity>local_verbosity_threshold){ cout << "\n\nRunCL::swap_costvol_pointers()_chk 0,  "<< flush; }
	cl_mem temp_mem;

	temp_mem 		= cdatabuf;
	cdatabuf		= temp_cdatabuf;
	temp_cdatabuf	= temp_mem;

	temp_mem 		= hdatabuf;
	hdatabuf		= temp_hdatabuf;
	temp_hdatabuf	= temp_mem;
}


void RunCL::transform_costvolume( cv::Matx44f K2K_)// , cl_mem old_cdata_mem,  cl_mem new_cdata_mem, cl_mem old_hdata_mem,  cl_mem new_hdata_mem       ){												// NB must be used _after_ initializing the new cost_volume.
{
	int local_verbosity_threshold = verbosity_mp["RunCL::transform_costvolume"];// 0;
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::transform_costvolume(..)_chk0 .    runcl.dataset_frame_num="<< dataset_frame_num<<flush;}
	cl_event writeEvt;
	cl_int status;
	float K2K_arry[16]; for (int i=0; i<16;i++){ K2K_arry[i] = K2K_.operator()(i/4,i%4); }

	status = clEnqueueWriteBuffer(uload_queue, invk2kbuf,	CL_FALSE, 0, 16 * sizeof(float), K2K_arry, 	0, NULL, &writeEvt);				if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: RunCL::transform_costvolume(..)_chk1\n" << endl;exit_(status); }
	clFlush(uload_queue); status = clFinish(uload_queue);
																																			if(verbosity>local_verbosity_threshold) {
																																				cout<<"\n\nRunCL::transform_costvolume(..)_  K2K_arry[16] = "<<flush;
																																				for (int i=0; i<16;i++){
																																					if (i%4==0) cout << "\n";
																																					cout << ",  " << K2K_arry[i] ;
																																				}cout << "\n";
																																			}
	cl_int res;
	// inputs
	//     __private	 uint layer, set in mipmap_call_kernel(..) below																																//__private	    uint	    layer,				//0
	res = clSetKernelArg(transform_costvolume_kernel,  1, sizeof(cl_mem), &mipmap_buf);				if(res!=CL_SUCCESS){cout<<"\nmipmap_buf = "			<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__constant    uint*	    mipmap_params,		//1
	res = clSetKernelArg(transform_costvolume_kernel,  2, sizeof(cl_mem), &uint_param_buf);			if(res!=CL_SUCCESS){cout<<"\nuint_param_buf = "		<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__constant	uint*		uint_params,		//2
	res = clSetKernelArg(transform_costvolume_kernel,  3, sizeof(cl_mem), &fp32_param_buf);			if(res!=CL_SUCCESS){cout<<"\nfp32_param_buf = "		<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__constant	uint*		fp32_params,		//3
	res = clSetKernelArg(transform_costvolume_kernel,  4, sizeof(cl_mem), &invk2kbuf);				if(res!=CL_SUCCESS){cout<<"\nk2kbuf = "				<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global		float16* 	k2k,				//4
	res = clSetKernelArg(transform_costvolume_kernel,  5, sizeof(cl_mem), &temp_cdatabuf);			if(res!=CL_SUCCESS){cout<<"\nold_cdata_mem = "		<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global		float*		old_cdata,			//5		photometric cost volume
	res = clSetKernelArg(transform_costvolume_kernel,  6, sizeof(cl_mem), &temp_hdatabuf);			if(res!=CL_SUCCESS){cout<<"\nold_hdata_mem = "		<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global		float*		old_hdata,			//7		hit count volume
	// outputs
	res = clSetKernelArg(transform_costvolume_kernel,  7, sizeof(cl_mem), &cdatabuf);				if(res!=CL_SUCCESS){cout<<"\nnew_cdata_mem = "		<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global		float*		new_cdata,			//6
	res = clSetKernelArg(transform_costvolume_kernel,  8, sizeof(cl_mem), &hdatabuf);				if(res!=CL_SUCCESS){cout<<"\nnew_hdata_mem = "		<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global		float*		new_hdata,			//8
	res = clSetKernelArg(transform_costvolume_kernel,  9, sizeof(cl_mem), &lomem);					if(res!=CL_SUCCESS){cout<<"\nlo_mem = "				<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global		float*		lo_,				//9		lo, hi, and mean of this ray of the cost volume.
	res = clSetKernelArg(transform_costvolume_kernel, 10, sizeof(cl_mem), &himem);					if(res!=CL_SUCCESS){cout<<"\nhi_mem = "				<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global		float*		hi_,				//10
	res = clSetKernelArg(transform_costvolume_kernel, 11, sizeof(cl_mem), &mean_mem);				if(res!=CL_SUCCESS){cout<<"\nmean_mem = "			<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global		float*		mean_				//11
	
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::transform_costvolume(..)_chk1 ."<<flush;}
/*
																																			if(verbosity>local_verbosity_threshold ) {
																																				stringstream ss;
																																				ss << "tempbufs before_transform_costvolume" << save_index;													// Save buffers to file ###########
																																				bool show = false;
																																				bool exception_tiff = false;
																																				if (verbosity_mp["RunCL::updateDepthCostVol::cdatabuf"])																// divisor max_range = 0 -> val/max_val.   max_range <0 -> grey=0
																																					DownloadAndSaveVolume(		temp_cdatabuf, 			ss.str(), paths.at("cdatabuf"), mm_size_bytes_C1,	mm_Image_size,   CV_32FC1,  show , 0 , exception_tiff );
																																				if (verbosity_mp["RunCL::updateDepthCostVol::hdatabuf"])
																																					DownloadAndSaveVolume(		temp_hdatabuf, 			ss.str(), paths.at("hdatabuf"), mm_size_bytes_C1,	mm_Image_size,   CV_32FC1,  show , 0 , exception_tiff );
																																			}
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::transform_costvolume(..)_chk2 ."<<flush;}
*/
	mipmap_call_kernel( transform_costvolume_kernel, m_queue );
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::transform_costvolume(..)_chk3 ."<<flush;}
	clFlush(m_queue); status = clFinish(m_queue);																							if(status!= CL_SUCCESS){cout << " status = " << checkerror(status) <<", Error: RunCL::transform_costvolume(..)_transform_costvolume_kernel\n" << flush;exit_(status);}

																																			if(verbosity>local_verbosity_threshold ) {
																																				stringstream ss;
																																				ss << "transform_costvolume" << save_index;													// Save buffers to file ###########
																																				bool show = false;
																																				bool exception_tiff = false;
																																				DownloadAndSave(	lomem,  ss.str(), paths.at("lomem"),  mm_size_bytes_C1,   mm_Image_size,   CV_32FC1, 	show , 8);	// a little more than the num images in costvol.
																																				DownloadAndSave(	himem,  ss.str(), paths.at("himem"),  mm_size_bytes_C1,   mm_Image_size,   CV_32FC1, 	show , 8);	//params[COSTVOL_LAYERS]

																																				if (verbosity_mp["RunCL::updateDepthCostVol::cdatabuf"])																// divisor max_range = 0 -> val/max_val.   max_range <0 -> grey=0
																																					DownloadAndSaveVolume(		cdatabuf, 			ss.str(), paths.at("cdatabuf"), mm_size_bytes_C1,	mm_Image_size,   CV_32FC1,  show , 0 /*float max_range*/ , exception_tiff );
																																				if (verbosity_mp["RunCL::updateDepthCostVol::hdatabuf"])
																																					DownloadAndSaveVolume(		hdatabuf, 			ss.str(), paths.at("hdatabuf"), mm_size_bytes_C1,	mm_Image_size,   CV_32FC1,  show , 0 /*float max_range*/ , exception_tiff );
																																			}
	
}


void RunCL::initializeDepthCostVol( cl_mem key_frame_depth_map_src){			 															// Uses the current frame as the keyframe for a new depth cost volume.
																																			// Dynamic_slam::initialize_from_GT(), Dynamic_slam::initialize_new_keyframe();
	int local_verbosity_threshold = verbosity_mp["RunCL::initializeDepthCostVol"];// -2;
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::initializeDepthCostVol(..)_chk0 ."<<flush;}
	costvol_frame_num = 0;
	cl_event writeEvt, ev;																													// Load keyframe
	cl_int res, status;
	status = clEnqueueCopyBuffer( m_queue, imgmem, 			keyframe_imgmem, 			0, 0, mm_size_bytes_C4, 0, NULL, &writeEvt);		if(status!= CL_SUCCESS){cout << " status = " << checkerror(status) <<", Error: RunCL::initializeDepthCostVol(..)_keyframe_imgmem\n" 	<< flush;exit_(status);}
	status = clEnqueueCopyBuffer( m_queue, HSV_grad_mem, 	keyframe_imgmem_HSV_grad, 	0, 0, mm_size_bytes_C8, 0, NULL, &writeEvt);		if(status!= CL_SUCCESS){cout << " status = " << checkerror(status) <<", Error: RunCL::initializeDepthCostVol(..)_keyframe_imgmem_HSV_grad\n" 	<< flush;exit_(status);}
	clFlush(m_queue); status = clFinish(m_queue);																							if(status!= CL_SUCCESS){cout << " status = " << checkerror(status) <<", Error: RunCL::initializeDepthCostVol(..)_clFinish(m_queue)_keyframe_imgmem, keyframe_imgmem_HSV_grad\n" 	<< flush;exit_(status);}

	/*if(vtp==true)*/ Store_keyframe();

	stringstream ss;
	ss << "__buildDepthCostVol";
	save_index = keyFrameCount*1000 + costvol_frame_num;
	ss << save_index;

	DownloadAndSave(		 	key_frame_depth_map_src,   	ss.str(),   paths.at("key_frame_depth_map_src"),   	mm_size_bytes_C1,   mm_Image_size,   CV_32FC1, 	false , fp32_params[MAX_INV_DEPTH]);
																																			if(verbosity>local_verbosity_threshold)	cout << "\nDownloadAndSave (.. key_frame_depth_map_src ..) finished\n"<<flush;

	status = clEnqueueCopyBuffer( m_queue, key_frame_depth_map_src, keyframe_depth_mem,			0, 0, mm_size_bytes_C1, 	0, NULL, &writeEvt);	if(status!= CL_SUCCESS){cout << "\n status = " << checkerror(status) <<", Error: RunCL::initializeDepthCostVol(..)_key_frame_depth_map_src\n" 				<< flush;exit_(status);}
	status = clEnqueueCopyBuffer( m_queue, depth_mem_GT, 			keyframe_depth_mem_GT,		0, 0, mm_size_bytes_C1, 	0, NULL, &writeEvt);	if(status!= CL_SUCCESS){cout << "\n status = " << checkerror(status) <<", Error: RunCL::initializeDepthCostVol(..)_key_frame_depth_map_src\n" 				<< flush;exit_(status);}
	//status = clEnqueueCopyBuffer( m_queue, g1mem, 					keyframe_g1mem, 			0, 0, mm_size_bytes_C8, 	0, NULL, &writeEvt);	if(status!= CL_SUCCESS){cout << "\n status = " << checkerror(status) <<", Error: RunCL::initializeDepthCostVol(..)_keyframe_g1mem\n" 						<< flush;exit_(status);} // Not req, see below.
	status = clEnqueueCopyBuffer( m_queue, SE3_grad_map_mem, 		keyframe_SE3_grad_map_mem, 	0, 0, mm_size_bytes_C1*6*8, 0, NULL, &writeEvt);	if(status!= CL_SUCCESS){cout << "\n status = " << checkerror(status) <<", Error: RunCL::initializeDepthCostVol(..)_keyframe_SE3_grad_map_mem\n" 			<< flush;exit_(status);}
	clFlush(m_queue); status = clFinish(m_queue);																							if(status!= CL_SUCCESS){cout << "\n status = " << checkerror(status) <<", Error: RunCL::initializeDepthCostVol(..)_clFinish(m_queue)\n" 	<< flush;exit_(status);}

																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::initializeDepthCostVol(..)_chk1 ."<<flush;}
	float depth = 1/( obj["max_depth"].asFloat() - obj["min_depth"].asFloat() );															// Zero the new cost vol. NB 'depth' _might_ be a useful start value when bootstrapping.
	float zero  = 0;
	status = clEnqueueFillBuffer(uload_queue, dbg_databuf, 	&zero, sizeof(float),   0, mm_vol_size_bytes, 0, NULL, &writeEvt);				if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: RunCL::initializeDepthCostVol_chk1.1\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	status = clEnqueueFillBuffer(uload_queue, cdatabuf, 	&zero, sizeof(float),   0, mm_vol_size_bytes, 0, NULL, &writeEvt);				if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: RunCL::initializeDepthCostVol_chk1.1\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	status = clEnqueueFillBuffer(uload_queue, hdatabuf, 	&zero, sizeof(float),   0, mm_vol_size_bytes, 0, NULL, &writeEvt);				if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: RunCL::initializeDepthCostVol_chk1.2\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	status = clEnqueueFillBuffer(uload_queue, img_sum_buf, 	&zero, sizeof(float),   0, mm_vol_size_bytes, 0, NULL, &writeEvt);				if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: RunCL::initializeDepthCostVol_chk1.3\n"<< endl;exit_(status);}		clFlush(uload_queue); status = clFinish(uload_queue);

	status = clEnqueueFillBuffer(uload_queue, dmem, 		&zero, sizeof(float),   0, mm_size_bytes_C1, 0, NULL, &writeEvt);				if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: RunCL::initializeDepthCostVol_chk1.4\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	status = clEnqueueFillBuffer(uload_queue, amem, 		&zero, sizeof(float),   0, mm_size_bytes_C1, 0, NULL, &writeEvt);				if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: RunCL::initializeDepthCostVol_chk1.5\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	status = clEnqueueFillBuffer(uload_queue, qmem, 		&zero, sizeof(float),   0, mm_size_bytes_C1, 0, NULL, &writeEvt);				if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: RunCL::initializeDepthCostVol_chk1.6\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	status = clEnqueueFillBuffer(uload_queue, qmem2, 		&zero, sizeof(float),   0, mm_size_bytes_C1, 0, NULL, &writeEvt);				if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: RunCL::initializeDepthCostVol_chk1.6\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);

	status = clEnqueueFillBuffer(uload_queue, lomem, 		&zero, sizeof(float),   0, mm_size_bytes_C1, 0, NULL, &writeEvt);				if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: RunCL::initializeDepthCostVol_chk1.7\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	status = clEnqueueFillBuffer(uload_queue, himem, 		&zero, sizeof(float),   0, mm_size_bytes_C1, 0, NULL, &writeEvt);				if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: RunCL::initializeDepthCostVol_chk1.8\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);

	clFlush(uload_queue); status = clFinish(uload_queue); 																					if (status != CL_SUCCESS)	{ cout << "\nclFinish(uload_queue)=" << status << checkerror(status) <<"\n"  << flush; exit_(status);}
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::initializeDepthCostVol(..)_chk1.5 ."<<flush;}

																																			if(verbosity>local_verbosity_threshold) {
																																				stringstream ss;
																																				ss << "initializeDepthCostVol";
																																				ss << save_index;													// Save buffers to file ###########
																																				cout<<"\n\nRunCL::initializeDepthCostVol(..)_chk1.5.1 ."<<flush;

																																				DownloadAndSave_3Channel(	keyframe_imgmem, 			ss.str(), paths.at("keyframe_imgmem"),  mm_size_bytes_C4, mm_Image_size,  CV_32FC4, 	false );
																																				DownloadAndSave_HSV_grad(  keyframe_imgmem_HSV_grad, 	ss.str(), paths.at("keyframe_imgmem_HSV_grad"),2*mm_size_bytes_C4, mm_Image_size, CV_32FC(8),	false, -1, 0 );
																																				cout<<"\n\nRunCL::initializeDepthCostVol(..)_chk1.6 ."<<flush;

																																				DownloadAndSave(		 	keyframe_depth_mem,   		ss.str(), paths.at("keyframe_depth_mem"),   		mm_size_bytes_C1,   mm_Image_size,   CV_32FC1, 	false , fp32_params[MAX_INV_DEPTH]);
																																				cout<<"\n\nRunCL::initializeDepthCostVol(..)_chk1.7 ."<<flush;

																																				//DownloadAndSave_3Channel(	keyframe_g1mem, ss.str(), paths.at("keyframe_g1mem"),  mm_size_bytes_C4, mm_Image_size,  CV_32FC4, 	false );
																																				//DownloadAndSave_HSV_grad(  keyframe_g1mem,	ss.str(), paths.at("keyframe_g1mem"), 	mm_size_bytes_C8, mm_Image_size,  CV_32FC(8),false, -1, 0 ); //  keyframe_g1mem is initialzed vu cacheGValues() at start of optimize depth.
																																				cout<<"\n\nRunCL::initializeDepthCostVol(..)_chk1.8 ."<<flush;

																																				//DownloadAndSave_3Channel(	g1mem, 	ss.str(), paths.at("g1mem"),  mm_size_bytes_C4, mm_Image_size,  CV_32FC4, 	false );
																																				//DownloadAndSave_HSV_grad(  g1mem,	ss.str(), paths.at("g1mem"),  mm_size_bytes_C8, mm_Image_size,  CV_32FC(8), false, -1, 0 ); 					 // NB g1mem not used, remains zero
																																				cout<<"\n\nRunCL::initializeDepthCostVol(..)_chk1.9 ."<<flush;

																																				DownloadAndSave_6Channel_volume(  keyframe_SE3_grad_map_mem, ss.str(), paths.at("keyframe_SE3_grad_map_mem"), mm_size_bytes_C4, mm_Image_size, CV_32FC4, false, -1, 6 );
																																			}
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::initializeDepthCostVol(..)_Finished ########################################## ."<<flush;}
}
