#include "RunCL.hpp"

void RunCL::loadFrame(cv::Mat image){ //getFrame();
	int local_verbosity_threshold = verbosity_mp["RunCL::loadFrame"];//-2;
                                                                                                                                            if(verbosity>local_verbosity_threshold) {cout << "\n RunCL::loadFrame_chk 0\n" << flush;}
	cl_int status;
	cl_event writeEvt;																										               // WriteBuffer basemem #########
	status = clEnqueueWriteBuffer(uload_queue, basemem, CL_FALSE, 0, image_size_bytes, image.data, 0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nclEnqueueWriteBuffer basemem status = " 				<< checkerror(status) <<"\n"<<flush; exit_(status); }
	status = clFlush( uload_queue );																					if (status != CL_SUCCESS)	{ cout << "\nclEnqueueWriteBuffer clFlush( uload_queue ) status = " << checkerror(status) <<"\n"<<flush; exit_(status); }
	waitForEventAndRelease( &writeEvt );
                                                                                                                                            if (verbosity>local_verbosity_threshold){
                                                                                                                                                stringstream ss;	ss << dataset_frame_num << "loadFrame";
                                                                                                                                                DownloadAndSave_3Channel(basemem, ss.str(), paths.at("basemem"), image_size_bytes, baseImage_size,  baseImage_type, 	false );
                                                                                                                                            }
}

void RunCL::cvt_color_space(){ //getFrame(); basemem(CV_8UC3, RGB)->imgmem(CV16FC3, HSV), NB we will use basemem for image upload, and imgmem for the MipMap. RGB is default for .png standard.
	int local_verbosity_threshold = verbosity_mp["RunCL::cvt_color_space"];// -1;
                                                                                                                                            if(verbosity>local_verbosity_threshold) {
                                                                                                                                                cout<<"\n\nRunCL::cvt_color_space()_chk0"<<flush;
                                                                                                                                                cout << "\n";
                                                                                                                                                cout << ",mm_Image_size = " << mm_Image_size << endl;
                                                                                                                                                cout << ",mm_Image_type = "	<< mm_Image_type << endl;
                                                                                                                                                cout << ",mm_size_bytes_C3 = " << mm_size_bytes_C3 << endl;
                                                                                                                                                cout << ",mm_size_bytes_C4 = " << mm_size_bytes_C4 << endl;
                                                                                                                                                cout << ",mm_size_bytes_C1 = " << mm_size_bytes_C1 << endl;
                                                                                                                                                cout << "\n";
                                                                                                                                                cout << ",baseImage_size, = " << baseImage_size << endl;
                                                                                                                                                cout << ",baseImage_type = " << baseImage_type << endl;
                                                                                                                                                cout << ",image_size_bytes = " << image_size_bytes	<< endl;
                                                                                                                                                cout << ",mm_vol_size_bytes = " << mm_vol_size_bytes << endl;
                                                                                                                                                cout << "\n" << flush;
                                                                                                                                            }
	//args
	cl_int res, status;
	cl_event ev;																																									// cvt_color_space_kernel  or  cvt_color_space_linear_kernel
	res = clSetKernelArg(cvt_color_space_linear_kernel, 0, sizeof(cl_mem), &basemem);					   if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	//__global uchar3*		base,			//0
	res = clSetKernelArg(cvt_color_space_linear_kernel, 1, sizeof(cl_mem), &imgmem);	   				   if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	//__global float4*		img,			//1
	res = clSetKernelArg(cvt_color_space_linear_kernel, 2, sizeof(cl_mem), &uint_param_buf);			   if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	//__global uint*		uint_params		//2
	res = clSetKernelArg(cvt_color_space_linear_kernel, 3, sizeof(cl_mem), &mipmap_buf);				   if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	//__constant uint*		mipmap_params,	//3 // NB layer = 0.
	res = clSetKernelArg(cvt_color_space_linear_kernel, 4, local_work_size*4*sizeof(float), 	NULL);	   if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	//__local  float4*		local_sum_pix	//4
	res = clSetKernelArg(cvt_color_space_linear_kernel, 5, sizeof(cl_mem), &pix_sum_mem);				   if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	//__local  float4*		global_sum_pix	//5

	status = clFlush(m_queue); 				if (status != CL_SUCCESS)	{ cout << "\nclFlush(m_queue) status = " << checkerror(status) <<"\n"<<flush; exit_(status);}
	status = clFinish(m_queue); 			if (status != CL_SUCCESS)	{ cout << "\nclFinish(m_queue)="<<status<<" "<<checkerror(status)<<"\n"<<flush; exit_(status);}
																															if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::cvt_color_space()_chk1,  global_work_size="<< global_work_size <<flush;
	res = clEnqueueNDRangeKernel(m_queue, cvt_color_space_linear_kernel, 1, 0, &global_work_size, &local_work_size, 0, NULL, &ev); 							// run cvt_color_space_kernel  aka cvt_color_space(..) ##### TODO which CommandQueue to use ? What events to check ?
	if (res != CL_SUCCESS)	{ cout << "\nres = " << checkerror(res) <<"\n"<<flush; exit_(res);}
	status = clFlush(m_queue);				                                                               if (status != CL_SUCCESS)	{ cout << "\nRunCL::cvt_color_space()_chk1.1, clFlush(m_queue) status  = "<<status<<" "<< checkerror(status) <<"\n"<<flush; exit_(status);}
	status = clWaitForEvents (1, &ev);		                                                               if (status != CL_SUCCESS)	{ cout << "\nRunCL::cvt_color_space()_chk1.2, clWaitForEventsh(1, &ev) = "<<status<<" "<<checkerror(status)  <<"\n"<<flush; exit_(status);}
                                                                                                                                            if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::cvt_color_space()_chk2"<<flush;
                                                                                                                                            if (verbosity>local_verbosity_threshold){
                                                                                                                                                stringstream ss;		ss << dataset_frame_num << "_cvt_color_space";
                                                                                                                                                stringstream ss_path;	ss_path << "imgmem";

                                                                                                                                                cv::Size new_Image_size = cv::Size(mm_width, mm_height);
                                                                                                                                                size_t   new_size_bytes = mm_width * mm_height * 4* 4;

                                                                                                                                                cout << "imgmem="<< imgmem << endl << flush;
                                                                                                                                                cout <<", ss.str()="<< ss.str() << endl << flush;
                                                                                                                                                cout <<", paths.at(\"imgmem\")="<< paths.at("imgmem") << endl << flush;

                                                                                                                                                cout <<", paths.at(" << ss_path.str() <<")="<< paths.at(ss_path.str()) << endl << flush;
                                                                                                                                                cout <<", new_size_bytes="<< new_size_bytes << endl << flush;
                                                                                                                                                cout <<", new_Image_size="<< new_Image_size <<"" << endl << flush;

                                                                                                                                                DownloadAndSave_3Channel(	imgmem, ss.str(), paths.at( ss_path.str() ), new_size_bytes/*mm_size_bytes_C4*/, new_Image_size/*mm_Image_size*/,  CV_32FC4 /*mm_Image_type*/, 	false );
                                                                                                                                                /*
                                                                                                                                                cout<<"\n\n,chk2.3,"<<flush;
                                                                                                                                                cout<<"\n img_sum_buf="<< img_sum_buf <<flush;
                                                                                                                                                cout<<"\n ss.str()="<< ss.str() <<flush;
                                                                                                                                                cout<<"\n paths.at(\"img_sum_buf\")="<< paths.at("img_sum_buf") <<flush;
                                                                                                                                                cout<<"\n mm_size_bytes_C1="<< mm_size_bytes_C1 <<flush;
                                                                                                                                                cout<<"\n mm_size_bytes_C3="<< mm_size_bytes_C3 <<flush;
                                                                                                                                                cout<<"\n mm_Image_size="<< mm_Image_size <<"\n\n"<<flush;
                                                                                                                                                */
                                                                                                                                                //DownloadAndSave_3Channel(	img_sum_buf, ss.str(), paths.at("img_sum_buf"),  mm_size_bytes_C3*2, mm_Image_size,  CV_32FC3 /*mm_Image_type*/, 	false ); // only when debugging.
                                                                                                                                            }
	cv::Mat pix_sum_mat = cv::Mat::zeros (pix_sum_size, 1, CV_32FC4); // cv::Mat::zeros (int rows, int cols, int type)						// NB the data returned is one float4 per group, for the base image, holding hsv channels plus entry[3]=pixel count.
	ReadOutput( pix_sum_mat.data, pix_sum_mem, pix_sum_size_bytes );                                                                        // se3_sum_size_bytes

                                                                                                                                            if(verbosity>local_verbosity_threshold+2) {cout<<"\n\nRunCL::cvt_color_space(..)_chk2 ."<<flush;
																																				cout << "\npix_sum_mat.size()="<<pix_sum_mat.size()<<flush;
																																				cout << "\npix_sum_size="<<pix_sum_size<<flush;
                                                                                                                                                cout << "\n pix_sum_mat.data = (\n";
                                                                                                                                                for (int i=0; i<pix_sum_size; i++){
                                                                                                                                                    cout << "\n group="<<i<<" : ( " << flush;
                                                                                                                                                    for (int j=0; j<4; j++){
                                                                                                                                                        cout << pix_sum_mat.at<float>(i,j) << " , " << flush;
                                                                                                                                                    }
                                                                                                                                                    cout << ")" << flush;
                                                                                                                                                }cout << "\n)\n" << flush;
                                                                                                                                            }
	float pix_sum_reults[4] = {0};
	uint groups_to_sum = pix_sum_mat.at<float>(0, 0);
	uint start_group   = 1;
	uint stop_group    = start_group + groups_to_sum;
																																			if(verbosity>local_verbosity_threshold+2) cout << "\ngroups_to_sum="<<groups_to_sum<<",  stop_group="<<stop_group<<endl<<flush;

	for (int j=start_group; j< stop_group  ; j++){
		for (int k=0; k<4; k++){
			pix_sum_reults[k] += pix_sum_mat.at<float>(j, k);
		}
	}
	uint layer =0;
	for (int i=0; i<3; i++){
		img_stats[layer*4 + IMG_MEAN + i ]	=	pix_sum_reults[i] / pix_sum_reults[3];
	}

	cl_event writeEvt;																										               // Upload img_mean to GPU
	status = clEnqueueWriteBuffer(uload_queue, img_stats_buf, CL_FALSE, 0, img_stats_size_bytes, img_stats, 0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nclEnqueueWriteBuffer imgmem status = " << checkerror(status) <<"\n"<<flush; exit_(status); }
																																			if(verbosity>local_verbosity_threshold+2){
																																				cout << "\n Pix_sum_results = (";
																																				for (int k=0; k<4; k++){
																																						cout << ", " << pix_sum_reults[k] ;
																																				}cout << ")";
																																				cout << endl;
																																				cout << "\n Pix_sum_results/num_groups = (";
																																				for (int k=0; k<4; k++){
																																					cout << ", " << pix_sum_reults[k]/pix_sum_reults[3] ;
																																				}cout << ")";
																																			}
	status = clFlush( uload_queue );																								if (status != CL_SUCCESS)	{ cout << "\nclEnqueueWriteBuffer clFlush( uload_queue ) status = " << checkerror(status) <<"\n"<<flush; exit_(status); }
	waitForEventAndRelease( &writeEvt );
																																		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::cvt_color_space()_chk3_Finished"<<flush;
	// TODO NB it would be faster to find the mean from the smallest layer, BUT only if there are no bugs e.g. the black bottom edge.
	// Variance however must be computed for each layer, because blurring may reduce contrast &=> variance.
}

void RunCL::img_variance(){
	int local_verbosity_threshold = verbosity_mp["RunCL::img_variance"];//-1;

	// TODO ? create a class for data, holding buffer, CPU data, stats about the data object, functions for write, read, save, display, & set_kernel_arg ?

	cl_int res, status;
	cl_event ev, writeEvt;																																								// cvt_color_space_kernel  or  img_variance_kernel
	res = clSetKernelArg(img_variance_kernel, 0, sizeof(cl_mem), &img_stats_buf);						   if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	//__global uchar3*		img_stats,		//0
	res = clSetKernelArg(img_variance_kernel, 1, sizeof(cl_mem), &imgmem);  			   				   if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	//__global float4*		img,			//1
	res = clSetKernelArg(img_variance_kernel, 2, sizeof(cl_mem), &uint_param_buf);						   if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	//__global uint*		uint_params		//2
	res = clSetKernelArg(img_variance_kernel, 3, sizeof(cl_mem), &mipmap_buf);							   if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	//__constant uint*		mipmap_params,	//3 // NB layer = 0.
	res = clSetKernelArg(img_variance_kernel, 4, local_work_size*4*sizeof(float), 	NULL);				   if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	//__local  float4*		local_sum_pix	//4
	res = clSetKernelArg(img_variance_kernel, 5, sizeof(cl_mem), &var_sum_mem);							   if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	//__local  float4*		global_sum_pix	//5

	status = clFlush(m_queue); 				if (status != CL_SUCCESS)	{ cout << "\nclFlush(m_queue) status = " << checkerror(status) <<"\n"<<flush; exit_(status);}
	status = clFinish(m_queue); 			if (status != CL_SUCCESS)	{ cout << "\nclFinish(m_queue)="<<status<<" "<<checkerror(status)<<"\n"<<flush; exit_(status);}
																																			if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::img_variance()_chk1,  global_work_size="<< global_work_size <<flush;
	res = clEnqueueNDRangeKernel(m_queue, img_variance_kernel, 1, 0, &global_work_size, &local_work_size, 0, NULL, &ev); 					// run img_variance _kernel  aka img_variance(..) ##### TODO which CommandQueue to use ? What events to check ?




	if (res != CL_SUCCESS)	{ cout << "\nres = " << checkerror(res) <<"\n"<<flush; exit_(res);}
	status = clFlush(m_queue);				                                                               if (status != CL_SUCCESS)	{ cout << "\nRunCL::img_variance()_chk1.1, clFlush(m_queue) status  = "<<status<<" "<< checkerror(status) <<"\n"<<flush; exit_(status);}
	status = clWaitForEvents (1, &ev);		                                                               if (status != CL_SUCCESS)	{ cout << "\nRunCL::img_variance()_chk1.2, clWaitForEventsh(1, &ev) = "<<status<<" "<<checkerror(status)  <<"\n"<<flush; exit_(status);}
                                                                                                                                            if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::img_variance()_chk2"<<flush;
	cv::Mat var_sum_mat = cv::Mat::zeros (pix_sum_size, 1, CV_32FC4); // cv::Mat::zeros (int rows, int cols, int type)						// NB the data returned is one float4 per group, for the base image, holding hsv channels plus entry[3]=pixel count.
	ReadOutput( var_sum_mat.data, var_sum_mem, pix_sum_size_bytes );                                                                        // se3_sum_size_bytes
                                                                                                                                            if(verbosity>local_verbosity_threshold+2) {cout<<"\n\nRunCL::img_variance(..)_chk2 ."<<flush;
																																				cout << "\nvar_sum_mat.size()="<<var_sum_mat.size()<<flush;
																																				cout << "\npix_sum_size="<<pix_sum_size<<flush;
                                                                                                                                                cout << "\n var_sum_mat.data = (\n";
                                                                                                                                                for (int i=0; i<pix_sum_size; i++){
                                                                                                                                                    cout << "\n group="<<i<<" : ( " << flush;
                                                                                                                                                    for (int j=0; j<4; j++){
                                                                                                                                                        cout << var_sum_mat.at<float>(i,j) << " , " << flush;
                                                                                                                                                    }
                                                                                                                                                    cout << ")" << flush;
                                                                                                                                                }cout << "\n)\n" << flush;
                                                                                                                                            }
	float var_sum_results[4] = {0};
	uint groups_to_sum = var_sum_mat.at<float>(0, 0);
	uint start_group   = 1;
	uint stop_group    = start_group + groups_to_sum;
																																			if(verbosity>local_verbosity_threshold+2) cout << "\ngroups_to_sum="<<groups_to_sum<<",  stop_group="<<stop_group<<endl<<flush;
	for (int j=start_group; j< stop_group; j++){
		for (int k=0; k<4; k++){
			var_sum_results[k] += var_sum_mat.at<float>(j, k);
		}
	}
	uint layer = 0; // TODO convert to mimpap version.
	for (int i=0; i<3; i++){
		img_stats[layer*4 + IMG_VAR + i ]	=	var_sum_results[i] / var_sum_results[3];
	}
																																			// Upload img_variance to GPU
	status = clEnqueueWriteBuffer(uload_queue, img_stats_buf, CL_FALSE, 0, img_stats_size_bytes, img_stats, 0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nclEnqueueWriteBuffer imgmem status = " << checkerror(status) <<"\n"<<flush; exit_(status); }

																																			if(verbosity>local_verbosity_threshold){
																																				cout << "\n Var_sum_results = (";
																																				for (int k=0; k<4; k++){
																																						cout << ", " << var_sum_results[k] ;
																																				}cout << ")";
																																				cout << endl;
																																				cout << "\n Var_sum_results/num_groups = (";
																																				for (int k=0; k<4; k++){
																																					cout << ", " << var_sum_results[k]/var_sum_results[3] ;
																																				}cout << ")";
																																			}
	status = clFlush( uload_queue );																								if (status != CL_SUCCESS)	{ cout << "\nclEnqueueWriteBuffer clFlush( uload_queue ) status = " << checkerror(status) <<"\n"<<flush; exit_(status); }
	waitForEventAndRelease( &writeEvt );
																																			if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::img_variance()_chk3_Finished"<<flush;
}

void RunCL::blur_image(){
	int local_verbosity_threshold = verbosity_mp["RunCL::blur_image"];// -1;

	cl_int res, status;
	cl_event ev, writeEvt;																																												// blur_image_kernel
	size_t local_size = local_work_size;																																								// set kernel args
	uint layer = 0;
	res = clSetKernelArg(blur_image_kernel, 0, sizeof(uint), 						&layer);					if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	;		//__constant uint*		mipmap_params,	//0
    res = clSetKernelArg(blur_image_kernel, 1, sizeof(cl_mem), 						&mipmap_buf);				if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	;		//__constant uint*		mipmap_params,	//1
	// res = clSetKernelArg(blur_image_kernel, 2, sizeof(cl_mem), 						&gaussian_buf);				if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	;		//__constant float*		gaussian,		//2
	res = clSetKernelArg(blur_image_kernel, 2, sizeof(cl_mem), 						&uint_param_buf);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	;		//__constant uint*		uint_params,	//3
	res = clSetKernelArg(blur_image_kernel, 3, sizeof(cl_mem), 						&imgmem);					if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	;		//__global   float4*	img,			//4
	res = clSetKernelArg(blur_image_kernel, 4, sizeof(cl_mem), 						&imgmem_blurred);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	;		//__global   float4*	img,			//4
	res = clSetKernelArg(blur_image_kernel, 5, (local_size+4) *5*4* sizeof(float), 	NULL);						if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	;		//__local    float4*	local_img_patch //5


	status = clFlush(m_queue);				if (status != CL_SUCCESS)	{ cout << "\nclFlush(m_queue) status = " << checkerror(status) <<"\n"<<flush; exit_(status);}
	status = clFinish(m_queue); 			if (status != CL_SUCCESS)	{ cout << "\nclFinish(m_queue)="<<status<<" "<<checkerror(status)<<"\n"<<flush; exit_(status);}
																																			if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::img_variance()_chk1,  global_work_size="<< global_work_size <<flush;
	res = clEnqueueNDRangeKernel(m_queue, blur_image_kernel, 1, 0, &global_work_size, &local_work_size, 0, NULL, &ev); 																					// run blur_image_kernel

	if (res != CL_SUCCESS)	{ cout << "\nres = " << checkerror(res) <<"\n"<<flush; exit_(res);}
	status = clFlush(m_queue);				                                                               if (status != CL_SUCCESS)	{ cout << "\nRunCL::img_variance()_chk1.1, clFlush(m_queue) status  = "<<status<<" "<< checkerror(status) <<"\n"<<flush; exit_(status);}
	status = clWaitForEvents (1, &ev);		                                                               if (status != CL_SUCCESS)	{ cout << "\nRunCL::img_variance()_chk1.2, clWaitForEventsh(1, &ev) = "<<status<<" "<<checkerror(status)  <<"\n"<<flush; exit_(status);}
                                                                                                                                            if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::img_variance()_chk2"<<flush;

																																			if (verbosity>local_verbosity_threshold){
                                                                                                                                                stringstream ss;		ss << dataset_frame_num << "_blur_image";
                                                                                                                                                stringstream ss_path;	ss_path << "imgmem_blurred";

                                                                                                                                                cv::Size new_Image_size = cv::Size(mm_width, mm_height);
                                                                                                                                                size_t   new_size_bytes = mm_width * mm_height * 4* 4;

                                                                                                                                                cout << "imgmem_blurred="<< imgmem_blurred << endl << flush;
                                                                                                                                                cout <<", ss.str()="<< ss.str() << endl << flush;
                                                                                                                                                cout <<", paths.at(\"imgmem_blurred\")="<< paths.at("imgmem_blurred") << endl << flush;

                                                                                                                                                cout <<", paths.at(" << ss_path.str() <<")="<< paths.at(ss_path.str()) << endl << flush;
                                                                                                                                                cout <<", new_size_bytes="<< new_size_bytes << endl << flush;
                                                                                                                                                cout <<", new_Image_size="<< new_Image_size <<"" << endl << flush;

                                                                                                                                                DownloadAndSave_3Channel(	imgmem_blurred, ss.str(), paths.at( ss_path.str() ), new_size_bytes/*mm_size_bytes_C4*/, new_Image_size/*mm_Image_size*/,  CV_32FC4 /*mm_Image_type*/, 	false );
																																			}
	res = clEnqueueCopyBuffer(
		m_queue,							// cl_command_queue command_queue
		imgmem_blurred, 					// cl_mem src_buffer
		imgmem, 							// cl_mem dst_buffer
		0,									// size_t src_offset
		0,									// size_t dst_offset
		mm_size_bytes_C4,					// size_t size
		0,									// cl_uint num_events_in_wait_list
		NULL,								// const cl_event* event_wait_list
		&writeEvt							// cl_event* event
	);

	if (res != CL_SUCCESS)	{ cout << "\nRunCL::blur_image() clEnqueueCopyBuffer(...)  res = " << checkerror(res) <<"\n"<<flush; exit_(res);}
}


void RunCL::mipmap_linear(){
	int local_verbosity_threshold = verbosity_mp["RunCL::mipmap_linear"];// -1;																										if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::mipmap_linear(..)_chk0"<<flush;}
	cl_event 	writeEvt;//, ev;
	cl_int 		res; //, status;
	/*
	//uint 		mipmap[8];
	float		a = float(0.0625);
	float		b = float(0.125);
	float		c = float(0.25);
	float 		gaussian[9] = {a, b, a, b, c, b, a , b, a };																																//  TODO load gaussian kernel & size from conf.json .
	if (mm_gaussian_size!=3) {cout<<"Error: (mm_gaussian_size!=3). Need to add code to malloc gaussian array. Probably with jsoncpp from 'conf.json' file." <<flush; exit(0); }

	status = clEnqueueWriteBuffer(uload_queue, gaussian_buf, CL_FALSE, 0, mm_gaussian_size*mm_gaussian_size*sizeof(float), gaussian, 0, NULL, &writeEvt);											// write mipmap_buf
	if (status != CL_SUCCESS){cout<<"\nstatus = "<<checkerror(status)<<"\n"<<flush; cout << "Error: RunCL::mipmap, clEnqueueWriteBuffer, mipmap_buf \n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	
	//res = clSetKernelArg(mipmap_float4_kernel, 2, sizeof(cl_mem), 					 	&gaussian_buf);				if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	;		//__constant float*		gaussian,		//2
	*/
	
	size_t local_size = local_work_size;																																							// set kernel args
	//      __private	 uint layer, set in mipmap_call_kernel(..) below                                                                                                                                      __private	 uint	    layer,		    //0
    res = clSetKernelArg(mipmap_float4_kernel, 1, sizeof(cl_mem), 					 	&mipmap_buf);				if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	;		//__constant uint*		mipmap_params,	//1
	res = clSetKernelArg(mipmap_float4_kernel, 2, sizeof(cl_mem), 					 	&uint_param_buf);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	;		//__constant uint*		uint_params,	//3
	res = clSetKernelArg(mipmap_float4_kernel, 3, sizeof(cl_mem), 						&imgmem);					if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	;		//__global   float4*	img,			//4
	res = clSetKernelArg(mipmap_float4_kernel, 4, (local_size+4) *5*4* sizeof(float), 	NULL);						if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	;		//__local    float4*	local_img_patch //5

	mipmap_call_kernel( mipmap_float4_kernel, m_queue, mm_start, mm_stop, true );   // TODO Start at first reduction, rehash __kernel void mipmap_linear_flt(..) and call only the num threads required. NB currently uses 4x as many threads as needed.

	//mipmap_call_kernel(cl_kernel kernel_to_call, cl_command_queue queue_to_call, uint start, uint stop, bool layers_sequential=false)


																																			if(verbosity>local_verbosity_threshold) {
																																				cout<<"\n\nRunCL::mipmap(..)_chk3 Finished all loops."<<flush;
																																				stringstream ss;	ss << dataset_frame_num << "_mipmap_linear";
																																				cv::Size new_Image_size = cv::Size(mm_width, mm_height);
																																				size_t   new_size_bytes = mm_width * mm_height * 4*4;
																																				ss << "_raw_";
																																				stringstream ss_path;	ss_path << "imgmem";
																																				DownloadAndSave_3Channel( imgmem, ss.str(), paths.at(ss_path.str()), new_size_bytes, new_Image_size, CV_32FC4, false );
																																				cout << "\n  (local_size+4) *5*4* sizeof(float) = "<<  (local_size+4) *5*4* sizeof(float) << " ,   (local_size+4) = " <<  (local_size+4) << endl << flush;
																																			}
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::mipmap_linear(..)_chk4 Finished"<<flush;}
}

void RunCL::img_gradients(){ //getFrame();
	int local_verbosity_threshold = verbosity_mp["RunCL::img_gradients"];// -2;																										if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::img_gradients(..)_chk0"<<flush;}
	cl_int res;
	size_t num_threads = ceil( (float)(mm_layerstep)/(float)local_work_size ) * local_work_size ;
																																			if(verbosity>local_verbosity_threshold) {cout << "\n num_threads = " << num_threads << ",   mm_layerstep = " << mm_layerstep << ",  local_work_size = " << local_work_size  <<endl << flush;}
	//      __private	 uint layer, set in mipmap_call_kernel(..) below                                                                                                                              __private	 uint	    layer,		//0
    res = clSetKernelArg(img_grad_kernel,  1, sizeof(cl_mem), &mipmap_buf);											if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__constant uint*	mipmap_params,	//1
	res = clSetKernelArg(img_grad_kernel,  2, sizeof(cl_mem), &uint_param_buf);										if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__constant uint*	uint_params		//2
	res = clSetKernelArg(img_grad_kernel,  3, sizeof(cl_mem), &fp32_param_buf);										if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__constant float*	fp32_params		//3
	res = clSetKernelArg(img_grad_kernel,  4, sizeof(cl_mem), &imgmem);												if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global   float4*	img,		//4
	res = clSetKernelArg(img_grad_kernel,  5, sizeof(cl_mem), &SE3_map_mem);										if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__constant float2*	SE3_map,	//8
	res = clSetKernelArg(img_grad_kernel,  6, sizeof(cl_mem), &SE3_grad_map_mem);									if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 	 float4*	SE3_grad_map//9
	res = clSetKernelArg(img_grad_kernel,  7, sizeof(cl_mem), &HSV_grad_mem);										if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}		//__global 	 float4*	HSV_grad_mem//10

																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::img_gradients(..)_chk2"<<flush;}
	mipmap_call_kernel( img_grad_kernel, m_queue );
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::img_gradients(..)_chk3 Finished all loops. Saving gxmem, gymem."<<flush;  // , g1mem
																																				stringstream ss;	ss << dataset_frame_num << "__img_grad_kernel";
																																				stringstream ss_path;
																																				///
																																				ss_path.str(std::string()); // reset ss_path
																																				ss_path << "SE3_grad_map_mem"<<flush;
																																				cout << "\n" << ss_path.str() <<flush;
																																				cout << "\n" <<  paths.at(ss_path.str()) <<flush;
																																				DownloadAndSave_6Channel_volume(  SE3_grad_map_mem, ss.str(), paths.at(ss_path.str()), mm_size_bytes_C4, mm_Image_size, CV_32FC4, false, -1, 6 );
																																				///
																																				ss_path.str(std::string()); // reset ss_path
																																				ss_path << "HSV_grad_mem"<<flush;
																																				cout << "\n" << ss_path.str() <<flush;
																																				cout << "\n" <<  paths.at(ss_path.str()) <<flush;
																																				DownloadAndSave_HSV_grad(  HSV_grad_mem, ss.str(), paths.at(ss_path.str()), mm_size_bytes_C8, mm_Image_size, CV_32FC(8), false, -1, 0 );

																																				cout << "\n\n SE3_grad_map_mem = SE3_grad_map_mem = "<<SE3_grad_map_mem;
																																			}
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::img_gradients(..)_chk4 Finished."<<flush;}
}

void RunCL::load_GT_depth(cv::Mat GT_depth, bool invert){ //getFrameData();, cv::Matx44f GT_K2K,   cv::Matx44f GT_pose2pose
    int local_verbosity_threshold = verbosity_mp["RunCL::load_GT_depth"];// -4;
																																		if(verbosity>local_verbosity_threshold) cout << "\nRunCL::load_GT_depth(..)_chk_0:"<<flush;
																																		if ( GT_depth.empty() ) {cerr << "\nRunCL::load_GT_depth(..)_chk_0:   Error  GT_depth.empty() "<<flush;  exit(1); }
	cl_event 		writeEvt;
	cl_int 	 		status;
	stringstream 	ss;
	ss << "__load_GT_depth" << (keyFrameCount*1000 + costvol_frame_num);

	float zero  = 0;
	status = clEnqueueFillBuffer(uload_queue, depth_mem, 	&zero, sizeof(float), 0, mm_size_bytes_C1, 	0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: RunCL::load_GT_depth_chk1\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	status = clEnqueueFillBuffer(uload_queue, depth_mem_GT, &zero, sizeof(float), 0, mm_size_bytes_C1, 	0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: RunCL::load_GT_depth_chk1\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
																																		if(verbosity>local_verbosity_threshold+1)
																																			{ DownloadAndSave( depth_mem_GT,   	ss.str(),   paths.at("depth_GT"),   	mm_size_bytes_C1,   mm_Image_size,   CV_32FC1, 	false , 1.0);	cout << "\nDownloadAndSave (.. depth_mem_GT ..)\n"<<flush;
																																			  DownloadAndSave( depth_mem,   	ss.str(),   paths.at("depth_mem"),   	mm_size_bytes_C1,   mm_Image_size,   CV_32FC1, 	false , 1.0);	cout << "\nDownloadAndSave (.. depth_mem_GT ..)\n"<<flush;


																																			}

    status = clEnqueueWriteBuffer(uload_queue, depth_mem, 		CL_FALSE, 0, image_size_bytes_C1,	 GT_depth.data, 0, NULL, &writeEvt);
																									if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: RunCL::load_GT_depth(..)_chk_2\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	ss << "__0";
	status = clFlush( uload_queue );																							if (status != CL_SUCCESS)	{ cout << "\nclEnqueueWriteBuffer clFlush( uload_queue ) status = " << checkerror(status) <<"\n"<<flush; exit_(status); }
	waitForEventAndRelease( &writeEvt );
	status = clFinish(uload_queue);
																																		if(verbosity>local_verbosity_threshold+1)
																																			{ DownloadAndSave( depth_mem_GT,   	ss.str(),   paths.at("depth_GT"),   	mm_size_bytes_C1,   mm_Image_size,   CV_32FC1, 	false , 1.0);	cout << "\nDownloadAndSave (.. depth_mem_GT ..)\n"<<flush;
																																			  DownloadAndSave( depth_mem,   	ss.str(),   paths.at("depth_mem"),   	mm_size_bytes_C1,   mm_Image_size,   CV_32FC1, 	false , 1.0);	cout << "\nDownloadAndSave (.. depth_mem_GT ..)\n"<<flush;
																																			}
	float factor = 1; //obj["min_depth"].asFloat(); // 1;//256;  // normalize depthmap as per conf.json file. Adjust for dataset.
	convert_depth( invert, factor);
	ss << "__1";
																																		if(verbosity>local_verbosity_threshold+1)
																																			{ DownloadAndSave( depth_mem_GT,   	ss.str(),   paths.at("depth_GT"),   	mm_size_bytes_C1,   mm_Image_size,   CV_32FC1, 	false , 1.0);	cout << "\nDownloadAndSave (.. depth_mem_GT ..)\n"<<flush;}
																																		if(verbosity>local_verbosity_threshold) cout << "\nRunCL::load_GT_depth(..)_chk_1:"<<flush;
	mipmap_depthmap(depth_mem_GT);
																																		if(verbosity>local_verbosity_threshold) cout << "\nRunCL::load_GT_depth(..)_chk_2:"<<flush;
	ss << "__2";
																																		//if(verbosity>local_verbosity_threshold) {
																																		if(costvol_frame_num > 0){
																																			bool old_vtp = vtp;
																																			vtp = true;
																																			DownloadAndSave( depth_mem_GT,   	ss.str(),   paths.at("depth_GT"),   	mm_size_bytes_C1,   mm_Image_size,   CV_32FC1, 	false , 1.0);	cout << "\nDownloadAndSave (.. depth_mem_GT ..)\n"<<flush;
																																			vtp = old_vtp;
																																		}

																																		if(verbosity>local_verbosity_threshold) cout << "\nRunCL::load_GT_depth(..)_chk_finished:##########################################################"<<flush;
}

void RunCL::convert_depth(uint invert, float factor){
	int local_verbosity_threshold = verbosity_mp["RunCL::convert_depth"];/* 0;*/															if(verbosity>local_verbosity_threshold) {
																																					cout<<"\n\nRunCL::convert_depth(uint invert, float factor)_chk0"<<flush;
																																					cout<<", invert="<<invert<<",  factor="<<factor<<flush;
																																			}
	cl_event 	ev;
	cl_int 		res, status;

	res = clSetKernelArg(convert_depth_kernel, 0, sizeof(uint),   &invert );			if(res!=CL_SUCCESS)		{cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);};	//__private	 bool 	invert,			//0
	res = clSetKernelArg(convert_depth_kernel, 1, sizeof(float),  &factor );			if(res!=CL_SUCCESS)		{cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);};	//__private	 float 	factor,			//1
	res = clSetKernelArg(convert_depth_kernel, 2, sizeof(cl_mem), &mipmap_buf);			if(res!=CL_SUCCESS)		{cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);};	//__constant uint*	mipmap_params,	//2
	res = clSetKernelArg(convert_depth_kernel, 3, sizeof(cl_mem), &uint_param_buf);		if(res!=CL_SUCCESS)		{cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);};	//__constant uint*	uint_params		//3
	res = clSetKernelArg(convert_depth_kernel, 4, sizeof(cl_mem), &depth_mem);			if(res!=CL_SUCCESS)		{cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);};	//__global	 float* depth_map,		//4
	res = clSetKernelArg(convert_depth_kernel, 5, sizeof(cl_mem), &depth_mem_GT);		if(res!=CL_SUCCESS)		{cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);};	//__global	 float* depth_map,		//5

	status = clFlush(m_queue); 				if (status != CL_SUCCESS)	{ cout << "\nclFlush(m_queue) status = " << checkerror(status) <<"\n"<<flush; exit_(status);}
	status = clFinish(m_queue); 			if (status != CL_SUCCESS)	{ cout << "\nclFinish(m_queue)="<<status<<" "<<checkerror(status)<<"\n"<<flush; exit_(status);}
																																			if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::convert_depth()_chk1,  global_work_size="<< global_work_size <<flush;

	res 	= clEnqueueNDRangeKernel(m_queue, convert_depth_kernel, 1, 0,  &global_work_size, &local_work_size, 0, NULL, &ev); // run mipmap_float4_kernel, NB wait for own previous iteration.
																						if (res    != CL_SUCCESS)	{ cout << "\nres = " << checkerror(res) <<"\n"<<flush; exit_(res);}
	status 	= clFlush(m_queue);															if (status != CL_SUCCESS)	{ cout << "\nclFlush(queue_to_call) status  = "<<status<<" "<< checkerror(status) <<"\n"<<flush; exit_(status);}
	status 	= clWaitForEvents (1, &ev);													if (status != CL_SUCCESS)	{ cout << "\nRunCL::convert_depth(..)  clWaitForEventsh(1, &ev) ="	<<status<<" "<<checkerror(status)  <<"\n"<<flush; exit_(status);}
}

void RunCL::mipmap_depthmap(cl_mem depthmap_){
	int local_verbosity_threshold = verbosity_mp["RunCL::mipmap_depthmap"];/*2;	*/									if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::mipmap_depthmap(..)_chk0"<<flush;}
	cl_event 	writeEvt;
	cl_int 		res;

	size_t local_size = local_work_size;																																							// set kernel args
	//      __private	 uint layer, set in mipmap_call_kernel(..) below                                                                                                                                      __private	 uint	    layer,		    //0
    res = clSetKernelArg(mipmap_float_kernel, 1, sizeof(cl_mem), 					 	&mipmap_buf);				if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	;		//__constant uint8*		mipmap_params,	//1
	res = clSetKernelArg(mipmap_float_kernel, 2, sizeof(cl_mem), 					 	&uint_param_buf);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	;		//__constant uint*		uint_params,	//3
	res = clSetKernelArg(mipmap_float_kernel, 3, sizeof(cl_mem), 						&depthmap_);				if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	;		//__global   float*		img,			//4
	res = clSetKernelArg(mipmap_float_kernel, 4, (local_size+4) *5*sizeof(float), 		NULL);						if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	;		//__local    float*		local_img_patch //5

	//void RunCL::mipmap_call_kernel( cl_kernel kernel_to_call,   cl_command_queue queue_to_call,   uint start, uint stop,   bool layers_sequential )

	mipmap_call_kernel( mipmap_float_kernel, m_queue, mm_start, mm_stop, true);// TODO Start at first reduction, rehash __kernel void mipmap_linear_flt(..) and call only the num threads required. NB currently uses 4x as many threads as needed.

																																			if(verbosity>local_verbosity_threshold) {
																																				cout<<"\n\nRunCL::mipmap_depthmap(..)_chk3 Finished all loops."<<flush;
																																				stringstream ss;	ss << dataset_frame_num << "_mipmap_depthmap";
																																				cv::Size new_Image_size = cv::Size(mm_width, mm_height);
																																				size_t   new_size_bytes = mm_width * mm_height * 4*4;
																																				ss << "_raw_";
																																				stringstream ss_path;	ss_path << "depth_GT";
																																				DownloadAndSave( depthmap_,   	ss.str(),   paths.at(ss_path.str()),   	mm_size_bytes_C1,   mm_Image_size,   CV_32FC1, 	false , fp32_params[MAX_INV_DEPTH]);

																																				cout << "\n  (local_size+4) *5*4* sizeof(float) = "<<  (local_size+4) *5*4* sizeof(float) << " ,   (local_size+4) = " <<  (local_size+4) << endl << flush;
																																			}
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::mipmap_depthmap(..)_chk4 Finished"<<flush;}
}
