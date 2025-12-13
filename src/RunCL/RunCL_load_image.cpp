#include "RunCL.hpp"

void RunCL::loadFrame(cv::Mat image){ //getFrame();																							// WriteBuffer basemem #########
	string fname = "RunCL::loadFrame(..)";
	int local_verbosity_threshold = V_RUNCL_LOADFRAME;
                                                                                                                                            if(verbosity>local_verbosity_threshold) {cout << "\n RunCL::loadFrame_chk 0\n" << flush;}
	_clEnqueueWriteBuffer(uload_queue, basemem, CL_FALSE, 0, image_size_bytes, image.data, fname);
                                                                                                                                            if (verbosity>local_verbosity_threshold){
                                                                                                                                                stringstream ss;	ss << dataset_frame_num << "loadFrame";
                                                                                                                                                DownloadAndSave_3Channel(basemem, ss.str(), paths.at("basemem"), image_size_bytes, baseImage_size,  baseImage_type, 	false );
                                                                                                                                            }
}

void RunCL::cvt_color_space(){ //getFrame(); basemem(CV_8UC3, RGB)->imgmem(CV16FC3, HSV), NB we will use basemem for image upload, and imgmem for the MipMap. RGB is default for .png standard.
	string fname = "RunCL::cvt_color_space()";
	int local_verbosity_threshold = V_RUNCL_CVT_COLOR_SPACE;
	//const cl_mem imgmem_   = current_frames[ current_frames_idx[0] ].img_buf;
	const cl_mem imgmem_   = current_frames[ current_frames_idx[0] ].img_buf;
                                                                                                                                            if(verbosity>local_verbosity_threshold) {
                                                                                                                                                cout<<"\n\nRunCL::cvt_color_space()_chk0"<<flush;
                                                                                                                                                cout << "\n";
                                                                                                                                                cout << ",mm_Image_size = " 	<< mm_Image_size << endl;
                                                                                                                                                cout << ",mm_Image_type = "		<< mm_Image_type << endl;
                                                                                                                                                cout << ",mm_size_bytes_C3 = " 	<< mm_size_bytes_C3 << endl;
                                                                                                                                                cout << ",mm_size_bytes_C4 = " 	<< mm_size_bytes_C4 << endl;
                                                                                                                                                cout << ",mm_size_bytes_C1 = " 	<< mm_size_bytes_C1 << endl;
                                                                                                                                                cout << "\n";
                                                                                                                                                cout << ",baseImage_size, = " 	<< baseImage_size << endl;
                                                                                                                                                cout << ",baseImage_type = " 	<< baseImage_type << endl;
                                                                                                                                                cout << ",image_size_bytes = " 	<< image_size_bytes	<< endl;
                                                                                                                                                cout << ",mm_vol_size_bytes = " << mm_vol_size_bytes << endl;
                                                                                                                                                cout << "\n" 					<< flush;
                                                                                                                                            }
	_clSetKernelArg(cvt_color_space_linear_kernel, 0, sizeof(cl_mem), &basemem, fname);														//__global uchar3*		base,			//0
	_clSetKernelArg(cvt_color_space_linear_kernel, 1, sizeof(cl_mem), &imgmem_, fname);	   												//__global float4*		img,			//1
	_clSetKernelArg(cvt_color_space_linear_kernel, 2, sizeof(cl_mem), &uint_param_buf, fname);												//__global uint*		uint_params		//2
	_clSetKernelArg(cvt_color_space_linear_kernel, 3, sizeof(cl_mem), &mipmap_buf, fname);													//__constant uint*		mipmap_params,	//3 // NB layer = 0.
	_clSetKernelArg(cvt_color_space_linear_kernel, 4, local_work_size*4*sizeof(float), 	NULL, fname);										//__local  float4*		local_sum_pix	//4
	_clSetKernelArg(cvt_color_space_linear_kernel, 5, sizeof(cl_mem), &pix_sum_mem, fname);													//__local  float4*		global_sum_pix	//5
																																			if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::cvt_color_space()_chk1,  global_work_size="<< global_work_size <<flush;
	_clEnqueueNDRangeKernel(m_queue, cvt_color_space_linear_kernel, 1, 0, &global_work_size, &local_work_size, fname);
                                                                                                                                            if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::cvt_color_space()_chk2"<<flush;
                                                                                                                                            if (verbosity>local_verbosity_threshold){
                                                                                                                                                stringstream ss;		ss << dataset_frame_num << "_cvt_color_space";
                                                                                                                                                stringstream ss_path;	ss_path << "imgmem";

                                                                                                                                                cv::Size new_Image_size = cv::Size(mm_width, mm_height);
                                                                                                                                                size_t   new_size_bytes = mm_width * mm_height * 4* 4;

                                                                                                                                                cout << "imgmem="<< imgmem_ << endl << flush;
                                                                                                                                                cout <<", ss.str()="<< ss.str() << endl << flush;
                                                                                                                                                cout <<", paths.at(\"imgmem\")="<< paths.at("imgmem") << endl << flush;

                                                                                                                                                cout <<", paths.at(" << ss_path.str() <<")="<< paths.at(ss_path.str()) << endl << flush;
                                                                                                                                                cout <<", new_size_bytes="<< new_size_bytes << endl << flush;
                                                                                                                                                cout <<", new_Image_size="<< new_Image_size <<"" << endl << flush;

                                                                                                                                                DownloadAndSave_3Channel(	imgmem_, ss.str(), paths.at( ss_path.str() ), new_size_bytes/*mm_size_bytes_C4*/, new_Image_size/*mm_Image_size*/,  CV_32FC4 /*mm_Image_type*/, 	false );
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
	float pix_sum_reults[4]		= {0};
	uint groups_to_sum			= pix_sum_mat.at<float>(0, 0);
	uint start_group			= 1;
	uint stop_group				= start_group + groups_to_sum;
																																			if(verbosity>local_verbosity_threshold+2) cout << "\ngroups_to_sum="<<groups_to_sum<<",  stop_group="<<stop_group<<endl<<flush;
	for (int j=start_group; j< stop_group  ; j++){
		for (int k=0; k<4; k++){
			pix_sum_reults[k] += pix_sum_mat.at<float>(j, k);
		}
	}
	uint layer =0;
	for (int i=0; i<3; i++){
		img_stats[layer*8 + IMG_MEAN*4 + i ]	=	pix_sum_reults[i] / pix_sum_reults[3];
	}
	_clEnqueueWriteBuffer(uload_queue, img_stats_buf, CL_FALSE, 0, img_stats_size_bytes, img_stats, fname);									// Upload img_mean to GPU
																																			if(verbosity>local_verbosity_threshold/*+2*/){
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
																																			uint start = mm_start, stop = mm_stop;
																																			float img_stats__[img_stats_size];
																																			ReadOutput(  (uchar*)img_stats__, img_stats_buf, img_stats_size_bytes );
																																			if (verbosity>local_verbosity_threshold){
																																				cout << "\n" << fname;
																																				for (uint layer=start; layer<=stop; layer++){
																																					cout << "\nlayer="<<layer<<" mean={ ";
																																					for (uint chan=0; chan<4; chan++){
																																						cout <<  img_stats__[layer*2*4+chan] << ",  \t";
																																					}
																																					cout << "},   \tvariance={ ";
																																					for (uint chan=0; chan<4; chan++){
																																						cout <<  img_stats__[layer*2*4 + IMG_VAR*4 + chan] << ",  \t";
																																					}
																																					cout << "}" << flush;
																																				}
																																			}
																																			if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::cvt_color_space()_chk3_Finished"<<flush;
	// TODO NB it would be faster to find the mean from the smallest layer, BUT only if there are no bugs e.g. the black bottom edge.
	// Variance however must be computed for each layer, because blurring may reduce contrast &=> variance.
}

void RunCL::sum_image_variance(){
	string fname = "RunCL::img_variance()";
	int local_verbosity_threshold = V_RUNCL_SUM_IMAGE_VARIANCE;//verbosity_mp["RunCL::img_variance"];//-1;
	// TODO ? create a class for data, holding buffer, CPU data, stats about the data object, functions for write, read, save, display, & set_kernel_arg ?

	const cl_kernel kernel		= sum_image_variance_kernel;
	const cl_mem imgmem_		= current_frames[ current_frames_idx[0] ].img_buf;
																																			// cvt_color_space_kernel  or  img_variance_kernel
	_clSetKernelArg( kernel, 0, sizeof(cl_mem), 					&img_stats_buf, 		fname);											//__global uchar3*		img_stats,		//0
	_clSetKernelArg( kernel, 1, sizeof(cl_mem), 					&imgmem_, 				fname);											//__global float4*		img,			//1
	_clSetKernelArg( kernel, 2, sizeof(cl_mem), 					&uint_param_buf, 		fname);											//__global uint*		uint_params		//2
	_clSetKernelArg( kernel, 3, sizeof(cl_mem), 					&mipmap_buf, 			fname);											//__constant uint*		mipmap_params,	//3 // NB layer = 0.
	_clSetKernelArg( kernel, 4, local_work_size*4*sizeof(float), 	NULL, 					fname);											//__local  float4*		local_sum_pix	//4
	_clSetKernelArg( kernel, 5, sizeof(cl_mem), 					&var_sum_mem, 			fname);											//__local  float4*		global_sum_pix	//5
																																			if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::img_variance()_chk1,  global_work_size="<< global_work_size <<flush;
	_clEnqueueNDRangeKernel(m_queue,  kernel, 1, 0, &global_work_size, &local_work_size, fname); 											// run img_variance _kernel  aka img_variance(..) ##### TODO which CommandQueue to use ? What events to check ?

	//mipmap_call_kernel( kernel, m_queue, true );

	//mipmap_call_kernel(  kernel, m_queue );  To run on all layers would req kernel edits.
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
		img_stats[layer*8 + IMG_VAR*4 + i ]	=	var_sum_results[i] / var_sum_results[3];
	}
	_clEnqueueWriteBuffer( uload_queue, img_stats_buf, CL_FALSE, 0, img_stats_size_bytes, img_stats, fname);									// Upload img_variance to GPU
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
																																			if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::img_variance()_chk3_Finished"<<flush;
}


void RunCL::sample_image_variance(){
	string fname = "RunCL::sample_image_variance()";
	int local_verbosity_threshold = V_RUNCL_SAMPLE_IMAGE_VARIANCE;
	uint start = mm_start, stop = mm_stop;																									// NB mm_start = 0, mm_stop = obj["num_reductions"].asUInt(); set in conf.json
	const cl_kernel kernel		= sample_image_variance_kernel;
	const cl_mem imgmem_		= current_frames[ current_frames_idx[0] ].img_buf;
																																			// cvt_color_space_kernel  or  img_variance_kernel
	_clSetKernelArg( kernel, 0, sizeof(uint), 						&start,					fname);											//__private	uint		start_layer,	//0
	_clSetKernelArg( kernel, 1, sizeof(cl_mem), 					&img_stats_buf, 		fname);											//__global uchar3*		img_stats,		//1
	_clSetKernelArg( kernel, 2, sizeof(cl_mem), 					&imgmem_, 				fname);											//__global float4*		img,			//2
	_clSetKernelArg( kernel, 3, sizeof(cl_mem), 					&uint_param_buf, 		fname);											//__global uint*		uint_params		//3
	_clSetKernelArg( kernel, 4, sizeof(cl_mem), 					&mipmap_buf, 			fname);											//__constant uint*		mipmap_params,	//4 // NB layer = 0.
	_clSetKernelArg( kernel, 5, local_work_size*4*sizeof(float), 	NULL, 					fname);											//__local  float4*		local_sum_pix	//5

	size_t global_work_size__	= (stop-start) * local_work_size;																			// One workgroup per layer of the image pyramid
	cout<<"\n"<<fname<<"  local_work_size="<<local_work_size<<"   global_work_size__="<<global_work_size__<<"   global_work_size="<<global_work_size__<<",  stop="<<stop<<",  start="<<start<<",  (stop-start)="<<(stop-start)<<flush;

	_clEnqueueNDRangeKernel(m_queue,  kernel, 1, 0, &global_work_size__, &local_work_size, fname);

	//cv::Mat var_sum_mat = cv::Mat::zeros (img_stats_size, 1, CV_32FC4); 																	// cv::Mat::zeros (int rows, int cols, int type)

	float img_stats__[img_stats_size];

	ReadOutput(  (uchar*)img_stats__, img_stats_buf, img_stats_size_bytes );
	for (uint idx=0; idx<img_stats_size; idx++  ){ img_stats[idx] = img_stats__[idx];}
																																			if (verbosity>local_verbosity_threshold){
																																				cout << "\n" << fname;
																																				for (uint layer=start; layer<=stop; layer++){
																																					cout << "\nlayer="<<layer<<" mean={ ";
																																					for (uint chan=0; chan<4; chan++){
																																						cout <<  img_stats__[layer*2*4+chan] << ",  \t";
																																					}
																																					cout << "},   \tvariance={ ";
																																					for (uint chan=0; chan<4; chan++){
																																						cout <<  img_stats__[layer*2*4 + IMG_VAR*4 + chan] << ",  \t";
																																					}
																																					cout << "}" << flush;
																																					for (uint chan=0; chan<4; chan++){
																																						cout <<  img_stats[layer*2*4 + IMG_VAR*4 + chan] << ",  \t";
																																					}
																																					cout << "}" << flush;
																																				}

																																				cout<<"\n\nRunCL::mipmap(..)_chk3 Finished all loops."<<flush;
																																				stringstream ss;	ss << dataset_frame_num << "_sample_image_variance";
																																				cv::Size new_Image_size = cv::Size(mm_width, mm_height);
																																				size_t   new_size_bytes = mm_width * mm_height * 4*4;
																																				ss << "_raw_";
																																				DownloadAndSave_3Channel( imgmem_, ss.str(), paths.at("imgmem"), new_size_bytes, new_Image_size, CV_32FC4, false, 1, 0, true );

																																			}
}

/*
void RunCL::blur_image(){//cl_mem in_buff, cl_mem blurred_buf, std::string folder ){
	string fname = "RunCL::blur_image()";
	int local_verbosity_threshold = V_RUNCL_BLUR_IMAGE;//verbosity_mp["RunCL::blur_image"];// -1;

	/ * const * / cl_mem imgmem_   = current_frames[ current_frames_idx[0] ].img_buf;

	size_t local_size = local_work_size;
	uint layer = 0;
	_clSetKernelArg(blur_image_kernel, 0, sizeof(uint), 						&layer, fname );											//__constant uint*		mipmap_params,	//0
    _clSetKernelArg(blur_image_kernel, 1, sizeof(cl_mem), 						&mipmap_buf, fname );										//__constant uint*		mipmap_params,	//1
	_clSetKernelArg(blur_image_kernel, 2, sizeof(cl_mem), 						&uint_param_buf, fname );									//__constant uint*		uint_params,	//2
	_clSetKernelArg(blur_image_kernel, 3, sizeof(cl_mem), 						&imgmem_, fname );											//__global   float4*	img,			//3
	_clSetKernelArg(blur_image_kernel, 4, sizeof(cl_mem), 						&imgmem_blurred, fname );									//__global   float4*	img,			//4
	_clSetKernelArg(blur_image_kernel, 5, (local_size+4) *5*4* sizeof(float), 	NULL, fname );												//__local    float4*	local_img_patch //5
																																			if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::img_variance()_chk1,  global_work_size="<< global_work_size <<flush;
	_clEnqueueNDRangeKernel(m_queue, blur_image_kernel, 1, 0, &global_work_size, &local_work_size, fname ); 								// run blur_image_kernel
                                                                                                                                            if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::img_variance()_chk2"<<flush;
	_clSetKernelArg(blur_image_kernel, 4, sizeof(cl_mem), 						&imgmem_, fname );											//__global   float4*	img,			//3
	_clSetKernelArg(blur_image_kernel, 3, sizeof(cl_mem), 						&imgmem_blurred, fname );									//__global   float4*	img,			//4

	_clEnqueueNDRangeKernel(m_queue, blur_image_kernel, 1, 0, &global_work_size, &local_work_size, fname ); 								// run blur_image_kernel
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

                                                                                                                                                DownloadAndSave_3Channel(	imgmem_blurred, ss.str(), paths.at( ss_path.str() ), new_size_bytes / * mm_size_bytes_C4 * / , new_Image_size / * mm_Image_size * / ,  CV_32FC4  / * mm_Image_type * / , 	false );
																																			}
	// swap( imgmem_blurred, imgmem_ );
																																			if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::img_variance()_Finished"<<flush;
}
*/

void RunCL::mipmap_linear(cl_mem image_buf, std::string folder){
	string fname = "RunCL::mipmap_linear()";
	int local_verbosity_threshold = V_RUNCL_MIPMAP_LINEAR;																					if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::mipmap_linear(..)_chk0"<<flush;}
	
	size_t local_size = local_work_size;																									// set kernel args
	//      __private	 uint layer, set in mipmap_call_kernel(..) below                                                                      __private	 uint	    layer,		    //0
    _clSetKernelArg(mipmap_float4_kernel, 1, sizeof(cl_mem), 					 	&mipmap_buf, fname);									//__constant uint*		mipmap_params,	//1
	_clSetKernelArg(mipmap_float4_kernel, 2, sizeof(cl_mem), 					 	&uint_param_buf, fname);								//__constant uint*		uint_params,	//3
	_clSetKernelArg(mipmap_float4_kernel, 3, sizeof(cl_mem), 						&image_buf, fname);										//__global   float4*	img,			//4
	_clSetKernelArg(mipmap_float4_kernel, 4, (local_size+4) *5*4* sizeof(float), 	NULL, fname);											//__local    float4*	local_img_patch //5

	mipmap_call_kernel( mipmap_float4_kernel, m_queue, true );   // TODO Start at first reduction, rehash __kernel void mipmap_linear_flt(..) and call only the num threads required. NB currently uses 4x as many threads as needed.

																																			if(verbosity>local_verbosity_threshold) {
																																				cout<<"\n\nRunCL::mipmap(..)_chk3 Finished all loops."<<flush;
																																				stringstream ss;	ss << dataset_frame_num << "_mipmap_linear";
																																				cv::Size new_Image_size = cv::Size(mm_width, mm_height);
																																				size_t   new_size_bytes = mm_width * mm_height * 4*4;
																																				ss << "_raw_";
																																				//stringstream ss_path;	ss_path << "imgmem";
																																				//
	//void DownloadAndSave_3Channel( buffer, count, folder_tiff, image_size_bytes, size_mat, type_mat, show,            max_range=1, offset=0, exception_tiff=false )
	//	   DownloadAndSave_3Channel( buffer, count, folder_tiff, image_size_bytes, size_mat, type_mat, show,  &bufImg,  max_range,   offset,   exception_tiff );
	//void DownloadAndSave_3Channel( buffer, count, folder_tiff, image_size_bytes, size_mat, type_mat, show,  *bufImg,  max_range=1, offset=0, exception_tiff=false );
																																				DownloadAndSave_3Channel( image_buf, ss.str(), paths.at(folder/*ss_path.str()*/), new_size_bytes, new_Image_size, CV_32FC4, false, 1, 0, true );
																																				cout << "\n  (local_size+4) *5*4* sizeof(float) = "<<  (local_size+4) *5*4* sizeof(float) << " ,   (local_size+4) = " <<  (local_size+4) << endl << flush;
																																			}
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::mipmap_linear(..)_chk4 Finished"<<flush;}
}
/*
void RunCL::mipmap_3x3blur_linear(cl_mem image_buf, std::string folder){
	string fname = "RunCL::mipmap_linear()";
	int local_verbosity_threshold = V_RUNCL_MIPMAP_LINEAR;																					if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::mipmap_linear(..)_chk0"<<flush;}

	size_t local_size = local_work_size;																									// set kernel args
	//      __private	 uint layer, set in mipmap_call_kernel(..) below                                                                      __private	 uint	    layer,		    //0
    _clSetKernelArg( mipmap_3x3blur_flt4_kernel, 1, sizeof(cl_mem), 					 	&mipmap_buf, fname);									//__constant uint*		mipmap_params,	//1
	_clSetKernelArg( mipmap_3x3blur_flt4_kernel, 2, sizeof(cl_mem), 					 	&uint_param_buf, fname);								//__constant uint*		uint_params,	//3
	_clSetKernelArg( mipmap_3x3blur_flt4_kernel, 3, sizeof(cl_mem), 						&image_buf, fname);										//__global   float4*	img,			//4
	_clSetKernelArg( mipmap_3x3blur_flt4_kernel, 4, (local_size+4) *5*4* sizeof(float), 	NULL, fname);											//__local    float4*	local_img_patch //5

	mipmap_call_kernel( mipmap_3x3blur_flt4_kernel, m_queue, true );   // TODO Start at first reduction, rehash __kernel void mipmap_linear_flt(..) and call only the num threads required. NB currently uses 4x as many threads as needed.

																																			if(verbosity>local_verbosity_threshold) {
																																				cout<<"\n\nRunCL::mipmap(..)_chk3 Finished all loops."<<flush;
																																				stringstream ss;	ss << dataset_frame_num << "_mipmap_linear";
																																				cv::Size new_Image_size = cv::Size(mm_width, mm_height);
																																				size_t   new_size_bytes = mm_width * mm_height * 4*4;
																																				ss << "_raw_";
																																				//stringstream ss_path;	ss_path << "imgmem";
																																				//
																																				DownloadAndSave_3Channel( image_buf, ss.str(), paths.at(folder / *  ss_path.str()  * / ), new_size_bytes, new_Image_size, CV_32FC4, false, 1, 0, true );
																																				cout << "\n  (local_size+4) *5*4* sizeof(float) = "<<  (local_size+4) *5*4* sizeof(float) << " ,   (local_size+4) = " <<  (local_size+4) << endl << flush;
																																			}
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::mipmap_linear(..)_chk4 Finished"<<flush;}
}
*/
/* void RunCL::img_gradients(){ //getFrame();
	string fname = "RunCL::img_gradients()";
	int local_verbosity_threshold = V_RUNCL_IMG_GRADIENTS;																					if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::img_gradients(..)_chk0"<<flush;}
	size_t num_threads = ceil( (float)(mm_layerstep)/(float)local_work_size ) * local_work_size ;

	const cl_mem imgmem_   = current_frames[ current_frames_idx[0] ].img_buf;
																																			if(verbosity>local_verbosity_threshold) {cout << "\n num_threads = " << num_threads << ",   mm_layerstep = " << mm_layerstep << ",  local_work_size = " << local_work_size  <<endl << flush;}
	//      __private	 uint layer, set in mipmap_call_kernel(..) below                                                                      __private	 uint	    layer,		//0
    _clSetKernelArg(img_grad_kernel,  1, sizeof(cl_mem), &mipmap_buf, fname);																//__constant uint*	mipmap_params,	//1
	_clSetKernelArg(img_grad_kernel,  2, sizeof(cl_mem), &uint_param_buf, fname);															//__constant uint*	uint_params		//2
	_clSetKernelArg(img_grad_kernel,  3, sizeof(cl_mem), &fp32_param_buf, fname);															//__constant float*	fp32_params		//3
	_clSetKernelArg(img_grad_kernel,  4, sizeof(cl_mem), &imgmem_, fname);																	//__global   float4*	img,		//4
	_clSetKernelArg(img_grad_kernel,  5, sizeof(cl_mem), &SE3_map_mem, fname);																//__constant float2*	SE3_map,	//8
	_clSetKernelArg(img_grad_kernel,  6, sizeof(cl_mem), &SE3_grad_map_mem, fname);															//__global 	 float4*	SE3_grad_map//9
	_clSetKernelArg(img_grad_kernel,  7, sizeof(cl_mem), &HSV_grad_mem, fname);																//__global 	 float4*	HSV_grad_mem//10
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
*/
void RunCL::load_GT_depth(cv::Mat GT_depth, bool invert){ //getFrameData();,  cv::Matx44f GT_K2K,   cv::Matx44f GT_pose2pose
	string fname = "RunCL::load_GT_depth(..)";
	int local_verbosity_threshold = V_RUNCL_LOAD_GT_DEPTH;
																																		if(verbosity>local_verbosity_threshold) cout << "\nRunCL::load_GT_depth(..)_chk_0:"<<flush;
																																		if ( GT_depth.empty() ) {cerr << "\nRunCL::load_GT_depth(..)_chk_0:   Error  GT_depth.empty() "<<flush;  exit_(1); }
	stringstream 	ss;
	ss << "__load_GT_depth";// << (keyFrameCount*1000 + costvol_frame_num);

	float default_depth  = (fp32_params[MAX_INV_DEPTH] + fp32_params[MIN_INV_DEPTH])/2.0;

	_clEnqueueFillBuffer(  uload_queue, depth_mem_temp, &default_depth, sizeof(float), 0, mm_size_bytes_C1,  fname );
	_clEnqueueFillBuffer(  uload_queue, depth_mem_GT, 	&default_depth, sizeof(float), 0, mm_size_bytes_C1,  fname );

	float max_range_ = 0.0f;																											// 0.0f => (temp_mat / maxVal) * 256*256 for .png; TODO move this to conf.json
																																		if(verbosity>local_verbosity_threshold+1){
																																			DownloadAndSave( depth_mem_GT,  ss.str(),   paths.at("depth_GT"),   	mm_size_bytes_C1,   mm_Image_size,   CV_32FC1, 	false , max_range_ );	cout << "\nDownloadAndSave (.. depth_mem_GT ..)\n"<<flush;
																																			DownloadAndSave( depth_mem_temp,   	ss.str(),   paths.at("depth_mem_temp"),   	image_size_bytes_C1,   baseImage_size,   CV_32FC1, 	false , max_range_ );	cout << "\nDownloadAndSave (.. depth_mem_GT ..)\n"<<flush;						// NB depth_mem_temp is just the raw image, with no margins nor mipmapping.
																																		}
	_clEnqueueWriteBuffer(uload_queue, depth_mem_temp, 		CL_FALSE, 0, image_size_bytes_C1,	 GT_depth.data,  fname);
	ss << "__0";
																																		if(verbosity>local_verbosity_threshold+1){
																																			DownloadAndSave( depth_mem_GT,  ss.str(),   paths.at("depth_GT"),   	mm_size_bytes_C1,   mm_Image_size,   CV_32FC1, 	false , max_range_ );	cout << "\nDownloadAndSave (.. depth_mem_GT ..)\n"<<flush;
																																			DownloadAndSave( depth_mem_temp,   	ss.str(),   paths.at("depth_mem_temp"),   	image_size_bytes_C1,   baseImage_size,   CV_32FC1, 	false , max_range_ );	cout << "\nDownloadAndSave (.. depth_mem_GT ..)\n"<<flush;
																																		}
	float factor = 1; //obj["min_depth"].asFloat(); // 1;//256;  // normalize depthmap as per conf.json file. Adjust for dataset.
	convert_depth( invert, factor);																										// calls convert_depth_kernel, reads depth_mem_temp, divides by "factor", takes inverse, then writes to depth_mem_GT
	ss << "__1";
																																		if(verbosity>local_verbosity_threshold+1){
																																			DownloadAndSave( depth_mem_GT,	ss.str(),   paths.at("depth_GT"),   	mm_size_bytes_C1,   mm_Image_size,   CV_32FC1, 	false , max_range_ );	cout << "\nDownloadAndSave (.. depth_mem_GT ..)\n"<<flush;
																																			DownloadAndSave( depth_mem_temp,   	ss.str(),   paths.at("depth_mem_temp"),   	image_size_bytes_C1,   baseImage_size,   CV_32FC1, 	false , max_range_ );	cout << "\nDownloadAndSave (.. depth_mem_GT ..)\n"<<flush;
																																		}
																																		if(verbosity>local_verbosity_threshold) cout << "\nRunCL::load_GT_depth(..)_chk_1:"<<flush;
	mipmap_depthmap(depth_mem_GT);
	ss << "__2";
																																		if(verbosity>local_verbosity_threshold+1){
																																			DownloadAndSave( depth_mem_GT,  ss.str(),   paths.at("depth_GT"),   	mm_size_bytes_C1,   mm_Image_size,   CV_32FC1, 	false , max_range_ );	cout << "\nDownloadAndSave (.. depth_mem_GT ..)\n"<<flush;
																																			DownloadAndSave( depth_mem_temp,   	ss.str(),   paths.at("depth_mem_temp"),   	image_size_bytes_C1,   baseImage_size,   CV_32FC1, 	false , max_range_ );	cout << "\nDownloadAndSave (.. depth_mem_GT ..)\n"<<flush;
																																		}
																																		if(verbosity>local_verbosity_threshold) cout << "\nRunCL::load_GT_depth(..)_chk_2:"<<flush;
	ss << "__3";
																																		if(verbosity>local_verbosity_threshold+1){	//(costvol_frame_num > 0)
																																			bool old_vtp = vtp;
																																			vtp = true;
																																			DownloadAndSave( depth_mem_GT,   	ss.str(),   paths.at("depth_GT"),   	mm_size_bytes_C1,   mm_Image_size,   CV_32FC1, 	false , max_range_ );	cout << "\nDownloadAndSave (.. depth_mem_GT ..)\n"<<flush;
																																			vtp = old_vtp;
																																		}

																																		if(verbosity>local_verbosity_threshold) cout << "\nRunCL::load_GT_depth(..)_chk_finished:##########################################################"<<flush;
}

void RunCL::convert_depth(uint invert, float factor){
	string fname = "RunCL::convert_depth(..)";
	int local_verbosity_threshold = V_RUNCL_CONVERT_DEPTH;//verbosity_mp["RunCL::convert_depth"];/* 0;*/
																																		if(verbosity>local_verbosity_threshold) {
																																			cout<<"\n\nRunCL::convert_depth(uint invert, float factor)_chk0"<<flush;
																																			cout<<", invert="<<invert<<",  factor="<<factor<<flush;
																																		}
	_clSetKernelArg(convert_depth_kernel, 0, sizeof(uint),   &invert,			fname);													//__private	 bool 	invert,			//0
	_clSetKernelArg(convert_depth_kernel, 1, sizeof(float),  &factor,			fname);													//__private	 float 	factor,			//1
	_clSetKernelArg(convert_depth_kernel, 2, sizeof(cl_mem), &mipmap_buf,		fname);													//__constant uint*	mipmap_params,	//2
	_clSetKernelArg(convert_depth_kernel, 3, sizeof(cl_mem), &uint_param_buf,	fname);													//__constant uint*	uint_params		//3
	_clSetKernelArg(convert_depth_kernel, 4, sizeof(cl_mem), &depth_mem_temp,	fname);													//__global	 float* depth_map,		//4
	_clSetKernelArg(convert_depth_kernel, 5, sizeof(cl_mem), &depth_mem_GT,		fname);													//__global	 float* depth_map,		//5
																																		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::convert_depth()_chk1,  global_work_size="<< global_work_size <<flush;
	_clEnqueueNDRangeKernel(m_queue, convert_depth_kernel, 1, 0,  &global_work_size, &local_work_size,		fname); 					// run mipmap_float4_kernel, NB wait for own previous iteration.
																																		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::convert_depth()_Finished:############################################################" <<flush;
}

void RunCL::mipmap_depthmap(cl_mem depthmap_){
	string fname = "RunCL::mipmap_depthmap(..)";
	int local_verbosity_threshold = V_RUNCL_MIPMAP_DEPTHMAP;																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::mipmap_depthmap(..)_chk0"<<flush;}

	size_t local_size = local_work_size;																								// set kernel args
	//      __private	 uint layer, set in mipmap_call_kernel(..) below																__private	 uint	    layer,		    //0
    _clSetKernelArg(mipmap_float_kernel, 1, sizeof(cl_mem), 					&mipmap_buf,		fname);								//__constant uint8*		mipmap_params,	//1
	_clSetKernelArg(mipmap_float_kernel, 2, sizeof(cl_mem), 					&uint_param_buf,	fname);								//__constant uint*		uint_params,	//3
	_clSetKernelArg(mipmap_float_kernel, 3, sizeof(cl_mem), 					&depthmap_,			fname);								//__global   float*		img,			//4
	_clSetKernelArg(mipmap_float_kernel, 4, (local_size+4) *5*sizeof(float), 	NULL,				fname);								//__local    float*		local_img_patch //5

	mipmap_call_kernel( mipmap_float_kernel, m_queue, true);// TODO Start at first reduction, rehash __kernel void mipmap_linear_flt(..) and call only the num threads required. NB currently uses 4x as many threads as needed.

																																		if(verbosity>local_verbosity_threshold) {
																																			cout<<"\n\nRunCL::mipmap_depthmap(..)_chk3 Finished all loops."<<flush;
																																			stringstream ss;	ss << dataset_frame_num << "_mipmap_depthmap";
																																			cv::Size new_Image_size = cv::Size(mm_width, mm_height);
																																			//size_t   new_size_bytes = mm_width * mm_height * 4*4;
																																			ss << "_raw_";
																																			stringstream ss_path;	ss_path << "depth_GT";
																																			DownloadAndSave( depthmap_,   	ss.str(),   paths.at(ss_path.str()),   	mm_size_bytes_C1,   mm_Image_size,   CV_32FC1, 	false , fp32_params[MAX_INV_DEPTH]);

																																			cout << "\n  (local_size+4) *5*4* sizeof(float) = "<<  (local_size+4) *5*4* sizeof(float) << " ,   (local_size+4) = " <<  (local_size+4) << endl << flush;
																																		}
																																		if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::mipmap_depthmap(..)_chk4 Finished:#######################################################"<<flush;}
}
