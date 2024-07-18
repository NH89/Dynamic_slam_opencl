#include "RunCL.hpp"

void RunCL::createFolders(){
	int local_verbosity_threshold = verbosity_mp["RunCL::createFolders"];
																																			if(verbosity>local_verbosity_threshold) cout << "\n createFolders_chk 0\n" << flush;
	std::string   out_dir = date_time_string();

	boost::filesystem::path 	out_path(boost::filesystem::current_path());
	boost::filesystem::path 	conf_outpath( obj["out_path"].asString() );
																																			//cout << "\nconf_outpath = " << conf_outpath ;
	if (conf_outpath.empty()  ) {
		out_path = out_path.parent_path().parent_path();																					// move "out_path" up two levels in the directory tree.
		out_path += conf_outpath;
																																			//cout << "  conf_outpath.empty()==true" ;
	}else {out_path = conf_outpath;}
	out_path += "/output/";
																																			if(verbosity>-2) cout << "\n\n  out_path = " << out_path << "\n\n" << flush;

	if(boost::filesystem::create_directory(out_path)) { if(verbosity>-2) std::cerr<< "Directory Created: "<<out_path<<std::endl;}
	else{ 												if(verbosity>-2) std::cerr<< "Output directory previously created: "<<out_path<<std::endl;}

	out_path +=  out_dir;																													if(verbosity>local_verbosity_threshold) cout <<"Creating output sub-directories: "<< out_path <<std::endl;
	boost::filesystem::create_directory(out_path);
	out_path += "/";																														if(verbosity>local_verbosity_threshold) cout << "\n createFolders_chk 1\n" << flush;

	boost::filesystem::path temp_path = out_path;																							// Vector of device buffer names
																																			// imgmem[2],  gxmem[2], gymem[2], g1mem[2],  k_map_mem[2], SE3_map_mem[2], dist_map_mem[2];
	/*
		 "imgmem",						mm_size_bytes_C4,  		.HSV 3chan image. Also receives HSV_grad (2chan per image, 4 images per map) from buildDepthCostVol
		 "imgmem_blurred",				mm_size_bytes_C4,  		.HSV 3chan, before mipmap
		 "keyframe_imgmem",				2*mm_size_bytes_C4,		.HSV_grad (2chan per image, 4 images per map)

		 "gxmem",						mm_size_bytes_C4, 		.rgba 4chan
		 "gymem",						.""
		 "g1mem",						.""
		 "keyframe_g1mem",				.""

		 "SE3_grad_map_mem",			mm_size_bytes_C1*6*8	.rgba 4chan, 6 gradient maps,
		 "keyframe_SE3_grad_map_mem",	.""

		 "SE3_map_mem",					mm_size_bytes_C1*12		.Green-Blue,  rgba 4chan
		 "SE3_incr_map_mem",			mm_size_bytes_C1*24		.4chan rgba, alpha continuous scale ############## broken,
		 "SE3_rho_map_mem",				mm_size_bytes_C4,  		.4chan rgba, alpha continuous scale ############## broken,

		 "SO3_incr_map_mem",			uses SE3 buffer			.4chan rgba, alpha continuous scale ############## broken,
		 "SO3_rho_map_mem",				""						.4chan rgba, alpha continuous scale ############## broken,

		 "basemem", 					image_size_bytes,  		.3chan rgb
		 "keyframe_basemem",			## no buffer allocated?  currently not used ?

		 "depth_mem",					mm_size_bytes_C1		.1chan?		// used in RunCL::load_GT_depth, convert_depth, precomp_param_maps, transform_depthmap, & Dynamic_slam::initialize_keyframe_from_tracking.
		 "keyframe_depth_mem",			mm_size_bytes_C1		.1chan		// used by tracking
		 "key_frame_depth_map_src",		mm_size_bytes_C1	 	.""			// used in RunCL::initializeDepthCostVol( cl_mem key_frame_depth_map_src)

		 "depth_GT",					mm_size_bytes_C1,		.1chan, before mipmap
		 "dmem",						mm_size_bytes_C1,		.1chan
		 "amem", 						.""

		 "lomem",						mm_size_bytes_C1,		.1chan
		 "himem",						.""

		 "qmem",						2 * mm_size_bytes_C1, 	.A_count, Q_count
		 "qmem2",						.""

		 "cdatabuf",					mm_vol_size_bytes, 		1chan, 64layer volume
		 "cdatabuf_8chan",				mm_vol_size_bytes*8, 	1 folder per vol/8chan/ 64layers

		 "hdatabuf",					1mm_vol_size_bytes, 	chan, 64layer volume

		 "img_sum_buf",					2 * mm_vol_size_bytes,	4chan rgba, only used when debugging. Written to by RunCL::updateDepthCostVol(..)
		 "HSV_grad_mem",				2*mm_size_bytes_C4,  	2chan per image, 4 images per map,

		 "dmem_disparity"				rgb image ?
		 */
	std::vector<std::string> names = {"imgmem", "imgmem_blurred", "keyframe_imgmem", "keyframe_imgmem_HSV_grad", "gxmem", "gymem", "g1mem", "keyframe_g1mem", \
										"SE3_grad_map_mem", "keyframe_SE3_grad_map_mem", \
										"SE3_map_mem", \
										"SE3_weight_map_mem",\
										"SE3_incr_map_mem", "SE3_rho_map_mem", \
										/*"SO3_incr_map_mem", "SO3_rho_map_mem", \*/
										\
										"basemem", "keyframe_basemem", "depth_mem", "keyframe_depth_mem", \
										"key_frame_depth_map_src", "depth_GT", \
										"dmem","amem","lomem","himem","qmem","qmem2","cdatabuf","cdatabuf_8chan","hdatabuf","dbg_databuf","img_sum_buf", \
										"HSV_grad_mem", "dmem_disparity" \
	};
	std::pair<std::string, boost::filesystem::path> tempPair;

	tempPair = {"folder", out_path};																										// Top level output folder, used to std::out file.
	paths.insert(tempPair);

	for (std::string key : names){
		temp_path = out_path;
		temp_path += key;
		tempPair = {key, temp_path};
		paths.insert(tempPair);
		boost::filesystem::create_directory(temp_path);
		temp_path += "/tiff/";									// "/png/";
		boost::filesystem::create_directory(temp_path);

	}
																																			if(verbosity>local_verbosity_threshold) {
																																				cout << "\nRunCL::createFolders() chk1\n";			// print the folder paths
																																				cout << "KEY\tPATH\n";
																																				for (auto itr=paths.begin(); itr!=paths.end(); ++itr) { cout<<"First:["<< itr->first << "]\t:\t Second:"<<itr->second<<"\n"; }
																																				cout<<"\npaths.at(\"basemem\")="<<paths.at("basemem")<<"\n"<<flush;
																																			}
}

void RunCL::ReadOutput(uchar* outmat) {
		ReadOutput(outmat, amem,  (baseImage_width * baseImage_height * sizeof(float)) );
}

void RunCL::ReadOutput(uchar* outmat, cl_mem buf_mem, size_t data_size, size_t offset/*=0*/) {
	int local_verbosity_threshold = verbosity_mp["RunCL::ReadOutput"];
																																			if(verbosity>local_verbosity_threshold) cout << "\nRunCL::ReadOutput chk_0"<<flush;
																																			if(verbosity>local_verbosity_threshold) cout << "\noutmat:"<< outmat <<",  buf_mem:"<<buf_mem <<",  data_size:"<<  data_size <<",  offset:"<<  offset <<","<<flush;
		cl_event readEvt;
		cl_int status;
														//cout<<"\nReadOutput: "<<flush;
														//cout<<"&outmat="<<&outmat<<", buf_mem="<<buf_mem<<", data_size="<<data_size<<", offset="<<offset<<"\n"<<flush;
		status = clEnqueueReadBuffer(dload_queue,			// command_queue
											buf_mem,		// buffer
											CL_FALSE,		// blocking_read
											offset,			// offset
											data_size,		// size
											outmat,			// pointer
											0,				// num_events_in_wait_list
											NULL,			// event_waitlist				needs to know about preceeding events:
											&readEvt);		// event
														if (status != CL_SUCCESS) { cout << "\nclEnqueueReadBuffer(..) status=" << checkerror(status) <<"\n"<<flush; exit_(status);}
														else 																				if(verbosity>local_verbosity_threshold) cout << "\nRunCL::ReadOutput chk_1"<<flush;
		status = clFlush(dload_queue);					if (status != CL_SUCCESS) { cout << "\nclFlush(m_queue) status = " 		<< checkerror(status) <<"\n"<<flush; exit_(status);}
		status = clFinish(dload_queue);					if (status != CL_SUCCESS) { cout << "\nclFinish(m_queue) status = " 	<< checkerror(status) <<"\n"<<flush; exit_(status);}
		clReleaseEvent(readEvt);
																																			if(verbosity>local_verbosity_threshold) cout << "\nRunCL::ReadOutput finish"<<flush;
}

void RunCL::saveCostVols(float max_range){
	int local_verbosity_threshold = verbosity_mp["RunCL::saveCostVols"];

																				if(verbosity>local_verbosity_threshold) cout<<"\nsaveCostVols: Calling DownloadAndSaveVolume";
	stringstream ss;
	ss << "saveCostVols";
	ss << (keyFrameCount*1000 + costvol_frame_num);
	DownloadAndSaveVolume(cdatabuf, 	ss.str(), paths.at("cdatabuf"), 	mm_size_bytes_C1,  mm_Image_size, CV_32FC1,  false  , max_range);
	//DownloadAndSaveVolume(hdatabuf, 	ss.str(), paths.at("hdatabuf"), 	mm_size_bytes_C1,  baseImage_size, CV_32FC1,  false  , max_range);
	//DownloadAndSaveVolume(img_sum_buf, 	ss.str(), paths.at("img_sum_buf"), 	mm_size_bytes_C1,  baseImage_size, CV_32FC1,  false  , max_range);
																				if(verbosity>local_verbosity_threshold) cout <<"\ncostvol_frame_num="<<costvol_frame_num << "\ncalcCostVol chk13_finished\n" << flush;
}

void RunCL::Store_keyframe(){
	int local_verbosity_threshold = verbosity_mp["RunCL::DownloadAndSave"];
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nRunCL::Store_keyframe chk0"<<flush;
	cv::Mat temp(uint_params[MM_ROWS], uint_params[MM_ROWS],  CV_32FC4);
	temp.copyTo(key_frame);
	ReadOutput(key_frame.data, keyframe_imgmem,  image_size_bytes);		// NB need to check this still holds data after the function finishes.
}

void RunCL::Save_vtk(cv::Mat mat, cv::Mat keyframe, boost::filesystem::path folder ){														// NB very slow. To examine depth maps for debugging only.
	int local_verbosity_threshold = verbosity_mp["RunCL::DownloadAndSave"];
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nRunCL::Save_vtk chk0"<<flush;
	// check mat types
	if (mat.type() 		!=CV_32FC1) 		{cout << "\n\nError RunCL::Save_vtk  (mat.type() !=CV_32FC1)  "				<< flush; exit(1);}
	if (keyframe.type() !=CV_32FC4) 		{cout << "\n\nError RunCL::Save_vtk  (keyframe.type() !=CV_32FC4)  keyframe.type() = "<<  checkCVtype(keyframe.type())	<< flush; return;}
	if (mat.total() != keyframe.total() )	{cout << "\n\nError RunCL::Save_vtk  (mat.total() "<<mat.total()<<" != keyframe.total() "<<keyframe.total()<<" )  "	<< flush; return;}

	// generate filename, by inserting folder "/csv", and adding ".csv" suffix
	folder.parent_path() += "/vtp/";

	// make csv folder, if necessary
	if(boost::filesystem::create_directory(folder.parent_path() )) { if(verbosity>-2) std::cerr<< "Directory Created: "<<folder.parent_path()<<std::endl;}

																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nRunCL::Save_vtk chk1"<<flush;
	// open file
	ofstream vtp_file_stream;
	vtp_file_stream.open(folder);

																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nRunCL::Save_vtk chk2"<<flush;
	vtp_file_stream << "<?xml version=\"1.0\"?>\n";
	vtp_file_stream << "<VTKFile type=\"PolyData\" version=\"0.1\" byte_order=\"LittleEndian\" header_type=\"UInt32\" compressor=\"vtkZLibDataCompressor\">\n";
	vtp_file_stream << "<Piece NumberOfPoints=\""<<mat.total()<<"\" NumberOfVerts=\"0\" NumberOfLines=\"0\" NumberOfStrips=\"0\" NumberOfPolys=\"0\">\n";
	vtp_file_stream << "<PointData>\n";
	vtp_file_stream << "<DataArray type=\"UInt32\" Name=\"FCLR\" NumberOfComponents=\"4\" format=\"ascii\" \">\n"; // RangeMin=\"0\" RangeMax=\""<<4278203136<<"
	for(int col=0; col<mat.cols ; col ++ ){
		for(int row=0; row<mat.rows ; row ++ ){
			cv::Vec4f clr =  keyframe.at<cv::Vec4f>(col,row);
			vtp_file_stream << clr[0] <<" "<< clr[1]  <<" "<<clr[2]  <<" "<<clr[3] <<"\n";
		}
	}

																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nRunCL::Save_vtk chk3"<<flush;
	vtp_file_stream << "</DataArray>\n";
	vtp_file_stream << "</PointData>\n";
	vtp_file_stream << "<Points>\n";
	vtp_file_stream << "<DataArray type=\"Float32\" Name=\"Points\" NumberOfComponents=\"3\" format=\"ascii\" \n"; // RangeMin=\"0\" RangeMax=\""<<30<<"\">
	for(int col=0; col<mat.cols ; col ++ ){
		for(int row=0; row<mat.rows ; row ++ ){
			float depth =  mat.at<float>(col,row);
			if (depth>0 && isfinite(depth) ) depth = 1/depth;
			vtp_file_stream << col <<" "<< row  <<" "<< depth <<"\n";
		}
	}
	vtp_file_stream << "</DataArray>\n";
	vtp_file_stream << "</Points>\n";
	vtp_file_stream << "</Piece>\n";
	vtp_file_stream << "</PolyData>\n";
	vtp_file_stream << "</VTKFile>\n";
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nRunCL::Save_vtk chk4"<<flush;
	// close file
	vtp_file_stream.close();
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nRunCL::Save_vtk finished"<<flush;
}

/*
// void CostVol::writePointCloud(cv::Mat depthMap)
// {
// 	cv::Mat tempImage = 255 * baseImage.reshape(0, rows*cols);
//
// 	cv::Mat pointCloud(depthMap.size(), CV_32FC3);
// 	if(verbosity>0) cout <<"\na)pointCloud.size = " << pointCloud.size() << "\n"<<flush;
//
// 	for (int u=0; u<rows ; u++){
// 		for (int v=0; v<cols ; v++){
// 			cv::Vec4f homogeneousPoint = inv_K * cv::Vec4f( u, v, 1, depthMap.at<float>(u,v) );
// 			pointCloud.at<Vec3f>(u,v) = cv::Vec3f(homogeneousPoint[0], homogeneousPoint[1], homogeneousPoint[2]) /homogeneousPoint[3];
// 		}
// 	}
// 	if(verbosity>0) cv::imshow("pointCloud depthmap", pointCloud );
// 	pointCloud = pointCloud.reshape(0, rows*cols);
//
// 	if(verbosity>0) cout << "\nb)pointCloud.size() = " << pointCloud.size() << "\tinv_K.size() = ‘class cv::Matx<float, 4, 4>’\n"<<flush;
//
// 	boost::filesystem::path folder_results  =  cvrc.paths.at("amem");
// 	stringstream ss;
// 	ss << folder_results.parent_path().string() << "/depthmap.pts"  ;
// 	if(verbosity>0) cout << "\nDepthmap file : " << ss.str() << "\n" << flush;
// 	char buf[256];
//     sprintf ( buf, "%s", ss.str().data() ); // /depthmap.xyz   folder_results.parent_path().string()
// 	if(verbosity>0) cout << "buf"<< buf << "\n" << flush;
// 	FILE* fp = fopen ( buf, "w" );
// 	if (fp == NULL) {
//         cout << "\nvoid CostVol::writePointCloud(cv::Mat depthMap)  Could not open file "<< fp <<"\n"<< std::flush;
//         assert(0);
//     }
//     fprintf(fp, "%u\n", rows*cols);
// 	for (int i=0; i<rows*cols; i++){		// (x,y,z) coordinates,  "intensity"{0-255},  "color" (rgb){0-255}.
// 		fprintf(fp, "%f %f %f %i %i %i %i\n", pointCloud.at<Vec3f>(i)[0], pointCloud.at<Vec3f>(i)[1], pointCloud.at<Vec3f>(i)[2], 100, (int)(0.5+tempImage.at<Vec3f>(i)[2]), (int)(0.5+tempImage.at<Vec3f>(i)[1]), (int)(0.5+tempImage.at<Vec3f>(i)[1]) );
// 	}
// 	fclose ( fp );
//     fflush ( fp );
// 	if(verbosity>0) cout << "\nwritePointCloud(..) finished\n" << flush;
// }
*/





void RunCL::DownloadAndSave(cl_mem buffer, std::string count, boost::filesystem::path folder_tiff, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range ){
	int local_verbosity_threshold = verbosity_mp["RunCL::DownloadAndSave"];// 1;
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave chk0"<<flush;
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave filename = ["<<folder_tiff.filename().string()<<"] "<<flush;

		cv::Mat temp_mat = cv::Mat::zeros (size_mat, type_mat);																				// (int rows, int cols, int type)
		ReadOutput(temp_mat.data, buffer,  image_size_bytes); 																				// NB contains elements of type_mat, (CV_32FC1 for most buffers)
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave chk1 finished ReadOutput\n\n"<<flush;
		if (temp_mat.type() == CV_16FC1)	temp_mat.convertTo(temp_mat, CV_32FC1);															// NB conversion to FP32 req for cv::sum(..).
		cv::Scalar sum = cv::sum(temp_mat);																									// NB always returns a 4 element vector.
																																		//	if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave chk1.1 finished ReadOutput\n\n"<<flush;
		double minVal=1, maxVal=1;
		cv::Point minLoc={0,0}, maxLoc{0,0};
		if (temp_mat.channels()==1) { cv::minMaxLoc(temp_mat, &minVal, &maxVal, &minLoc, &maxLoc); }
																																		//	if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave chk1.2 finished ReadOutput\n\n"<<flush;
		string type_string = checkCVtype(type_mat);
		stringstream ss;
		stringstream png_ss;
		std::string  date_time_str = date_time_string();

		ss << "/" << folder_tiff.filename().string() << "_" << count <<"_sum"<<sum<<"type_"<<type_string<<"min"<<minVal<<"max"<<maxVal<<"maxRange"<<max_range;
		png_ss << "/" << folder_tiff.filename().string() << "_" << count << date_time_str;
																																		//	if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave chk1.3 finished ReadOutput\n\n"<<flush;
		boost::filesystem::path folder_png = folder_tiff;
																																		//	if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave chk1.4 finished ReadOutput\n\n"<<flush;
		folder_tiff += "/tiff/";
		folder_tiff += ss.str();
		folder_tiff += ".tiff";
		//folder_png  += "/png/";
																																		//	if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave chk1.5 finished ReadOutput\n\n"<<flush;
		folder_png  += png_ss.str();
		folder_png  += ".png";
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave chk2 filename = ["<<ss.str()<<"]"<<flush;
		cv::Mat outMat;
		if (type_mat != CV_32FC1 && type_mat != CV_16FC1 ) {
			cout << "\n\n## Error  (type_mat != CV_32FC1 or CV_16FC1) ##\n\n" << flush;
			return;
		}																																//	if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave chk3 filename = ["<<ss.str()<<"]"<<flush;
		if (max_range == 0){ temp_mat /= maxVal;}																							// Squash/stretch & shift to 0.0-1.0 range
		else if (max_range <0.0){
			temp_mat /=(-2*max_range);
			temp_mat +=0.5;
		}else{ temp_mat /=max_range;}
																																		//	if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave chk4 filename = ["<<ss.str()<<"]"<<flush;
		if(tiff==true) cv::imwrite(folder_tiff.string(), temp_mat );
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave chk5 filename = ["<<ss.str()<<"]"<<flush;
		if(vtp==true) Save_vtk(temp_mat, key_frame, folder_png );

		temp_mat *= 256*256;
		temp_mat.convertTo(outMat, CV_16UC1);
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave chk6 filename = ["<<ss.str()<<"]"<<flush;

		if(png==true) cv::imwrite(folder_png.string(), outMat );
																																			//if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave chk7 filename = ["<<ss.str()<<"],  show="<<show<<flush;

		if(show==true) {cv::imshow( ss.str(), outMat );}
																																			//if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave chk finished filename = ["<<ss.str()<<"]"<<flush;
		return;
}

void RunCL::DownloadAndSave_2Channel_volume(cl_mem buffer, std::string count, boost::filesystem::path folder_tiff, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range, uint vol_layers ){
	int local_verbosity_threshold = verbosity_mp["RunCL::DownloadAndSave_2Channel_volume"];// 1;
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave_2Channel_volume() vol_layers="<<vol_layers<<", max_range="<<max_range<<", folder = ["<<folder_tiff.filename().string()<<"] "<<flush;
	if (type_mat != CV_32FC2){cout <<"Error (type_mat != CV_32FC2)"<<flush; return;}

	for (uint layer=0; layer<vol_layers; layer++  ) {

		uint offset = layer * image_size_bytes;

		cv::Mat temp_mat = cv::Mat::zeros (size_mat, type_mat);																				// (int rows, int cols, int type)
		ReadOutput(temp_mat.data, buffer,  image_size_bytes, offset); 																		// NB contains elements of type_mat, (CV_32FC1 for most buffers)
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave_2Channel_volume()_Chk_1, layer="<<layer<<flush;
		cv::Mat channels[4];
		split(temp_mat, channels);																											// Split u and v (col and row) channels.
		cv::Scalar sum_u = cv::sum(channels[0]);
		cv::Scalar sum_v = cv::sum(channels[1]);
		//cv::Scalar sum_w = cv::sum(channels[2]);
		channels[2] = cv::Mat::zeros( channels[1].size(), channels[1].type() );
		channels[3] = cv::Mat::ones( channels[1].size(), channels[1].type() );

																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave_2Channel_volume()_Chk_2"<<flush;
		cv::Mat temp_mat_u, temp_mat_v;
		cv::Mat channels_u[4] = {channels[0],channels[0]*(-1),channels[2],channels[3] };													// Compose BGR 3channel for each of u & v.
		cv::Mat channels_v[4] = {channels[1],channels[1]*(-1),channels[2],channels[3] };													// NB Blue = +ve, , Green = -ve , Red not used.
		merge(channels_u,4,temp_mat_u);																										// Origin is top right corner of the image.
		merge(channels_v,4,temp_mat_v);
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave_2Channel_volume()_Chk_3"<<flush;
		double minVal_u=1, maxVal_u=1,  minVal_v=1, maxVal_v=1;
		cv::Point minLoc_u={0,0}, maxLoc_u{0,0}, minLoc_v={0,0}, maxLoc_v{0,0};
		cv::minMaxLoc(channels[0], &minVal_u, &maxVal_u, &minLoc_u, &maxLoc_u);
		cv::minMaxLoc(channels[1], &minVal_v, &maxVal_v, &minLoc_v, &maxLoc_v);
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave_2Channel_volume()_Chk_4"<<flush;
		string type_string = checkCVtype(type_mat);
		stringstream ss_u, ss_v;
		stringstream png_ss_u, png_ss_v;
		std::string  date_time_str = date_time_string();

		ss_u << "/" << folder_tiff.filename().string() << "layer_"<<layer<<"_U_" << count <<"_sum"<<sum_u<<"_type_"<<type_string<<"min"<<minVal_u<<"_max"<<maxVal_u<<"_maxRange"<<max_range;
		ss_v << "/" << folder_tiff.filename().string() << "layer_"<<layer<<"_V_" << count <<"_sum"<<sum_v<<"_type_"<<type_string<<"min"<<minVal_v<<"_max"<<maxVal_u<<"_maxRange"<<max_range;

		png_ss_u << "/" << folder_tiff.filename().string() << "layer_"<<layer<<"_U_" << count << date_time_str;
		png_ss_v << "/" << folder_tiff.filename().string() << "layer_"<<layer<<"_V_" << count << date_time_str;

		boost::filesystem::path folder_png_u = folder_tiff, folder_png_v = folder_tiff;
		//folder_tiff += "/tiff/";
		boost::filesystem::path folder_tiff_u = folder_tiff, folder_tiff_v = folder_tiff;
		folder_tiff_u += "/tiff/";
		folder_tiff_u += ss_u.str();
		folder_tiff_u += ".tiff";

		folder_tiff_v += "/tiff/";
		folder_tiff_v += ss_v.str();
		folder_tiff_v += ".tiff";

		//folder_png_u  += "/png/";
		folder_png_u  += png_ss_u.str();
		folder_png_u  += ".png";

		//folder_png_v  += "/png/";
		folder_png_v  += png_ss_v.str();
		folder_png_v  += ".png";
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave_2Channel_volume(), max_range="<<max_range<<",   filepath = ["<<folder_png_u.string()<<" ,\t "<<folder_png_v.string()<<"]";
		cv::Mat outMat_u, outMat_v;
		if (type_mat != CV_32FC2 && type_mat != CV_16FC2 ) {
			cout << "\n\n## Error  (type_mat != CV_32FC2 or CV_16FC2) ##\n\n" << flush;
			return;
		}
		if (max_range == 0){ temp_mat_u /= maxVal_u;  temp_mat_v /= maxVal_v; }																// Squash/stretch & shift to 0.0-1.0 range
		else if (max_range <0.0){
			temp_mat_u /=(-2*max_range);
			temp_mat_v /=(-2*max_range);
			temp_mat_u +=0.5;
			temp_mat_v +=0.5;
		}else{
			temp_mat_u /=max_range;
			temp_mat_v /=max_range;
		}
		if(tiff==true){
			cv::imwrite(folder_tiff_u.string(), temp_mat_u );
			cv::imwrite(folder_tiff_v.string(), temp_mat_v );
		}
		temp_mat_u *= 256*256 *5;																											// NB This is mosty for SE3_map_mem, which is dark due to small increment of each DoF.
		temp_mat_v *= 256*256 *5;
		temp_mat_u.convertTo(outMat_u, CV_16UC4);
		temp_mat_v.convertTo(outMat_v, CV_16UC4);
		if(png==true){
			cv::imwrite(folder_png_u.string(), outMat_u );
			cv::imwrite(folder_png_v.string(), outMat_v );
		}
		if(show){
			cv::imshow( ss_u.str(), outMat_u );
			cv::imshow( ss_v.str(), outMat_v );
		}
	}
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave_2Channel_volume()_finished"<<flush;
}

void RunCL::DownloadAndSave_3Channel(cl_mem buffer, std::string count, boost::filesystem::path folder_tiff, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, cv::Mat *bufImg, float max_range /*=1*/, uint offset /*=0*/, bool exception_tiff /*=false*/){
	int local_verbosity_threshold = verbosity_mp["RunCL::DownloadAndSave_3Channel"];// 2;																										// bufImg will hold a pointer to the version written to .png
	bool old_tiff = tiff;
	if (exception_tiff == true) tiff = exception_tiff;
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave_3Channel_Chk_0    filename = ["<<folder_tiff.filename()
																																				<<"] folder="<<folder_tiff<<", image_size_bytes="<<image_size_bytes<<", size_mat="<<size_mat
																																				<<", type_mat="<<type_mat<<" : "<<checkCVtype(type_mat)<<"\t"<<flush;
		cv::Mat temp_mat, temp_mat2;

		if (type_mat == CV_16FC3)	{
			temp_mat2 = cv::Mat::zeros (size_mat, CV_16FC3);																				//cout << "\nReading CV_16FC3. size_mat="<< size_mat<<",   temp_mat2.total()*temp_mat2.elemSize()="<< temp_mat2.total()*temp_mat2.elemSize() << flush;
			ReadOutput(temp_mat2.data, buffer,  temp_mat2.total()*temp_mat2.elemSize(),   offset );  										// baseImage.total() * baseImage.elemSize()
																																			// void ReadOutput(   uchar* outmat,   cl_mem buf_mem,   size_t data_size,   size_t offset=0)
			temp_mat = cv::Mat::zeros (size_mat, CV_32FC3);
			temp_mat2.convertTo(temp_mat, CV_32FC3);																						// NB conversion to FP32 req for cv::sum(..).
		} else {
			temp_mat = cv::Mat::zeros (size_mat, type_mat);
			ReadOutput(temp_mat.data, buffer,  image_size_bytes,   offset);
		}
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave_3Channel_Chk_1, "<<flush;
		cv::Scalar 	sum = cv::sum(temp_mat);																								// NB always returns a 4 element vector.
		string 		type_string=checkCVtype(type_mat);
		double 		minVal[3]={1,1,1}, 					maxVal[3]={0,0,0};
		cv::Point 	minLoc[3]={{0,0},{0,0},{0,0}}, 		maxLoc[3]={{0,0},{0,0},{0,0}};
		vector<cv::Mat> spl;
		split(temp_mat, spl);																												// process - extract only the correct channel
		double max = 0;
		for (int i =0; i < 3; ++i){
			cv::minMaxLoc(spl[i], &minVal[i], &maxVal[i], &minLoc[i], &maxLoc[i]);
			if (maxVal[i] > max) max = maxVal[i];
		}
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave_3Channel_Chk_2, "<<flush;
		stringstream ss;
		stringstream png_ss;
		std::string  date_time_str = date_time_string();

		ss<<"/"<<folder_tiff.filename().string()<<"_"<<count<<"_sum"<<sum<<"type_"<<type_string<<"min("<<minVal[0]<<","<<minVal[1]<<","<<minVal[2]<<")_max("<<maxVal[0]<<","<<maxVal[1]<<","<<maxVal[2]<<")";
		png_ss<< "/" << folder_tiff.filename().string() << "_" << count << date_time_str;
		if(show){
			cv::Mat temp;
			temp_mat.convertTo(temp, CV_8U, 256);																								// NB need CV_U8 for imshow(..)
			cv::namedWindow( ss.str() , 0 );
			cv::imshow( ss.str(), temp);
			cv::waitKey(-1);
			cout << "\nDownloadAndSave_3Channel:  temp.size()= "<< temp.size() << ", temp_mat.size()= "<< temp_mat.size() << ", temp_mat.type()= "<< temp_mat.type() <<"  "<< checkCVtype(temp_mat.type()) << flush;
			destroyWindow( ss.str() );
		}
		boost::filesystem::path folder_png = folder_tiff;
		//folder_png  += "/png/";
		folder_png  += png_ss.str();
		folder_png  += ".png";

		folder_tiff += "/tiff/";
		folder_tiff += ss.str();
		folder_tiff += ".tiff";

		if (max_range == 0){
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave_3Channel_Chk_2.1, (max_range == 0)    spl[0] /= "<<maxVal[0]<<";  spl[1] /= "<<maxVal[1]<<";  spl[2] /= "<<maxVal[2]<<";"<<flush;
			spl[0] /= maxVal[0];  spl[1] /= maxVal[1];  spl[2] /= maxVal[2];
			//spl[3] = cv::Mat::ones (size_mat, CV_32FC1);																					// set alpha=1
			cv::merge(spl, temp_mat);
		}	// Squash/stretch & shift to 0.0-1.0 range
		else if (max_range <0.0){																											if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave_3Channel_Chk_2.2, (max_range <0.0)    squeeze and shift to 0.0-1.0 "<<flush;
			spl[0] /=(-2*max_range);  spl[1] /=(-2*max_range);  spl[2] /=(-2*max_range);
			spl[0] +=0.5;  spl[1] +=0.5;  spl[2] +=0.5;
			cv::merge(spl, temp_mat);
		}else{ 																																if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave_3Channel_Chk_2.3, (max_range > 0)     temp_mat /=max_range;"<<flush;
			temp_mat /=max_range;
		}
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave_3Channel_Chk_3, "<<flush;
		cv::Mat outMat;
		if ((type_mat == CV_32FC3) || (type_mat == CV_32FC4)){
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave_3Channel_Chk_4, "<<flush;
			if(tiff==true) cv::imwrite(folder_tiff.string(), temp_mat );
			temp_mat *=256;
			temp_mat.convertTo(outMat, CV_8U);

			if (type_mat == CV_32FC4){
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave_3Channel_Chk_5, "<<flush;
				std::vector<cv::Mat> matChannels;
				cv::split(outMat, matChannels);
				//matChannels.at(3)=255;																									// set alpha=1
				cv::merge(matChannels, outMat);
			}
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave_3Channel_Chk_6,  folder_png.string()="<< folder_png.string() <<flush;
			if(png==true)  cv::imwrite(folder_png.string(), (outMat) );																					// Has "Grayscale 16-bit gamma integer"
			outMat.copyTo(*bufImg);
		}else if (type_mat == CV_8UC3){
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave_3Channel_Chk_7, "<<flush;
			if(tiff==true) cv::imwrite(folder_tiff.string(), temp_mat );
			if(png==true)  cv::imwrite(folder_png.string(),  temp_mat );
			temp_mat.copyTo(*bufImg);
		}else if (type_mat == CV_16FC3) {																									// This demonstrates that <cv::float16_t> != <cl_half> and the read/write up/download of these types needs more debugging. NB Cannot use <cv::float16_t>  to prepare  <cl_half> data to the GPU.
			/*
			//cout << "\n Writing CV_16FC3 to .tiff & .png .\n"<< flush;
			//cout << "\n temp_mat2.at<cv::float16_t>(101,100-105) = " << temp_mat2.at<cv::float16_t>(101,100) << "," << temp_mat2.at<cv::float16_t>(101,101) << ","<< temp_mat2.at<cv::float16_t>(101,102) << ","<< temp_mat2.at<cv::float16_t>(101,103) << ","<< temp_mat2.at<cv::float16_t>(101,104) << ","<< temp_mat2.at<cv::float16_t>(101,105) << ","<< flush;
			//cout << "\n temp_mat2.at<cl_half>(101,100-105) = " << temp_mat2.at<cl_half>(101,100) << "," << temp_mat2.at<cl_half>(101,101) << ","<< temp_mat2.at<cl_half>(101,102) << ","<< temp_mat2.at<cl_half>(101,103) << ","<< temp_mat2.at<cl_half>(101,104) << ","<< temp_mat2.at<cl_half>(101,105) << ","<< flush;
			//cout << "\n temp_mat2.at<cl_half>(101,100) x,y,z,w,s0,s3 = " << temp_mat2.at<cl_half3>(101,100).x << "," << temp_mat2.at<cl_half3>(101,100).y << ","<< temp_mat2.at<cl_half3>(101,100).z << ","<< temp_mat2.at<cl_half3>(101,100).w << ","<< temp_mat2.at<cl_half3>(101,100).s0 << ","<< temp_mat2.at<cl_half3>(101,100).s3 << ","<< flush;
			*/
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave_3Channel_Chk_8, "<<flush;
			temp_mat2 *=256;
			if(tiff==true) cv::imwrite(folder_tiff.string(), temp_mat2 );

			temp_mat2.convertTo(outMat, CV_8UC3);

			if(png==true) cv::imwrite(folder_png.string(), (outMat) );
			outMat.copyTo(*bufImg);
		}else {cout << "\n\nError RunCL::DownloadAndSave_3Channel(..)  needs new code for "<<checkCVtype(type_mat)<<endl<<flush; exit(0);}
																																			//if(verbosity>local_verbosity_threshold) cout << "\nDownloadAndSave_3Channel  bufImg->type() = "<<bufImg->type()<<"\t "<< checkCVtype(bufImg->type())  <<" \n" << flush;
	tiff = old_tiff;
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave_3Channel_Chk_9, finished "<<flush;
}

void RunCL::DownloadAndSave_3Channel_volume(cl_mem buffer, std::string count, boost::filesystem::path folder, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range, uint vol_layers,  bool exception_tiff /*=false*/, float iter, bool display){
	int local_verbosity_threshold = verbosity_mp["RunCL::DownloadAndSave_3Channel_volume"];//2;
																																			if(verbosity> local_verbosity_threshold) {
																																				cout<<"\n\nDownloadAndSave_3Channel_volume_chk_0   \tcount=\""<<count<<"\", \tfilename = ["<<folder.filename().string()<<"]"<<flush;
																																				cout<<"\t folder="<<folder.string()<<",\t image_size_bytes="<<image_size_bytes<<",\t size_mat="<<size_mat<<",\t type_mat="<<type_mat<<
																																				"\t show="<<show<<"\t max_range="<<max_range<<"\t vol_layers="<<vol_layers<<"\t exception_tiff="<<exception_tiff<<"\t iter="<<iter<<"\t display="<<display<<flush;
																																			}
	cv::Mat bufImg;
	int row_of_images = 0;  if ( vol_layers > 1) row_of_images=1;
	for (uint i=0; i<vol_layers; i++) {
		stringstream ss;	ss << count << i;
		DownloadAndSave_3Channel(buffer, ss.str(), folder, image_size_bytes, size_mat, type_mat, show, &bufImg, max_range, i*image_size_bytes, exception_tiff);
																																			if(verbosity> local_verbosity_threshold) { cout << "\n\nDownloadAndSave_3Channel_volume_chk_1 (display==true)   bufImg.type() = "<<bufImg.type()
																																				<<"\t "<< checkCVtype(bufImg.type()) <<"\t row_of_images+i="<<row_of_images+i<<"\t  iter="<<iter<<"   \n" << flush;}
		if (display){
			cv::namedWindow( "RunCL::DownloadAndSave_3Channel_volume: bufImg" , 0 );
			cv::imshow( "RunCL::DownloadAndSave_3Channel_volume: bufImg" , bufImg  );
			cv::waitKey(-1);
			destroyWindow( "RunCL::DownloadAndSave_3Channel_volume: bufImg" );
		}																																	if(verbosity> local_verbosity_threshold) { cout << "\n\nDownloadAndSave_3Channel_volume_chk_2"<<flush;}
		writeToResultsMat(&bufImg , iter , row_of_images+i );																				// Add patch from bufImg to resultsMat  TODO this is a bad idea, tangled code.
																																			// DownloadAndSave_3Channel_volume(..) is called for several differnt buffers. !
																																			// Onlly valid when called by RunCL::tracking_result
	}
																																			if(verbosity> local_verbosity_threshold){cout << "\nDownloadAndSave_3Channel_volume_chk_3  finished" << flush;}
}

void RunCL::PrepareResults_3Channel(cl_mem buffer, size_t image_size_bytes, cv::Size size_mat, int type_mat, cv::Mat *bufImg, float max_range /*=1*/, uint offset /*=0*/ ){
	int local_verbosity_threshold = verbosity_mp["RunCL::PrepareResults_3Channel"];// 2;																										// bufImg will hold a pointer to the version written to .png
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nPrepareResults_3Channel_Chk_0, image_size_bytes="<<image_size_bytes<<", size_mat="<<size_mat<<", type_mat="<<type_mat<<" : "<<checkCVtype(type_mat)<<"\t"<<flush;
		cv::Mat temp_mat, temp_mat2;
		temp_mat = cv::Mat::zeros (size_mat, type_mat);
		ReadOutput(temp_mat.data, buffer,  image_size_bytes,   offset);
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nPrepareResults_3Channel_Chk_1, "<<flush;
		cv::Scalar 	sum = cv::sum(temp_mat);																								// NB always returns a 4 element vector.

		double 		minVal[3]={1,1,1}, 					maxVal[3]={0,0,0};
		cv::Point 	minLoc[3]={{0,0},{0,0},{0,0}}, 		maxLoc[3]={{0,0},{0,0},{0,0}};
		vector<cv::Mat> spl;
		split(temp_mat, spl);																												// process - extract only the correct channel
		double max = 0;
		for (int i =0; i < 3; ++i){
			cv::minMaxLoc(spl[i], &minVal[i], &maxVal[i], &minLoc[i], &maxLoc[i]);
			if (maxVal[i] > max) max = maxVal[i];
		}

		if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave_3Channel_Chk_2, "<<flush;
		if (max_range <0.0){																												if(verbosity>local_verbosity_threshold) cout<<"\n\nPrepareResults_3Channel_Chk_2.2, (max_range <0.0)    squeeze and shift to 0.0-1.0 "<<flush;
			spl[0] /=(-2*max_range);  spl[1] /=(-2*max_range);  spl[2] /=(-2*max_range);
			spl[0] +=0.5;  spl[1] +=0.5;  spl[2] +=0.5;
			cv::merge(spl, temp_mat);
		}else{ 																																if(verbosity>local_verbosity_threshold) cout<<"\n\nPrepareResults_3Channel_Chk_2.3, (max_range > 0)     temp_mat /=max_range;"<<flush;
			temp_mat /=max_range;
		}
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nPrepareResults_3Channel_Chk_3, "<<flush;
		cv::Mat outMat;
		if ((type_mat == CV_32FC3) || (type_mat == CV_32FC4)){
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nPrepareResults_3Channel_Chk_4, "<<flush;
			temp_mat *=256;
			temp_mat.convertTo(outMat, CV_8U);

			if (type_mat == CV_32FC4){
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nPrepareResults_3Channel_Chk_5, "<<flush;
				std::vector<cv::Mat> matChannels;
				cv::split(outMat, matChannels);
				cv::merge(matChannels, outMat);
			}
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nPrepareResults_3Channel_Chk_6, "<<flush;
			outMat.copyTo(*bufImg);
		}else if (type_mat == CV_8UC3){
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nPrepareResults_3Channel_Chk_7, "<<flush;
			temp_mat.copyTo(*bufImg);

		}else {cout << "\n\nError RunCL::PrepareResults_3Channel(..)  needs new code for "<<checkCVtype(type_mat)<<endl<<flush; exit(0);}
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nPrepareResults_3Channel_Chk_9, finished "<<flush;
}

void RunCL::PrepareResults_3Channel_volume(cl_mem buffer, size_t image_size_bytes, cv::Size size_mat, int type_mat, float max_range, uint vol_layers,  float iter){
	int local_verbosity_threshold = verbosity_mp["RunCL::PrepareResults_3Channel_volume"];//2;
																																			if(verbosity> local_verbosity_threshold) {
																																				cout<<"\n\nPrepareResults_3Channel_volume_chk_0 "<<flush;
																																				cout<<"\t  image_size_bytes="<<image_size_bytes<<",\t size_mat="<<size_mat<<",\t type_mat="<<type_mat<<
																																				"\t max_range="<<max_range<<"\t vol_layers="<<vol_layers<<"\t iter="<<iter<<flush;
																																			}
	cv::Mat bufImg;
	int row_of_images = 0;  if ( vol_layers > 1) row_of_images=1;
	for (uint i=0; i<vol_layers; i++) {
		PrepareResults_3Channel(buffer, image_size_bytes, size_mat, type_mat, &bufImg, max_range, i*image_size_bytes);

		if (verbosity> verbosity_mp["RunCL::PrepareResults_3Channel_volume"]){
			cv::namedWindow( "RunCL::PrepareResults_3Channel_volume: bufImg" , 0 );
			cv::imshow( "RunCL::PrepareResults_3Channel_volume: bufImg" , bufImg  );
			cv::waitKey(-1);
			destroyWindow( "RunCL::PrepareResults_3Channel_volume: bufImg" );
		}
		writeToResultsMat(&bufImg , iter , row_of_images+i );																				// Add patch from bufImg to resultsMat  TODO this is a bad idea, tangled code.
																																			// PrepareResults_3Channel_volume(..) is called for several differnt buffers. !
																																			// Onlly valid when called by RunCL::tracking_result
	}
																																			if(verbosity> local_verbosity_threshold){cout << "\nDownloadAndSave_3Channel_volume_chk_2  finished" << flush;}
}

void RunCL::DownloadAndSave_6Channel(cl_mem buffer, std::string count, boost::filesystem::path folder_tiff, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range /*=1*/, uint offset /*=0*/){
	int local_verbosity_threshold = verbosity_mp["RunCL::DownloadAndSave_6Channel"];// 1;
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave_6Channel_Chk_0    filename = ["<<folder_tiff.filename()<<"] folder="<<folder_tiff<<", image_size_bytes="<<image_size_bytes<<", size_mat="<<size_mat<<", type_mat="<<type_mat<<" : "<<checkCVtype(type_mat)<<"\t"<<flush;
		cv::Mat temp_mat, temp_mat2;

		if (type_mat == CV_16FC3)	{
			temp_mat2 = cv::Mat::zeros (size_mat.height, 2*size_mat.width, CV_16FC3);														//cout << "\nReading CV_16FC3. size_mat="<< size_mat<<",   temp_mat2.total()*temp_mat2.elemSize()="<< temp_mat2.total()*temp_mat2.elemSize() << flush;
			ReadOutput(temp_mat2.data, buffer,  2*temp_mat2.total()*temp_mat2.elemSize(),   offset );										// baseImage.total() * baseImage.elemSize()    // void ReadOutput(   uchar* outmat,   cl_mem buf_mem,   size_t data_size,   size_t offset=0)
			temp_mat = cv::Mat::zeros (size_mat.height, 2*size_mat.width, CV_32FC3);
			temp_mat2.convertTo(temp_mat, CV_32FC3);																						// NB conversion to FP32 req for cv::sum(..).
		} else {
			temp_mat = cv::Mat::zeros (size_mat.height, 2*size_mat.width, type_mat);
			ReadOutput(temp_mat.data, buffer,  2*image_size_bytes,   2*offset);
		}

		cv::Mat mat_u, mat_v;
		mat_u = cv::Mat::zeros (size_mat, type_mat);
		mat_v = cv::Mat::zeros (size_mat, type_mat);
		for (int i=0; i<mat_u.total(); i++){
			float data[8];
			for (int j=0; j<8; j++){ data[j] = temp_mat.at<float>(i*8  + j) ;}

			for (int j=0; j<4; j++){
				float alpha = ( (data[j] != 0) || (data[j+4] != 0) );
				mat_u.at<float>(i*4  + j) = data[j] ;																						// NB in buffer, alphachan carries
				mat_u.at<float>(i*4  + 3) = alpha;																							// sets alpha=0 when , else alpha=1.
				mat_v.at<float>(i*4  + j) = data[j+4] ;
				mat_v.at<float>(i*4  + 3) = alpha;
			}
		}
																																			if(verbosity>local_verbosity_threshold){
																																				//cv::imshow("mat_u", mat_u);
																																				//cv::imshow("mat_v", mat_v);
																																				//cv::waitKey(-1);
																																				cout << "\nmat_v alpha = ";
																																				for (int px=0; px< (mat_v.rows * mat_v.cols ) ; px += 1000){
																																					cout <<", " << mat_v.at<float>(px*4  + 3);
																																				}cout << flush;
																																			}
		SaveMat(mat_u, type_mat,  folder_tiff,  show,  max_range, "mat_u", count);
		SaveMat(mat_v, type_mat,  folder_tiff,  show,  max_range, "mat_v", count);
}

void RunCL::writeToResultsMat(cv::Mat *bufImg , uint column_of_images , uint row_of_images ){													// writeToResultsMat(buffer , column of images = iteration, row of images );
	int local_verbosity_threshold = verbosity_mp["RunCL::writeToResultsMat"];
	if (obj["sample_se3_incr"]==false) {cout<<"RunCL::writeToResultsMat:  ####  No resultsMat ####   (obj[\"sample_se3_incr\"]==false)  "<<flush; return;}
																																			if(verbosity> local_verbosity_threshold) {cout<<"\n\nRunCL::writeToResultsMat(..)_chk0,"<<flush;
																																				cout<<"\n\t column_of_images="<<column_of_images<<flush;
																																				cout<<"\n\t row_of_images="<<row_of_images<<" "<<flush;
																																			}
	uint reduction 			= obj["sample_layer"].asUInt();
	int patch_rows 			= MipMap[reduction*8 + MiM_READ_ROWS];
	int patch_cols 			= MipMap[reduction*8 + MiM_READ_COLS];
	int row_offset 			= mm_margin * reduction;																						if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::writeToResultsMat(..)_chk3"<<flush;}
	for (int layer = 0; layer < reduction; layer ++)  { row_offset += MipMap[layer*8 + MiM_READ_ROWS]; }									if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::writeToResultsMat(..)_chk4"<<flush;}
	int col_offset 			= mm_margin ;																									if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::writeToResultsMat(..)_chk5"<<flush;}
	int row_offset2			= mm_margin + row_of_images * ( mm_margin + patch_rows );																// paste patch
	int col_offset2			= mm_margin + column_of_images * ( mm_margin + patch_cols );													if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::writeToResultsMat(..)_chk5.1"<<flush;}

																																			if(verbosity>local_verbosity_threshold) cout <<"\nresultsMat.size()="<<resultsMat.size()<<", bufImg->size()="<< bufImg->size()<<flush;
	if ((resultsMat.size().width < patch_cols) || (resultsMat.size().height < patch_rows)) {cout<<"RunCL::writeToResultsMat: ####  No resultsMat ####  ((resultsMat.size().width < patch_cols) || (resultsMat.size().height < patch_rows))  "<<flush; return;}

	for (int col = 0; col < patch_cols ; col++ ) {
		for (int row = 0; row < patch_rows ; row ++) {
																																			if(verbosity>local_verbosity_threshold)cout<<"\nrow="<< row <<",  row_offset2"<< row_offset2 <<",  col"<< col
																																				<<",  col_offset2"<< col_offset2 <<",  patch_cols="<<patch_cols<<",  patch_rows="<<patch_rows<<flush;
			resultsMat.at<Vec4b>(row+row_offset2 , col+col_offset2) = bufImg->at<Vec4b>(row+row_offset , col+col_offset);					// <Vec4b> = <uchar, 4>, ie 4 channel 8 bit img, eg rgba .png .
		}
	}																																		if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::writeToResultsMat(..)_chk finished"<<flush;}
}


void RunCL::DownloadAndSave_HSV_grad(cl_mem buffer, std::string count, boost::filesystem::path folder_tiff, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range /*=1*/, uint offset /*=0*/){
	int local_verbosity_threshold = verbosity_mp["RunCL::DownloadAndSave_HSV_grad"];// 1;
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave_HSV_grad_Chk_0    filename = ["<<folder_tiff.filename()<<"] folder="<<folder_tiff<<", image_size_bytes="<<image_size_bytes<<", size_mat="<<size_mat<<", type_mat="<<type_mat<<" : "<<checkCVtype(type_mat)<<"\t"<<flush;
		cv::Mat temp_mat, temp_mat2;

		if (type_mat != CV_32FC(8))	{cout <<"\nRunCL::DownloadAndSave_HSV_grad(...)  type_mat != CV_32FC(8)\n"<< flush; return;}

		temp_mat = cv::Mat::zeros (size_mat.height, size_mat.width, type_mat);
		ReadOutput(temp_mat.data, buffer,  image_size_bytes,   offset);

		cv::Mat mat_H, mat_SV, mat_Sgrad, mat_Vgrad;

		mat_H 		= cv::Mat::zeros (size_mat, CV_32FC4);
		mat_SV 		= cv::Mat::zeros (size_mat, CV_32FC3);
		mat_Sgrad 	= cv::Mat::zeros (size_mat, CV_32FC3);
		mat_Vgrad 	= cv::Mat::zeros (size_mat, CV_32FC3);

		//uint data_elem_size = 4*sizeof(float);
		for (int i=0; i<mat_H.total(); i++){
			float data[8];
			for (int j=0; j<8; j++){ data[j] = temp_mat.at<float>(i*8  + j) ;}

			mat_H.at<float>(i*4 ) 		= data[0];
			mat_H.at<float>(i*4 +1) 	= data[1];
			mat_H.at<float>(i*4 +3) 	= 1.0;			// alpha

			mat_SV.at<float>(i*3 ) 		= data[2];
			mat_SV.at<float>(i*3 +1) 	= data[3];
			//mat_SV.at<float>(i*3 +3) 	= 1.0;			// alpha

			mat_Sgrad.at<float>(i*3 ) 	= data[4];
			mat_Sgrad.at<float>(i*3 +1) = data[5];
			//mat_Sgrad.at<float>(i*3 +3) = 1.0;		// alpha

			mat_Vgrad.at<float>(i*3 ) 	= data[6];
			mat_Vgrad.at<float>(i*3 +1) = data[7];
			mat_Vgrad.at<float>(i*3 +3) = 1.0;			// alpha
		}
		bool old_tiff = tiff;
		tiff=true;
		SaveMat(mat_H,     CV_32FC4,  folder_tiff,  show,  max_range, "mat_H", count);
		tiff=old_tiff;

		SaveMat(mat_SV,    CV_32FC3,  folder_tiff,  show,  max_range, "mat_SV", count);
		SaveMat(mat_Sgrad, CV_32FC3,  folder_tiff,  show,  max_range, "mat_Sgrad", count);
		SaveMat(mat_Vgrad, CV_32FC3,  folder_tiff,  show,  max_range, "mat_Vgrad", count);
}

void RunCL::SaveMat(cv::Mat temp_mat, int type_mat, boost::filesystem::path folder_tiff, bool show, float max_range, std::string mat_name, std::string count){
	int local_verbosity_threshold = verbosity_mp["RunCL::SaveMat"];// 1;
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nSaveMat_Chk_1, "<<flush;
		cv::Scalar 	sum = cv::sum(temp_mat);																								// NB always returns a 4 element vector.
		string 		type_string=checkCVtype(type_mat);
		double 		minVal[3]={1,1,1}, 					maxVal[3]={0,0,0};
		cv::Point 	minLoc[3]={{0,0},{0,0},{0,0}}, 		maxLoc[3]={{0,0},{0,0},{0,0}};
		vector<cv::Mat> spl;
		split(temp_mat, spl);																												// process - extract only the correct channel
		double max = 0;

		for (int i =0; i < 3; ++i){
																																			if(verbosity>local_verbosity_threshold)
																																				cout << "\n spl[i].empty() = "<<spl[i].empty()<<", spl[i].type() = "<< spl[i].type() << flush; //", cn = "<<cn<<", minIdx = "<< minIdx <<",  maxIdx = "<<maxIdx << flush;
			cv::minMaxLoc(spl[i], &minVal[i], &maxVal[i], &minLoc[i], &maxLoc[i]);
			if (maxVal[i] > max) max = maxVal[i];
		}
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nSaveMat_Chk_2, "<<flush;
		stringstream ss;
		stringstream png_ss;
		ss<<"/"<<folder_tiff.filename().string()<<"_"<<mat_name<<"_"<<count<<"__sum"<<sum<<"type_"<<type_string<<"min("<<minVal[0]<<","<<minVal[1]<<","<<minVal[2]<<")_max("<<maxVal[0]<<","<<maxVal[1]<<","<<maxVal[2]<<")";
		png_ss<< "/" << folder_tiff.filename().string() <<"_"<<mat_name<< "_" << count;
		if(show){
			cv::Mat temp;
			temp_mat.convertTo(temp, CV_8U);																								// NB need CV_U8 for imshow(..)
			cv::imshow( ss.str(), temp);
		}
		boost::filesystem::path folder_png = folder_tiff;
		//folder_png  += "/png";
		folder_png  += png_ss.str();
		folder_png  += ".png";

		folder_tiff += "/tiff/";
		folder_tiff += ss.str();
		folder_tiff += ".tiff";

		if (max_range == 0){ 																												if(verbosity>local_verbosity_threshold) cout<<"\n\nSaveMat_Chk_2.1, (max_range == 0)    spl[0] /= maxVal[0];  spl[1] /= maxVal[1];  spl[2] /= maxVal[2];"<<flush;
			spl[0] /= maxVal[0];  spl[1] /= maxVal[1];  spl[2] /= maxVal[2]; 																// Squash/stretch & shift to 0.0-1.0 range
			cv::merge(spl, temp_mat);
		}
		else if (max_range <0.0){																											if(verbosity>local_verbosity_threshold) cout<<"\n\nSaveMat_Chk_2.2, (max_range <0.0)    squeeze and shift to 0.0-1.0 "<<flush;
			spl[0] /=(-2*max_range);  spl[1] /=(-2*max_range);  spl[2] /=(-2*max_range);
			spl[0] +=0.5;  spl[1] +=0.5;  spl[2] +=0.5;
			cv::merge(spl, temp_mat);
		}else{ 																																if(verbosity>local_verbosity_threshold) cout<<"\n\nSaveMat_Chk_2.3, (max_range > 0)     temp_mat /=max_range;"<<flush;
			temp_mat /=max_range;
		}
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nSaveMat_Chk_3, "<<flush;
		cv::Mat outMat;
		if ((type_mat == CV_32FC3) || (type_mat == CV_32FC4)){																				if(verbosity>local_verbosity_threshold) cout<<"\n\nSaveMat_Chk_4, "<<flush;
			if(tiff==true) cv::imwrite(folder_tiff.string(), temp_mat );

			temp_mat.convertTo(outMat, CV_8U, 255);																							if(verbosity>local_verbosity_threshold) cout<<"\n\nSaveMat_Chk_6,  folder_png.string()="<< folder_png.string() <<flush;
			if(png==true)  cv::imwrite(folder_png.string(), (outMat) );																		// Has "Grayscale 16-bit gamma integer" ?
		}else if (type_mat == CV_8UC3){
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nSaveMat_Chk_7, "<<flush;
			if(tiff==true) cv::imwrite(folder_tiff.string(), temp_mat );
			if(png==true)  cv::imwrite(folder_png.string(),  temp_mat );
		}																																	else {cout << "\n\nError RunCL::SaveMat(..)  needs new code for "<<checkCVtype(type_mat)<<endl<<flush; exit(0);}
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nSaveMat_Chk_9, finished "<<flush;
}

void RunCL::SaveMat_1chan(cv::Mat temp_mat, int type_mat, boost::filesystem::path folder_tiff, bool show, float max_range, std::string mat_name, std::string count){
	int local_verbosity_threshold = verbosity_mp["RunCL::SaveMat_1chan"];// 1;
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nSaveMat_1chan_Chk_1, "<<flush;
		cv::Scalar 	sum = cv::sum(temp_mat);																								// NB always returns a 4 element vector.
		string 		type_string=checkCVtype(type_mat);
		double 		minVal=1, 			maxVal=0;
		cv::Point 	minLoc={0,0}, 		maxLoc={0,0};
		double max = 0;
		cv::minMaxLoc(temp_mat, &minVal, &maxVal, &minLoc, &maxLoc);
		if (maxVal > max) max = maxVal;
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nSaveMat_1chan_Chk_2, "<<flush;
		stringstream ss;
		stringstream png_ss;
		ss<<"/" << mat_name<<"__sum"<<sum<<"type_"<<type_string<<"min("<<minVal<<")_max("<<maxVal<<")";
		png_ss<< "/" << mat_name;
		if(show){
			cv::Mat temp;
			temp_mat.convertTo(temp, CV_8U);																								// NB need CV_U8 for imshow(..)
			cv::imshow( ss.str(), temp);
		}
		boost::filesystem::path folder_png = folder_tiff;
		//folder_png  += "/png";
		folder_png  += png_ss.str();
		folder_png  += ".png";

		folder_tiff += "/tiff/";
		folder_tiff += ss.str();
		folder_tiff += ".tiff";

		if (max_range == 0){ 																												if(verbosity>local_verbosity_threshold) cout<<"\n\nSaveMat_1chan_Chk_2.1, (max_range == 0)    spl[0] /= maxVal[0];  spl[1] /= maxVal[1];  spl[2] /= maxVal[2];"<<flush;
			temp_mat /= maxVal;
		}
		else if (max_range <0.0){																											if(verbosity>local_verbosity_threshold) cout<<"\n\nSaveMat_1chan_Chk_2.2, (max_range <0.0)    squeeze and shift to 0.0-1.0 "<<flush;
			temp_mat /=(-2*max_range);
			temp_mat +=0.5;
		}else{ 																																if(verbosity>local_verbosity_threshold) cout<<"\n\nSaveMat_1chan_Chk_2.3, (max_range > 0)     temp_mat /=max_range;"<<flush;
			temp_mat /=max_range;
		}
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nSaveMat_1chan_Chk_3, "<<flush;
		cv::Mat outMat;
		if (type_mat == CV_32FC1){																											if(verbosity>local_verbosity_threshold) cout<<"\n\nSaveMat_1chan_Chk_4, "<<flush;
			if(tiff==true) cv::imwrite(folder_tiff.string(), temp_mat );

			temp_mat.convertTo(outMat, CV_8U, 255);																							if(verbosity>local_verbosity_threshold) cout<<"\n\nSaveMat_1chan_Chk_6,  folder_png.string()="<< folder_png.string() <<flush;
			if(png==true)  cv::imwrite(folder_png.string(), (outMat) );																		// Has "Grayscale 16-bit gamma integer" ?
		}																																	else {cout << "\n\nError RunCL::SaveMat_1chan(..)  needs new code for "<<checkCVtype(type_mat)<<endl<<flush; exit(0);}

																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nSaveMat_1chan_Chk_7, finished "<<flush;
}

void RunCL::DownloadAndSave_6Channel_volume(cl_mem buffer, std::string count, boost::filesystem::path folder, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range, uint vol_layers ){
	int local_verbosity_threshold = verbosity_mp["RunCL::DownloadAndSave_6Channel_volume"];// 1;
																																			if(verbosity> local_verbosity_threshold) {
																																				cout<<"\n\nDownloadAndSave_6Channel_volume_chk_0   costVolLayers="<<costVolLayers<<", filename = ["<<folder.filename().string()<<"]"<<flush;
																																				cout<<"\n folder="<<folder.string()<<",\t image_size_bytes="<<image_size_bytes<<",\t size_mat="<<size_mat<<",\t type_mat="<<size_mat<<"\t"<<flush;
																																			}
	for (uint i=0; i<vol_layers; i++) {
		stringstream ss;	ss << count <<"_"<< i <<"_";
		DownloadAndSave_6Channel(buffer, ss.str(), folder, image_size_bytes, size_mat, type_mat, show, max_range, i*image_size_bytes);
	}
																																			if(verbosity> local_verbosity_threshold){cout << "DownloadAndSave_6Channel_volume_chk_1  finished" << flush;}
}

void RunCL::DownloadAndSave_8Channel(cl_mem buffer, std::string count, std::map< std::string, boost::filesystem::path > folder_tiff, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range /*=1*/, uint offset /*=0*/){
	int local_verbosity_threshold = verbosity_mp["RunCL::DownloadAndSave_8Channel"];// 1;
																																			if(verbosity>local_verbosity_threshold) {
																																				cout<<"\n\nDownloadAndSave_8Channel_Chk_0    filename = ["<<folder_tiff["0"].filename()<<"] folder="<<folder_tiff["0"]<<", image_size_bytes="<<image_size_bytes<<", size_mat="<<size_mat<<", type_mat="<<type_mat<<" : "<<checkCVtype(type_mat)<<",  offset= "<< offset <<"\t"<<flush;
																																			}
	cv::Mat temp_mat;
	temp_mat = cv::Mat::zeros (size_mat.height, size_mat.width, CV_32FC(8) );
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave_8Channel_Chk_1"
																																				<<"   temp_mat.rows = "			<< temp_mat.rows
																																				<<",  temp_mat.cols = "			<< temp_mat.cols
																																				<<",  temp_mat.elemSize() = "	<< temp_mat.elemSize()
																																				<<",  temp_mat.channels() = "	<< temp_mat.channels()
																																				<<",  size_bytes = "			<< temp_mat.total() * temp_mat.elemSize() * temp_mat.channels()
																																				<<",  8*image_size_bytes = "	<< 8*image_size_bytes
																																				<<",  mm_vol_size_bytes*8 = "	<< mm_vol_size_bytes*8 <<flush;
	ReadOutput(temp_mat.data, buffer,  image_size_bytes,   offset);

	cv::Mat mat[8];
	for (int i=0; i<8; i++){
		mat[i] = cv::Mat::zeros (size_mat, CV_32FC1);				// zero mat[i]
	}
	for (int i=0; i<mat[0].total(); i++){
		for (int j=0; j<8; j++){
			mat[j].at<float>(i) = temp_mat.at<float>(i*8  + j) ;				// copy 8 channel pixel
		}
	}
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave_8Channel_Chk_2";
	std::vector<std::string> names = {"0","1","2","3","4","5","6","7"};

	for (int i=0; i<8; i++){																												// for each channel
		boost::filesystem::path temp_path2 = folder_tiff[names[i]];

		stringstream ss;	ss << "channel" << names[i] << "_" << count ;
		SaveMat_1chan(mat[i], CV_32FC1,  temp_path2,  show,  max_range, ss.str(), count);
																																			if(verbosity>local_verbosity_threshold) cout<<"\n\nDownloadAndSave_8Channel_Chk_3, i = "<< i <<",  temp_path2 = "<<temp_path2<<", ss.str() = "<<ss.str()<<", count = "<<count<< flush;
	}
}

void RunCL::DownloadAndSave_8Channel_volume(cl_mem buffer, std::string count, boost::filesystem::path folder, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range, uint vol_layers ){
	int local_verbosity_threshold = verbosity_mp["RunCL::DownloadAndSave_8Channel_volume"];// 1;
																																			if(verbosity> local_verbosity_threshold) {
																																				cout<<"\n\nDownloadAndSave_8Channel_volume_chk_0   costVolLayers="<<costVolLayers<<", filename = ["<<folder.filename().string()<<"]"<<flush;
																																				cout<<"\n folder="<<folder.string()<<",\t image_size_bytes="<<image_size_bytes<<",\t size_mat="<<size_mat<<",\t type_mat="<<type_mat<<" : "<<checkCVtype(type_mat)<<"\t"<<flush;
																																			}
	boost::filesystem::path temp_path1 = folder;
	temp_path1 += "/";
	temp_path1 += count;
	boost::filesystem::create_directory(temp_path1);																							// make new folder for this cost volume

	std::map< std::string, boost::filesystem::path > channel_paths;
	std::vector<std::string> names = {"0","1","2","3","4","5","6","7"};
	std::pair<std::string, boost::filesystem::path> tempPair;

	for (std::string key : names){
		boost::filesystem::path temp_path2 = temp_path1;
		temp_path2 += "/channel";
		temp_path2 += key;
		tempPair = {key, temp_path2};
		channel_paths.insert(tempPair);
		boost::filesystem::create_directory(temp_path2);																						// make new folders for each channel of this cost volume
		temp_path2 += "/tiff/";
		boost::filesystem::create_directory(temp_path2);
	}

	for (uint i=0; i<vol_layers; i++) {
		stringstream ss;	ss << count <<"__layer_"<< i <<"_";
		DownloadAndSave_8Channel(buffer, ss.str(), channel_paths, image_size_bytes, size_mat, type_mat, show, max_range, i*image_size_bytes);
																																			if(verbosity> local_verbosity_threshold){cout << "DownloadAndSave_3Channel_volume_chk_1  : ss.str() = "<< ss.str()<<",  temp_path1 = "<<temp_path1 << flush;}
	}
																																			if(verbosity> local_verbosity_threshold){cout << "DownloadAndSave_3Channel_volume_chk_2  finished " << flush;}
}

void RunCL::DownloadAndSaveVolume(cl_mem buffer, std::string count, boost::filesystem::path folder, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range, bool exception_tiff /*=false*/){
	int local_verbosity_threshold = verbosity_mp["RunCL::DownloadAndSaveVolume"];// 0;
	bool old_tiff = tiff;
	if (exception_tiff == true) tiff = exception_tiff;
																																			if(verbosity>local_verbosity_threshold) {
																																				cout<<"\n\nDownloadAndSaveVolume, costVolLayers="<<costVolLayers<<", filename = ["<<folder.filename().string()<<"]";
																																				cout<<"\n folder="<<folder.string()<<",\t image_size_bytes="<<image_size_bytes<<",\t size_mat="<<size_mat<<",\t type_mat="<<type_mat<<" : "<<checkCVtype(type_mat)<<"\t"<<flush;
																																			}

	for(int i=0; i<costVolLayers; i++){
																																			if(verbosity>local_verbosity_threshold) cout << "\ncostVolLayers="<<costVolLayers<<", i="<<i<<"\t";
		size_t offset = i * image_size_bytes;
		cv::Mat temp_mat = cv::Mat::zeros (size_mat, type_mat);																				//(int rows, int cols, int type)

		ReadOutput(temp_mat.data, buffer,  image_size_bytes, offset);
																																			if(verbosity>local_verbosity_threshold) cout << "\nRunCL::DownloadAndSaveVolume, ReadOutput completed\n"<<flush;
		if (temp_mat.type() == CV_16FC1)	temp_mat.convertTo(temp_mat, CV_32FC1);															// NB conversion to FP32 req for cv::sum(..).
		cv::Scalar sum = cv::sum(temp_mat);

		double minVal=1, maxVal=1;
		cv::Point minLoc={0,0}, maxLoc{0,0};
		if (type_mat == CV_32FC1) cv::minMaxLoc(temp_mat, &minVal, &maxVal, &minLoc, &maxLoc);

		boost::filesystem::path new_filepath = folder;
		boost::filesystem::path folder_png   = folder;
		new_filepath += "/tiff/";

		string type_string = checkCVtype(type_mat);
		stringstream ss;
		stringstream png_ss;
		ss << "/"<< folder.filename().string() << "_" << count << "_layer"<< i <<"_sum"<<sum<<"type_"<<type_string<< "min"<<minVal<<"max"<<maxVal;
		png_ss << "/"<< folder.filename().string() << "_" << count << "_layer_"<< setw(4) << setfill('0') << i;
		if(show){
			cv::Mat temp;
			temp_mat.convertTo(temp, CV_8U);																								// NB need CV_U8 for imshow(..)
			cv::imshow(ss.str(), temp);
		}
		new_filepath += ss.str();
		new_filepath += ".tiff";
		//folder_png += "/png/";
		folder_png += png_ss.str();
		folder_png += ".png";
																																			if(verbosity>local_verbosity_threshold) cout << "\nnew_filepath.string() = "<<new_filepath.string() <<"\n";
		cv::Mat outMat;

		if (type_mat != CV_32FC1 && type_mat != CV_16FC1 ) {
			cout << "\n\n## Error  (type_mat != CV_32FC1 or CV_16FC1) ##\n\n" << flush;
			return;
		}
		if (max_range == 0){ temp_mat /= maxVal;}																							// Squash/stretch & shift to 0.0-1.0 range
		else if (max_range <0.0){
			temp_mat /=(-2*max_range);
			temp_mat +=0.5;
		}else{ temp_mat /=max_range;}

		if(tiff==true) cv::imwrite(new_filepath.string(), temp_mat );
		temp_mat *= 256*256;
		temp_mat.convertTo(outMat, CV_16UC1);
		if(png==true)  cv::imwrite(folder_png.string(), outMat );
		if(show) cv::imshow( ss.str(), outMat );
	}
	tiff = old_tiff;
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////// # TODO New way: generic RunCL::DownloadAndSave_buffer(...) , and specialized calling functions.

/*
 Need to store 2D array of linear mipmap params

 When reading linearMipMap,
 1) download data
 2) copy data array into new image
    Use mipmap_params to locate data in new Mat.
 3) save png (&tiff)

 Issues
 1) data type
 2) num channels
 3) num maps in volume

 When writing CPU data to MipMap buffer
 1)

 When reading/writing linear Mipmap in kernels
 1) Minimize calculation
        read index = UID + offset
 2) Avoid smearing margins

 Make set of function aliases for each buffer.
 */
/*
//void DownloadAndSave_linear_Mipmap(cl_mem buffer, std::string count, boost::filesystem::path folder_tiff, int type_mat, bool show );

// void RunCL::DownloadAndSave_buffer (
//     cl_mem buffer,
//     std::string count,
//     //boost::filesystem::path folder_tiff,
//     std::map< std::string, boost::filesystem::path > folder_tiff,   // used in DownloadAndSave_8Channel, when called by DownloadAndSave_8Channel_volume
//
//     size_t      image_size_bytes,
//     cv::Size    size_mat,
//
//     int type_mat_out,           / *(data size and channels)* /
//     int num_channels_out,       // channels per image to write to file
//     int num_channels_in,        // channels in the buffer
//     int maps_in_vol,            // layers of depth cost vol,  //uint vol_layers,
//
//     int start_layer,            // of the mipmap
//     int stop_layer,
//
//     float max_range,            // used to scale the values in the png for visualization
//     bool exception_tiff,        // generate tiff, when tiffs are globally turned off
//     bool show                   // display the image(s).
//
//     //uint offset =0,       // passed to ReadOutput(uchar* outmat, cl_mem buf_mem, size_t data_size, size_t offset=0)
//     //cv::Mat temp_mat,           // mat passed to SaveMat or SaveMat_1chan
//     //std::string mat_name        //
//
// ){
//     int type_mat_in = CV_MAKETYPE(CV_32F, num_channels_in);                                 // create temp_mat1
//     cv::Mat tempMat1(size_mat, type_mat_in);
//     float * data1 = (float*)tempMat1.data;
//     ReadOutput(tempMat1.data, buffer, image_size_bytes / *,size_t offset=0 * /);               // download buffer, to *uchar
//
//     std::vector<cv::Mat>    tempMat2_vec;                                                   // create tempMat2_vec
//     std::vector<float*>     data2;
//
//     int num_Mats = num_channels_in/num_channels_out;
//     for (int i=0; i<num_Mats; i++){
//         tempMat2_vec.push_back(cv::Mat::zeros(size_mat, CV_32FC4));
//         data2.push_back( (float*)tempMat2_vec[i].data );
//     }
//
//     for (int i=0; i<tempMat1.total(); i++){                                                 // write temp_mat1 to vector_temp_mat2
//         int a = i * num_channels_in;
//         int b = i * num_channels_out;
//         for (int j=0; j<num_Mats; j++){
//             a += j*num_channels_out;
//             for (int k=0; k<num_channels_out; k++){
//                 data2[j][b+k] = data1[a+k];
//             }
//         }
//     }
//
//     std::vector<cv::Mat>    tempMat3_vec;                                                   // create tempMat3_vec
//     std::vector<float*>     data3;
//     cv::Size size_mipmap_mat(mm_height, mm_width );
//
//     for (int i=0; i<num_Mats; i++){
//         tempMat3_vec.push_back(cv::Mat::zeros(size_mipmap_mat, CV_32FC4));
//         data3.push_back( (float*)tempMat3_vec[i].data );
//     }
//
//     for (int i=0; i<num_Mats; i++){
//         for (int j=start_layer; j<stop_layer; j++){                                             // reconstruct mipmap
//             cv:Size size_mmlayer_mat(   );
//             cv::Mat tempMat4 = cv::Mat::zeros(size_mmlayer_mat, CV_32FC4);
//
//             int offset_2    = ;  // Location of the mipmap layer in each Mat.
//             int offset_3    = ;
//             int rows        = ;
//             int cols        = ;
//             int margin      = ;
//
//             for (int k=0; k<rows; k++){
//                 for (int l=0; l<cols; l++){
//                     tempMat3_vec[i].at<Vec4f>(row,col)      = {data2[i].data[], data2[i].data[], data2[i].data[], data2[i].data[] };
//                 }
//             }
//         }
//     }
//
//     // save png(s)
//
//     // save tiff(s)
//
//     // display image
// }
*/
/*
// void RunCL::DownloadAndSave_... (){
//     DownloadAndSave_linearMipMap (
//     buffer              =,          / * cl_mem * /
//     count               =,          / * std::string * /
//     folder_tiff         =,          / * boost::filesystem::path * /
//     folder_tiff         =,          / * std::map< std::string, boost::filesystem::path > * /    // used in DownloadAndSave_8Channel, when called by DownloadAndSave_8Channel_volume
//
//     image_size_bytes    =,          / * size_t * /
//     size_mat            =,          / * cv::Size * /
//
//     type_mat            =,          / * int * /                                                 // data size and channels
//     num_channels_out    =,          / * int * /                                                 // channels per image to write to file
//     num_channels_in     =,          / * int * /                                                 // channels in the buffer
//     maps_in_vol         =,          / * int * /                                                 // layers of depth cost vol,  //uint vol_layers,
//
//     start_layer         =,          / * int * /                                                 // of the mipmap
//     stop_layer          =,          / * int * /
//
//     //uint offset / * =0 * /,                                                                   // passed to ReadOutput(uchar* outmat, cl_mem buf_mem, size_t data_size, size_t offset=0)
//     max_range           =,          / * float * /                                               // used to scale the values in the png for visualization
//
//     exception_tiff      =,          / * bool * /                                                // generate tiff, when tiffs are globally turned off
//     show                =,          / * bool * /                                                // display the image(s).
//
//     temp_mat            =,          / * cv::Mat * /                                             // mat passed to SaveMat or SaveMat_1chan
//     mat_name            =           / * std::string * /
//     )
// }
*/
/*
 * The old system, all are currently 32bit float, (could be 16bit in future, would require some brand specific code)
 *
    void DownloadAndSave(cl_mem buffer, std::string count, boost::filesystem::path folder, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range );
    depthmap_GT, keyframe_depth_mem, key_frame_depth_map_src, lomem, himem, amem, dmem, qmem, gxmem
# 1Channel: 1 channel_out, 1 channel_in, 1 map.

	void DownloadAndSave_2Channel_volume(cl_mem buffer, std::string count, boost::filesystem::path folder_tiff, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range, uint vol_layers );
	SE3_map_mem
# 2Channel: 3or4 channels_out,

	void DownloadAndSave_3Channel(cl_mem buffer, std::string count, boost::filesystem::path folder_tiff, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range=1, uint offset=0, bool exception_tiff=false );
	basemem, imgmem, imgmem_blurred, gxmem, gymem, g1mem, keyframe_g1mem, dmem_disparity, buffer in DownloadAndSave_3Channel_volume,
# 3Channel: 4 channels_out (inc alpha), 4 channels_in, 1 map

	void DownloadAndSave_3Channel_volume(cl_mem buffer, std::string count, boost::filesystem::path folder, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range, uint vol_layers );
	SE3_incr_map_mem, SE3_rho_map_mem   ## bug in .png output.
# 3Channel_volume: 2 channels_out (+ve & -ve), 3 channels_in,

	void DownloadAndSave_6Channel(cl_mem buffer, std::string count, boost::filesystem::path folder_tiff, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range, uint offset=0);
	buffer in DownloadAndSave_6ChannelVolume(..)

	void DownloadAndSave_6Channel_volume(cl_mem buffer, std::string count, boost::filesystem::path folder, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range, uint vol_layers );
	SE3_grad_map_mem, keyframe_SE3_grad_map_mem

	void DownloadAndSave_HSV_grad(cl_mem buffer, std::string count, boost::filesystem::path folder_tiff, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range, uint offset=0 );
	HSV_grad_mem, keyframe_imgmem,

	void SaveMat(cv::Mat temp_mat, int type_mat, boost::filesystem::path folder_tiff, bool show, float max_range, std::string mat_name, std::string count);
	mat_u & mat_v                          in DownloadAndSave_6Channel(..),
	Mat_H, mat_SV, mat_Sgrad, mat_Vgrad    in DownloadAndSave_HSV_grad(..)
# HSV_grad:         2 channels_out (per image),  8 channels_in, 1 map
# 6Channel_volume:  1 channels_out (per image),  6 channels_in, 1 map


	void DownloadAndSave_8Channel(cl_mem buffer, std::string count, std::map< std::string, boost::filesystem::path > folder_tiff, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range , uint offset );
	buffer in DownloadAndSave_8ChannelVolume(..)

	void DownloadAndSave_8Channel_volume(cl_mem buffer, std::string count, boost::filesystem::path folder, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range, uint vol_layers );
	cdatabuf_8chan

	void SaveMat_1chan(cv::Mat temp_mat, int type_mat, boost::filesystem::path folder_tiff, bool show, float max_range, std::string mat_name, std::string count);
	mat[i] in DownloadAndSave_8Channel(..)
# 8 channels, 1 map


	void DownloadAndSaveVolume(cl_mem buffer, std::string count, boost::filesystem::path folder, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range, bool exception_tiff=false );
	cdatabuf, hdatabuf
# 1 channel, many maps
 *
 */
