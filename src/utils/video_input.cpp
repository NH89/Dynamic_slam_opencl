#include "../Dynamic_slam/Dynamic_slam.hpp"



    // return the filenames of all files that have the specified extension
    // in the specified directory and all subdirectories
void Dynamic_slam::get_all(const fs::path& root , const string& ext, vector<fs::path>& ret ) {
      if (!fs::exists(root))        return;
      if (fs::is_directory(root))   {
        typedef std::set<std::filesystem::path> Files;
        Files files;
        fs::recursive_directory_iterator it0(root);
        fs::recursive_directory_iterator endit0;
        std::copy(it0, endit0, std::inserter(files, files.begin()));
        Files::iterator it= files.begin();
        Files::iterator endit= files.end();

        while(it != endit)
        {
          if (fs::is_regular_file(*it) && (*it).extension() == ext)
          {
            ret.push_back(*it);
          }
          ++it;
        }
      }
    }

cv::Mat Dynamic_slam::capture_(){
	cv::Mat image;

	if( use_image_dataset==true){
		image = imread( dataset_img_file_vec[ runcl.dataset_frame_num ].string() );
																				cerr 	<<"\nDynamic_slam::capture_() : use_image_dataset==true\n"
																						<<"\nruncl.dataset_frame_num = "									<<runcl.dataset_frame_num
																						<<"\ndataset_img_file_vec[ runcl.dataset_frame_num ].string() = "	<<dataset_img_file_vec[ runcl.dataset_frame_num ].string()
																						<<"\ndataset_img_file_vec.len()	= "									<<dataset_img_file_vec.size()
																						<<flush;
	}else{
		capture >> image;
	}
					cerr 	<< "\nDynamic_slam::capture_() image.size()= "	<<image.size()
							<< "\nimage.type()= "							<<image.type()
							<< "\n\n"										<<flush;
	return image;
}
/*
void Dynamic_slam::capture_( cv::Mat &image ){
	if( use_image_dataset==true){
		imread( dataset_img_file_vec[ runcl.dataset_frame_num ].string() );
	}else{
		capture >> image;
	}
}
*/
void Dynamic_slam::start_data_capture(){
	int local_verbosity_threshold 		= V_DYNAMIC_SLAM_DYNAMIC_SLAM;
																																			if(verbosity>local_verbosity_threshold) cout << "\n Dynamic_slam::start_data_capture_chk 1\n" << flush;
	string arg;

	if( obj["GT_available"].asBool() ){
		stringstream  ss0;
		ss0 << obj["data_path"].asString()  <<  obj["data_file"].asString();																	// Collect the filenames of all the input images, plus ground truth files for camera data and depth maps- #####################
		rootpath_string		= ss0.str();
		root_fs_path		= rootpath_string;

		if ( exists(		root_fs_path )==false )		{ cout << "Data folder "<< ss0.str()  <<" does not exist.\n" <<flush; runcl.exit_(0); }
		if ( is_directory(	root_fs_path )==false )		{ cout << "Data folder "<< ss0.str()  <<" is not a folder.\n"<<flush; runcl.exit_(0); }
		if ( empty(			root_fs_path )==true )		{ cout << "Data folder "<< ss0.str()  <<" is empty.\n"		 <<flush; runcl.exit_(0); }
																																			if(verbosity>local_verbosity_threshold) cout << "\n Dynamic_slam::start_data_capture_chk 2\n" << flush;
		get_all( root_fs_path, ".txt",   txt);																											// Get lists of files. Gathers all filepaths with each suffix, into c++ vectors.
		get_all( root_fs_path, ".depth", depth);
		if (txt.size()	<=0){	GT_available = false;		cout<<",  WARNING no gound truth .txt file."<<flush;}
		if (depth.size()<=0){	GT_available = false;		cout<<",  WARNING no gound truth .depth file."<<flush;}

																																			if(verbosity>local_verbosity_threshold){cout << "\n Dynamic_slam::start_data_capture_chk 3\n"	<<flush;
																																				cout<<"\ntxt.size() 	= "	<<txt.size()		<<" GT params files found in data folder."	<<flush;
																																				cout<<"\ndepth.size()	= "	<<depth.size()		<<" GT depth files found in data folder."	<<flush;
																																				cout<<"\ndataset_path	= " <<rootpath_string	<<flush;
																																			}
	}

	if( obj["use_image_dataset"].asBool() ){																									// use dataset of images
		stringstream  ss0;
		ss0 << obj["data_path"].asString()  <<  obj["data_file"].asString();																	// Collect the filenames of all the input images, plus ground truth files for camera data and depth maps- #####################
		rootpath_string		= ss0.str();
		root_fs_path		= rootpath_string;

		if ( exists(		root_fs_path )==false )		{ cout << "Data folder "<< ss0.str()  <<" does not exist.\n" <<flush; runcl.exit_(0); }
		if ( is_directory(	root_fs_path )==false )		{ cout << "Data folder "<< ss0.str()  <<" is not a folder.\n"<<flush; runcl.exit_(0); }
		if ( empty(			root_fs_path )==true )		{ cout << "Data folder "<< ss0.str()  <<" is empty.\n"		 <<flush; runcl.exit_(0); }
																																			if(verbosity>local_verbosity_threshold) cout << "\n Dynamic_slam::start_data_capture_chk 2\n" << flush;
		get_all( root_fs_path, obj["image_file_suffix"].asString(),   dataset_img_file_vec );																									// Get lists of files. Gathers all filepaths with each suffix, into c++ vectors.

		if ( dataset_img_file_vec.size() <=0){
											cerr << "\nDynamic_slam::start_data_capture()  No image files in dataset !\n" << endl;
											runcl.exit_( 1 );
		}
																																			if(verbosity>local_verbosity_threshold){cout << "\n Dynamic_slam::start_data_capture_chk 3\n" << flush;
																																				cout<<"\n txt.size() = "					<<txt.size()	<<flush;
																																				cout <<"\nDynamic_slam::Dynamic_slam(): "	<< dataset_img_file_vec.size()	<<" .png images found in data folder.\t"
																																				<<"png[runcl.dataset_frame_num].string()="	<< dataset_img_file_vec[runcl.dataset_frame_num].string()	<<flush;
																																			}
		arg									= dataset_img_file_vec[ runcl.dataset_frame_num ].string();

	}else if( obj["use_video_file"].asBool() ){																									// use video file

		arg									= obj["video_file"].asString();

	}else if( obj["use_camera"].asBool() ){																										// use camera

		arg									= obj["camera_index"].asString();

	}else if( obj["use_ROS_topic"].asBool() ){																									// use ROS topic	TODO implement ROS topic option.

		// obj["ros_topic"].asString();
											cerr << "\nDynamic_slam::start_data_capture()  ROS topic option not yet implemented !\n" << endl;
											runcl.exit_( 1 );

	}else{																																		// no video source specified
											cerr << "\nDynamic_slam::start_data_capture()  No video source specified !\n" << endl;
											runcl.exit_( 1 );
	}


	capture									= VideoCapture(arg);																				//try to open string, this will attempt to open it as a video file or image sequence
	if (!capture.isOpened() )				capture.open(atoi(arg.c_str()));																	//if this fails, try to open as a video camera, through the use of an integer param
	if (!capture.isOpened() ){
											cerr << "\nDynamic_slam::start_data_capture()  Failed to open the video device, video file or image sequence!\n" << endl;
											runcl.exit_( 1 );
	}

}
