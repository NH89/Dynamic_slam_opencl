//#include "Dynamic_slam/Dynamic_slam.hpp"
//#include "utils/fileLoader.hpp"

#include <filesystem>
#include <iostream>
#include <iomanip>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <iostream>

#include <map>
#include <string>
#include <string_view>

#include <string>
#include <sstream>
#include "Dynamic_slam/Dynamic_slam.hpp"
#include "utils/conf_params.hpp"

using namespace cv;
using namespace std;


void copy_conf(Json::String source_filepath,  Json::String infile,  string outfile   ){
	filesystem::path in_path_verbosity(  source_filepath  +  infile  );
	filesystem::path out_path_verbosity( outfile   );
	out_path_verbosity.replace_filename( in_path_verbosity.filename() );
	cerr << "\nin_path="<<in_path_verbosity<<",  out_path="<<out_path_verbosity<<endl<<flush;
	filesystem::copy( in_path_verbosity , out_path_verbosity );
}

int main(int argc, char *argv[])
{
	if (argc !=2) { cout << "\n\nUsage : DTAM_OpenCL <config_file.json>\n\n" << flush; exit(1); }
cout << "\nmain_chk_1" <<flush;
	Json::Value obj;

cout << "\n\nmain_chk_2" <<flush;
	conf_params j_params(argv[1], obj);

cout << "\n\nmain_chk_3" <<flush;
	j_params.display_params();

cout << "\n\nmain_chk_4" <<flush;
	int verbosity_ 		= j_params.verbosity_mp["verbosity"]; 				// obj["verbosity"]["verbosity"].asInt() ;	// -1= none, 0=errors only, 1=basic, 2=lots.
	int imagesPerCV 	= obj["imagesPerCV"].asUInt() ;						// j_params.int_mp["imagesPerCV"]; 			//
	int max_frame_count = obj["max_frame_count"].asUInt();					// j_params.int_mp["max_frame_count"]; 		//
	int frame_count 	= 0;
	int ds_error 		= 0;

cout << "\n\nmain_chk_5" <<flush;
																			if(verbosity_>0) cout << "\n\n main_chk 0\n" << flush;
																			cout <<"\nconf file = "		<< argv[1];
																			cout <<"\nverbosity_ = "	<<verbosity_;
																			cout <<"\nimagesPerCV = "	<<imagesPerCV;
																			cout <<"\noutpath = " 		<< j_params.paths_mp["out_path"];
cout << "\n\nmain_chk_6" <<flush;
	Dynamic_slam   dynamic_slam(obj, j_params.verbosity_mp);				// Instantiate Dynamic_slam object before while loop.

	stringstream outfile;													// Redirecting cout to write to "output.txt"
	outfile << dynamic_slam.runcl.paths.at("folder").c_str() <<  "Dynamic_slam_output.txt";
    cout << "\nOutfile = " << outfile.str() << endl;
    fflush (stdout);
    freopen (outfile.str().c_str(), "w", stdout);
	cout << "Dynamic_slam started. Objects created. Initializing.\n\n" << flush;
																			// Copy conf files to output.
	copy_conf( obj["source_filepath"].asString(),  obj["params_conf"].asString(),		outfile.str().c_str()  );
	copy_conf( obj["source_filepath"].asString(),  obj["verbosity_conf"].asString(),	outfile.str().c_str()  );
	copy_conf( ""								,  argv[1],    							outfile.str().c_str()  );

																			if(verbosity_>0) cout << "\n main_chk 1\n" << flush;
																			// New continuous while loop: load next (image + data), Dynamic_slam::nextFrame(..)
																			// NB need to initialize the cost volume with a key frame, and tracking with a depth map.
																			// Also need a test for when to start a new keyframe.
	cerr << "\n\nmain()  dynamic_slam.initialize_keyframe_from_GT()"<<flush;
	dynamic_slam.initialize_keyframe_vec();
	//dynamic_slam.initialize_keyframe_from_GT();
	frame_count++;
																			if(verbosity_>0) cout << "\n main_chk 2\n" << flush;
	do{																		// Long do while not yet crashed loop.
		for (int i=0; i<imagesPerCV ; i++){									// Inner loop per keyframe.params
			cerr << "\n\nmain()  dynamic_slam.nextFrame();   frame_count="<<frame_count<<flush;
			ds_error = dynamic_slam.nextFrame();
			frame_count ++;
		}
		cerr << "\n\nmain()  dynamic_slam.optimize_depth();"<<flush;
		dynamic_slam.optimize_depth();
																			if(verbosity_>0) dynamic_slam.runcl.saveCostVols(imagesPerCV);
		dynamic_slam.initialize_keyframe();
		//dynamic_slam.initialize_keyframe_from_GT();
		//dynamic_slam.initialize_keyframe_from_tracking();
		//break; // TODO write new depthmap transformation based on bin sort from fluids_v3 & Morphogenesis.
		
	}while(!ds_error && ((frame_count<max_frame_count) || (max_frame_count==-1)) );
																			if(verbosity_>0) cout << "\n main_chk 3\n" << flush;
	cerr << "\n\nmain() starting  dynamic_slam.getResult();"<<flush;
	dynamic_slam.getResult();												// also calls RunCL::CleanUp()

	cerr << "\n\nmain() Dynamic_slam finished. Exiting."<<flush;
	cout << "\n\nDynamic_slam finished. Exiting."<<flush;
	fflush (stdout);
    fclose (stdout);
	exit(0);																// guarantees class destructors are called.
}
