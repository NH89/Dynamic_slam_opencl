#include <jsoncpp/json/json.h>

#include "Dynamic_slam/Dynamic_slam.hpp"
#include "utils/conf_params.hpp"

#include <iomanip>
#include <ctime>
#include <map>
#include <string_view>
#include <sstream>
#include <string>

using namespace cv;
using namespace std;

int main(int argc, char *argv[])
{
	if (argc !=2) { cout << "\n\nUsage : DTAM_OpenCL <config_file.json>\n\n" << flush; exit(1); }
																														cout << "\n main_chk 0\n" << flush;
	Json::Value obj;
	conf_params j_params(argv[1], obj);																					// Construct conf_params object j_params,  i.e. read all three .json files.#################################

	int verbosity_ 		= obj["verbosity"].asUInt() ;																	// Global verbosity: -1= none, 0=errors only, 1=basic, 2=lots.
	int imagesPerCV 	= obj["imagesPerCV"].asUInt() ;																	// j_params.int_mp["imagesPerCV"]; 			//
	int max_frame_count = obj["max_frame_count"].asUInt();																// j_params.int_mp["max_frame_count"]; 		//
	int frame_count 	= 0;
	int ds_error 		= 0;
																														if(verbosity_>0) cout << "\n\n main_chk 1\n" << flush;
																														cout 	<<"\nconf file = "		<< argv[1]
																																<<"\nverbosity_ = "		<<verbosity_
																																<<"\nimagesPerCV = "	<<imagesPerCV
																																<<"\noutpath = " 		<<obj["out_path"].asString()
																																<<"\n## main.cpp line 32 ##"<< flush; //j_params.paths_mp
	j_params.save_stdout( obj,  "Dynamic_slam_startup_output.txt");
	Dynamic_slam 	dynamic_slam( obj );																				// Construct Dynamic_slam object (including RunCL object) before while loop.################################
	frame_count++;
	j_params.save_stdout( obj,  "Dynamic_slam_output.txt");
																														if(verbosity_>0) cout << "\n main_chk 2\n" << flush;
	do{																													// Long do-while not yet crashed loop. #####################################################################
		for (int i=0; i<imagesPerCV ; i++){																				// Inner loop per keyframe.params
																														cerr << "\n\nmain()  dynamic_slam.nextFrame();   frame_count="<<frame_count<<flush;
			ds_error = dynamic_slam.nextFrame();
			frame_count ++;
		}
																														cerr << "\n\nmain()  dynamic_slam.optimize_depth();"<<flush;
		//dynamic_slam.optimize_depth();
																														//if(verbosity_>1) dynamic_slam.runcl.saveCostVols(imagesPerCV);
		//dynamic_slam.initialize_keyframe_vec();																			// next keyframe
																														// TODO write new depthmap transformation based on bin sort from fluids_v3 & Morphogenesis.
		
	}while(!ds_error && ((frame_count<max_frame_count) || (max_frame_count==-1)) );										// #########################################################################################################
																														if(verbosity_>0) cout << "\n main_chk 3\n" << flush;
																														cerr << "\n\nmain() starting  dynamic_sla-1.getResult();"<<flush;
	//dynamic_slam.print_pose_vectors( 0, 20);
	//dynamic_slam.getResult();																							// also calls RunCL::CleanUp()
																														cerr << "\n\nmain() Dynamic_slam finished. Exiting."<<flush;
																														cout << "\n\nDynamic_slam finished. Exiting."<<flush;
	fflush (stdout);
    fclose (stdout);
	dynamic_slam.runcl.exit_(0);																						// guarantees class destructors are called.
}
