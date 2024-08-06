#include "conf_params.hpp"
#include "../utils/verbosity.hpp"


conf_params::conf_params(char * arg, Json::Value &val){							// "arg" contains the path to the "local_conf/filepaths_<computer name>.json" file, which includes "conf.json" and "verbosity.json".
	int local_verbosity_threshold = V_CONF_PARAMS_CONF_PARAMS;
                                                                                cout << "\nconf_params::conf_params(char * \""<<arg<<"\", arg, Json::Value &val) chk 1"<<flush;
	ifstream ifs(arg);
	Json::Reader 	reader;
	Json::Value 	paths_obj, 		params_obj, 	verbosity_obj;

    bool b;
	b = reader.parse(ifs, paths_obj); 											if (!b) { cout << "Error: " << reader.getFormattedErrorMessages()<<flush; cerr << flush;  exit(1) ;}   else {cout << "\nconf_params::conf_params(..) chk_2: \tNB lists .json file entries alphabetically: \npaths_obj = \n" << paths_obj ;}

	ifstream ifs_params(	paths_obj["source_filepath"].asString()	 +  paths_obj["params_conf"].asString() 	);
																													if (!ifs_params.is_open()) {
																															cout << "\njson_params::json_params(char * arg):  ifs_params  is NOT open !"<< flush;
																															cout << "\nfilepath + "<< paths_obj["source_filepath"].asString()	 +  paths_obj["params_conf"].asString() << endl<<flush;
																															cerr << flush;
																															exit(1);
																													}

	ifstream ifs_verbosity( paths_obj["source_filepath"].asString()	 +  paths_obj["verbosity_conf"].asString()	);
																													if (!ifs_verbosity.is_open()) {
																															cout << "\njson_params::json_params(char * arg):  ifs_verbosity  is NOT open !"<< flush;
																															cout << "\nfilepath + "<< paths_obj["source_filepath"].asString()	 +  paths_obj["verbosity_conf"].asString() << endl<<flush;
																															cerr << flush;
																															exit(1);
																													}

	b = reader.parse(ifs_params, 	params_obj); 								if (!b) { cout << "Error: " << reader.getFormattedErrorMessages()<< flush;; cerr << flush; exit(1) ;}   else {cout << "\nconf_params::conf_params(..) chk_3: \tNB lists .json file entries alphabetically: \nparams_obj = \n" << params_obj ;}

	b = reader.parse(ifs_verbosity, verbosity_obj); 							if (!b) { cout << "Error: " << reader.getFormattedErrorMessages()<< flush;; cerr << flush; exit(1) ;}   else {cout << "\nconf_params::conf_params(..) chk_4: \tNB lists .json file entries alphabetically: \nverbosity_obj = \n" << verbosity_obj ;}

	val = params_obj ;																								// Copies the local "params_obj" to "val" passed by reference to this function.
    Json::Value::ArrayIndex 	size 	= paths_obj.size();
	Json::Value::Members 		members = paths_obj.getMemberNames();
    string member;
	for (int index=0; index<size; index++){																			// Appends "paths_obj" contents to "val". Now "val" contains "conf.json" + "filepaths<computer_name>.json"
		member 					=	members[index];
		val[  members[index] ]	=	paths_obj[member];
    }
																													cout << "\njson_params::json_params(char * arg) finished\n"<<flush;
}
