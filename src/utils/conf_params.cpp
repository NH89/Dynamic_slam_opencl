#include "conf_params.hpp"
#include "../utils/verbosity.hpp"
#include "time_utils.hpp"


void copy_conf(Json::String source_filepath,  Json::String infile,  string outfile   ){
	filesystem::path in_path_verbosity(   source_filepath + infile    );
	filesystem::path out_path_verbosity( outfile   );
	out_path_verbosity.replace_filename( in_path_verbosity.filename() );
																														cerr << "\nin_path="<<in_path_verbosity<<",  out_path="<<out_path_verbosity<<endl<<flush;
	filesystem::copy( in_path_verbosity , out_path_verbosity );															// Used to save a copy of each .conf file to the output folder.
}


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
	b = reader.parse(ifs_params, 	params_obj); 								if (!b) { cout << "Error: " << reader.getFormattedErrorMessages()<< flush;; cerr << flush; exit(1) ;}   else {cout << "\nconf_params::conf_params(..) chk_3: \tNB lists .json file entries alphabetically: \nparams_obj = \n" << params_obj ;}

	val = params_obj ;																								// Copies the local "params_obj" to "val" passed by reference to this function.
    Json::Value::ArrayIndex 	size 	= paths_obj.size();
	Json::Value::Members 		members = paths_obj.getMemberNames();
    string member;
	for (int index=0; index<size; index++){																			// Appends "paths_obj" contents to "val". Now "val" contains "conf.json" + "filepaths<computer_name>.json"
		member 					=	members[index];
		val[  members[index] ]	=	paths_obj[member];
    }
    create_out_folder(val);

	copy_conf( val["source_filepath"].asString(),  val["params_conf"].asString(),	val["out_path"].asString()  );
	copy_conf( ""								,  arg,    							val["out_path"].asString()  );
																													cout << "\njson_params::json_params(char * arg) finished\n"<<flush;
}


void conf_params::create_out_folder(Json::Value& val){
	int local_verbosity_threshold = V_CONF_PARAMS_CONF_PARAMS;

	std::string   out_dir = date_time_string();

	std::filesystem::path 	out_path(std::filesystem::current_path());
	std::filesystem::path 	conf_outpath( val["out_path"].asString() );
																													// cout << "\nconf_outpath = " << conf_outpath ;
	if (conf_outpath.empty()  ) {
		out_path = out_path.parent_path().parent_path();															// move "out_path" up two levels in the directory tree.
		out_path += conf_outpath;
																													// cout << "  conf_outpath.empty()==true" ;
	}else {out_path = conf_outpath;}
	out_path += "/output/";
	out_path += date_time_string();

	if(std::filesystem::create_directory(out_path)) { 							cerr<< "Directory Created: "<<out_path<<std::endl;}
	else{ 																		cerr<< "Output directory previously created: "<<out_path<<std::endl;}

	out_path += "/";
	val["out_path"] = out_path.c_str();
}

void conf_params::save_stdout(Json::Value& val,  string outfile){
	std::filesystem::path 	out_path( val["out_path"].asString() );
	out_path += outfile;
																				cerr << "\nOutfile = " << out_path.string() << endl;
	fflush (stdout);
	freopen (out_path.string().c_str(), "w", stdout);
}
