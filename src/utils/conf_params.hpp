#ifndef CONF_PARAMS
#define CONF_PARAMS

#include <jsoncpp/json/json.h>

#include <iostream>
#include <iomanip>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <iostream>

#include <map>
#include <string>
#include <string_view>

#include <vector>
#include <sstream>

#include <filesystem>

using namespace std;


void copy_conf(Json::String source_filepath,  Json::String infile,  string outfile   );

class conf_params {
    public:

	conf_params(char * arg, Json::Value &val);

	void create_out_folder(	Json::Value &val);
	void save_stdout(Json::Value& val,  string outfile);

	void read_paths(		Json::Value paths_obj);
	void read_jparams(		Json::Value params_obj);

	void readVecVecFloat(  	string member, Json::Value params_obj );
	void readVecFloat( 		string member, Json::Value params_obj );
	void readVecString( 	string member, Json::Value params_obj );

	void display_params();
};



#endif
