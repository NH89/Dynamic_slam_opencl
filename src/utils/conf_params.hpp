#ifndef CONF_PARAMS
#define CONF_PARAMS

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
#include <jsoncpp/json/json.h>

using namespace std;

class conf_params {
    public:

	conf_params(char * arg, Json::Value &val);
	//void read_verbosity(	Json::Value verbosity_obj);
	void read_paths(		Json::Value paths_obj);
	void read_jparams(		Json::Value params_obj);

	void readVecVecFloat(  	string member, Json::Value params_obj );
	void readVecFloat( 		string member, Json::Value params_obj );
	void readVecString( 	string member, Json::Value params_obj );

	void display_params();
};

#endif
