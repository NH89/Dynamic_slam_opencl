#include "conf_params.hpp"

/////// functions
conf_params::conf_params(char * arg, Json::Value &val){															// "arg" contains the path to the "local_conf/filepaths_<computer name>.json" file, which includes "conf.json" and "verbosity.json".
	int local_verbosity_threshold = val["conf_params::conf_params"].asInt();
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

	read_verbosity( verbosity_obj);																					// Writes the verbosity values to an Map<int> object. This makes checking local verbosity at every function faster than reading a json string.
/*
	//read_paths(   paths_obj);        // Need updating given reorganization of conf.json and filepaths.json .
	//read_jparams(	params_obj);
*/
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


void conf_params::read_verbosity(Json::Value verbosity_obj){	                                                   // iterate over all entries in the verbosity.json file.
	// NB Need to include classname. to prevent function name clashes between classes.
																													cout << "\njson_params::read_verbosity(..) : chk_0\n"<<flush;
	Json::Value::ArrayIndex size = verbosity_obj.size();
	Json::Value::Members members = verbosity_obj.getMemberNames();
																													cout << "\njson_params::read_verbosity(..) : chk_2\n"<<flush;
																													cout << "\nsize = "<< size << flush;
																													cout << "\nmembers[0] = "<< members[0] <<flush;
	string member;
	for (int index=0; index<size; index++){
		member = members[index];
		verbosity_mp[  members[index] ]   =    verbosity_obj[member].asInt();
	}
																													cout << "\njson_params::read_verbosity(..) : finished\n"<<flush;
}

/*
void conf_params::read_paths(Json::Value paths_obj){	                                                           //string_map path_map = const_cast<string_map&>(paths);

	Json::ArrayIndex size = paths_obj.size();
	Json::Value::Members members = paths_obj.getMemberNames();
	string member;
	for (int index=0; index<size; index++){
		member = members[index];
		paths_mp[  members[index] ]   =    paths_obj[member].asString();

	}
}


void conf_params::read_jparams(Json::Value params_obj){   // , int_map	&int_params, float_map &flt_params, float_vec_map &flt_arry_params  // int_,	float_,	float_vec;
																													cout << "\njson_params::read_jparams(..) : chk_0\n"<<flush;
	Json::ArrayIndex size = params_obj.size();
	Json::Value::Members members = params_obj.getMemberNames();
	string member;
																													//cout << "\njson_params::read_jparams(..) : chk_1\n"<<flush;
	for (int index=0; index<size; index++){
		member = members[index];
																													//cout << "\nindex="<<index<<flush; cout<< "\tmember="<<member<<" ,"<<flush;
		if (params_obj[member].isArray() ){ 																		//cout << "\n isArray() " << members[index] <<" : "<< flush;
			if (params_obj[member][0].isString() ){																	//cout << "\n isArray() " << members[index] <<" : "<< flush;
				readVecString( member, params_obj  );
			}
			else if (params_obj[member][0].isArray() ){																//cout << "\n isArray() " << members[index] <<" : "<< flush;
				if (params_obj[member][0][0].isNumeric() ) {
					readVecVecFloat(  member, params_obj );
				}
				else cout << "\n " << member << " is not float_vecvec  :    "  << flush;
			}
			else if (params_obj[member][0].isNumeric() ) {															//cout << "\nisNumeric"<<flush;
				readVecFloat( member, params_obj );
			}
			else {
				cout << "\nparams_obj[\""<< members[index] <<"\"] : "<<  params_obj[member].asString()  << " , is not a float_array. break;"<<flush;
			}
		}
		else if (params_obj[member].isBool()){																		//cout << "\n isBool() " << members[index] <<" : "<< params_obj[member].asBool() <<" , "<< flush;
			bool_mp[  members[index] ]	= params_obj[member].asBool();
		}
		else if (params_obj[member].isInt() ){																		//cout << "\n isInt() " << members[index] <<" : "<< params_obj[member].asInt() <<" , "<< flush;
			int_mp[  members[index] ]   	= params_obj[member].asInt();
		}
		else if (params_obj[member].isDouble() && ! params_obj[member].isInt() ){  									//cout << "\n isDouble() " << members[index] <<" : "<< params_obj[member].asDouble() <<" , "<< flush;
			float_mp[  members[index] ]	= params_obj[member].asFloat();
		}
		else {
			cout << "\nparams_obj[\""<< members[index] <<"\"] : "<<  params_obj[member].asString()  << " , is not a float_array, float or int."<<flush;
		}
	}
																													cout << "\njson_params::read_jparams(..) : finished\n"<<flush;
}


void conf_params::readVecVecFloat( string member, Json::Value params_obj  ){
	vector<vector<float>> vec0;
	Json::ArrayIndex size2 = params_obj[member].size();																cout << "\n array size = "<< size2 << " , " << flush;
	for (int index2=0; index2<size2; index2++){																		cout << "\nindex2 = "<< index2 ;
		vector<float> vec1;
		Json::ArrayIndex size3 = params_obj[member].size();															cout << "\n array size = "<< size3 << " , " << flush;

		for (int index3=0; index3<size2; index3++){																	cout << "\nindex3 = "<< index3 ;
			if (params_obj[member][index2][index3].isNumeric() ) {													cout << "\nisNumeric"<<flush;
																													cout << "\nparams_obj[member][index2][index3] = " << params_obj[member][index2][index3].asFloat() << " , " << flush;
			vec1.push_back( params_obj[member][index2][index3].asFloat() );
			}
		}
		vec0.push_back(vec1);
	}
	float_vecvec_mp[  member ] = vec0;
}

void conf_params::readVecFloat( string member, Json::Value params_obj  ){
	vector<float> vec0;
	Json::ArrayIndex size2 = params_obj[member].size();																cout << "\n array size = "<< size2 << " , " << flush;
	for (int index2=0; index2<size2; index2++){																		cout << "\nindex2 = "<< index2 ;
		if (params_obj[member][index2].isNumeric() ) {																cout << "\nisNumeric"<<flush;
																													cout << "\nparams_obj[member][index2][index3] = " << params_obj[member][index2].asFloat() << " , " << flush;
			vec0.push_back( params_obj[member][index2].asFloat() );
		}
	}
	float_vec_mp[  member ] = vec0;
}


void conf_params::readVecString( string member, Json::Value params_obj  ){
	vector<string> vec0;
	Json::ArrayIndex size2 = params_obj[member].size();																cout << "\n array size = "<< size2 << " , " << flush;
	for (int index2=0; index2<size2; index2++){																		cout << "\nindex2 = "<< index2 ;
																													cout << "\nparams_obj[member][index2][index3] = " << params_obj[member][index2].asString() << " , " << flush;
			vec0.push_back( params_obj[member][index2].asString() );
	}
	string_vec_mp[  member ] = vec0;
}
*/

void conf_params::display_params(  ) {
	cout << "\n\n params.paths   = "  << flush;
	for (auto elem : paths_mp) cout << "\n " <<  elem.first <<" : "<< elem.second << flush;

	cout << "\n\n params.int_   = "  << flush;
	for (auto elem : int_mp) cout << "\n " <<  elem.first <<" : "<< elem.second << flush;

	cout << "\n\n params.float_   = "  << flush;
	for (auto elem : float_mp) cout << "\n " <<  elem.first <<" : "<< elem.second << flush;

	cout << "\n\n params.float_vec   = "  << flush;
	for (auto elem : float_vec_mp) {
		cout << "\n " <<  elem.first <<" :  {";
		for (auto elem2 : elem.second) {
			cout << elem2 << ", ";
		}
		cout << " } " << flush;
	}

	cout << "\n\n params.float_vecvec   = "  << flush;
	for (auto elem : float_vecvec_mp) {
		cout << "\n " <<  elem.first <<" :  {";
		for (auto elem2 : elem.second) {
			cout <<"\t [ " ;
			for (auto elem3 : elem2){
				cout << elem3 << ", ";
			}
			cout <<"], ";
		}
		cout << " } " << flush;
	}

	cout << "\n\n params.string_vec   = "  << flush;
	for (auto elem : string_vec_mp) {
		cout << "\n " <<  elem.first <<" :  {";
		for (auto elem2 : elem.second) {
			cout << elem2 << ", ";
		}
		cout << " } " << flush;
	}

	cout << "\n\n params.verbosity   = "  << flush;
	for (auto elem : verbosity_mp) cout << "\n " <<  elem.first <<" : "<< elem.second << flush;
}

