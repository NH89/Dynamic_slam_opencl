#include "RunCL.hpp"

/*
 * NB there are memory leaks assoc with the Intel and AMD  OpenCL-ICD
 https://stackoverflow.com/questions/47869162/opencl-clgetplatformids-gives-around-230-valgrind-memcheck-errors
 https://community.khronos.org/t/opencl-clgetplatformids-gives-around-230-valgrind-memcheck-errors/7388
 https://community.intel.com/t5/OpenCL-for-CPU/OpenCL-clGetPlatformIDs-gives-26-valgrind-memcheck-errors/td-p/1174915
 Memory leak in icd.c #13
 https://github.com/KhronosGroup/OpenCL-ICD-Loader/issues/13
 */

RunCL::RunCL( Json::Value obj_  ){ //, int_map verbosity_mp_
	obj 		 	= obj_;																													// NB save obj_ to class member obj, so that it persists within this RunCL object.
	//verbosity_mp 	= verbosity_mp_;
	verbosity 						= obj["verbosity"].asInt();
	int local_verbosity_threshold 	= V_RUNCL_RUNCL;//verbosity_mp["RunCL::RunCL"];
	tiff 							= obj["tiff"].asBool();
	png 							= obj["png"].asBool();
	vtp 							= obj["vtp"].asBool();
	max_depth						= obj["max_depth"].asFloat();
																																			if(verbosity>local_verbosity_threshold) {
																																				cout << "\nRunCL_chk 0\n" << flush;
																																				cout << "\nverbosity = "<<verbosity<< flush;
																																			}
																																			/*Step1: Getting platforms and choose an available one.*/////////###############################
	//testOpencl();																															// Displays available OpenCL Platforms and Devices.
	cl_uint 		numPlatforms;																											//the NO. of platforms
	cl_platform_id 	platform 		= NULL;																									//the chosen platform
	cl_int			status 			= clGetPlatformIDs(0, NULL, &numPlatforms);				if (status != CL_SUCCESS){ cout << "Error: Getting platforms!" << endl; exit_(status); };
	uint			conf_platform	= obj["opencl_platform"].asUInt();																		if(verbosity>local_verbosity_threshold) cout << "numPlatforms = " << numPlatforms << ", conf_platform=" << conf_platform << "\n" << flush;



	if (numPlatforms > conf_platform){
		cl_platform_id* platforms 	= (cl_platform_id*)malloc(numPlatforms * sizeof(cl_platform_id));

		status 	 					= clGetPlatformIDs(numPlatforms, platforms, NULL);		if (status != CL_SUCCESS){ cout << "Error: Getting platformsIDs" << endl; exit_(status); }

		platform 					= platforms[ conf_platform ];																			if(verbosity>local_verbosity_threshold){ for(int i=0; i<numPlatforms; i++) { cout << "\nplatforms["<<i<<"] = "<<platforms[i]; }
																																								 cout <<"\nSelected platform number :"<<conf_platform<<", cl_platform_id platform = " << platform<<"\n"<<flush;
																																							}
		free(platforms);
	} else {																																cout<<"Error: Platform num "<<conf_platform<<" not available."<<flush; exit_(0);}

	cl_uint			numDevices		= 0;																									/*Step 2:Query the platform.*//////////////////////////////////################################
	cl_device_id    *devices;
	status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);			if (status != CL_SUCCESS) {cout << "\n3 status = " << checkerror(status) <<"\n"<<flush; exit_(status);}
	uint conf_device = obj["opencl_device"].asUInt();


	if (numDevices <= conf_device){                                                         cout << "\n\nRunCL::RunCL(..), (numDevices <= conf_device)\n" << flush; exit_(status); }
	devices = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));																		/*Choose the device*/
	status  = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);  if (status != CL_SUCCESS) {cout << "\n4 status = " << checkerror(status) <<"\n"<<flush; exit_(status);}


	cl_context_properties cps[3]={CL_CONTEXT_PLATFORM,(cl_context_properties)platform,0};													/*Step 3: Create context.*////////////////////////////////////##################################
	m_context 	= clCreateContextFromType( cps, CL_DEVICE_TYPE_GPU, NULL, NULL, &status);	if(status!=0) {cout<<"\n5 status="<<checkerror(status)<<"\n"<<flush;exit_(status);}

	deviceId  	= devices[conf_device];																										/*Step 4: Create command queue & associate context.*///////////#################################
																																			if(verbosity>local_verbosity_threshold){
																																				cout << "\ndeviceId = " << deviceId <<"\n" <<flush;
																																				cl_int err;
																																				cl_uint addr_data;
																																				char name_data[48], ext_data[4096];
																																				err = clGetDeviceInfo(deviceId, CL_DEVICE_NAME, sizeof(name_data), name_data, NULL);
																																				if(err < 0) {perror("Couldn't read extension data"); exit_(1); }
																																				clGetDeviceInfo(deviceId, CL_DEVICE_ADDRESS_BITS, sizeof(ext_data), &addr_data, NULL);
																																				clGetDeviceInfo(deviceId, CL_DEVICE_EXTENSIONS, sizeof(ext_data), ext_data, NULL);
																																				printf("\nDevice num: %i \nNAME: %s\nADDRESS_WIDTH: %u\nEXTENSIONS: %s \n", conf_device, name_data, addr_data, ext_data);
																																			}
	createQueues();
																																			// Multiple queues for latency hiding: Upload, Download, Mapping, Tracking,... autocalibration, SIRFS, SPMP
																																			// NB Might want to create command queues on multiple platforms & devices.
																																			// NB might want to divde a task across multiple MPI Ranks on a multi-GPU WS or cluster.

	createAndBulidProgramFromSource( devices ); 																							/*Step 5: Create program object*/////////////###################################################
																																			/*Step 6: Build program.*////////////////////###################################################
																																			/*Step 7: Create kernel objects.*////////////###################################################
	createKernels();
	/*
	 * Given the "apparent memory leaks wrt the Intel ocl-icd,  this is the "valgrind --leak-check=full" result upto this point on Intel IrisXe GPU, on ubuntu 23.04"
	==264229== LEAK SUMMARY:
	==264229==    definitely lost: 126,434 bytes in 577 blocks
	==264229==    indirectly lost: 16,608 bytes in 3 blocks
	==264229==      possibly lost: 529,189 bytes in 37 blocks
	==264229==    still reachable: 2,102,638 bytes in 8,690 blocks
	==264229==                       of which reachable via heuristic:
	==264229==                         newarray           : 1,032 bytes in 1 blocks
	==264229==                         multipleinheritance: 21,536 bytes in 4 blocks
	==264229==         suppressed: 0 bytes in 0 blocks
	==264229== Reachable blocks (those to which a pointer was found) are not shown.
	==264229== To see them, rerun with: --leak-check=full --show-leak-kinds=all
	==264229==
	==264229== For lists of detected and suppressed errors, rerun with: -s
	==264229== ERROR SUMMARY: 11 errors from 11 contexts (suppressed: 0 from 0)
	*/
	for (uint i=0; i<num_current_frames; i++ ) {imgmem[i]=0; velmap[i]=0;}
	basemem=k2kbuf=0;																														// Set device pointers to zero
	createFolders( );																														// Create the folders to which the output will be written.

	free(devices);
																																			if(verbosity>local_verbosity_threshold) cout << "RunCL_constructor finished ##########################\n" << flush;
}

void RunCL::testOpencl(){
	int local_verbosity_threshold = V_RUNCL_TESTOPENCL;//verbosity_mp["RunCL::testOpencl"];
																																			if(verbosity>local_verbosity_threshold) cout << "\n\nRunCL::testOpencl() ############################################################\n\n" << flush;
	cl_platform_id *platforms;
	cl_uint num_platforms;
	cl_int i, err, platform_index = -1;
	char* ext_data;

	size_t ext_size;
	const char icd_ext[] = "cl_khr_icd";
	/*
	cl_int clGetPlatformIDs(	cl_uint num_entries,
								cl_platform_id *platforms,
								cl_uint *num_platforms)
	*/
																																			// Find number of platforms
	err = clGetPlatformIDs(1, NULL, &num_platforms);										if(err < 0) { perror("Couldn't find any platforms."); exit_(1); }
	platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * num_platforms);															// Allocate platform array
	clGetPlatformIDs(num_platforms, platforms, NULL);																						// Initialize platform array
																																			if(verbosity>local_verbosity_threshold) cout << "\nnum_platforms="<<num_platforms<<"\n" << flush;
	for(i=0; i<num_platforms; i++) {
																																			if(verbosity>local_verbosity_threshold) cout << "\n##Platform num="<<i<<" #####################################################\n" << flush;
		char* name_data;
		/*
		cl_int clGetPlatformInfo(	cl_platform_id platform,
									cl_platform_info param_name,
									size_t param_value_size,
									void *param_value,
									size_t *param_value_size_ret)
		*/
																																			// Find size of name data
		err = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 0, NULL, &ext_size);		if(err < 0) { perror("Couldn't read platform name data."); exit_(1); }

		name_data = (char*)malloc(ext_size);
		clGetPlatformInfo( platforms[i],  CL_PLATFORM_NAME, ext_size, name_data, NULL);														printf("Platform %d name: %s\n", i, name_data);
		free(name_data);
																																			// Find size of extension data
		err = clGetPlatformInfo(platforms[i], CL_PLATFORM_EXTENSIONS, 0, NULL, &ext_size);	if(err < 0) { perror("Couldn't read extension data."); exit_(1); }

		ext_data = (char*)malloc(ext_size);																									// Read data extension
		clGetPlatformInfo( platforms[i],  CL_PLATFORM_EXTENSIONS, ext_size, ext_data, NULL);
																																			printf("Platform %d supports extensions: %s\n", i, ext_data);
		free(ext_data);
		/*
		cl_int clGetDeviceIDs(	cl_platform_id platform,
								cl_device_type device_type,
								cl_uint num_entries,
								cl_device_id *devices,
								cl_uint *num_devices)
		*/
		cl_uint			num_devices		= 0;																								/*Step 2:Query the platform.*//////////////////////////////////
		cl_device_id    *devices;																											// Find number of platforms
		err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);		if(err < 0) { perror("Couldn't find any platforms."); continue; }

		devices = (cl_device_id*) malloc(sizeof(cl_device_id) * num_devices);																// Allocate platform array
		clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);														// Initialize platform array
																																			if(verbosity>local_verbosity_threshold) cout << "\nnum_devices="<<num_devices<<"\n" << flush;
		free(devices);
		getDeviceInfoOpencl(platforms[i]);
	}
																																			if(platform_index > -1) printf("Platform %d supports the %s extension.\n", platform_index, icd_ext);
																																			else printf("No platforms support the %s extension.\n", icd_ext);
	free(platforms);
																																			if(verbosity>local_verbosity_threshold) cout << "\nRunCL::testOpencl() finished ##################################################\n\n" << flush;
}

void RunCL::getDeviceInfoOpencl(cl_platform_id platform){
	int local_verbosity_threshold = V_RUNCL_GETDEVICEINFOOPENCL;//verbosity_mp["RunCL::getDeviceInfoOpencl"];
																																			if(verbosity>local_verbosity_threshold) cout << "\n#RunCL::getDeviceInfoOpencl("<< platform <<")" << "\n" << flush;
	cl_device_id *devices;
	cl_uint num_devices, addr_data;
	cl_int i, err;
	char name_data[48], ext_data[4096];
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, NULL, &num_devices); 																if(err < 0) {perror("Couldn't find any devices"); exit_(1); }
	devices = (cl_device_id*) malloc(sizeof(cl_device_id) * num_devices);
	clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);
	for(i=0; i<num_devices; i++) {
		err = clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(name_data), name_data, NULL); 												if(err < 0) {perror("Couldn't read extension data"); exit_(1); }
		clGetDeviceInfo(devices[i], CL_DEVICE_ADDRESS_BITS, sizeof(ext_data), &addr_data, NULL);
		clGetDeviceInfo(devices[i], CL_DEVICE_EXTENSIONS, sizeof(ext_data), ext_data, NULL);
																																			printf("\nDevice num: %i \nNAME: %s\nADDRESS_WIDTH: %u\nEXTENSIONS: %s \n", i, name_data, addr_data, ext_data);
	}
	free(devices);
																																			if(verbosity>local_verbosity_threshold) cout << "\nRunCL::getDeviceInfoOpencl("<< platform <<") finished\n" <<flush;
}

void RunCL::createQueues(){
    int local_verbosity_threshold = V_RUNCL_CREATEQUEUES;//verbosity_mp["RunCL::createQueues"];
																																			if(verbosity>local_verbosity_threshold)  cout << "\nRunCL::createQueues(..) chk 0\n" << flush;
	cl_int	status;
    cl_command_queue_properties prop[] = { 0 };																								//  NB Device (GPU) queues are out-of-order execution -> need synchronization.
    m_queue 	= clCreateCommandQueueWithProperties(m_context, deviceId, prop, &status);	if(status!=CL_SUCCESS)	{ cout<<"\n6 status="<<checkerror(status)<<"\n"<<flush;exit_(status);}
	uload_queue = clCreateCommandQueueWithProperties(m_context, deviceId, prop, &status);	if(status!=CL_SUCCESS)	{ cout<<"\n7 status="<<checkerror(status)<<"\n"<<flush;exit_(status);}
	dload_queue = clCreateCommandQueueWithProperties(m_context, deviceId, prop, &status);	if(status!=CL_SUCCESS)	{ cout<<"\n8 status="<<checkerror(status)<<"\n"<<flush;exit_(status);}
	track_queue = clCreateCommandQueueWithProperties(m_context, deviceId, prop, &status);	if(status!=CL_SUCCESS)	{ cout<<"\n9 status="<<checkerror(status)<<"\n"<<flush;exit_(status);}
}

void RunCL::createAndBulidProgramFromSource(cl_device_id *devices){
	int local_verbosity_threshold = V_RUNCL_CREATEANDBULIDPROGRAMFROMSOURCE;//verbosity_mp["RunCL::createAndBulidProgramFromSource"];
																																			if(verbosity>local_verbosity_threshold)  cout << "\nRunCL::createAndBulidProgramFromSource(..) chk 0\n" << flush;
	cl_int 	status;
	cl_uint	num_files;
	char** 	strings;
	size_t*	lengths;

	const char	*basepath = obj["source_filepath"].asCString();
	const char	*foldername = obj["kernel_folder"].asCString();
	num_files	= obj["kernel_files"].size();
	lengths		= (size_t*)malloc( (num_files+1)*sizeof(size_t) );																				// allocate array for file lengths
	strings		= (char**)malloc( (num_files+1)*sizeof(size_t) );																				// allocate outer array for kernel files
	FILE		*program_handle;

	for (int i=0; i<num_files; i++){																										// for kernel source files in obj array
	const char *filename = obj["kernel_files"][i].asCString();
		stringstream filepath; filepath << basepath << foldername << filename;
		const std::string tmp =  filepath.str();
		const char* char_filepath = tmp.c_str();
		program_handle = fopen(char_filepath, "r");
																								if(program_handle == NULL) { perror("Couldn't find the program file");
																															cout << "\tchar_filepath = "<< char_filepath << flush;
																															exit_(1);
																								}
		fseek(program_handle, 0, SEEK_END);
		lengths[i] = ftell(program_handle);
		rewind(program_handle);

		strings[i] = (char*)malloc(lengths[i]+1);																							// allocate inner array for this kernel file
		strings[i][lengths[i]] = '\0';
		const size_t ret_code = fread( strings[i], sizeof(char), lengths[i], program_handle );
																								if (ret_code != lengths[i]){ perror("Couldn't read the program file");
																															cout << "\tchar_filepath = "<< char_filepath << flush;
																															exit_(1);
																								}
		fclose(program_handle);
	}
	m_program 	= clCreateProgramWithSource( m_context, num_files, (const char**)strings, lengths, &status );								// Create program object /////////////
																								if(status!=CL_SUCCESS)	{cout<<"\n11 status="<<checkerror(status)<<"\n"<<flush;exit_(status);}
	const char * include_dir = obj["kernel_build_options"].asCString();																		if(verbosity>local_verbosity_threshold) cout << "\n" << include_dir << "\n" << flush;

	status = clBuildProgram(m_program, 1, devices, include_dir , NULL, NULL);																// Build program. /////////////////////
	/*
		cl_int clBuildProgram(
								cl_program 				program,
								cl_uint 				num_devices,
								const cl_device_id* 	device_list,
								const char* 			options,
								void (CL_CALLBACK* pfn_notify)(cl_program program, void* user_data),
								void* 					user_data
								);
	*/
																								if (status != CL_SUCCESS){
																									printf("\nclBuildProgram failed: %d\n", status);
																									char buf[0x10000];
																									clGetProgramBuildInfo(m_program, deviceId, CL_PROGRAM_BUILD_LOG, 0x10000, buf, NULL);
																									printf("\n%s\n", buf);
																									exit_(status);
																								}
	for(int i=0; i<num_files; i++) { free(strings[i]); }
	free(strings);
	free(lengths);
																																			if(verbosity>local_verbosity_threshold) cout << "RunCL::createAndBulidProgramFromSource finished ##########################\n" << flush;
}

void RunCL::createKernels(){
	//int local_verbosity_threshold = V_RUNCL_CREATEKERNELS;

	cl_int err_code;
	// RunCL_load_image.cpp
	convert_depth_kernel			= clCreateKernel(m_program, "convert_depth", 				&err_code);			if (err_code != CL_SUCCESS)  {cout << "\nError 'convert_depth'  kernel not built.\n"			<<flush; exit_(0);   }
	mipmap_float_kernel				= clCreateKernel(m_program, "mipmap_linear_flt", 			&err_code);			if (err_code != CL_SUCCESS)  {cout << "\nError 'mipmap_linear_flt'  kernel not built.\n"		<<flush; exit_(0);   }
	cvt_color_space_linear_kernel 	= clCreateKernel(m_program, "cvt_color_space_linear", 		&err_code);			if (err_code != CL_SUCCESS)  {cout << "\nError 'cvt_color_space_linear'  kernel not built.\n"	<<flush; exit_(0);   }
	comp_param_maps_kernel			= clCreateKernel(m_program, "compute_param_maps", 			&err_code);			if (err_code != CL_SUCCESS)  {cout << "\nError 'compute_param_maps'  kernel not built.\n"		<<flush; exit_(0);   }

	//  1st gen patch kernels ?
	rho_sq_kernel					= clCreateKernel(m_program, "Rho_sq",				 		&err_code);			if (err_code != CL_SUCCESS)  {cout << "\nError 'rho_sq_kernel'  kernel not built.\n"				<<flush; exit_(0);   }
	reduce_patch_Rho_kernel			= clCreateKernel(m_program, "reduce_patch_Rho",		 		&err_code);			if (err_code != CL_SUCCESS)  {cout << "\nError 'update_SE3'  kernel not built.\n"					<<flush; exit_(0);   }
	update_k2k_kernel				= clCreateKernel(m_program, "update_k2k",			 		&err_code);			if (err_code != CL_SUCCESS)  {cout << "\nError 'update_k2k'  kernel not built.\n"					<<flush; exit_(0);   }

	// RunCL_patchslam.cpp
	compute_patch_lookup_table_kernel	= clCreateKernel(m_program, "compute_patch_lookup_table",	&err_code);		if (err_code != CL_SUCCESS)  {cout << "\nError 'compute_patch_lookup_table'  kernel not built.\n"	<<flush; exit_(0);   }
	patch_img_grad_kernel				= clCreateKernel(m_program, "patch_img_grad",				&err_code);		if (err_code != CL_SUCCESS)  {cout << "\nError 'patch_img_grad'  kernel not built.\n"				<<flush; exit_(0);   }
	patch_hessian_reduce_kernel			= clCreateKernel(m_program, "patch_hessian_reduce",			&err_code);		if (err_code != CL_SUCCESS)  {cout << "\nError 'patch_hessian_reduce'  kernel not built.\n"			<<flush; exit_(0);   }

	// RunCL_patch_tracking.cpp
	pad_image_top_bottom2_kernel		= clCreateKernel(m_program, "pad_image_top_bottom2",		&err_code);		if (err_code != CL_SUCCESS)  {cout << "\nError 'pad_image_top5'  kernel not built.\n"				<<flush; exit_(0);   }
	vertcal_blur5_kernel				= clCreateKernel(m_program, "vertcal_blur5",				&err_code);		if (err_code != CL_SUCCESS)  {cout << "\nError 'vertcal_blur5_kernel'  kernel not built.\n"			<<flush; exit_(0);   }
	pad_image_left_right2_kernel		= clCreateKernel(m_program, "pad_image_left_right2",		&err_code);		if (err_code != CL_SUCCESS)  {cout << "\nError 'pad_image_left_right2'  kernel not built.\n"		<<flush; exit_(0);   }
	horiz_blur5_kernel					= clCreateKernel(m_program, "horiz_blur5",					&err_code);		if (err_code != CL_SUCCESS)  {cout << "\nError 'patch_hessian_reduce'  kernel not built.\n"			<<flush; exit_(0);   }
	reduce_img_kernel					= clCreateKernel(m_program, "reduce_img",					&err_code);		if (err_code != CL_SUCCESS)  {cout << "\nError 'reduce_img'  kernel not built.\n"					<<flush; exit_(0);   }

	// RunCL_depth.cpp
	update_depth_kernel					= clCreateKernel(m_program, "update_depth",					&err_code);		if (err_code != CL_SUCCESS)  {cout << "\nError 'update_depth'  kernel not built.\n"					<<flush; exit_(0);   }
	update_depth_2_kernel				= clCreateKernel(m_program, "update_depth_2",				&err_code);		if (err_code != CL_SUCCESS)  {cout << "\nError 'update_depth'  kernel not built.\n"					<<flush; exit_(0);   }
	regularize_depth_kernel				= clCreateKernel(m_program, "regularize_depth",				&err_code);		if (err_code != CL_SUCCESS)  {cout << "\nError 'regularize_depth'  kernel not built.\n"				<<flush; exit_(0);   }
	enlarge_layer_float_kernel			= clCreateKernel(m_program, "enlarge_layer_float",			&err_code);		if (err_code != CL_SUCCESS)  {cout << "\nError 'enlarge_layer_float'  kernel not built.\n"			<<flush; exit_(0);   }
}

void RunCL::initialize_fp32_params(){
	int local_verbosity_threshold = V_RUNCL_INITIALIZE_FP32_PARAMS;
																																			if(verbosity>local_verbosity_threshold) cout << "\n\nRunCL::initialize_fp32_params_chk_0,  \n" << flush;
	if ( obj["max_depth_infinity"].asBool()  ) { fp32_params[MIN_INV_DEPTH]	= 0;
	} else { fp32_params[MIN_INV_DEPTH]	=  1/obj["max_depth"].asFloat()		;   }

	fp32_params[MAX_INV_DEPTH]	=  1/obj["min_depth"].asFloat()		;																		// This works: Initialize 'params[]' from conf.json .
	fp32_params[INV_DEPTH_STEP]	=	 ( fp32_params[MAX_INV_DEPTH] - fp32_params[MIN_INV_DEPTH] ) /  uint_params[COSTVOL_LAYERS]	;
																																			if(verbosity>local_verbosity_threshold){ cout << "\n\nRunCL::initialize_fp32_params_finished,  "
																																				<<"fp32_params[MIN_INV_DEPTH] = "<<fp32_params[MIN_INV_DEPTH]<<"\n\n" << flush;
																																			}
}

void RunCL::initialize_patch_depthmap_offset(){
	int local_verbosity_threshold = V_RUNCL_INITIALIZE_PATCH_DEPTH_MAP_OFFSET;
																																			if(verbosity>local_verbosity_threshold) cout << "\n\nRunCL::initialize_patch_depthmap_offset_chk_0,  \n" << flush;
/*
																																			//uint	read_cols					= MipMap[ MiM_READ_COLS]/out_block_size;	// NB constexpr uint out_block_size	= OUT_BLOCK_SIZE	 4
	//uint	read_rows					= MipMap[ MiM_READ_ROWS]/out_block_size;
	//uint	margin						= uint_params[MARGIN];
*/
	uint	dm_margin							= obj["depthmap_margin"].asUInt();
/*
	//depthmap_params[0].DM_MARGIN		= depthmap_params[0].DM_MARGIN;
	// depthmap_params[0].DM_WIN_COLS		= read_cols + 2*dm_margin;
	// depthmap_params[0].DM_DATA_COLS		= read_cols;
	// depthmap_params[0].DM_WIN_ROWS		= read_rows + 2*dm_margin;
	// depthmap_params[0].DM_DATA_ROWS		= read_rows;
	// depthmap_params[0].DM_WIN_OFFSET	= 0;
	// depthmap_params[0].DM_DATA_OFFSET	= depthmap_params[0].DM_WIN_OFFSET	+ depthmap_params[0].DM_WIN_COLS + dm_margin;
	// depthmap_params[0].DM_WIN_BYTES		= depthmap_params[0].DM_WIN_COLS	* depthmap_params[0].DM_WIN_ROWS * sizeof(cl_float2);
 //
	// ///
	// // depth_save_offset[0] 				= 0;
	// // patch_depthmap_width[0]				= read_cols + 2*dm_margin;
	// // patch_depthmap_offset[0] 			= depth_save_offset[0] + patch_depthmap_width[0] + dm_margin;
	// 																																		if(verbosity>local_verbosity_threshold) cout << "\n\nRunCL::initialize_patch_depthmap_offset_chk_1, layer=0\n"
	// 																																			<<"\n depthmap_params[layer].DM_MARGIN="		<<depthmap_params[0].DM_MARGIN
	// 																																			<<"\n depthmap_params[layer].DM_WIN_COLS="		<<depthmap_params[0].DM_WIN_COLS
	// 																																			<<"\n depthmap_params[layer].DM_DATA_COLS="		<<depthmap_params[0].DM_DATA_COLS
	// 																																			<<"\n depthmap_params[layer].DM_WIN_ROWS="		<<depthmap_params[0].DM_WIN_ROWS
	// 																																			<<"\n depthmap_params[layer].DM_DATA_ROWS="		<<depthmap_params[0].DM_DATA_ROWS
	// 																																			<<"\n depthmap_params[layer].DM_WIN_OFFSET="	<<depthmap_params[0].DM_WIN_OFFSET
	// 																																			<<"\n depthmap_params[layer].DM_DATA_OFFSET="	<<depthmap_params[0].DM_DATA_OFFSET
	// 																																			<<"\n depthmap_params[layer].DM_WIN_BYTES="		<<depthmap_params[0].DM_WIN_BYTES
	// 																																			// <<"\n read_cols="					<<read_cols
	// 																																			// <<"\n read_rows="					<<read_rows
	// 																																			// <<"\n dm_margin="					<<dm_margin
	// 																																			// <<"\n depth_save_offset[0]="		<<depth_save_offset[0]
	// 																																			// <<"\n patch_depthmap_width[0]="		<<patch_depthmap_width[0]
	// 																																			// <<"\n patch_depthmap_offset[0]="	<<patch_depthmap_offset[0]
	// 																																			<< flush;
*/
	uint	win_offset_prev_layer				= 0;
	uint	tot_elem_prev_layer					= 0;
	for(uint layer=0; layer<max_mipmap_layers; layer++){
		uint read_cols							= MipMap[ layer*8 + MiM_READ_COLS]	/out_block_size;														cout<<" ,1"<<flush;
		uint read_rows							= MipMap[ layer*8 + MiM_READ_ROWS]	/out_block_size;														cout<<",2"<<flush;
		depthmap_params[layer].DM_MARGIN		= dm_margin;																								cout<<",3"<<flush;
		depthmap_params[layer].DM_WIN_COLS		= read_cols + 2*dm_margin;																					cout<<",4"<<flush;
		depthmap_params[layer].DM_DATA_COLS		= read_cols;																								cout<<",5"<<flush;
		depthmap_params[layer].DM_WIN_ROWS		= read_rows + 2*dm_margin;																					cout<<",6"<<flush;
		depthmap_params[layer].DM_DATA_ROWS		= read_rows;																								cout<<",7"<<flush;
		depthmap_params[layer].DM_WIN_OFFSET	= win_offset_prev_layer						+ tot_elem_prev_layer;											cout<<",8"<<flush;
		depthmap_params[layer].DM_DATA_OFFSET	= depthmap_params[layer].DM_WIN_OFFSET		+ depthmap_params[layer].DM_WIN_COLS	+ dm_margin;			cout<<",9"<<flush;
		depthmap_params[layer].DM_WIN_BYTES		= depthmap_params[layer].DM_WIN_COLS		* depthmap_params[layer].DM_WIN_ROWS	* sizeof(cl_float2);	cout<<",10"<<flush;
		tot_elem_prev_layer						= depthmap_params[layer].DM_WIN_COLS		* depthmap_params[layer].DM_WIN_ROWS;							cout<<",11"<<flush;
		win_offset_prev_layer					= depthmap_params[layer].DM_WIN_OFFSET;
/*		///
// 		uint	depthmap_width			= patch_depthmap_width[ (layer-1) ];
// 		uint	tot_elem_prev_layer		= (read_rows + 2*dm_margin )		* depthmap_width;
// 		depth_save_offset[ layer]		= depth_save_offset[ layer -1]		+ tot_elem_prev_layer;
// 		patch_depthmap_offset[ layer]	= depth_save_offset[layer]			+ patch_depthmap_width[layer] + dm_margin;
//
// 		read_cols						= MipMap[ layer*8 + MiM_READ_COLS]	/out_block_size;
// 		read_rows						= MipMap[ layer*8 + MiM_READ_ROWS]	/out_block_size;
// 		patch_depthmap_width[ layer ]	= read_cols + 2*dm_margin;
*/
																																			if(verbosity>local_verbosity_threshold) cout << "\n\nRunCL::initialize_patch_depthmap_offset_chk_2, layer="<<layer
																																				<<"\n depthmap_params[layer].DM_MARGIN="		<<depthmap_params[layer].DM_MARGIN
																																				<<"\n depthmap_params[layer].DM_WIN_COLS="		<<depthmap_params[layer].DM_WIN_COLS
																																				<<"\n depthmap_params[layer].DM_DATA_COLS="		<<depthmap_params[layer].DM_DATA_COLS
																																				<<"\n depthmap_params[layer].DM_WIN_ROWS="		<<depthmap_params[layer].DM_WIN_ROWS
																																				<<"\n depthmap_params[layer].DM_DATA_ROWS="		<<depthmap_params[layer].DM_DATA_ROWS
																																				<<"\n depthmap_params[layer].DM_WIN_OFFSET="	<<depthmap_params[layer].DM_WIN_OFFSET
																																				<<"\n depthmap_params[layer].DM_DATA_OFFSET="	<<depthmap_params[layer].DM_DATA_OFFSET
																																				<<"\n depthmap_params[layer].DM_WIN_BYTES="		<<depthmap_params[layer].DM_WIN_BYTES
																																				/*
																																				// <<"\n depthmap_width="						<<depthmap_width
																																				// <<"\n tot_elem_prev_layer="					<<tot_elem_prev_layer
																																				// <<"\n depth_save_offset[ layer]="			<< depth_save_offset[ layer]
																																				// <<"\n patch_depthmap_offset[ layer]("		<< patch_depthmap_offset[ layer]
																																				// <<"\n read_cols="							<< read_cols
																																				// <<"\n read_rows="							<< read_rows
																																				// <<"\n patch_depthmap_width[ layer ]="		<< patch_depthmap_width[ layer ]
																																				*/
																																				<< flush;
		// ### TODO set size for depth_temp buffer NB check where it is used. Ensure it is always large enough.
		// NB the final "mipmap" layer is a dummy for offsetting regularized depthmas from the raw depthmaps.
	}
}

void RunCL::initialize_RunCL(cv::Mat baseImage_){
	int local_verbosity_threshold = V_RUNCL_INITIALIZE_RUNCL;
																																			if(verbosity>local_verbosity_threshold) cout << "\n\nRunCL::initialize_RunCL_chk_0\n\n" << flush;
	baseImage =  baseImage_;
																																	if( baseImage.empty() ){cerr <<"\nError RunCL::initialize() : runcl.baseImage.empty()"<<flush; exit_(0); }
																																			if (verbosity>local_verbosity_threshold) {
																																				cout << "\n"
																																				<< "RunCL::initialize_RunCL_chk_1: runcl.baseImage.size() = "<< baseImage.size() \
																																				<<" runcl.baseImage.type() = \"" << baseImage.type() << "\" = "<< checkCVtype(baseImage.type()) <<flush;
																																			}
																																	if( baseImage.type() != CV_8UC3 ) {  //
																																		cerr << "RunCL::initialize_RunCL(cv::Mat baseImage_) "
																																		<<"Error: ( baseImage.type() != CV_8UC3 ),  NB current image loading kernels depend on CV_8UC3 input."
																																		<<"runcl.baseImage.type() = \"" << baseImage.type() << "\" = "<< checkCVtype(baseImage.type()) <<flush;
																																		exit_(EXIT_FAILURE);
																																	}
																																			//if(verbosity>1) { imshow("runcl.baseImage",baseImage); cv::waitKey(-1); }
	image_size_bytes	= baseImage.total() * baseImage.elemSize();																			// Constant parameters of the base image
	image_size_bytes_C1	= baseImage.total() * sizeof(float);
	costVolLayers 		=( 1 + obj["layers"].asUInt() ); // TO DO  2;
	baseImage_size 		= baseImage.size();
	baseImage_type 		= baseImage.type();
	baseImage_width		= baseImage.cols;
	baseImage_height	= baseImage.rows;
	layerstep 			= baseImage_width * baseImage_height;

	uint num_reductions_width		= log2(baseImage_width/5);
	uint num_reductions_height		= log2(baseImage_height/5);
	mm_num_reductions				= min(num_reductions_width, num_reductions_height);														if(verbosity>local_verbosity_threshold){ cout << "\nRunCL::initialize_RunCL_chk0.4"
																																				<< "\nnum_reductions_width="<<num_reductions_width
																																				<<",  num_reductions_height="<<num_reductions_height
																																				<<", mm_num_reductions="<<mm_num_reductions	<<flush;
																																			}
																																			// Constant parameters of the mipmap, (as opposed to per-layer mipmap_buf)
	if (mm_num_reductions >= max_mipmap_layers) {
		cout << "\n\n BEWARE (mm_num_reductions="<<mm_num_reductions<<" >= max_mipmap_layers="<<max_mipmap_layers<<")  #############\n\n"<<flush;
		cerr << "\n\n BEWARE (mm_num_reductions="<<mm_num_reductions<<" >= max_mipmap_layers="<<max_mipmap_layers<<")  #############\n\n"<<flush;
		mm_num_reductions = max_mipmap_layers-1;
	}
	uint apexImage_width	= baseImage_width	/ pow(2, mm_num_reductions);
	uint apexImage_height	= baseImage_height	/ pow(2, mm_num_reductions);
																																			if(verbosity>local_verbosity_threshold){ cout << "\nRunCL::initialize_RunCL_chk0.4"
																																				<< "\nmm_num_reductions = "<<mm_num_reductions
																																				<<",  apexImage_width = "<<apexImage_width
																																				<<",  apexImage_height = "<<apexImage_height<<flush;
																																			}
	mm_start			= 0;
	float short_side	= fmin(baseImage_width, baseImage_height);
	uint num_reductions	= floor(log2(short_side)) -2;																						// i.e. at least 4 pixels remain on short side of img at apex of img pyramid.
//	if( (short_side/pow(2,num_reductions)) < 6) num_reductions--;																			// i.e. at least 6 pixels remain on short side of img at apex of img pyramid.
//	mm_stop				= num_reductions; /*mm_num_reductions;// + mm_num_blur_layers;*/
	mm_stop				= min(num_reductions, max_mipmap_layers-2);
																																			if(verbosity>local_verbosity_threshold){ cout << "\nRunCL::initialize_RunCL_chk0.5"
																																						<<",  mm_start="<<mm_start
																																						<<",  mm_stop="<<mm_stop
																																						<<",  short_side="<<short_side
																																						<<",  log2(short_side)="<<log2(short_side)
																																						<<" \n" << flush;
																																			}
	mm_gaussian_size	= obj["gaussian_size"].asUInt();
	mm_margin			= obj["MipMap_margin"].asUInt() * mm_num_reductions;
	mm_width 			= baseImage_width  + 2 * mm_margin;
	mm_height 			= baseImage_height * 2.1  + 2 * mm_margin;  // 1.5
	mm_layerstep		= mm_width * mm_height;

	cv::Mat temp(mm_height, mm_width, CV_32FC3);
	mm_Image_size		= temp.size();
	mm_Image_type		= temp.type();
	mm_size_bytes_C3	= temp.total() * temp.elemSize() ;																					// for mipmaps with CV_16FC3  mm_width*mm_height*fp16_size;// for FP16 'half', or BF16 on Tensor cores
	mm_size_bytes_C4	= temp.total() * 4 * sizeof(float);
	mm_size_bytes_C8	= temp.total() * 8 * sizeof(float);
	cv::Mat temp2(mm_height, mm_width, CV_32FC1);
	mm_size_bytes_C1	= temp.total()	   * sizeof(float);			//temp2.total() * temp2.elemSize(); // NB elemSize() -> size bytes _per_ channel.
	mm_vol_size_bytes	= mm_size_bytes_C1 * costVolLayers;
																																			if(verbosity>local_verbosity_threshold) cout << "\n\nRunCL::initialize_RunCL_chk1  "
																																				<<"\nmm_gaussian_size="<<mm_gaussian_size
																																				<<"\nmm_Image_size="<<mm_Image_size
																																				<<"\ntemp.total()="<<temp.total()
																																				<<"\nmm_size_bytes_C3="<<mm_size_bytes_C3
																																				<<"\nmm_size_bytes_C4="<<mm_size_bytes_C4
																																				<<"\nmm_size_bytes_C1="<<mm_size_bytes_C1
																																				<<" \n\n" << flush;
																																			// Get the maximum work group size for executing the kernel on the device ///////
																																			// From https://github.com/rsnemmen/OpenCL-examples/blob/e2c34f1dfefbd265cfb607c2dd6c82c799eb322a/square_array/square.c
	cl_int 				status;
	status = clGetKernelWorkGroupInfo(cvt_color_space_linear_kernel, deviceId, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local_work_size), &local_work_size, NULL); 										if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; exit_(status);}
	status = clGetKernelWorkGroupInfo(cvt_color_space_linear_kernel, deviceId, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(kernel_work_size_multiple), &kernel_work_size_multiple, NULL); 	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; exit_(status);}
	status = clGetDeviceInfo( deviceId, CL_DEVICE_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(device_work_size_multiple), &device_work_size_multiple, NULL); 										if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; exit_(status);}
/*
	// cl_int clGetDeviceInfo(
	// 	cl_device_id 	device,
	// 	cl_device_info 	param_name,
	// 	size_t 			param_value_size,
	// 	void* 			param_value,
	// 	size_t* 		param_value_size_ret
	// );
*/
																																			// Number of total work items, calculated here after 1st image is loaded &=> know the size.
																																			// NB localSize must be devisor
																																			// NB global_work_size must be a whole number of "Preferred work group size multiple" for Nvidia.
																																			// i.e. global_work_size should be slightly more than the number of point or pixels to be processed.
	global_work_size 	= ceil( (float)layerstep/(float)local_work_size ) * local_work_size;
	mm_global_work_size = ceil( (float)mm_layerstep/(float)local_work_size ) * local_work_size;
																																			if(verbosity>local_verbosity_threshold){
																																				cout << "\n\nRunCL::initialize_chk1.2,\t global_work_size="<<global_work_size<<",\t mm_global_work_size="<<mm_global_work_size <<"\n" << flush;
																																				cout << "layerstep="<<layerstep <<",\t mm_layerstep"<< mm_layerstep<<"\n\n" << flush;
																																				cout<<"\nglobal_work_size="<<global_work_size<<", local_work_size="<<local_work_size<<", deviceId="<<deviceId<<"\n"<<flush;
																																				cout<<"\nlayerstep=mm_width*mm_height="<<mm_width<<"*"<<mm_height<<"="<<layerstep<<",\tsizeof(layerstep)="<< sizeof(layerstep) <<",\tsizeof(int)="<< sizeof(int) <<flush;
																																				cout<<"\n";
																																				cout<<"\nRunCL::initialize, baseImage.total()=" << baseImage.total() << ", sizeof(float)="<< sizeof(float)<<flush;
																																				cout<<"\nbaseImage.elemSize()="<< baseImage.elemSize()<<", baseImage.elemSize1()="<<baseImage.elemSize1()<<flush;
																																				cout<<"\nbaseImage.type()="<< baseImage.type() <<", sizeof(baseImage.type())="<< sizeof(baseImage.type())<<flush;
																																				cout<<"\n";
																																				cout<<"\nRunCL::initialize, image_size_bytes="<< image_size_bytes <<  ", sizeof(float)="<< sizeof(float)<<flush;
																																				cout<<"\n";

																																				cout<<"\n"<<", mm_margin="     << mm_margin       <<", mm_width ="     <<  mm_width       <<flush;  //  ", fp16_size ="<< fp16_size   <<
																																				cout<<"\n"<<", mm_height ="<< mm_height   <<", mm_Image_size ="<<  mm_Image_size  <<", mm_Image_type ="<< mm_Image_type   <<flush;
																																				cout<<"\n"<<", mm_size_bytes_C1="<< mm_size_bytes_C1  <<", mm_size_bytes_C3="<< mm_size_bytes_C3 <<", mm_size_bytes_C4="<< mm_size_bytes_C4 << ", mm_size_bytes_C8="<< mm_size_bytes_C8 <<", mm_vol_size_bytes ="<<  mm_vol_size_bytes  <<flush;
																																				cout<<"\n";
																																				cout<<"\n"<<", temp.elemSize() ="<< temp.elemSize()   <<", temp2.elemSize()="<< temp2.elemSize() <<flush;
																																				cout<<"\n"<<", temp.total() ="<< temp.total()         <<", temp2.total()="   << temp2.total()    <<flush;

																																			}
																																			if(verbosity>local_verbosity_threshold) cout <<"\n\nRunCL::initialize_RunCL_chk3.8\n\n" << flush;
	uint_params[PIXELS]			= 	baseImage_height * baseImage_width ;
	uint_params[ROWS]			= 	baseImage_height ;
	uint_params[COLS]			= 	baseImage_width ;
	uint_params[COSTVOL_LAYERS]	= 	obj["layers"].asUInt() ;
	uint_params[MARGIN]			= 	mm_margin ;
	uint_params[MM_PIXELS]		= 	mm_height * mm_width ;
	uint_params[MM_ROWS]		= 	mm_height ;
	uint_params[MM_COLS]		= 	mm_width ;

	initialize_fp32_params();																												// Requires uint_params[COSTVOL_LAYERS]	;
																																			if(verbosity>local_verbosity_threshold) cout <<"\n\nRunCL::initialize_RunCL_chk3.9\n\n" << flush;
																																			if(verbosity>local_verbosity_threshold){
																																				cout << "\n\nRunCL::initialize  Checking fp32_params[]" << flush;
																																				cout << "\nfp32_params[0 MAX_INV_DEPTH]="	<<fp32_params[MAX_INV_DEPTH]		<<"\t\t1/obj[\"min_depth\"].asFloat()="	<<1/obj["min_depth"].asFloat();
																																				cout << "\nfp32_params[1 MIN_INV_DEPTH]="	<<fp32_params[MIN_INV_DEPTH]		<<"\t\t1/obj[\"max_depth\"].asFloat()="	<<1/obj["max_depth"].asFloat();
																																				cout << "\nfp32_params[2 INV_DEPTH_STEP]="	<<fp32_params[INV_DEPTH_STEP];
																																				cout << "\n" << flush;
																																			}
	set_mimpmap_offsets();
																																			if(verbosity>local_verbosity_threshold) {
																																				cout << "\n";
																																				cout << ",mm_Image_size = " << mm_Image_size << endl;
																																				cout << ",mm_Image_type = "	<< mm_Image_type << endl;
																																				cout << ",mm_size_bytes_C3 = " << mm_size_bytes_C3 << endl;
																																				cout << ",mm_size_bytes_C4 = " << mm_size_bytes_C4 << endl;
																																				cout << ",mm_size_bytes_C1 = " << mm_size_bytes_C1 << endl;
																																				cout << "\n";
																																				cout << ",baseImage_size, = " << baseImage_size << endl;
																																				cout << ",baseImage_type = " << baseImage_type << endl;
																																				cout << ",image_size_bytes = " << image_size_bytes	<< endl;
																																				cout << ",mm_vol_size_bytes = " << mm_vol_size_bytes << endl;
																																				cout << "\n" << flush;
																																			}
																																			// Summation buffer sizes
	se3_sum_size 			= 1 + ceil( (float)(MipMap[(mm_num_reductions+1)*8 + MiM_READ_OFFSET]) / (float)local_work_size ) ;				// i.e. num workgroups used = MiM_READ_OFFSET for 1 layer more than used / local_work_size,   will give one row of vector per group.
	se3_sum_size 			*= 2;  																											// *2 incr num grps for reduced groupsize
	uint num_DoFs			= 6 ; 																											// 6 DoF of float4 channels, + 1 DoF to compute global Rho.
	se3_sum_size_bytes		= se3_sum_size * sizeof(float) * 4 * num_DoFs ;																	if(verbosity>local_verbosity_threshold) cout <<"\n\n se3_sum_size="<< se3_sum_size<<",    se3_sum_size_bytes="<<se3_sum_size_bytes<<flush;
	se3_sum2_size_bytes 	= 2 * mm_num_reductions * sizeof(float) * 4 * num_DoFs;															// NB the data returned is 6xfloat4 per group, holding one float4 per 6DoF of SE3, where alpha channel=pixel count.
	se3_sum2_size_bytes 	= ((se3_sum2_size_bytes%32) + 1) * 32;																			// Needed for Nvidia, to ensure memory allocations are multiples of 32bytes.

	so3_sum_size_bytes		= se3_sum_size_bytes / 2;
	so3_sum_size			= se3_sum_size ;

	pix_sum_size			= se3_sum_size;
	pix_sum_size_bytes		= pix_sum_size * sizeof(float) * 4;																				// NB the data returned is one float4 per group, for the base image, holding hsv channels plus entry[3]=pixel count.
																																			if(verbosity>local_verbosity_threshold) cout <<"\nRunCL::initialize_RunCL_chk finished -1 ############################################################\n"<<flush;
	allocatemem();																													// Allocate buffers on the GPU ######
	initialize_patch_depthmap_offset();
	initialize_patch_params();
	compute_patch_lookup_table();
																																			if(verbosity>local_verbosity_threshold){ cout <<"\nRunCL::initialize_RunCL_chk finished -0.5 ############################################################\n"<<flush;
																																				for(uint layer = 0; layer <= mm_stop; layer++) {
																																					cout <<"\npatch_local_work_size["<<layer<<"] = "<<patch_local_work_size[layer]<< flush;
																																				}
																																			}
	patch_img_gradients_set_params();		//TO DO set in .conf file,  uint out_block_sizefor ST3_hessian		// will need a runcl.set_patch_kernels_params() function
																																			if(verbosity>local_verbosity_threshold) cout <<"\nRunCL::initialize_RunCL_chk finished ############################################################\n"<<flush;
}


void RunCL::set_mimpmap_offsets(){
	int local_verbosity_threshold = V_RUNCL_INITIALIZE_RUNCL;
	// #### Set offsets for the mipmap of img pyr + blur layers ####################################################################################################################################################################
																																			if(verbosity>local_verbosity_threshold) {
																																				cout <<"	\nRunCL::set_mimpmap_offsets()"<<endl;
																																				cout <<"	#define MiM_PIXELS			0	// for mipmap_buf, 				when launching one kernel per layer. 	Updated for each layer."<<endl;
																																				cout <<"	#define MiM_READ_OFFSET		1	// for ths layer, 				start of image data"<<endl;
																																				cout <<"	#define MiM_WRITE_OFFSET	2"<<endl;
																																				cout <<"	#define MiM_READ_COLS		3	// cols without margins"<<endl;
																																				cout <<"	#define MiM_WRITE_COLS		4"<<endl;
																																				///cout <<"	#define MiM_GAUSSIAN_SIZE	5	// filter box size"<<endl;
																																				cout <<"	#define MiM_READ_ROWS		6	// rows without margins"<<endl;
																																				cout <<"	#define MiM_WRITE_ROWS		7"<<endl;
																																				cout <<"	"<<endl;
																																			}
	uint 							mipmap[8];
	mipmap[MiM_READ_ROWS] 			= baseImage_height;
	uint write_rows 				= mipmap[MiM_READ_ROWS] /2;
	mipmap[MiM_WRITE_ROWS]			= write_rows;
	uint margin						= mm_margin;
	uint read_cols_with_margin 		= mm_width ;
	uint read_rows_with_margin		= mipmap[MiM_READ_ROWS] + margin;
	mipmap[MiM_READ_OFFSET]			= margin*mm_width + margin;
	mipmap[MiM_WRITE_OFFSET]		= read_cols_with_margin * read_rows_with_margin + mipmap[MiM_READ_OFFSET];
	mipmap[MiM_READ_COLS]			= baseImage_width;
	mipmap[MiM_WRITE_COLS]			= mipmap[MiM_READ_COLS]/2;
	mipmap[MiM_PIXELS]				= mipmap[MiM_READ_COLS] * mipmap[MiM_READ_ROWS];
																																			// TO DO compute required reduction and blur depending on img size
																													// #### Img pyr layers ##########
	int reduction = 0;
	num_threads[reduction]		= ceil( (float)(mipmap[MiM_PIXELS])/(float)local_work_size ) * local_work_size ;							// global_work_size formula for num_treads req for this layer.
	for (int i=0; i<8; i++)		{																											// Initialize the global MipMap[8*8] array.
		MipMap[reduction*8 +i]	= mipmap[i];																								if(verbosity>local_verbosity_threshold) { cout << "\nMipMap["<<reduction<<"*8 +"<<i<<"]="<<MipMap[reduction*8 +i] ;}
	}																																		if(verbosity>local_verbosity_threshold) { cout << endl << flush; }

	for(; reduction <= mm_stop/*stop1*/; reduction++) {
		mipmap[MiM_READ_OFFSET]		= mipmap[MiM_WRITE_OFFSET];
		mipmap[MiM_WRITE_OFFSET]	= mipmap[MiM_WRITE_OFFSET] + read_cols_with_margin * (margin + write_rows);
		mipmap[MiM_READ_ROWS]		= write_rows;
		write_rows					= write_rows/2;
		mipmap[MiM_WRITE_ROWS]		= write_rows;
		mipmap[MiM_READ_COLS]		= mipmap[MiM_WRITE_COLS];
		mipmap[MiM_WRITE_COLS]		= mipmap[MiM_WRITE_COLS]/2;
		mipmap[MiM_PIXELS]			= mipmap[MiM_READ_COLS] * mipmap[MiM_READ_ROWS];

		num_threads[reduction+1]	= ceil( (float)(mipmap[MiM_PIXELS])/(float)local_work_size ) * local_work_size ;						// global_work_size formula for num_treads req for this layer.
		for (int i=0; i<8; i++)		{																										// Initialize the global MipMap[8*8] array.
			MipMap[(reduction+1)*8 +i]	= mipmap[i];																						if(verbosity>local_verbosity_threshold) { cout << "\nMipMap["<<reduction<<"*8 +"<<i<<"]="<<MipMap[reduction*8 +i] ;}
		}																																	if(verbosity>local_verbosity_threshold) { cout << endl << flush; }

	}
																																			if(verbosity>local_verbosity_threshold) {
																																				cout <<"	\nRunCL::set_mimpmap_offsets()"<<endl;
																																				cout << "\n\nImg pyr layers"<<flush;
																																				for(int reduction = 0; reduction < max_mipmap_layers; reduction++) {
																																					cout << "\n\n reduction = " 		<< reduction;
																																					cout << "\n MiM_PIXELS = " 			<< MipMap[reduction*8 +MiM_PIXELS];
																																					cout << "\n MiM_READ_OFFSET = " 	<< MipMap[reduction*8 +MiM_READ_OFFSET] ;
																																					cout << "\n MiM_WRITE_OFFSET = " 	<< MipMap[reduction*8 +MiM_WRITE_OFFSET];
																																					cout << "\n MiM_READ_COLS = " 		<< MipMap[reduction*8 +MiM_READ_COLS];
																																					cout << "\n MiM_WRITE_COLS = " 		<< MipMap[reduction*8 +MiM_WRITE_COLS];
																																					//cout << "\n MiM_GAUSSIAN_SIZE = " 	<< MipMap[reduction*8 +MiM_GAUSSIAN_SIZE];
																																					cout << "\n MiM_READ_ROWS = " 		<< MipMap[reduction*8 +MiM_READ_ROWS];
																																					cout << "\n MiM_WRITE_ROWS = " 		<< MipMap[reduction*8 +MiM_WRITE_ROWS];
																																					cout << "\n offset row = "			<< MipMap[reduction*8 +MiM_READ_OFFSET] / mm_width;
																																					cout << "\n offset col = "			<< MipMap[reduction*8 +MiM_READ_OFFSET] % mm_width;

																																					if(reduction == mm_stop ) { cout << "\n\nunused layers"<<flush;}
																																				}
																																			}
	// ## set array of arrays for workgoup offsets, for patch kernels on mipmaps ########################################################### TO DO Replace with fixed arrays capable of 10K images
	// Allocate array of arrays, and set counter array.
	uint num_levels		= mm_num_reductions;

	for (int iter = 0; iter<num_levels; iter ++){									// #### Image pyramid ###############
		int wg_cols			= ceil((float)MipMap[ iter*8 +  MiM_READ_COLS] / (float)local_work_size);
		int wg_rows			= ceil((float)MipMap[ iter*8 +  MiM_READ_ROWS] / (float)patch_size);

		wg_counter[iter]	=  wg_cols * wg_rows;
		//wg_offsets[iter]	= (uint*)calloc( wg_counter[iter], sizeof(uint) );

		int iter2 			= 0;
		int offset			= MipMap[ iter*8 +  MiM_READ_OFFSET];
		for (int row=0; row<wg_rows ; row++){
			offset			+= patch_size * mm_width;
			for (int col=0; col<wg_cols; col++){
				wg_offsets[iter][iter2]			= offset + local_work_size * col;		// offset for each workgroup, assuming no wrapping. NB Wrapping would req lookup table.
			}
		}
	}

}

void RunCL::set_cam_bufs( cv::Matx44f k,  cv::Matx44f inv_k,  cv::Matx44f pose,  cv::Matx44f k2k ){
	int local_verbosity_threshold = V_RUNCL_SET_CAM_BUFS;
	string fname = "RunCL::set_cam_bufs( )";
																																			if(verbosity>local_verbosity_threshold) { cout<<"\n"<<fname<<"(  )_chk0"<<flush; }
																																			// NB Orthographic camera, See notes in convertTransforms.cpp , cv::Matx44f generate_invK_(cv::Matx44f K_, int verbosity){..}
																																			// 4x4 perspective matrix is not invertable for points at infinity. We correct ortho->perspective in the kernel by dividing by Z.
	float k_arry[16], inv_k_arry[16], pose_arry[16], k2k_arry[16];
	Matx44f_To_float16arry( k,		k_arry		);																							if(verbosity>local_verbosity_threshold) { cout<<"\n"<<fname<< endl; PRINT_FLOAT_16(k_arry, ) }
	Matx44f_To_float16arry( inv_k,	inv_k_arry	);
	Matx44f_To_float16arry( pose,	pose_arry	);
	Matx44f_To_float16arry( k2k,	k2k_arry	);

	current_frames[ current_frames_idx[0] ].K		= k;
	current_frames[ current_frames_idx[0] ].inv_K	= inv_k;

	_clEnqueueWriteBuffer( uload_queue, 	K_buf,		CL_FALSE, 0, 16 * sizeof( float), k_arry, 			fname);
	_clEnqueueWriteBuffer( uload_queue, 	inv_K_buf,	CL_FALSE, 0, 16 * sizeof( float), inv_k_arry, 		fname);
																																			if(verbosity>local_verbosity_threshold) {cout<<"\nRunCL::"<<fname<<"(  )_chk1"<<flush;
																																				PRINT_MATX44F(k,);
																																				PRINT_MATX44F(inv_k,);
																																				PRINT_MATX44F(pose,);
																																				PRINT_MATX44F(k2k,);
																																			}
	update_k2k_buf( k2k_arry, pose_arry );
																																			if(verbosity>local_verbosity_threshold) {cout<<"\nRunCL::"<<fname<<"(  )_chk2"<<flush;
																																				float k_buf_arr[16];
																																				void * ptr = k_buf_arr;
																																				ReadOutput( (uchar*)ptr, K_buf, sizeof(float)*16, 0 );
																																				PRINT_FLOAT_16(k_buf_arr, )
																																				cout<<"\n"<<fname<<"_finished"<<flush;
																																			}
}


void RunCL::mipmap_call_kernel(cl_kernel kernel_to_call, cl_command_queue queue_to_call, uint start, uint stop, bool layers_sequential, const size_t local_work_size){
	int local_verbosity_threshold = V_RUNCL_MIPMAP_CALL_KERNEL;//verbosity_mp["RunCL::mipmap_call_kernel"];// -2;
																																			if(verbosity>local_verbosity_threshold) {
																																				cout<<"\nRunCL::mipmap_call_kernel( cl_kernel "	<<kernel_to_call
																																				<<",  cl_command_queue "						<<queue_to_call
																																				<<",  start="									<<start
																																				<<",  stop="									<<stop
																																				<<",  layers_sequential="						<<layers_sequential
																																				<<",  local_work_size="							<<local_work_size
																																				<<" )_chk0"<<flush;
																																				//cout <<"\nmm_num_reductions+1="<<mm_num_reductions+1<< ",  start="<<start<<",  stop="<<stop <<flush;
																																			}
	cl_event						ev;
	cl_int							res, status;
	for(uint reduction = start; reduction <= stop; reduction++) {																			// NB processes largest layer first.
																																			if(verbosity>local_verbosity_threshold) { cout<<"\nRunCL::mipmap_call_kernel(..)_chk1,  reduction="<<reduction<<",  num_threads[reduction]="<<num_threads[reduction]<<"  local_work_size="<<local_work_size<<flush; }
		//if (reduction>=start && reduction<stop){																							// compute num threads to launch & num_pixels in reduction
			res 	= clSetKernelArg(kernel_to_call, 0, sizeof(int), &reduction);							if (res    !=CL_SUCCESS)	{ cout <<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	;
			res 	= clEnqueueNDRangeKernel(queue_to_call, kernel_to_call, 1, 0, &num_threads[reduction], &local_work_size, 0, NULL, &ev); // run mipmap_float4_kernel, NB wait for own previous iteration.
																											if (res    != CL_SUCCESS)	{ cout << "\nres = " << checkerror(res) <<"\n"<<flush; exit_(res);}
			status 	= clFlush(queue_to_call);	status 	= clFinish(queue_to_call);															if (status != CL_SUCCESS)	{ cout << "\nRunCL::mipmap_call_kernel( cl_kernel "<<kernel_to_call<<",  clFlush(queue_to_call) status  = "		<<status<<" "<< checkerror(status) <<"\n"<<flush; exit_(status);}
			if (layers_sequential==true) status 	= clWaitForEvents (1, &ev);								if (status != CL_SUCCESS)	{ cout << "\nRunCL::mipmap_call_kernel( cl_kernel "<<kernel_to_call<<") for loop,  clWaitForEventsh(1, &ev) ="	<<status<<" "<<checkerror(status)  <<"\n"<<flush; exit_(status);}

		//} 																							// TO DO execute layers in asynchronous parallel. i.e. relax clWaitForEvents.
	}if (layers_sequential==false) status 	= clWaitForEvents (1, &ev);										if (status != CL_SUCCESS)	{ cout << "\nRunCL::mipmap_call_kernel( cl_kernel "<<kernel_to_call<<") final,  clWaitForEventsh(1, &ev) ="		<<status<<" "<<checkerror(status)  <<"\n"<<flush; exit_(status);}
}


void RunCL::allocatemem(){
	int local_verbosity_threshold = V_RUNCL_ALLOCATEMEM;//verbosity_mp["RunCL::allocatemem"];// 0;
																																			if(verbosity>local_verbosity_threshold) cout <<"\n\nRunCL::allocatemem()_chk0\n"<<flush;
	stringstream 	ss;
	ss 				<< "allocatemem";
	cl_int 			status;
	cl_event 		writeEvt;
	cl_int 			res;

	for (uint i=0; i<num_current_frames; i++ ) {
		imgmem[i]		= clCreateBuffer(m_context, CL_MEM_READ_WRITE  						, mm_size_bytes_C4,  		0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 1= "<<checkerror(res)<<"\n"<<flush;exit_(res);}
		velmap[i]		= clCreateBuffer(m_context, CL_MEM_READ_WRITE  						, mm_size_bytes_C4,  		0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 1= "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	}
																																			if(verbosity>local_verbosity_threshold){ cout <<"\nRunCL::allocatemem()_chk1"<<flush;
																																				cout <<"\nmm_size_bytes_C4 = "<<mm_size_bytes_C4<<flush;
																																				for (uint i=0; i<num_current_frames; i++ ) {
																																					cout<<"\nimgmem["<<i<<"] = "<<imgmem[i]<<flush;
																																				}
																																			}
	initialize_current_frames();
	// test_update_current_frames_idx(64);	// NB Only for debugging.

	imgmem_blurred		= clCreateBuffer(m_context, CL_MEM_READ_WRITE  						, mm_size_bytes_C4,  		0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 1= "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	SE3_grad_map_mem 	= clCreateBuffer(m_context, CL_MEM_READ_WRITE 		,num_SE3_DoF *	  mm_size_bytes_C4,			0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 7= "<<checkerror(res)<<"\n"<<flush;exit_(res);} // SE3_map * img grad, 6DoF*8channels=48     // 6DoF*3channels=18,but 4*6=24 because hsv img gradient is held in float4
	SE3_weight_map_mem	= clCreateBuffer(m_context, CL_MEM_READ_WRITE 						, mm_size_bytes_C4,			0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 9= "<<checkerror(res)<<"\n"<<flush;exit_(res);}

	SE3_incr_map_mem	= clCreateBuffer(m_context, CL_MEM_READ_WRITE 					, 2 * mm_size_bytes_C1,			0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 9= "<<checkerror(res)<<"\n"<<flush;exit_(res);} // For debugging before summation.
	SE3_map_mem			= clCreateBuffer(m_context, CL_MEM_READ_WRITE 		,num_SE3_DoF *2	* mm_size_bytes_C1,			0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 10= "<<checkerror(res)<<"\n"<<flush;exit_(res);}	// (row, col) increment fo each parameter.
	basemem				= clCreateBuffer(m_context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, image_size_bytes,			0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 11= "<<checkerror(res)<<"\n"<<flush;exit_(res);} // Original image CV_8UC3
	depth_mem_temp		= clCreateBuffer(m_context, CL_MEM_READ_WRITE 						, mm_size_bytes_C1,			0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 12= "<<checkerror(res)<<"\n"<<flush;exit_(res);} // Used to be : Copy used by tracing & auto-calib. Now spare buffer for upload & computations
	depth_mem_GT		= clCreateBuffer(m_context, CL_MEM_READ_WRITE /*2*mm_size_bytes_C1*/, mm_size_bytes_C1,			0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 13= "<<checkerror(res)<<"\n"<<flush;exit_(res);} // Where depthmap GT mimpap is constructed.

	img_grad_mem		= clCreateBuffer(m_context, CL_MEM_READ_WRITE 					, 2 * mm_size_bytes_C1, 		0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 16= "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	g1mem				= clCreateBuffer(m_context, CL_MEM_READ_WRITE 						, mm_size_bytes_C8, 		0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 16= "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	depth_mem			= clCreateBuffer(m_context, CL_MEM_READ_WRITE 						, mm_size_bytes_C1,			0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 17= "<<checkerror(res)<<"\n"<<flush;exit_(res);}

	fp32_param_buf		= clCreateBuffer(m_context, CL_MEM_READ_ONLY  					, 16* sizeof(float),			0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 25= "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	uint_param_buf		= clCreateBuffer(m_context, CL_MEM_READ_ONLY  					, 8 * sizeof(uint), 			0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 29= "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	mipmap_buf			= clCreateBuffer(m_context, CL_MEM_READ_ONLY  					, 8*8*sizeof(uint), 			0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 30= "<<checkerror(res)<<"\n"<<flush;exit_(res);}

	k2kbuf				= clCreateBuffer(m_context, CL_MEM_READ_ONLY  ,	 num_current_frames*16*sizeof(float),			0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 26= "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	SE3_k2kbuf			= clCreateBuffer(m_context, CL_MEM_READ_ONLY  ,max_mipmap_layers*num_SE3_DoF*16*sizeof(float),	0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 28= "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	cur_frames_k2kbuf	= clCreateBuffer(m_context, CL_MEM_READ_ONLY  ,	 num_current_frames*16*sizeof(float),			0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 28= "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	cur_frames_st3buf	= clCreateBuffer(m_context, CL_MEM_READ_ONLY  ,	 num_current_frames* 4*sizeof(float),			0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 28= "<<checkerror(res)<<"\n"<<flush;exit_(res);}


	SE3_rho_map_mem		= clCreateBuffer(m_context, CL_MEM_READ_WRITE  , tracking_num_samples*2*mm_size_bytes_C4,  		0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 34= "<<checkerror(res)<<"\n"<<flush;exit_(res);}

	img_stats_buf		= clCreateBuffer(m_context, CL_MEM_READ_WRITE  						, img_stats_size_bytes,		0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 36= "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	pix_sum_mem			= clCreateBuffer(m_context, CL_MEM_READ_WRITE 						, pix_sum_size_bytes,		0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 37= "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	var_sum_mem			= clCreateBuffer(m_context, CL_MEM_READ_WRITE 						, pix_sum_size_bytes,		0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 38= "<<checkerror(res)<<"\n"<<flush;exit_(res);}

	HSV_grad_mem		= clCreateBuffer(m_context, CL_MEM_READ_WRITE  						, mm_size_bytes_C8,  		0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 39= "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	ST3_img_grad_mem	= clCreateBuffer(m_context, CL_MEM_READ_WRITE  						,3* mm_size_bytes_C4,  		0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 39= "<<checkerror(res)<<"\n"<<flush;exit_(res);}

	// buffers for patch kernel based Dynamic_slam
	pose_buf						= clCreateBuffer(m_context, CL_MEM_READ_WRITE 		, sizeof(float)*16,				0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 41= "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	pose_update_buf					= clCreateBuffer(m_context, CL_MEM_READ_WRITE 		, sizeof(float)*16,				0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 41= "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	distorsion_update_buf			= clCreateBuffer(m_context, CL_MEM_READ_WRITE 		, sizeof(float)*6,				0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 41= "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	old_results_buf					= clCreateBuffer(m_context, CL_MEM_READ_WRITE 		, sizeof(float)*6*4,			0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 41= "<<checkerror(res)<<"\n"<<flush;exit_(res);}

	K_buf							= clCreateBuffer(m_context, CL_MEM_READ_WRITE 		, sizeof(float)*16,				0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 41= "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	inv_K_buf						= clCreateBuffer(m_context, CL_MEM_READ_WRITE 		, sizeof(float)*16,				0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 41= "<<checkerror(res)<<"\n"<<flush;exit_(res);}

	patch_lookup_table_buf			= clCreateBuffer(m_context, CL_MEM_READ_WRITE 			, mm_size_bytes_C4,			0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 41= "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	SE3_hessian_map_mem				= clCreateBuffer(m_context, CL_MEM_READ_WRITE 		, 2 * mm_size_bytes_C4,			0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 41= "<<checkerror(res)<<"\n"<<flush;exit_(res);}

																																		if(verbosity>local_verbosity_threshold) {
																																			cout << "\n\nRunCL::allocatemem_chk3\n\n" << flush;

																																			cout << ",imgmem = " 		<< imgmem << endl;
																																			cout << ",basemem = " 		<< basemem << endl;
																																			cout << ",fp32_param_buf = "<< fp32_param_buf << endl;
																																			cout << ",k2kbuf = " 		<< k2kbuf << endl;
																																			cout << ",uint_param_buf = "<< uint_param_buf << endl;
																																			cout << "\n" << flush;
																																		}

																																		if(verbosity>local_verbosity_threshold) {
																																			cout << "\n\nRunCL::allocatemem_chk3.1\n\n" << flush;
																																			cout << "fp32_params[MIN_INV_DEPTH] = " << fp32_params[MIN_INV_DEPTH] << flush;
																																			cout << "\n" << flush;
																																		}

	status = clEnqueueWriteBuffer(uload_queue, fp32_param_buf, 	CL_FALSE, 0, 16 * sizeof(float), fp32_params, 			0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.5\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	status = clEnqueueWriteBuffer(uload_queue, uint_param_buf,	CL_FALSE, 0,  8 * sizeof(uint),	 uint_params, 			0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.5\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	status = clEnqueueWriteBuffer(uload_queue, mipmap_buf,		CL_FALSE, 0,  8*8* sizeof(uint), MipMap, 				0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.5\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	status = clEnqueueWriteBuffer(uload_queue, basemem, 		CL_FALSE, 0, image_size_bytes, 	baseImage.data, 		0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.6\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
																																		if(verbosity>local_verbosity_threshold) cout <<"\n\nRunCL::allocatemem_chk4.2\n\n" << flush;
	float depth = 1/( obj["max_depth"].asFloat() - (obj["min_depth"].asFloat() / 2)  );
	float default_depth = fp32_params[MAX_INV_DEPTH] / 2.0f;

	status = clEnqueueFillBuffer(uload_queue, depth_mem, 				&depth,			sizeof(float),   0, mm_size_bytes_C1,		0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.6\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	status = clEnqueueFillBuffer(uload_queue, depth_mem_temp, 			&default_depth, sizeof(float),   0, mm_size_bytes_C1,		0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.8\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	status = clEnqueueFillBuffer(uload_queue, depth_mem_GT, 			&default_depth, sizeof(float),   0, mm_size_bytes_C1,		0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.8\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	status = clEnqueueFillBuffer(uload_queue, HSV_grad_mem, 			&zero_flt,		sizeof(float),   0, mm_size_bytes_C8,		0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.3\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	status = clEnqueueFillBuffer(uload_queue, ST3_img_grad_mem,			&zero_flt,		sizeof(float),   0, 3*mm_size_bytes_C4,		0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.3\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	status = clEnqueueFillBuffer(uload_queue, patch_lookup_table_buf,	&zero_uint,		sizeof(uint),   0, mm_size_bytes_C4, 		0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.3\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);

	clFlush(uload_queue); status = clFinish(uload_queue); 																				if (status != CL_SUCCESS)	{ cout << "\nclFinish(uload_queue)=" << status << checkerror(status) <<"\n"  << flush; exit_(status);}

																																		if(verbosity>local_verbosity_threshold) {
																																			cout << "\n\nRunCL::allocatemem_chk5\n\n" << flush;
																																			cout << ",mm_Image_size = " << mm_Image_size << endl;
																																			cout << ",mm_Image_type = "	<< mm_Image_type << endl;
																																			cout << ",mm_size_bytes_C3 = " << mm_size_bytes_C3 << endl;
																																			cout << ",mm_size_bytes_C4 = " << mm_size_bytes_C4 << endl;
																																			cout << ",mm_size_bytes_C1 = " << mm_size_bytes_C1 << endl;
																																			cout << "\n";
																																			cout << ",baseImage_size, = " << baseImage_size << endl;
																																			cout << ",baseImage_type = " << baseImage_type << endl;
																																			cout << ",image_size_bytes = " << image_size_bytes	<< endl;
																																			cout << ",mm_vol_size_bytes = " << mm_vol_size_bytes << endl;
																																			cout << "\n" << flush;
																																		}
																																		if(verbosity>local_verbosity_threshold) {
																																			DownloadAndSave_3Channel( 	basemem,	ss.str(), paths.at("basemem"),		image_size_bytes, 	baseImage_size, 	baseImage_type, false ); 	cout << "\nbasemem,"	<< flush;
																																		}
																																		if(verbosity>local_verbosity_threshold) cout << "RunCL::allocatemem_finished #############################################################################\n\n" << flush;
}

RunCL::~RunCL(){  // TO DO  ? Replace individual buffer clearance with the large array method from Morphogenesis &  fluids_v3 ? OR a C++ vector ?
	int local_verbosity_threshold = V__RUNCL;//verbosity_mp["RunCL::allocatemem"];																	cout<<"\nRunCL::~RunCL_chk0_called"<<flush;
	cl_int status;																														// release memory

	for (uint i=0; i<num_current_frames; i++ ) {
		status = clReleaseMemObject(imgmem[i]);					if (status != CL_SUCCESS)	{ cout << "\nimgmem                         status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_01"<<flush;
		status = clReleaseMemObject(velmap[i]);					if (status != CL_SUCCESS)	{ cout << "\nimgmem                         status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_01"<<flush;
	}
	status = clReleaseMemObject(imgmem_blurred);				if (status != CL_SUCCESS)	{ cout << "\nimgmem_blurred                 status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_02"<<flush;
	status = clReleaseMemObject(SE3_grad_map_mem);				if (status != CL_SUCCESS)	{ cout << "\nSE3_grad_map_mem               status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_08"<<flush;
	status = clReleaseMemObject(SE3_weight_map_mem);			if (status != CL_SUCCESS)	{ cout << "\nSE3_weight_map_mem             status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_10"<<flush;

	status = clReleaseMemObject(SE3_incr_map_mem);				if (status != CL_SUCCESS)	{ cout << "\nSE3_incr_map_mem               status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_11"<<flush;
	status = clReleaseMemObject(SE3_map_mem);					if (status != CL_SUCCESS)	{ cout << "\nSE3_map_mem                    status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_12"<<flush;
	status = clReleaseMemObject(basemem);						if (status != CL_SUCCESS)	{ cout << "\nbasemem                        status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_13"<<flush;
	status = clReleaseMemObject(depth_mem_temp);				if (status != CL_SUCCESS)	{ cout << "\ndepth_mem                      status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_14"<<flush;
	status = clReleaseMemObject(depth_mem_GT);					if (status != CL_SUCCESS)	{ cout << "\ndepth_mem_GT                   status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_15"<<flush;

	status = clReleaseMemObject(img_grad_mem);					if (status != CL_SUCCESS)	{ cout << "\nimg_grad_mem                   status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_20"<<flush;
	status = clReleaseMemObject(g1mem);							if (status != CL_SUCCESS)	{ cout << "\ng1mem                          status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_20"<<flush;
	status = clReleaseMemObject(depth_mem);						if (status != CL_SUCCESS)	{ cout << "\ndepth_mem                      status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_21"<<flush;

	status = clReleaseMemObject(fp32_param_buf);				if (status != CL_SUCCESS)	{ cout << "\nfp32_param_buf                 status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_30"<<flush;
	status = clReleaseMemObject(k2kbuf);						if (status != CL_SUCCESS)	{ cout << "\nk2kbuf                         status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_31"<<flush;

	status = clReleaseMemObject(SE3_k2kbuf);					if (status != CL_SUCCESS)	{ cout << "\nSE3_k2kbuf                     status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_33"<<flush;
	status = clReleaseMemObject(uint_param_buf);				if (status != CL_SUCCESS)	{ cout << "\nuint_param_buf                 status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_34"<<flush;
	status = clReleaseMemObject(mipmap_buf);					if (status != CL_SUCCESS)	{ cout << "\nmipmap_buf                     status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_35"<<flush;

	status = clReleaseMemObject(SE3_rho_map_mem	);				if (status != CL_SUCCESS)	{ cout << "\nSE3_rho_map_mem                status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_40"<<flush;
	status = clReleaseMemObject(img_stats_buf);					if (status != CL_SUCCESS)	{ cout << "\nimg_stats_buf                  status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_42"<<flush;
	status = clReleaseMemObject(pix_sum_mem);					if (status != CL_SUCCESS)	{ cout << "\npix_sum_mem                    status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_43"<<flush;
	status = clReleaseMemObject(var_sum_mem);					if (status != CL_SUCCESS)	{ cout << "\nvar_sum_mem                    status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_44"<<flush;
	status = clReleaseMemObject(HSV_grad_mem);					if (status != CL_SUCCESS)	{ cout << "\nHSV_grad_mem                   status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_45"<<flush;
	status = clReleaseMemObject(ST3_img_grad_mem);				if (status != CL_SUCCESS)	{ cout << "\nST3_img_grad_mem               status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_45"<<flush;


	// buffers for patch kernel based Dynamic_slam
	status = clReleaseMemObject(pose_buf);						if (status != CL_SUCCESS)	{ cout << "\npose_buf                       status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_48"<<flush;
	status = clReleaseMemObject(pose_update_buf);				if (status != CL_SUCCESS)	{ cout << "\npose_update_buf                status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_48"<<flush;
	status = clReleaseMemObject(distorsion_update_buf);			if (status != CL_SUCCESS)	{ cout << "\ndistorsion_update_buf          status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_48"<<flush;
	status = clReleaseMemObject(old_results_buf);				if (status != CL_SUCCESS)	{ cout << "\nold_result_buf                 status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_48"<<flush;

	status = clReleaseMemObject(K_buf);							if (status != CL_SUCCESS)	{ cout << "\nK_buf                          status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_48"<<flush;
	status = clReleaseMemObject(inv_K_buf);						if (status != CL_SUCCESS)	{ cout << "\ninv_K_buf                      status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_48"<<flush;

	status = clReleaseMemObject(patch_lookup_table_buf);		if (status != CL_SUCCESS)	{ cout << "\npatch_lookup_table_buf         status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_48"<<flush;
	status = clReleaseMemObject(SE3_hessian_map_mem);			if (status != CL_SUCCESS)	{ cout << "\nSE3_hessian_map_mem            status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_48"<<flush;


	// release kernels
	status = clReleaseKernel(convert_depth_kernel);					if (status != CL_SUCCESS)	{ cout << "\nconvert_depth_kernel				status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_59"<<flush;
	status = clReleaseKernel(mipmap_float_kernel);					if (status != CL_SUCCESS)	{ cout << "\nmipmap_float_kernel				status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_59"<<flush;
	status = clReleaseKernel(cvt_color_space_linear_kernel);		if (status != CL_SUCCESS)	{ cout << "\ncvt_color_space_linear_kernel 		status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_49"<<flush;
	status = clReleaseKernel(comp_param_maps_kernel);				if (status != CL_SUCCESS)	{ cout << "\ncomp_param_maps_kernel 			status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_54"<<flush;
	//
	status = clReleaseKernel(rho_sq_kernel);						if (status != CL_SUCCESS)	{ cout << "\nrho_sq_kernel						status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_66"<<flush;
	status = clReleaseKernel(reduce_patch_Rho_kernel);				if (status != CL_SUCCESS)	{ cout << "\nupdate_SE3_kernel					status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_66"<<flush;
	status = clReleaseKernel(update_k2k_kernel);					if (status != CL_SUCCESS)	{ cout << "\nupdate_k2k_kernel					status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_66"<<flush;

	status = clReleaseKernel(compute_patch_lookup_table_kernel);	if (status != CL_SUCCESS)	{ cout << "\ncompute_patch_lookup_table_kernel	status = " << checkerror(status) <<"\n"<<flush; }	if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_66"<<flush;
	status = clReleaseKernel(patch_img_grad_kernel);				if (status != CL_SUCCESS)	{ cout << "\npatch_img_grad_kernel				status = " << checkerror(status) <<"\n"<<flush; }	if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_66"<<flush;
	status = clReleaseKernel(patch_hessian_reduce_kernel);			if (status != CL_SUCCESS)	{ cout << "\npatch_hessian_reduce_kernel		status = " << checkerror(status) <<"\n"<<flush; }	if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_66"<<flush;
	// RunCL_patch_tracking.cpp
	status = clReleaseKernel(pad_image_top_bottom2_kernel);			if (status != CL_SUCCESS)	{ cout << "\npad_image_top_bottom2_kernel		status = " << checkerror(status) <<"\n"<<flush; }	if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_66"<<flush;
	status = clReleaseKernel(vertcal_blur5_kernel);					if (status != CL_SUCCESS)	{ cout << "\nvertcal_blur5_kernel				status = " << checkerror(status) <<"\n"<<flush; }	if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_66"<<flush;
	status = clReleaseKernel(pad_image_left_right2_kernel);			if (status != CL_SUCCESS)	{ cout << "\npad_image_left_right2_kernel		status = " << checkerror(status) <<"\n"<<flush; }	if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_66"<<flush;
	status = clReleaseKernel(horiz_blur5_kernel);					if (status != CL_SUCCESS)	{ cout << "\nhoriz_blur5_kernel					status = " << checkerror(status) <<"\n"<<flush; }	if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_66"<<flush;
	status = clReleaseKernel(reduce_img_kernel);					if (status != CL_SUCCESS)	{ cout << "\nreduce_img_kernel					status = " << checkerror(status) <<"\n"<<flush; }	if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_66"<<flush;
	// RunCL_depth.cpp
	status = clReleaseKernel(update_depth_kernel);					if (status != CL_SUCCESS)	{ cout << "\nupdate_depth_kernel				status = " << checkerror(status) <<"\n"<<flush; }	if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_66"<<flush;
	status = clReleaseKernel(update_depth_2_kernel);				if (status != CL_SUCCESS)	{ cout << "\nupdate_depth_2_kernel				status = " << checkerror(status) <<"\n"<<flush; }	if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_66"<<flush;
	status = clReleaseKernel(regularize_depth_kernel);				if (status != CL_SUCCESS)	{ cout << "\nregularize_depth_kernel			status = " << checkerror(status) <<"\n"<<flush; }	if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_66"<<flush;
	status = clReleaseKernel(enlarge_layer_float_kernel);			if (status != CL_SUCCESS)	{ cout << "\nenlarge_layer_float_kernel			status = " << checkerror(status) <<"\n"<<flush; }	if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_66"<<flush;


	// release command queues
	status = clReleaseCommandQueue(m_queue);                   if (status != CL_SUCCESS)	{ cout << "\nm_queue                        status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_67"<<flush;
	status = clReleaseCommandQueue(uload_queue);               if (status != CL_SUCCESS)	{ cout << "\nuload_queue 	                status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_68"<<flush;
	status = clReleaseCommandQueue(dload_queue);               if (status != CL_SUCCESS)	{ cout << "\ndload_queue 	                status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_69"<<flush;
	status = clReleaseCommandQueue(track_queue);               if (status != CL_SUCCESS)	{ cout << "\ntrack_queue 	                status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_70"<<flush;

	// release Program
	clReleaseProgram(m_program);	if (status != CL_SUCCESS)	{ cout << "\nm_program 	status = " << checkerror(status) <<"\n"<<flush; }	if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_71"<<flush;

	// release context
	clReleaseContext(m_context);	if (status != CL_SUCCESS)	{ cout << "\nm_context 	status = " << checkerror(status) <<"\n"<<flush; }	if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_72"<<flush;

	// free_wg_offsets();
	cout<<"\nRunCL::~RunCL_chk1_finished"<<flush;
}

void RunCL::exit_(int res)   // TO DO convert all uses to exit_(res); Will call RunCL::~RunCL() automatically.
{
	cout <<endl<< flush;
	cerr <<endl<< flush;
	exit(res);
}
