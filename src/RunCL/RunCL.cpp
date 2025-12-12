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
	basemem=k2kbuf=0; // dbg_databuf=cdatabuf=hdatabuf=temp_cdatabuf=temp_hdatabuf=	dmem=amem=gxmem=gymem=lomem=himem=mean_mem=				// Set device pointers to zero
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

	const char *basepath = obj["source_filepath"].asCString();
    const char *foldername = obj["kernel_folder"].asCString();
    num_files = obj["kernel_files"].size();
    lengths = (size_t*)malloc( (num_files+1)*sizeof(size_t) );                                                                              // allocate array for file lengths
	strings = (char**)malloc( (num_files+1)*sizeof(size_t) );																				// allocate outer array for kernel files
    FILE *program_handle;

    for (int i=0; i<num_files; i++){                                                                                                        // for kernel source files in obj array
        const char *filename = obj["kernel_files"][i].asCString();
        stringstream filepath; filepath << basepath << foldername << filename;
		const std::string tmp =  filepath.str();
		const char* char_filepath = tmp.c_str();
		program_handle = fopen(char_filepath, "r");                                          if(program_handle == NULL) { perror("Couldn't find the program file");
																															cout << "\tchar_filepath = "<< char_filepath << flush;
																															exit_(1); }
        fseek(program_handle, 0, SEEK_END);
        lengths[i] = ftell(program_handle);
        rewind(program_handle);

        strings[i] = (char*)malloc(lengths[i]+1);																							// allocate inner array for this kernel file
        strings[i][lengths[i]] = '\0';
        const size_t ret_code = fread( strings[i], sizeof(char), lengths[i], program_handle );if (ret_code != lengths[i]){ perror("Couldn't read the program file");
																															cout << "\tchar_filepath = "<< char_filepath << flush;
																															exit_(1); }
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
	int local_verbosity_threshold = V_RUNCL_CREATEKERNELS;//verbosity_mp["RunCL::createKernels"];

	cl_int err_code;
	convert_depth_kernel			= clCreateKernel(m_program, "convert_depth", 				&err_code);			if (err_code != CL_SUCCESS)  {cout << "\nError 'convert_depth'  kernel not built.\n"			<<flush; exit_(0);   }
	mipmap_float_kernel				= clCreateKernel(m_program, "mipmap_linear_flt", 			&err_code);			if (err_code != CL_SUCCESS)  {cout << "\nError 'mipmap_linear_flt'  kernel not built.\n"		<<flush; exit_(0);   }

	cvt_color_space_linear_kernel 	= clCreateKernel(m_program, "cvt_color_space_linear", 		&err_code);			if (err_code != CL_SUCCESS)  {cout << "\nError 'cvt_color_space_linear'  kernel not built.\n"	<<flush; exit_(0);   }
	sum_image_variance_kernel		= clCreateKernel(m_program, "sum_image_variance",			&err_code);			if (err_code != CL_SUCCESS)  {cout << "\nError 'sum_image_variance'  kernel not built.\n"		<<flush; exit_(0);   }
	sample_image_variance_kernel	= clCreateKernel(m_program, "sample_image_variance",		&err_code);			if (err_code != CL_SUCCESS)  {cout << "\nError 'sample_image_variance'  kernel not built.\n"	<<flush; exit_(0);   }

	img_grad_kernel					= clCreateKernel(m_program, "img_grad", 					&err_code);			if (err_code != CL_SUCCESS)  {cout << "\nError 'img_grad'  kernel not built.\n"					<<flush; exit_(0);   }
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

/*
	blur_image_kernel				= clCreateKernel(m_program, "blur_image",					&err_code);			if (err_code != CL_SUCCESS)  {cout << "\nError 'blur_image'  kernel not built.\n"				<<flush; exit_(0);   }
	reduce_kernel					= clCreateKernel(m_program, "reduce", 						&err_code);			if (err_code != CL_SUCCESS)  {cout << "\nError 'reduce'  kernel not built.\n"					<<flush; exit_(0);   }
	mipmap_float4_kernel			= clCreateKernel(m_program, "mipmap_linear_flt4", 			&err_code);			if (err_code != CL_SUCCESS)  {cout << "\nError 'mipmap_linear_flt4'  kernel not built.\n"		<<flush; exit_(0);   }
//	mipmap_3x3blur_flt4_kernel		= clCreateKernel(m_program, "mipmap_3x3blur_flt4", 			&err_code);			if (err_code != CL_SUCCESS)  {cout << "\nError 'mipmap_linear_flt4'  kernel not built.\n"		<<flush; exit_(0);   }

	se3_rho_sq_kernel				= clCreateKernel(m_program, "se3_Rho_sq", 					&err_code);			if (err_code != CL_SUCCESS)  {cout << "\nError 'se3_Rho_sq'  kernel not built.\n"				<<flush; exit_(0);   }
	se3_lk_grad_kernel				= clCreateKernel(m_program, "se3_LK_grad", 					&err_code);			if (err_code != CL_SUCCESS)  {cout << "\nError 'se3_LK_grad'  kernel not built.\n"				<<flush; exit_(0);   }

	atomic_test1_kernel				= clCreateKernel(m_program, "atomic_test1", 				&err_code);			if (err_code != CL_SUCCESS)  {cout << "\nError 'atomic_test1'  kernel not built.\n"				<<flush; exit_(0);   }
	atomic_test2_kernel				= clCreateKernel(m_program, "atomic_test2", 				&err_code);			if (err_code != CL_SUCCESS)  {cout << "\nError 'atomic_test1'  kernel not built.\n"				<<flush; exit_(0);   }

	transform_depthmap_kernel		= clCreateKernel(m_program, "transform_depthmap", 			&err_code);			if (err_code != CL_SUCCESS)  {cout << "\nError 'transform_depthmap'  kernel not built.\n"		<<flush; exit_(0);   }
	transform_costvolume_kernel 	= clCreateKernel(m_program, "transform_cost_volume", 		&err_code);			if (err_code != CL_SUCCESS)  {cout << "\nError 'transform_cost_volume'  kernel not built.\n"	<<flush; exit_(0);   }

	depth_cost_vol_kernel			= clCreateKernel(m_program, "DepthCostVol", 				&err_code);			if (err_code != CL_SUCCESS)  {cout << "\nError 'DepthCostVol'  kernel not built.\n"				<<flush; exit_(0);   }
	updateQD_kernel 				= clCreateKernel(m_program, "UpdateQD", 					&err_code);			if (err_code != CL_SUCCESS)  {cout << "\nError 'UpdateQD'  kernel not built.\n"					<<flush; exit_(0);   }
	updateG_kernel  				= clCreateKernel(m_program, "UpdateG", 						&err_code);			if (err_code != CL_SUCCESS)  {cout << "\nError 'UpdateG'  kernel not built.\n"					<<flush; exit_(0);   }
	updateA_kernel  				= clCreateKernel(m_program, "UpdateA", 						&err_code);			if (err_code != CL_SUCCESS)  {cout << "\nError 'UpdateA'  kernel not built.\n"					<<flush; exit_(0);   }

	measureDepthFit_kernel			= clCreateKernel(m_program, "MeasureDepthFit", 				&err_code);			if (err_code != CL_SUCCESS)  {cout << "\nError 'MeasureDepthFit'  kernel not built.\n"			<<flush; exit_(0);   }

	// stereo disparity kernels
	compute_lookup_table_kernel		= clCreateKernel(m_program, "compute_lookup_table", 		&err_code);			if (err_code != CL_SUCCESS)  {cout << "\nError 'compute_lookup_table_kernel'  kernel not built.\n"	<<flush; exit_(0);   }
	disparity_load_frame_kernel		= clCreateKernel(m_program, "disparity_load_frame", 		&err_code);			if (err_code != CL_SUCCESS)  {cout << "\nError 'disparity_load_frame_kernel'  kernel not built.\n"	<<flush; exit_(0);   }
	set_warp_new_image_kernel		= clCreateKernel(m_program, "set_warp_new_image",			&err_code);			if (err_code != CL_SUCCESS)  {cout << "\nError 'set_warp_new_image_kernel'  kernel not built.\n"	<<flush; exit_(0);   }
	warp_image_kernel				= clCreateKernel(m_program, "warp_image", 					&err_code);			if (err_code != CL_SUCCESS)  {cout << "\nError 'warp_image_kernel'  kernel not built.\n"			<<flush; exit_(0);   }

	correlation_one_step_kernel		= clCreateKernel(m_program, "correlation_one_step", 		&err_code);			if (err_code != CL_SUCCESS)  {cout << "\nError 'correlation_one_step'  kernel not built.\n"			<<flush; exit_(0);   }
	correlation_2nd_step_kernel		= clCreateKernel(m_program, "correlation_2nd_step", 		&err_code);			if (err_code != CL_SUCCESS)  {cout << "\nError 'correlation_2nd_step'  kernel not built.\n"			<<flush; exit_(0);   }
	blur_volume_kernel				= clCreateKernel(m_program, "blur_volume",			 		&err_code);			if (err_code != CL_SUCCESS)  {cout << "\nError 'blur_volume_kernel'  kernel not built.\n"			<<flush; exit_(0);   }

	mean_sq_3rows_kernel			= clCreateKernel(m_program, "mean_3rows", 					&err_code);			if (err_code != CL_SUCCESS)  {cout << "\nError 'mean_sq_3rows_kernel'  kernel not built.\n"			<<flush; exit_(0);   }
	mean_sq_cols_kernel				= clCreateKernel(m_program, "mean_3cols",	 				&err_code);			if (err_code != CL_SUCCESS)  {cout << "\nError 'mean_sq_3cols_kernel'  kernel not built.\n"			<<flush; exit_(0);   }
	sigma_3rows_kernel				= clCreateKernel(m_program, "sigma_3rows",	 				&err_code);			if (err_code != CL_SUCCESS)  {cout << "\nError 'sigma_3rows_kernel'  kernel not built.\n"			<<flush; exit_(0);   }
	sigma_3cols_kernel				= clCreateKernel(m_program, "sigma_3cols",	 				&err_code);			if (err_code != CL_SUCCESS)  {cout << "\nError 'sigma_3cols_kernel'  kernel not built.\n"			<<flush; exit_(0);   }

	covariance_3rows_kernel			= clCreateKernel(m_program, "covariance_3rows",				&err_code);			if (err_code != CL_SUCCESS)  {cout << "\nError 'covariance_3rows_kernel'  kernel not built.\n"		<<flush; exit_(0);   }
	covariance_cols_kernel			= clCreateKernel(m_program, "covariance_3cols", 			&err_code);			if (err_code != CL_SUCCESS)  {cout << "\nError 'covariance_cols_kernel'  kernel not built.\n"		<<flush; exit_(0);   }
	correlation_kernel				= clCreateKernel(m_program, "correlation",		 			&err_code);			if (err_code != CL_SUCCESS)  {cout << "\nError 'correlation_kernel'  kernel not built.\n"			<<flush; exit_(0);   }

	compute_warp_kernel				= clCreateKernel(m_program, "compute_warp", 				&err_code);			if (err_code != CL_SUCCESS)  {cout << "\nError 'regularize_warp_kernel'  kernel not built.\n"		<<flush; exit_(0);   }
	regularize_warp_kernel			= clCreateKernel(m_program, "regularize_warp", 				&err_code);			if (err_code != CL_SUCCESS)  {cout << "\nError 'regularize_warp_kernel'  kernel not built.\n"		<<flush; exit_(0);   }
	propagate_warp_kernel			= clCreateKernel(m_program, "propagate_warp", 				&err_code);			if (err_code != CL_SUCCESS)  {cout << "\nError 'propagate_warp_kernel'  kernel not built.\n"		<<flush; exit_(0);   }

	warp_and_depth_error_kernel		= clCreateKernel(m_program, "warp_and_depth_error", 		&err_code);			if (err_code != CL_SUCCESS)  {cout << "\nError 'warp_and_depth_error_kernel'  kernel not built.\n"	<<flush; exit_(0);   }
*/
}

int RunCL::convertToString(const char *filename, std::string& s){
	int local_verbosity_threshold = V_RUNCL_CONVERTTOSTRING;//verbosity_mp["RunCL::convertToString"];

	size_t size;
	char*  str;
	std::fstream f(filename, (std::fstream::in | std::fstream::binary));
	if (f.is_open() ) {
		size_t fileSize;
		f.seekg(0, std::fstream::end);
		size = fileSize = (size_t)f.tellg();
		f.seekg(0, std::fstream::beg);
		str = new char[size + 1];
		if (!str) {
			f.close();
			return 0;
		}
		f.read(str, fileSize);
		f.close();
		str[size] = '\0';
		s = str;
		delete[] str;
		return 0;
	}
										cout << "Error: failed to open file\n:" << filename << endl;
	return 1;
}

void RunCL::initialize_fp32_params(){
	int local_verbosity_threshold = V_RUNCL_INITIALIZE_FP32_PARAMS;//verbosity_mp["RunCL::initialize_fp32_params"];
																																			if(verbosity>local_verbosity_threshold) cout << "\n\nRunCL::initialize_fp32_params_chk_0,  \n" << flush;
	if ( obj["max_depth_infinity"].asBool()  ) { fp32_params[MIN_INV_DEPTH]	= 0;
	} else { fp32_params[MIN_INV_DEPTH]	=  1/obj["max_depth"].asFloat()		;   }

	fp32_params[MAX_INV_DEPTH]	=  1/obj["min_depth"].asFloat()		;																		// This works: Initialize 'params[]' from conf.json .
	//fp32_params[MIN_INV_DEPTH]	=  1/obj["max_depth"].asFloat()		;

	fp32_params[INV_DEPTH_STEP]	=	 ( fp32_params[MAX_INV_DEPTH] - fp32_params[MIN_INV_DEPTH] ) /  uint_params[COSTVOL_LAYERS]	;

	fp32_params[ALPHA_G]		=    obj["alpha_g"].asFloat()		;
	fp32_params[BETA_G]			=    obj["beta_g"].asFloat()		;
	fp32_params[EPSILON]		=    obj["epsilon"].asFloat()		;
				//SIGMA_Q ;
				//SIGMA_D ;
	fp32_params[THETA]			=    obj["thetaStart"].asFloat()	;
	fp32_params[OLD_THETA]		=    fp32_params[THETA];

	fp32_params[LAMBDA]			=    obj["lambda"].asFloat()		;
	fp32_params[SCALE_EAUX]		=    obj["scale_E_aux"].asFloat()	;
	fp32_params[SE3_LM_A]		=    obj["SE3_LM_A"].asFloat()		;
	fp32_params[SE3_LM_B]		=    obj["SE3_LM_B"].asFloat()		;
																																			if(verbosity>local_verbosity_threshold) cout << "\n\nRunCL::initialize_fp32_params_finished,  fp32_params[MIN_INV_DEPTH] = "<<fp32_params[MIN_INV_DEPTH]<<"\n\n" << flush;
}

void RunCL::initialize_RunCL(cv::Mat baseImage_){
	int local_verbosity_threshold = V_RUNCL_INITIALIZE_RUNCL;//verbosity_mp["RunCL::initialize_RunCL"];// -1;
																																			if(verbosity>local_verbosity_threshold) cout << "\n\nRunCL::initialize_RunCL_chk_0\n\n" << flush;
	baseImage =  baseImage_;
																																			if( baseImage.empty() ){cout <<"\nError RunCL::initialize() : runcl.baseImage.empty()"<<flush; exit_(0); }
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
																																			if(verbosity>1) { imshow("runcl.baseImage",baseImage); cv::waitKey(-1); }
	image_size_bytes	= baseImage.total() * baseImage.elemSize();																			// Constant parameters of the base image
	image_size_bytes_C1	= baseImage.total() * sizeof(float);
	costVolLayers 		=( 1 + obj["layers"].asUInt() ); // TODO  2;
	baseImage_size 		= baseImage.size();
	baseImage_type 		= baseImage.type();
	baseImage_width		= baseImage.cols;
	baseImage_height	= baseImage.rows;
	layerstep 			= baseImage_width * baseImage_height;

	mm_num_reductions	= obj["num_reductions"].asUInt();																					// Constant parameters of the mipmap, (as opposed to per-layer mipmap_buf)
	mm_num_blur_layers	= obj["num_blur_layers"].asUInt();
	mm_start			= 0;
	mm_stop				= mm_num_reductions;																								if(verbosity>local_verbosity_threshold) cout << "\nRunCL::initialize_RunCL_chk0.5,  mm_start="<<mm_start<<",  mm_stop="<<mm_stop<<" \n" << flush;
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
	mm_size_bytes_C1	= temp2.total() * temp2.elemSize();
	mm_vol_size_bytes	= mm_size_bytes_C1 * costVolLayers;
																																			if(verbosity>local_verbosity_threshold) cout << "\n\nRunCL::initialize_RunCL_chk1  mm_gaussian_size="<<mm_gaussian_size<<" \n\n" << flush;
																																			// Get the maximum work group size for executing the kernel on the device ///////
																																			// From https://github.com/rsnemmen/OpenCL-examples/blob/e2c34f1dfefbd265cfb607c2dd6c82c799eb322a/square_array/square.c
	cl_int 				status;
	status = clGetKernelWorkGroupInfo(cvt_color_space_linear_kernel, deviceId, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local_work_size), &local_work_size, NULL); 										if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; exit_(status);}
	status = clGetKernelWorkGroupInfo(cvt_color_space_linear_kernel, deviceId, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(kernel_work_size_multiple), &kernel_work_size_multiple, NULL); 	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; exit_(status);}
	status = clGetDeviceInfo( deviceId, CL_DEVICE_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(device_work_size_multiple), &device_work_size_multiple, NULL); 										if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; exit_(status);}

	// cl_int clGetDeviceInfo(
	// 	cl_device_id 	device,
	// 	cl_device_info 	param_name,
	// 	size_t 			param_value_size,
	// 	void* 			param_value,
	// 	size_t* 		param_value_size_ret
	// );

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
// TODO  replace with patch slam.  NB sigmas were part of dram mapping.
//	computeSigmas( obj["epsilon"].asFloat(), obj["thetaStart"].asFloat(), obj["L"].asFloat(), fp32_params[SIGMA_Q], fp32_params[SIGMA_D] );
																																			//computeSigmas( obj["epsilon"].asFloat(), obj["thetaStart"].asFloat(), obj["L"].asFloat(), cl_half_params[SIGMA_Q], cl_half_params[SIGMA_D] );
																																			if(verbosity>local_verbosity_threshold){
																																				cout << "\n\nRunCL::initialize  Checking fp32_params[]" << flush;
																																				cout << "\nfp32_params[0 MAX_INV_DEPTH]="	<<fp32_params[MAX_INV_DEPTH]		<<"\t\t1/obj[\"min_depth\"].asFloat()="	<<1/obj["min_depth"].asFloat();
																																				cout << "\nfp32_params[1 MIN_INV_DEPTH]="	<<fp32_params[MIN_INV_DEPTH]		<<"\t\t1/obj[\"max_depth\"].asFloat()="	<<1/obj["max_depth"].asFloat();
																																				cout << "\nfp32_params[2 INV_DEPTH_STEP]="	<<fp32_params[INV_DEPTH_STEP];
																																				cout << "\nfp32_params[3 ALPHA_G]="			<<fp32_params[ALPHA_G]				<<"\t\tobj[\"alpha_g\"].asFloat()="		<<obj["alpha_g"].asFloat();
																																				cout << "\nfp32_params[4 BETA_G]="			<<fp32_params[BETA_G]				<<"\t\tobj[\"beta_g\"].asFloat()="		<<obj["beta_g"].asFloat();
																																				cout << "\nfp32_params[5 EPSILON]="			<<fp32_params[EPSILON]				<<"\t\tobj[\"epsilon\"].asFloat()="		<<obj["epsilon"].asFloat();
																																				cout << "\nfp32_params[6 SIGMA_Q]="			<<fp32_params[SIGMA_Q];
																																				cout << "\nfp32_params[7 SIGMA_D ]="		<<fp32_params[SIGMA_D ];
																																				cout << "\nfp32_params[8 THETA]="			<<fp32_params[THETA]				<<"\t\tobj[\"thetaStart\"].asFloat()="	<<obj["thetaStart"].asFloat();
																																				cout << "\nfp32_params[9 LAMBDA]="			<<fp32_params[LAMBDA]				<<"\t\tobj[\"lambda\"].asFloat()="		<<obj["lambda"].asFloat();
																																				cout << "\nfp32_params[10 SCALE_EAUX]="		<<fp32_params[SCALE_EAUX]			<<"\t\tobj[\"scale_E_aux\"].asFloat()="	<<obj["scale_E_aux"].asFloat();
																																				cout << "\n" << flush;
																																			}
	for (int i=0; i<3; i++){ fp32_so3_k2k[i+ i*3]		=1.0; }																				// initialize fp32_so3_k2k & fp32_k2k as 'unity' transform, i.e. zero rotation & zero translation.
	//for (int i=0; i<4; i++){ fp32_k2k[i+ i*4]    		=1.0; }																				// NB instantiated as {{0}}.
	for (int i=0; i<4; i++){ fp32_k2keyframe[i+ i*4]    =1.0; }																				// NB instantiated as {{0}}.
	/*
	fp32_k2k[0]  =  1.0 ;	// (1,0,0,0)
	fp32_k2k[5]  =  1.0 ;	// (0,1,0,0)
	fp32_k2k[10] =  1.0 ;	// (0,0,1,0)
	fp32_k2k[15] =  1.0 ;	// (0,0,0,1)
	*/
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
	// ####################################################################################################################################################################
																																			if(verbosity>local_verbosity_threshold) {
																																				cout <<"	\nRunCL::initialize"<<endl;
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

	int stop 						= min(mm_num_reductions, max_mipmap_layers-1);	// TODO compute required reduction and blur depending on img size
	int reduction = 0;
	for(; reduction <= stop; reduction++) {
		num_threads[reduction]		= ceil( (float)(mipmap[MiM_PIXELS])/(float)local_work_size ) * local_work_size ;						// global_work_size formula for num_treads req for this layer.
		for (int i=0; i<8; i++) 	{																										// Initialize the global MipMap[8*8] array.
			MipMap[reduction*8 +i] = mipmap[i];
																																			if(verbosity>local_verbosity_threshold) { cout << "\nMipMap["<<reduction<<"*8 +"<<i<<"]="<<MipMap[reduction*8 +i] ;}
		}																																	if(verbosity>local_verbosity_threshold) { cout << endl << flush; }
		mipmap[MiM_READ_OFFSET] 	= mipmap[MiM_WRITE_OFFSET];
		mipmap[MiM_WRITE_OFFSET] 	= mipmap[MiM_WRITE_OFFSET] + read_cols_with_margin * (margin + write_rows);
		mipmap[MiM_READ_ROWS] 		= write_rows;
		write_rows					= write_rows/2;
		mipmap[MiM_WRITE_ROWS]		= write_rows;
		mipmap[MiM_READ_COLS] 		= mipmap[MiM_WRITE_COLS];
		mipmap[MiM_WRITE_COLS] 		= mipmap[MiM_WRITE_COLS]/2;
		mipmap[MiM_PIXELS]			= mipmap[MiM_READ_COLS] * mipmap[MiM_READ_ROWS];
	}
	mipmap[MiM_WRITE_ROWS]			= mipmap[MiM_READ_ROWS];
	mipmap[MiM_WRITE_COLS] 			= mipmap[MiM_READ_COLS];
	mipmap[MiM_WRITE_OFFSET] 		= mipmap[MiM_READ_OFFSET] + mipmap[MiM_READ_COLS] + 2*margin;

	stop 							= min( (mm_num_reductions + mm_num_blur_layers),  max_mipmap_layers-1);
	int layer 						= reduction;
	for(; layer <= stop; layer++) {
		for (int i=0; i<8; i++) 	{																										// Initialize the global MipMap[8*8] array.
			MipMap[layer*8 +i] = mipmap[i];
																																			if(verbosity>local_verbosity_threshold) { cout << "\nMipMap["<<layer<<"*8 +"<<i<<"]="<<MipMap[layer*8 +i] ;}
		}																																	if(verbosity>local_verbosity_threshold) { cout << endl << flush; }
		mipmap[MiM_READ_OFFSET]		= mipmap[MiM_WRITE_OFFSET];
		mipmap[MiM_WRITE_OFFSET] 	= mipmap[MiM_READ_OFFSET] + mipmap[MiM_READ_COLS] + 2*margin;
	}

																																			if(verbosity>local_verbosity_threshold) {
																																				cout <<"	\nRunCL::initialize"<<endl;
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
																																					cout << "\n row = "					<< MipMap[reduction*8 +MiM_READ_OFFSET] / mm_width;
																																					cout << "\n col = "					<< MipMap[reduction*8 +MiM_READ_OFFSET] % mm_width;
																																				}
																																			}
																																			/*
																																			#define MiM_PIXELS			0	// for mipmap_buf, 				when launching one kernel per layer. 	Updated for each layer.
																																			#define MiM_READ_OFFSET		1	// for ths layer, 				start of image data
																																			#define MiM_WRITE_OFFSET	2
																																			#define MiM_READ_COLS		3	// cols without margins
																																			#define MiM_WRITE_COLS		4
																																			// #define MiM_GAUSSIAN_SIZE	5	// filter box size
																																			#define MiM_READ_ROWS		6	// rows without margins
																																			#define MiM_WRITE_ROWS		7
																																			*/
																																			// Summation buffer sizes
	se3_sum_size 			= 1 + ceil( (float)(MipMap[(mm_num_reductions+1)*8 + MiM_READ_OFFSET]) / (float)local_work_size ) ;				// i.e. num workgroups used = MiM_READ_OFFSET for 1 layer more than used / local_work_size,   will give one row of vector per group.
	se3_sum_size 			*= 2;  																											// *2 incr num grps for reduced groupsize
	uint num_DoFs			= 6 ; 																											// 6 DoF of float4 channels, + 1 DoF to compute global Rho.
	se3_sum_size_bytes		= se3_sum_size * sizeof(float) * 4 * num_DoFs ;																	if(verbosity>local_verbosity_threshold) cout <<"\n\n se3_sum_size="<< se3_sum_size<<",    se3_sum_size_bytes="<<se3_sum_size_bytes<<flush;
	se3_sum2_size_bytes 	= 2 * mm_num_reductions * sizeof(float) * 4 * num_DoFs;															// NB the data returned is 6xfloat4 per group, holding one float4 per 6DoF of SE3, where alpha channel=pixel count.
	se3_sum2_size_bytes 	= ((se3_sum2_size_bytes%32) + 1) * 32;																			// Needed for Nvidia, to ensure memory allocations are multiples of 32bytes.

	so3_sum_size_bytes		= se3_sum_size_bytes / 2;
	so3_sum_size			= se3_sum_size ;
	//pix_sum_size			= 1 + ceil( (float)(baseImage_width * baseImage_height) / (float)local_work_size ) ;  							// i.e. num workgroups used = baseImage_width * baseImage_height / local_work_size,   will give one row of vector per group.
	pix_sum_size			= se3_sum_size;
	pix_sum_size_bytes		= pix_sum_size * sizeof(float) * 4;																				// NB the data returned is one float4 per group, for the base image, holding hsv channels plus entry[3]=pixel count.


	// d_disp_sum_size			=  1 + ceil( (float)( MipMap[(mm_num_reductions+1) + MiM_READ_OFFSET] ) / (float)local_work_size ) ;			// mm_size_bytes_C1 =	mm_size_bytes_C1	= temp2.total() * temp2.elemSize();
	// d_disp_sum_size_bytes	=  d_disp_sum_size * sizeof(float) * 4;
																																			if(verbosity>local_verbosity_threshold) cout <<"\nRunCL::initialize_RunCL_chk finished ############################################################\n"<<flush;
}


void RunCL::set_cam_bufs( cv::Matx44f k,  cv::Matx44f inv_k,  cv::Matx44f pose,  cv::Matx44f k2k ){
	int local_verbosity_threshold = -2; //V_RUNCL_SET_CAM_BUFS;
	string fname = "RunCL::set_cam_bufs( )";
																																			if(verbosity>local_verbosity_threshold) {
																																				cout<<"\nRunCL::"<<fname<<"(  )_chk0"<<flush;
																																			}
																																			// NB Orthographic camera, See notes in convertTransforms.cpp , cv::Matx44f generate_invK_(cv::Matx44f K_, int verbosity){..}
																																			// 4x4 perspective matrix is not invertable for points at infinity. We correct ortho->perspective in the kernel by dividing by Z.
	//Matx44f_To_float16arry(Matx44f matx, float arry[16]);
	//PRINT_FLOAT_16(a, txt)
	float k_arry[16], inv_k_arry[16], pose_arry[16], k2k_arry[16];
	Matx44f_To_float16arry( k,		k_arry		);		PRINT_FLOAT_16(k_arry, )
	Matx44f_To_float16arry( inv_k,	inv_k_arry	);
	Matx44f_To_float16arry( pose,	pose_arry	);
	Matx44f_To_float16arry( k2k,	k2k_arry	);

	float k_buf_arr[16];
	void * ptr = k_buf_arr;
																																			PRINT_MATX44F(k,);
	_clEnqueueWriteBuffer( uload_queue, 	K_buf,		CL_FALSE, 0, 16 * sizeof( float), k_arry, 			fname);							cout<<"\nRunCL::"<<fname<<"(  )_chk1"<<flush;
	//ReadOutput( (uchar*)ptr, K_buf, sizeof(float)*16, 0 );
	//PRINT_FLOAT_16(k_buf_arr, )
																																			PRINT_MATX44F(inv_k,);
	_clEnqueueWriteBuffer( uload_queue, 	inv_K_buf,	CL_FALSE, 0, 16 * sizeof( float), inv_k_arry, 		fname);							cout<<"\nRunCL::"<<fname<<"(  )_chk2"<<flush;
																																			PRINT_MATX44F(pose,);
	_clEnqueueWriteBuffer( uload_queue, 	pose_buf,	CL_FALSE, 0, 16 * sizeof( float), pose_arry, 		fname);							cout<<"\nRunCL::"<<fname<<"(  )_chk3"<<flush;
																																			PRINT_MATX44F(k2k,);
	_clEnqueueWriteBuffer( uload_queue, 	k2kbuf,		CL_FALSE, 0, 16 * sizeof( float), k2k_arry,			fname);							cout<<"\nRunCL::"<<fname<<"(  )_chk4"<<flush;

	// void RunCL::ReadOutput(uchar* outmat, cl_mem buf_mem, size_t data_size, size_t offset/*=0*/)

	ReadOutput( (uchar*)ptr, K_buf, sizeof(float)*16, 0 );
	PRINT_FLOAT_16(k_buf_arr, )
	/*
	clEnqueueReadBuffer(cl_command_queue    command_queue,
                    cl_mem              buffer,
                    cl_bool             blocking_read,
                    size_t              offset,
                    size_t              size,
                    void *              ptr,
                    cl_uint             num_events_in_wait_list,
                    const cl_event *    event_wait_list,
                    cl_event *          event)
	*//*
	cl_event readEvt;
	cl_int status;

	status = clEnqueueReadBuffer( uload_queue,  K_buf, CL_FALSE, (size_t)0, (size_t)16*sizeof(float), ptr, 0, NULL, &readEvt);
	_cl_flush_finish(dload_queue, fname);
	if (status != CL_SUCCESS) { cout << "\nclEnqueueReadBuffer(..) status=" << checkerror(status) <<"\n"<<flush; exit_(status);}
	clReleaseEvent(readEvt);

	PRINT_FLOAT_16(k_buf_arr, )
	*/


}


void RunCL::mipmap_call_kernel(cl_kernel kernel_to_call, cl_command_queue queue_to_call, uint start, uint stop, bool layers_sequential, const size_t local_work_size){
	int local_verbosity_threshold = V_RUNCL_MIPMAP_CALL_KERNEL;//verbosity_mp["RunCL::mipmap_call_kernel"];// -2;
																																			if(verbosity>local_verbosity_threshold) {
																																				cout<<"\nRunCL::mipmap_call_kernel( cl_kernel "<<kernel_to_call<<",  cl_command_queue "<<queue_to_call<<",   start="<<start<<",   stop="<<stop<<
																																				", layers_sequential="<<layers_sequential<<",  local_work_size="<<local_work_size<<" )_chk0"<<flush;
																																				//cout <<"\nmm_num_reductions+1="<<mm_num_reductions+1<< ",  start="<<start<<",  stop="<<stop <<flush;
																																			}
	cl_event						ev;
	cl_int							res, status;
	for(uint reduction = start; reduction <= stop; reduction++) {																			// NB processes largest layer first.
																																			if(verbosity>local_verbosity_threshold) { cout<<"\nRunCL::mipmap_call_kernel(..)_chk1,  reduction="<<reduction<<",  num_threads[reduction]="<<num_threads[reduction]<<"  local_work_size="<<local_work_size<<flush; }
		//if (reduction>=start && reduction<stop){																							// compute num threads to launch & num_pixels in reduction
																																			//if(verbosity>local_verbosity_threshold) { cout<<"\nRunCL::mipmap_call_kernel(..)_chk2 :  num_threads[reduction]="<<num_threads[reduction]<<"  local_work_size="<<local_work_size<<flush; }
			res 	= clSetKernelArg(kernel_to_call, 0, sizeof(int), &reduction);							if (res    !=CL_SUCCESS)	{ cout <<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	;
			res 	= clEnqueueNDRangeKernel(queue_to_call, kernel_to_call, 1, 0, &num_threads[reduction], &local_work_size, 0, NULL, &ev); // run mipmap_float4_kernel, NB wait for own previous iteration.
																											if (res    != CL_SUCCESS)	{ cout << "\nres = " << checkerror(res) <<"\n"<<flush; exit_(res);}
			status 	= clFlush(queue_to_call);																if (status != CL_SUCCESS)	{ cout << "\nRunCL::mipmap_call_kernel( cl_kernel "<<kernel_to_call<<",  clFlush(queue_to_call) status  = "		<<status<<" "<< checkerror(status) <<"\n"<<flush; exit_(status);}
			if (layers_sequential==true) status 	= clWaitForEvents (1, &ev);								if (status != CL_SUCCESS)	{ cout << "\nRunCL::mipmap_call_kernel( cl_kernel "<<kernel_to_call<<") for loop,  clWaitForEventsh(1, &ev) ="	<<status<<" "<<checkerror(status)  <<"\n"<<flush; exit_(status);}

		//} 																																	// TODO execute layers in asynchronous parallel. i.e. relax clWaitForEvents.
	}if (layers_sequential==false) status 	= clWaitForEvents (1, &ev);										if (status != CL_SUCCESS)	{ cout << "\nRunCL::mipmap_call_kernel( cl_kernel "<<kernel_to_call<<") final,  clWaitForEventsh(1, &ev) ="		<<status<<" "<<checkerror(status)  <<"\n"<<flush; exit_(status);}
}

/*
void RunCL::layer_call_kernel(cl_kernel kernel_to_call, cl_command_queue queue_to_call, uint layer, const size_t local_work_size){
	int local_verbosity_threshold = V_RUNCL_LAYER_CALL_KERNEL;
	string fname = "RunCL::layer_call_kernel( )";
																																			if(verbosity>local_verbosity_threshold) {
																																				cout<<"\nRunCL::layer_call_kernel( cl_kernel "<<kernel_to_call<<",  cl_command_queue "<<queue_to_call<<",   layer="<<layer<<
																																				",  local_work_size="<<local_work_size<<" )_chk0"
																																				",  mm_layerstep="<<mm_layerstep<<flush;
																																			}
	cl_event		ev;
	cl_int			res, status;
	uint 			reduction 		= layer;
	uint 			read_offset		= lookup_table_offset[reduction];																		cout << "\nread_offset = "<<read_offset<<"  "<<flush;

	_clSetKernelArg( kernel_to_call,  			0, sizeof( uint), &read_offset,				fname);										// 	__private	uint	read_offset,
																																			if(verbosity>local_verbosity_threshold) { cout<<"\nRunCL::layer_call_kernel(..)_chk1,  reduction="<<reduction<<",  num_threads[reduction]="<<num_threads[reduction]<<"  local_work_size="<<local_work_size<<flush; }

	res 	= clEnqueueNDRangeKernel(queue_to_call, kernel_to_call, 1, 0, &num_threads[reduction], &local_work_size, 0, NULL, &ev); // run mipmap_float4_kernel, NB wait for own previous iteration.

																	if (res    != CL_SUCCESS)	{ cout << "\nres = " << checkerror(res) <<"\n"<<flush; exit_(res);}
	status 	= clFlush(queue_to_call);								if (status != CL_SUCCESS)	{ cout << "\nRunCL::layer_call_kernel( cl_kernel "<<kernel_to_call<<",  clFlush(queue_to_call) status  = "		<<status<<" "<< checkerror(status) <<"\n"<<flush; exit_(status);}
	status 	= clWaitForEvents (1, &ev);								if (status != CL_SUCCESS)	{ cout << "\nRunCL::layer_call_kernel( cl_kernel "<<kernel_to_call<<") for loop,  clWaitForEventsh(1, &ev) ="	<<status<<" "<<checkerror(status)  <<"\n"<<flush; exit_(status);}
																																			if(verbosity>local_verbosity_threshold) { cout<<"\nRunCL::layer_call_kernel(..)_finished\n"<<flush; }
}
*/

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
	initialize_current_frames();
	// test_update_current_frames_idx(64);	// NB Only for debugging.
	//imgmem				= clCreateBuffer(m_context, CL_MEM_READ_ONLY  						, mm_size_bytes_C4,  		0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 1= "<<checkerror(res)<<"\n"<<flush;exit_(res);}

	imgmem_blurred		= clCreateBuffer(m_context, CL_MEM_READ_WRITE  						, mm_size_bytes_C4,  		0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 1= "<<checkerror(res)<<"\n"<<flush;exit_(res);}
/*
	gxmem				= clCreateBuffer(m_context, CL_MEM_READ_WRITE 						, mm_size_bytes_C8, 		0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 2= "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	gymem				= clCreateBuffer(m_context, CL_MEM_READ_WRITE 						, mm_size_bytes_C8, 		0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 3= "<<checkerror(res)<<"\n"<<flush;exit_(res);}

	k_map_mem			= clCreateBuffer(m_context, CL_MEM_READ_WRITE 					,10 * mm_size_bytes_C1,			0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 5= "<<checkerror(res)<<"\n"<<flush;exit_(res);} // camera intrinsic map
	dist_map_mem		= clCreateBuffer(m_context, CL_MEM_READ_WRITE 					,28 * mm_size_bytes_C1,			0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 6= "<<checkerror(res)<<"\n"<<flush;exit_(res);} // distorsion map
*/
	SE3_grad_map_mem 	= clCreateBuffer(m_context, CL_MEM_READ_WRITE 					,6*2* mm_size_bytes_C8,			0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 7= "<<checkerror(res)<<"\n"<<flush;exit_(res);} // SE3_map * img grad, 6DoF*8channels=48     // 6DoF*3channels=18,but 4*6=24 because hsv img gradient is held in float4

//	keyframe_SE3_grad_map_mem = clCreateBuffer(m_context, CL_MEM_READ_WRITE 			,6*2* mm_size_bytes_C8,			0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 8= "<<checkerror(res)<<"\n"<<flush;exit_(res);}

	SE3_weight_map_mem	= clCreateBuffer(m_context, CL_MEM_READ_WRITE 					,24 * mm_size_bytes_C1,			0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 9= "<<checkerror(res)<<"\n"<<flush;exit_(res);}

	SE3_incr_map_mem	= clCreateBuffer(m_context, CL_MEM_READ_WRITE 					,24 * mm_size_bytes_C1,			0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 9= "<<checkerror(res)<<"\n"<<flush;exit_(res);} // For debugging before summation.
	SE3_map_mem			= clCreateBuffer(m_context, CL_MEM_READ_WRITE 					,12 * mm_size_bytes_C1,			0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 10= "<<checkerror(res)<<"\n"<<flush;exit_(res);}	// (row, col) increment fo each parameter.
	basemem				= clCreateBuffer(m_context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, image_size_bytes,			0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 11= "<<checkerror(res)<<"\n"<<flush;exit_(res);} // Original image CV_8UC3
	depth_mem_temp		= clCreateBuffer(m_context, CL_MEM_READ_WRITE 						, mm_size_bytes_C1,			0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 12= "<<checkerror(res)<<"\n"<<flush;exit_(res);} // Used to be : Copy used by tracing & auto-calib. Now spare buffer for upload & computations
	depth_mem_GT		= clCreateBuffer(m_context, CL_MEM_READ_WRITE 						, mm_size_bytes_C1,			0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 13= "<<checkerror(res)<<"\n"<<flush;exit_(res);} // Where depthmap GT mimpap is constructed.
/*
	keyframe_imgmem		= clCreateBuffer(m_context, CL_MEM_READ_WRITE  						, mm_size_bytes_C4,			0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 14= "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	keyframe_imgmem_HSV_grad = clCreateBuffer(m_context, CL_MEM_READ_WRITE  				, mm_size_bytes_C8,			0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 14= "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	keyframe_depth_mem	= clCreateBuffer(m_context, CL_MEM_READ_WRITE 						, mm_size_bytes_C1,			0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 15= "<<checkerror(res)<<"\n"<<flush;exit_(res);} // The depth map for tracking, i.e. used when adding frames to the cost volume.
	keyframe_depth_mem_GT= clCreateBuffer(m_context, CL_MEM_READ_WRITE 						, mm_size_bytes_C1,			0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 15.5= "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	//keyframe_basemem	= clCreateBuffer(m_context, CL_MEM_READ_ONLY  						, mm_size_bytes_C4,  		0, &res);			if(res!=CL_SUCCESS){cout<<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	// Depth mapping buffers
	keyframe_g1mem		= clCreateBuffer(m_context, CL_MEM_READ_WRITE 						, mm_size_bytes_C8, 		0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 16= "<<checkerror(res)<<"\n"<<flush;exit_(res);}
*/
	g1mem				= clCreateBuffer(m_context, CL_MEM_READ_WRITE 						, mm_size_bytes_C8, 		0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 16= "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	depth_mem			= clCreateBuffer(m_context, CL_MEM_READ_WRITE 						, mm_size_bytes_C1,			0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 17= "<<checkerror(res)<<"\n"<<flush;exit_(res);}
/*
	dmem				= clCreateBuffer(m_context, CL_MEM_READ_WRITE 						, mm_size_bytes_C1,			0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 17= "<<checkerror(res)<<"\n"<<flush;exit_(res);} // depth in the mapping calculation.
	amem				= clCreateBuffer(m_context, CL_MEM_READ_WRITE 						, mm_size_bytes_C1, 		0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 18= "<<checkerror(res)<<"\n"<<flush;exit_(res);} // 'auxiliary variable to depth" in the mapping calculation.
	lomem				= clCreateBuffer(m_context, CL_MEM_READ_WRITE 						, mm_size_bytes_C1, 		0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 19= "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	himem				= clCreateBuffer(m_context, CL_MEM_READ_WRITE 						, mm_size_bytes_C1, 		0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 20= "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	mean_mem			= clCreateBuffer(m_context, CL_MEM_READ_WRITE 						, mm_size_bytes_C1, 		0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 20= "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	
	qmem				= clCreateBuffer(m_context, CL_MEM_READ_WRITE					, 2 * mm_size_bytes_C1, 		0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 21= "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	qmem2				= clCreateBuffer(m_context, CL_MEM_READ_WRITE					, 2 * mm_size_bytes_C1, 		0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 21= "<<checkerror(res)<<"\n"<<flush;exit_(res);}


	dbg_databuf			= clCreateBuffer(m_context, CL_MEM_READ_WRITE 						, mm_vol_size_bytes, 		0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 22= "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	cdatabuf			= clCreateBuffer(m_context, CL_MEM_READ_WRITE 						, mm_vol_size_bytes, 		0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 22= "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	temp_cdatabuf		= clCreateBuffer(m_context, CL_MEM_READ_WRITE 						, mm_vol_size_bytes, 		0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 22= "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	cdatabuf_8chan		= clCreateBuffer(m_context, CL_MEM_READ_WRITE 					, 8 * mm_vol_size_bytes,		0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 22= "<<checkerror(res)<<"\n"<<flush;exit_(res);}	// For investigating & tuning cost
	hdatabuf 			= clCreateBuffer(m_context, CL_MEM_READ_WRITE 						, mm_vol_size_bytes, 		0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 23= "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	temp_hdatabuf 		= clCreateBuffer(m_context, CL_MEM_READ_WRITE 						, mm_vol_size_bytes, 		0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 23= "<<checkerror(res)<<"\n"<<flush;exit_(res);}

	img_sum_buf 		= clCreateBuffer(m_context, CL_MEM_READ_WRITE 					, 2 * mm_vol_size_bytes,		0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 24= "<<checkerror(res)<<"\n"<<flush;exit_(res);}	// float debug buffer.
*/
	fp32_param_buf		= clCreateBuffer(m_context, CL_MEM_READ_ONLY  					, 16* sizeof(float),			0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 25= "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	k2kbuf				= clCreateBuffer(m_context, CL_MEM_READ_ONLY  ,tracking_num_samples*16*sizeof(float),			0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 26= "<<checkerror(res)<<"\n"<<flush;exit_(res);}
/*
	invk2kbuf			= clCreateBuffer(m_context, CL_MEM_READ_ONLY  					, 16* sizeof(float),			0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 26= "<<checkerror(res)<<"\n"<<flush;exit_(res);}

	SO3_k2kbuf			= clCreateBuffer(m_context, CL_MEM_READ_ONLY  					, 9 * sizeof(float),  			0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 27= "<<checkerror(res)<<"\n"<<flush;exit_(res);} // NB used in place of k2kbuf for RunCL::estimateSO3(..)
*/
	SE3_k2kbuf			= clCreateBuffer(m_context, CL_MEM_READ_ONLY  					,6*16*sizeof(float),			0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 28= "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	uint_param_buf		= clCreateBuffer(m_context, CL_MEM_READ_ONLY  					, 8 * sizeof(uint), 			0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 29= "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	mipmap_buf			= clCreateBuffer(m_context, CL_MEM_READ_ONLY  					, 8*8*sizeof(uint), 			0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 30= "<<checkerror(res)<<"\n"<<flush;exit_(res);}
/*
	gaussian_buf		= clCreateBuffer(m_context, CL_MEM_READ_ONLY  					, 9 * sizeof(float),			0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 31= "<<checkerror(res)<<"\n"<<flush;exit_(res);}	//  TODO load gaussian kernel & size from conf.json .

	se3_sum_mem			= clCreateBuffer(m_context, CL_MEM_READ_WRITE 						, se3_sum_size_bytes,		0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 32= "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	se3_sum2_mem		= clCreateBuffer(m_context, CL_MEM_READ_WRITE 						, se3_sum2_size_bytes,		0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 33= "<<checkerror(res)<<"\n"<<flush;exit_(res);}

	se3_weight_sum_mem	= clCreateBuffer(m_context, CL_MEM_READ_WRITE 						, se3_sum_size_bytes,		0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 32= "<<checkerror(res)<<"\n"<<flush;exit_(res);}
*/
	SE3_rho_map_mem		= clCreateBuffer(m_context, CL_MEM_READ_WRITE  , tracking_num_samples*2*mm_size_bytes_C4,  		0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 34= "<<checkerror(res)<<"\n"<<flush;exit_(res);}
//	se3_sum_rho_sq_mem	= clCreateBuffer(m_context, CL_MEM_READ_WRITE  , tracking_num_samples*pix_sum_size_bytes,  		0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 35= "<<checkerror(res)<<"\n"<<flush;exit_(res);}  // TODO what size should this be ?

	img_stats_buf		= clCreateBuffer(m_context, CL_MEM_READ_WRITE  						, img_stats_size_bytes,		0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 36= "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	pix_sum_mem			= clCreateBuffer(m_context, CL_MEM_READ_WRITE 						, pix_sum_size_bytes,		0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 37= "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	var_sum_mem			= clCreateBuffer(m_context, CL_MEM_READ_WRITE 						, pix_sum_size_bytes,		0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 38= "<<checkerror(res)<<"\n"<<flush;exit_(res);}

	HSV_grad_mem		= clCreateBuffer(m_context, CL_MEM_READ_WRITE  						, mm_size_bytes_C8,  		0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 39= "<<checkerror(res)<<"\n"<<flush;exit_(res);}
/*
	dmem_disparity		= clCreateBuffer(m_context, CL_MEM_READ_WRITE 						, mm_size_bytes_C4,			0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 40= "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	dmem_disparity_sum	= clCreateBuffer(m_context, CL_MEM_READ_WRITE 						, pix_sum_size_bytes,		0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 41= "<<checkerror(res)<<"\n"<<flush;exit_(res);}
*/

/*	// buffers for stereo disparity
	lookup_table_buf				= clCreateBuffer(m_context, CL_MEM_READ_WRITE 			, mm_size_bytes_C4,			0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 41= "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	ref_img_buf						= clCreateBuffer(m_context, CL_MEM_READ_WRITE 			, mm_size_bytes_C4,			0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 41= "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	new_img_buf						= clCreateBuffer(m_context, CL_MEM_READ_WRITE 			, mm_size_bytes_C4,			0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 41= "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	warped_img_buf					= clCreateBuffer(m_context, CL_MEM_READ_WRITE 			, mm_size_bytes_C4,			0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 41= "<<checkerror(res)<<"\n"<<flush;exit_(res);}

	ref_img_mean_rows_buf			= clCreateBuffer(m_context, CL_MEM_READ_WRITE 			, mm_size_bytes_C4,			0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 41= "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	ref_img_mean_buf				= clCreateBuffer(m_context, CL_MEM_READ_WRITE 			, mm_size_bytes_C4,			0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 41= "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	warped_img_mean_rows_buf		= clCreateBuffer(m_context, CL_MEM_READ_WRITE 			, mm_size_bytes_C4,			0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 41= "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	warped_img_mean_buf				= clCreateBuffer(m_context, CL_MEM_READ_WRITE 			, mm_size_bytes_C4,			0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 41= "<<checkerror(res)<<"\n"<<flush;exit_(res);}

	ref_img_diff_buf				= clCreateBuffer(m_context, CL_MEM_READ_WRITE 			, mm_size_bytes_C4,			0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 41= "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	warped_img_diff_buf				= clCreateBuffer(m_context, CL_MEM_READ_WRITE 			, mm_size_bytes_C4,			0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 41= "<<checkerror(res)<<"\n"<<flush;exit_(res);}

	ref_img_sigma_rows_buf			= clCreateBuffer(m_context, CL_MEM_READ_WRITE 			, mm_size_bytes_C4,			0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 41= "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	ref_img_mean_sigma_buf			= clCreateBuffer(m_context, CL_MEM_READ_WRITE 			, mm_size_bytes_C4,			0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 41= "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	warped_img_sigma_rows_buf		= clCreateBuffer(m_context, CL_MEM_READ_WRITE 			, mm_size_bytes_C4,			0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 41= "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	warped_img_mean_sigma_buf		= clCreateBuffer(m_context, CL_MEM_READ_WRITE 			, mm_size_bytes_C4,			0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 41= "<<checkerror(res)<<"\n"<<flush;exit_(res);}

	covariance_rows_buf				= clCreateBuffer(m_context, CL_MEM_READ_WRITE 		, 5 * mm_size_bytes_C4,			0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 41= "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	covariance_buf					= clCreateBuffer(m_context, CL_MEM_READ_WRITE 		, 5 * mm_size_bytes_C4,			0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 41= "<<checkerror(res)<<"\n"<<flush;exit_(res);}

	correlation_buf					= clCreateBuffer(m_context, CL_MEM_READ_WRITE 		, 5 * mm_size_bytes_C4,			0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 41= "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	correlation_blurred_buf			= clCreateBuffer(m_context, CL_MEM_READ_WRITE 		, 5 * mm_size_bytes_C4,			0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 41= "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	warp_buf						= clCreateBuffer(m_context, CL_MEM_READ_WRITE 		, 2 * mm_size_bytes_C4 / * C1 * / ,	0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 41= "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	warp_ref_buf					= clCreateBuffer(m_context, CL_MEM_READ_WRITE 		, 2 * mm_size_bytes_C4 / * C1 * / ,	0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 41= "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	confidence_buf					= clCreateBuffer(m_context, CL_MEM_READ_WRITE 			, mm_size_bytes_C4,			0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 41= "<<checkerror(res)<<"\n"<<flush;exit_(res);}

	warp_buf_regularized			= clCreateBuffer(m_context, CL_MEM_READ_WRITE 		, 2 * mm_size_bytes_C4 / * C1 * / ,	0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 41= "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	confidence_buf_regularized		= clCreateBuffer(m_context, CL_MEM_READ_WRITE 			, mm_size_bytes_C4,			0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 41= "<<checkerror(res)<<"\n"<<flush;exit_(res);}


	warp_GT_buf						= clCreateBuffer(m_context, CL_MEM_READ_WRITE 		, 2 * mm_size_bytes_C4 / * C1 * / ,	0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 41= "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	warp_error_buf					= clCreateBuffer(m_context, CL_MEM_READ_WRITE 		, 2 * mm_size_bytes_C4 / * C1 * / ,	0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 41= "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	depth_error_buf					= clCreateBuffer(m_context, CL_MEM_READ_WRITE 		, 2 * mm_size_bytes_C4 / * C1 * / ,	0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 41= "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	depth_est_buf					= clCreateBuffer(m_context, CL_MEM_READ_WRITE 		, 2 * mm_size_bytes_C4 / * C1 * / ,	0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 41= "<<checkerror(res)<<"\n"<<flush;exit_(res);}
*/
	// buffers for patch kernel based Dynamic_slam
	pose_buf						= clCreateBuffer(m_context, CL_MEM_READ_WRITE 		, sizeof(float)*16,				0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 41= "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	pose_update_buf					= clCreateBuffer(m_context, CL_MEM_READ_WRITE 		, sizeof(float)*16,				0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 41= "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	distorsion_update_buf			= clCreateBuffer(m_context, CL_MEM_READ_WRITE 		, sizeof(float)*6,				0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 41= "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	old_results_buf					= clCreateBuffer(m_context, CL_MEM_READ_WRITE 		, sizeof(float)*6*4,			0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 41= "<<checkerror(res)<<"\n"<<flush;exit_(res);}

	K_buf							= clCreateBuffer(m_context, CL_MEM_READ_WRITE 		, sizeof(float)*16,				0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 41= "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	inv_K_buf						= clCreateBuffer(m_context, CL_MEM_READ_WRITE 		, sizeof(float)*16,				0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 41= "<<checkerror(res)<<"\n"<<flush;exit_(res);}

	patch_lookup_table_buf			= clCreateBuffer(m_context, CL_MEM_READ_WRITE 			, mm_size_bytes_C4,			0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 41= "<<checkerror(res)<<"\n"<<flush;exit_(res);}
	SE3_hessian_pinv_map_mem		= clCreateBuffer(m_context, CL_MEM_READ_WRITE 		, 2 * mm_size_bytes_C4,			0, &res);			if(res!=CL_SUCCESS){cout<<"\nres 41= "<<checkerror(res)<<"\n"<<flush;exit_(res);}

																																		if(verbosity>local_verbosity_threshold) {
																																			cout << "\n\nRunCL::allocatemem_chk3\n\n" << flush;
/*																																			cout << ",dmem = " 			<< dmem << endl;
																																			cout << ",amem = " 			<< amem << endl;
																																			cout << ",gxmem = " 		<< gxmem << endl;
																																			cout << ",gymem = " 		<< gymem << endl;
																																			cout << ",qmem = " 			<< qmem << endl;
																																			//cout << ",g1mem = " 		<< g1mem << endl;
																																			cout << ",lomem = " 		<< lomem << endl;
																																			cout << ",himem = " 		<< himem << endl;
																																			cout << ",cdatabuf = " 		<< cdatabuf << endl;
																																			cout << ",hdatabuf = " 		<< hdatabuf << endl;
*/
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
	status = clEnqueueWriteBuffer(uload_queue, k2kbuf,			CL_FALSE, 0, 16 * sizeof(float), fp32_k2keyframe, 		0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.5\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	status = clEnqueueWriteBuffer(uload_queue, uint_param_buf,	CL_FALSE, 0,  8 * sizeof(uint),	 uint_params, 			0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.5\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	status = clEnqueueWriteBuffer(uload_queue, mipmap_buf,		CL_FALSE, 0,  8*8* sizeof(uint), MipMap, 				0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.5\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);

	status = clEnqueueWriteBuffer(uload_queue, basemem, 		CL_FALSE, 0, image_size_bytes, 	baseImage.data, 		0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.6\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
																																		if(verbosity>local_verbosity_threshold) cout <<"\n\nRunCL::allocatemem_chk4.2\n\n" << flush;
	float depth = 1/( obj["max_depth"].asFloat() - (obj["min_depth"].asFloat() / 2)  );
	float zero  = 0;
	float one   = 1;
	float default_depth = fp32_params[MAX_INV_DEPTH] / 2.0f;
	//																																	if(verbosity>local_verbosity_threshold) cout << "\n\nRunCL::allocatemem_chk4 \t Initial inverse depth = "<< depth <<"\n\n" << flush;
/*
	status = clEnqueueFillBuffer(uload_queue, gxmem, 				&zero, 			sizeof(float), 	0, mm_size_bytes_C4, 		0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.3\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	status = clEnqueueFillBuffer(uload_queue, gymem, 				&zero, 			sizeof(float), 	0, mm_size_bytes_C4, 		0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.4\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
*/																																		if(verbosity>local_verbosity_threshold) cout <<"\n\nRunCL::allocatemem_chk4.1\n\n" << flush;

	status = clEnqueueFillBuffer(uload_queue, depth_mem, 			&depth, 		sizeof(float),   0, mm_size_bytes_C1,		0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.6\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
//	status = clEnqueueFillBuffer(uload_queue, dmem, 				&default_depth, sizeof(float),   0, mm_size_bytes_C1, 		0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.8\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);

	status = clEnqueueFillBuffer(uload_queue, depth_mem_temp, 		&default_depth, sizeof(float),   0, mm_size_bytes_C1, 		0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.8\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	status = clEnqueueFillBuffer(uload_queue, depth_mem_GT, 		&default_depth, sizeof(float),   0, mm_size_bytes_C1, 		0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.8\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
/*
	status = clEnqueueFillBuffer(uload_queue, keyframe_depth_mem, 	&default_depth, sizeof(float),   0, mm_size_bytes_C1, 		0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.8\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	status = clEnqueueFillBuffer(uload_queue, keyframe_depth_mem_GT,&default_depth, sizeof(float),   0, mm_size_bytes_C1, 		0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.8\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);

	status = clEnqueueFillBuffer(uload_queue, cdatabuf, 			&default_depth, sizeof(float),   0, mm_vol_size_bytes, 		0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.8\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	status = clEnqueueFillBuffer(uload_queue, cdatabuf_8chan, 		&zero, 			sizeof(float),   0, mm_vol_size_bytes*8, 	0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.8\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);

	status = clEnqueueFillBuffer(uload_queue, hdatabuf, 			&zero, 			sizeof(float),   0, mm_vol_size_bytes, 		0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.9\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	status = clEnqueueFillBuffer(uload_queue, img_sum_buf, 			&zero, 			sizeof(float),   0, mm_vol_size_bytes, 		0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.10\n"<< endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	//status = clEnqueueFillBuffer(uload_queue, se3_sum_mem, 		&zero, 			sizeof(float),   0, se3_sum_size_bytes,		0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.6\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
*/
	status = clEnqueueFillBuffer(uload_queue, HSV_grad_mem, 		&zero, 			sizeof(float),   0, mm_size_bytes_C8, 		0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.3\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
/*
	status = clEnqueueFillBuffer(uload_queue, dmem_disparity, 		&zero, 			sizeof(float),   0, mm_size_bytes_C4, 		0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.8\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	status = clEnqueueFillBuffer(uload_queue, dmem_disparity_sum, 	&zero, 			sizeof(float),   0, pix_sum_size_bytes, 	0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.8\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);

	//status = clEnqueueFillBuffer(uload_queue, binocular_disparity, 	&zero, 			sizeof(float),   0, mm_size_bytes_C4, 		0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.8\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	//status = clEnqueueFillBuffer(uload_queue, binocular_rho, 		&zero, 			sizeof(float),   0, mm_size_bytes_C4, 		0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.8\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
*/
/*
	status = clEnqueueFillBuffer(uload_queue, lookup_table_buf, 	&zero, 			sizeof(float),   0, mm_size_bytes_C4, 		0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.8\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	// status = clEnqueueFillBuffer(uload_queue, curr_img_buf, 		&zero, 			sizeof(float),   0, mm_size_bytes_C4, 		0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.8\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	// status = clEnqueueFillBuffer(uload_queue, curr_img_sq_buf, 		&zero, 			sizeof(float),   0, mm_size_bytes_C4, 		0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.8\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	// status = clEnqueueFillBuffer(uload_queue, curr_img_var_buf,		&zero, 			sizeof(float),   0, mm_size_bytes_C4, 		0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.8\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	status = clEnqueueFillBuffer(uload_queue, new_img_buf,	 		&zero, 			sizeof(float),   0, mm_size_bytes_C4, 		0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.8\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	// status = clEnqueueFillBuffer(uload_queue, new_img_warped_buf,	&zero, 			sizeof(float),   0, mm_size_bytes_C4, 		0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.8\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	// status = clEnqueueFillBuffer(uload_queue, new_img_sq_buf, 		&zero, 			sizeof(float),   0, mm_size_bytes_C4, 		0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.8\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	// status = clEnqueueFillBuffer(uload_queue, new_img_var_buf, 		&zero, 			sizeof(float),   0, mm_size_bytes_C4, 		0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.8\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);

	// status = clEnqueueFillBuffer(uload_queue, img_covar_buf, 		&zero, 			sizeof(float),   0, mm_size_bytes_C4 * 5,	0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.8\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	// status = clEnqueueFillBuffer(uload_queue, img_corr_buf, 		&zero, 			sizeof(float),   0, mm_size_bytes_C4 * 5,	0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.8\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
	status = clEnqueueFillBuffer(uload_queue, warp_buf,		 		&zero, 			sizeof(float),   0, mm_size_bytes_C1 * 2,	0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.8\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);
*/
	status = clEnqueueFillBuffer(uload_queue, patch_lookup_table_buf,	&zero, 		sizeof(float), 	0, mm_size_bytes_C4, 		0, NULL, &writeEvt);	if (status != CL_SUCCESS)	{ cout << "\nstatus = " << checkerror(status) <<"\n"<<flush; cout << "Error: allocatemem_chk1.3\n" << endl;exit_(status);}	clFlush(uload_queue); status = clFinish(uload_queue);

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
																																		//	DownloadAndSave( 			gxmem,		ss.str(), paths.at("gxmem"), 		mm_size_bytes_C1, 	mm_Image_size, 		CV_32FC1, 		false , 1);	cout << "\ngxmem,"		<< flush;
																																		//	DownloadAndSaveVolume(		cdatabuf, 	ss.str(), paths.at("cdatabuf"), 	mm_size_bytes_C1,	mm_Image_size, 		CV_32FC1,  		false , 1);	cout << "\ncdatabuf,"	<< flush;
																																		}
	/*	// TODO  update and reactivate the old kernels
	// set kernelArg. NB "0 &k2kbuf" & "2 &imgmem" set in calcCostVol(..)
	res = clSetKernelArg(cost_kernel, 1, sizeof(cl_mem),  &basemem);		if(res!=CL_SUCCESS){cout<<"\nbasemem res= "   		<<checkerror(res)<<"\n"<<flush;exit_(res);} // base
	res = clSetKernelArg(cost_kernel, 3, sizeof(cl_mem),  &cdatabuf);		if(res!=CL_SUCCESS){cout<<"\ncdatabuf res = " 		<<checkerror(res)<<"\n"<<flush;exit_(res);} // cdata
	res = clSetKernelArg(cost_kernel, 4, sizeof(cl_mem),  &hdatabuf);		if(res!=CL_SUCCESS){cout<<"\nhdatabuf res = " 		<<checkerror(res)<<"\n"<<flush;exit_(res);} // hdata
	res = clSetKernelArg(cost_kernel, 5, sizeof(cl_mem),  &lomem);			if(res!=CL_SUCCESS){cout<<"\nlomem res = "    		<<checkerror(res)<<"\n"<<flush;exit_(res);} // lo
	res = clSetKernelArg(cost_kernel, 6, sizeof(cl_mem),  &himem);			if(res!=CL_SUCCESS){cout<<"\nhimem res = "    		<<checkerror(res)<<"\n"<<flush;exit_(res);} // hi
	res = clSetKernelArg(cost_kernel, 7, sizeof(cl_mem),  &amem);			if(res!=CL_SUCCESS){cout<<"\namem res = "     		<<checkerror(res)<<"\n"<<flush;exit_(res);} // a
	res = clSetKernelArg(cost_kernel, 8, sizeof(cl_mem),  &dmem);			if(res!=CL_SUCCESS){cout<<"\ndmem res = "     		<<checkerror(res)<<"\n"<<flush;exit_(res);} // d
	res = clSetKernelArg(cost_kernel, 9, sizeof(cl_mem),  &fp16_param_buf);	if(res!=CL_SUCCESS){cout<<"\nparam_buf res = "		<<checkerror(res)<<"\n"<<flush;exit_(res);} // param_buf
	res = clSetKernelArg(cost_kernel,10, sizeof(cl_mem),  &img_sum_buf);	if(res!=CL_SUCCESS){cout<<"\nimg_sum_buf res = " 	<<checkerror(res)<<"\n"<<flush;exit_(res);} // cdata
	*/
																																		if(verbosity>local_verbosity_threshold) {
																																			cout << "\n\nRunCL::allocatemem_chk6\n\n" << flush;
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
																																		if(verbosity>local_verbosity_threshold) cout << "RunCL::allocatemem_finished #############################################################################\n\n" << flush;
}

RunCL::~RunCL(){  // TODO  ? Replace individual buffer clearance with the large array method from Morphogenesis &  fluids_v3 ? OR a C++ vector ?
	int local_verbosity_threshold = V__RUNCL;//verbosity_mp["RunCL::allocatemem"];																	cout<<"\nRunCL::~RunCL_chk0_called"<<flush;
	cl_int status;																														// release memory

	for (uint i=0; i<num_current_frames; i++ ) {
		status = clReleaseMemObject(imgmem[i]);					if (status != CL_SUCCESS)	{ cout << "\nimgmem                         status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_01"<<flush;
		status = clReleaseMemObject(velmap[i]);					if (status != CL_SUCCESS)	{ cout << "\nimgmem                         status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_01"<<flush;
	}
	//status = clReleaseMemObject(imgmem);						if (status != CL_SUCCESS)	{ cout << "\nimgmem                         status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_01"<<flush;
	status = clReleaseMemObject(imgmem_blurred);				if (status != CL_SUCCESS)	{ cout << "\nimgmem_blurred                 status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_02"<<flush;
/*
	status = clReleaseMemObject(gxmem);							if (status != CL_SUCCESS)	{ cout << "\ngxmem                          status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_03"<<flush;
	status = clReleaseMemObject(gymem);							if (status != CL_SUCCESS)	{ cout << "\ngymem                          status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_04"<<flush;

	status = clReleaseMemObject(k_map_mem);						if (status != CL_SUCCESS)	{ cout << "\nk_map_mem                      status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_06"<<flush;
	status = clReleaseMemObject(dist_map_mem);					if (status != CL_SUCCESS)	{ cout << "\ndist_map_mem                   status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_07"<<flush;
*/
	status = clReleaseMemObject(SE3_grad_map_mem);				if (status != CL_SUCCESS)	{ cout << "\nSE3_grad_map_mem               status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_08"<<flush;
//	status = clReleaseMemObject(keyframe_SE3_grad_map_mem);		if (status != CL_SUCCESS)	{ cout << "\nkeyframe_SE3_grad_map_mem      status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_09"<<flush;
	status = clReleaseMemObject(SE3_weight_map_mem);			if (status != CL_SUCCESS)	{ cout << "\nSE3_weight_map_mem             status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_10"<<flush;

	status = clReleaseMemObject(SE3_incr_map_mem);				if (status != CL_SUCCESS)	{ cout << "\nSE3_incr_map_mem               status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_11"<<flush;
	status = clReleaseMemObject(SE3_map_mem);					if (status != CL_SUCCESS)	{ cout << "\nSE3_map_mem                    status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_12"<<flush;
	status = clReleaseMemObject(basemem);						if (status != CL_SUCCESS)	{ cout << "\nbasemem                        status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_13"<<flush;
	status = clReleaseMemObject(depth_mem_temp);				if (status != CL_SUCCESS)	{ cout << "\ndepth_mem                      status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_14"<<flush;
	status = clReleaseMemObject(depth_mem_GT);					if (status != CL_SUCCESS)	{ cout << "\ndepth_mem_GT                   status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_15"<<flush;
/*
	status = clReleaseMemObject(keyframe_depth_mem);			if (status != CL_SUCCESS)	{ cout << "\nkeyframe_depth_mem             status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_16"<<flush;
	status = clReleaseMemObject(keyframe_depth_mem_GT);			if (status != CL_SUCCESS)	{ cout << "\nkeyframe_depth_mem_GT          status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_17"<<flush;
	status = clReleaseMemObject(keyframe_imgmem);				if (status != CL_SUCCESS)	{ cout << "\nkeyframe_imgmem                status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_18"<<flush;
	status = clReleaseMemObject(keyframe_imgmem_HSV_grad);		if (status != CL_SUCCESS)	{ cout << "\nkeyframe_imgmem                status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_19"<<flush;
	status = clReleaseMemObject(keyframe_g1mem);				if (status != CL_SUCCESS)	{ cout << "\nkeyframe_g1mem                 status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_20"<<flush;
*/
	status = clReleaseMemObject(g1mem);							if (status != CL_SUCCESS)	{ cout << "\ng1mem                          status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_20"<<flush;
	status = clReleaseMemObject(depth_mem);						if (status != CL_SUCCESS)	{ cout << "\ndepth_mem                      status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_21"<<flush;
/*
	status = clReleaseMemObject(dmem);							if (status != CL_SUCCESS)	{ cout << "\ndmem                           status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_21"<<flush;
	status = clReleaseMemObject(amem);							if (status != CL_SUCCESS)	{ cout << "\namem                           status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_22"<<flush;
	status = clReleaseMemObject(lomem);							if (status != CL_SUCCESS)	{ cout << "\nlomem                          status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_23"<<flush;
	status = clReleaseMemObject(himem);							if (status != CL_SUCCESS)	{ cout << "\nhimem                          status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_24"<<flush;
	status = clReleaseMemObject(mean_mem);						if (status != CL_SUCCESS)	{ cout << "\nmean_mem                       status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_24.5"<<flush;
	
	status = clReleaseMemObject(qmem);							if (status != CL_SUCCESS)	{ cout << "\nqmem                           status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_25"<<flush;
	status = clReleaseMemObject(qmem2);							if (status != CL_SUCCESS)	{ cout << "\nqmem2                          status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_25.1"<<flush;

	status = clReleaseMemObject(dbg_databuf);					if (status != CL_SUCCESS)	{ cout << "\ncdatabuf                       status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_25.5"<<flush;
	status = clReleaseMemObject(cdatabuf);						if (status != CL_SUCCESS)	{ cout << "\ncdatabuf                       status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_26"<<flush;
	status = clReleaseMemObject(temp_cdatabuf);					if (status != CL_SUCCESS)	{ cout << "\ntemp_cdatabuf                  status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_26.5"<<flush;
	status = clReleaseMemObject(cdatabuf_8chan);				if (status != CL_SUCCESS)	{ cout << "\ncdatabuf_8chan                 status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_27"<<flush;
	status = clReleaseMemObject(hdatabuf);						if (status != CL_SUCCESS)	{ cout << "\nhdatabuf                       status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_28"<<flush;
	status = clReleaseMemObject(temp_hdatabuf);					if (status != CL_SUCCESS)	{ cout << "\ntemp_hdatabuf                  status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_28.5"<<flush;


	status = clReleaseMemObject(img_sum_buf);					if (status != CL_SUCCESS)	{ cout << "\nimg_sum_buf                    status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_29"<<flush;
*/
	status = clReleaseMemObject(fp32_param_buf);				if (status != CL_SUCCESS)	{ cout << "\nfp32_param_buf                 status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_30"<<flush;
	status = clReleaseMemObject(k2kbuf);						if (status != CL_SUCCESS)	{ cout << "\nk2kbuf                         status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_31"<<flush;
/*
	status = clReleaseMemObject(invk2kbuf);						if (status != CL_SUCCESS)	{ cout << "\ninvk2kbuf                      status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_31"<<flush;

	status = clReleaseMemObject(SO3_k2kbuf);					if (status != CL_SUCCESS)	{ cout << "\nSO3_k2kbuf                     status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_32"<<flush;
*/
	status = clReleaseMemObject(SE3_k2kbuf);					if (status != CL_SUCCESS)	{ cout << "\nSE3_k2kbuf                     status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_33"<<flush;
	status = clReleaseMemObject(uint_param_buf);				if (status != CL_SUCCESS)	{ cout << "\nuint_param_buf                 status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_34"<<flush;
	status = clReleaseMemObject(mipmap_buf);					if (status != CL_SUCCESS)	{ cout << "\nmipmap_buf                     status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_35"<<flush;
/*
	status = clReleaseMemObject(gaussian_buf);					if (status != CL_SUCCESS)	{ cout << "\ngaussian_buf                   status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_36"<<flush;
	status = clReleaseMemObject(se3_sum_mem);					if (status != CL_SUCCESS)	{ cout << "\nse3_sum_mem                    status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_37"<<flush;
	status = clReleaseMemObject(se3_sum2_mem);					if (status != CL_SUCCESS)	{ cout << "\nse3_sum2_mem                   status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_38"<<flush;
	status = clReleaseMemObject(se3_weight_sum_mem);			if (status != CL_SUCCESS)	{ cout << "\nse3_weight_sum_mem             status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_39"<<flush;
*/
	status = clReleaseMemObject(SE3_rho_map_mem	);				if (status != CL_SUCCESS)	{ cout << "\nSE3_rho_map_mem                status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_40"<<flush;
//	status = clReleaseMemObject(se3_sum_rho_sq_mem);			if (status != CL_SUCCESS)	{ cout << "\nse3_sum_rho_sq_mem             status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_41"<<flush;
	status = clReleaseMemObject(img_stats_buf);					if (status != CL_SUCCESS)	{ cout << "\nimg_stats_buf                  status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_42"<<flush;
	status = clReleaseMemObject(pix_sum_mem);					if (status != CL_SUCCESS)	{ cout << "\npix_sum_mem                    status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_43"<<flush;
	status = clReleaseMemObject(var_sum_mem);					if (status != CL_SUCCESS)	{ cout << "\nvar_sum_mem                    status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_44"<<flush;
	status = clReleaseMemObject(HSV_grad_mem);					if (status != CL_SUCCESS)	{ cout << "\nHSV_grad_mem                   status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_45"<<flush;
/*
	status = clReleaseMemObject(dmem_disparity);				if (status != CL_SUCCESS)	{ cout << "\ndmem_disparity                 status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_45"<<flush;
	status = clReleaseMemObject(dmem_disparity_sum);			if (status != CL_SUCCESS)	{ cout << "\ndmem_disparity_sum             status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_47"<<flush;
*/

/*	// buffers for stereo disparity
	status = clReleaseMemObject(lookup_table_buf);				if (status != CL_SUCCESS)	{ cout << "\nlookup_table_buf               status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_48"<<flush;
	status = clReleaseMemObject(ref_img_buf);					if (status != CL_SUCCESS)	{ cout << "\nref_img_mean_rows_buf          status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_48"<<flush;
	status = clReleaseMemObject(new_img_buf);					if (status != CL_SUCCESS)	{ cout << "\nnew_img_buf                    status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_48"<<flush;
	status = clReleaseMemObject(warped_img_buf);				if (status != CL_SUCCESS)	{ cout << "\nref_img_mean_rows_buf          status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_48"<<flush;

	status = clReleaseMemObject(ref_img_mean_rows_buf);			if (status != CL_SUCCESS)	{ cout << "\nref_img_mean_rows_buf          status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_48"<<flush;
	status = clReleaseMemObject(ref_img_mean_buf);				if (status != CL_SUCCESS)	{ cout << "\nref_img_mean_buf               status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_48"<<flush;
	status = clReleaseMemObject(warped_img_mean_rows_buf);		if (status != CL_SUCCESS)	{ cout << "\nwarped_img_mean_rows_buf    	status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_48"<<flush;
	status = clReleaseMemObject(warped_img_mean_buf);			if (status != CL_SUCCESS)	{ cout << "\nwarped_img_mean_buf        	 status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_48"<<flush;

	status = clReleaseMemObject(ref_img_diff_buf);				if (status != CL_SUCCESS)	{ cout << "\nref_img_diff_buf               status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_48"<<flush;
	status = clReleaseMemObject(warped_img_diff_buf);			if (status != CL_SUCCESS)	{ cout << "\nwarped_img_diff_buf            status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_48"<<flush;

	status = clReleaseMemObject(ref_img_sigma_rows_buf);		if (status != CL_SUCCESS)	{ cout << "\nref_img_sigma_rows_buf         status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_48"<<flush;
	status = clReleaseMemObject(ref_img_mean_sigma_buf);		if (status != CL_SUCCESS)	{ cout << "\nref_img_mean_sigma_buf         status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_48"<<flush;
	status = clReleaseMemObject(warped_img_sigma_rows_buf);		if (status != CL_SUCCESS)	{ cout << "\nwarped_img_sigma_rows_buf      status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_48"<<flush;
	status = clReleaseMemObject(warped_img_mean_sigma_buf);		if (status != CL_SUCCESS)	{ cout << "\nwarped_img_mean_sigma_buf      status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_48"<<flush;

	status = clReleaseMemObject(covariance_rows_buf);			if (status != CL_SUCCESS)	{ cout << "\ncovariance_rows_buf            status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_48"<<flush;
	status = clReleaseMemObject(covariance_buf);				if (status != CL_SUCCESS)	{ cout << "\ncovariance_buf                 status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_48"<<flush;

	status = clReleaseMemObject(correlation_buf);				if (status != CL_SUCCESS)	{ cout << "\ncorrelation_buf                status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_48"<<flush;
	status = clReleaseMemObject(correlation_blurred_buf);		if (status != CL_SUCCESS)	{ cout << "\ncorrelation_blurred_buf        status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_48"<<flush;

	status = clReleaseMemObject(warp_buf);						if (status != CL_SUCCESS)	{ cout << "\nwarp_buf                       status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_48"<<flush;
	status = clReleaseMemObject(confidence_buf);				if (status != CL_SUCCESS)	{ cout << "\nconfidence_buf                 status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_48"<<flush;
	status = clReleaseMemObject(warp_ref_buf);					if (status != CL_SUCCESS)	{ cout << "\nwarp_ref_buf                   status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_48"<<flush;

	status = clReleaseMemObject(warp_buf_regularized);			if (status != CL_SUCCESS)	{ cout << "\nwarp_buf_regularized           status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_48"<<flush;
	status = clReleaseMemObject(confidence_buf_regularized);	if (status != CL_SUCCESS)	{ cout << "\nconfidence_buf_regularized     status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_48"<<flush;

	status = clReleaseMemObject(warp_GT_buf);					if (status != CL_SUCCESS)	{ cout << "\nwarp_GT_buf                    status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_48"<<flush;
	status = clReleaseMemObject(warp_error_buf);				if (status != CL_SUCCESS)	{ cout << "\nwarp_error_buf                 status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_48"<<flush;
	status = clReleaseMemObject(depth_error_buf);				if (status != CL_SUCCESS)	{ cout << "\ndepth_error_buf                status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_48"<<flush;
	status = clReleaseMemObject(depth_est_buf);					if (status != CL_SUCCESS)	{ cout << "\ndepth_est_buf                  status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_48"<<flush;
*/
	// buffers for patch kernel based Dynamic_slam
	status = clReleaseMemObject(pose_buf);						if (status != CL_SUCCESS)	{ cout << "\npose_buf                       status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_48"<<flush;
	status = clReleaseMemObject(pose_update_buf);				if (status != CL_SUCCESS)	{ cout << "\npose_update_buf                status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_48"<<flush;
	status = clReleaseMemObject(distorsion_update_buf);			if (status != CL_SUCCESS)	{ cout << "\ndistorsion_update_buf          status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_48"<<flush;
	status = clReleaseMemObject(old_results_buf);				if (status != CL_SUCCESS)	{ cout << "\nold_result_buf                 status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_48"<<flush;

	status = clReleaseMemObject(K_buf);							if (status != CL_SUCCESS)	{ cout << "\nK_buf                          status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_48"<<flush;
	status = clReleaseMemObject(inv_K_buf);						if (status != CL_SUCCESS)	{ cout << "\ninv_K_buf                      status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_48"<<flush;

	status = clReleaseMemObject(patch_lookup_table_buf);		if (status != CL_SUCCESS)	{ cout << "\npatch_lookup_table_buf         status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_48"<<flush;
	status = clReleaseMemObject(SE3_hessian_pinv_map_mem);		if (status != CL_SUCCESS)	{ cout << "\nSE3_hessian_map_mem            status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_48"<<flush;



	// release kernels
	status = clReleaseKernel(convert_depth_kernel);					if (status != CL_SUCCESS)	{ cout << "\nconvert_depth_kernel				status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_59"<<flush;
	status = clReleaseKernel(mipmap_float_kernel);					if (status != CL_SUCCESS)	{ cout << "\nmipmap_float_kernel				status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_59"<<flush;

	status = clReleaseKernel(cvt_color_space_linear_kernel);		if (status != CL_SUCCESS)	{ cout << "\ncvt_color_space_linear_kernel 		status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_49"<<flush;
	status = clReleaseKernel(sum_image_variance_kernel);			if (status != CL_SUCCESS)	{ cout << "\nsum_image_variance_kernel 			status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_50"<<flush;
	status = clReleaseKernel(sample_image_variance_kernel);			if (status != CL_SUCCESS)	{ cout << "\nsample_image_variance_kernel		status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_50"<<flush;
	status = clReleaseKernel(reduce_kernel);						if (status != CL_SUCCESS)	{ cout << "\nreduce_kernel 						status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_51"<<flush;
	status = clReleaseKernel(mipmap_float4_kernel);					if (status != CL_SUCCESS)	{ cout << "\nmipmap_float4_kernel 				status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_52"<<flush;

	status = clReleaseKernel(img_grad_kernel);						if (status != CL_SUCCESS)	{ cout << "\nimg_grad_kernel 					status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_53"<<flush;
	status = clReleaseKernel(comp_param_maps_kernel);				if (status != CL_SUCCESS)	{ cout << "\ncomp_param_maps_kernel 			status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_54"<<flush;
	status = clReleaseKernel(se3_rho_sq_kernel);					if (status != CL_SUCCESS)	{ cout << "\nse3_rho_sq_kernel 					status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_55"<<flush;

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

/*
	status = clReleaseKernel(mipmap_3x3blur_flt4_kernel);		if (status != CL_SUCCESS)	{ cout << "\nmipmap_3x3blur_flt4 			status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_52"<<flush;

	status = clReleaseKernel(se3_lk_grad_kernel);				if (status != CL_SUCCESS)	{ cout << "\nse3_lk_grad_kernel 			status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_58"<<flush;

	status = clReleaseKernel(invert_depth_kernel);				if (status != CL_SUCCESS)	{ cout << "\ninvert_depth_kernel			status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_59"<<flush;
	status = clReleaseKernel(transform_depthmap_kernel);		if (status != CL_SUCCESS)	{ cout << "\ntransform_depthmap_kernel 		status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_60"<<flush;
	status = clReleaseKernel(transform_costvolume_kernel);		if (status != CL_SUCCESS)	{ cout << "\ntransform_costvolume_kernel 	status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_60.5"<<flush;
	
	status = clReleaseKernel(depth_cost_vol_kernel);			if (status != CL_SUCCESS)	{ cout << "\ndepth_cost_vol_kernel			status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_61"<<flush;
	status = clReleaseKernel(updateQD_kernel);					if (status != CL_SUCCESS)	{ cout << "\nupdateQD_kernel 				status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_62"<<flush;
	status = clReleaseKernel(updateG_kernel);					if (status != CL_SUCCESS)	{ cout << "\nupdateG_kernel 				status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_63"<<flush;
	status = clReleaseKernel(updateA_kernel);					if (status != CL_SUCCESS)	{ cout << "\nupdateA_kernel 				status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_64"<<flush;
	status = clReleaseKernel(measureDepthFit_kernel);			if (status != CL_SUCCESS)	{ cout << "\nmeasureDepthFit_kernel			status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_65"<<flush;

	status = clReleaseKernel(disparity_kernel);					if (status != CL_SUCCESS)	{ cout << "\nmeasureDepthFit_kernel			status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_65"<<flush;

	status = clReleaseKernel(atomic_test1_kernel);				if (status != CL_SUCCESS)	{ cout << "\natomic_test1_kernel			status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_66"<<flush;
	status = clReleaseKernel(atomic_test2_kernel);				if (status != CL_SUCCESS)	{ cout << "\natomic_test1_kernel			status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_66"<<flush;

	// kernels for stereo disparity
	status = clReleaseKernel(compute_lookup_table_kernel);		if (status != CL_SUCCESS)	{ cout << "\ncompute_lookup_table_kernel	status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_66"<<flush;
	status = clReleaseKernel(disparity_load_frame_kernel);		if (status != CL_SUCCESS)	{ cout << "\ndisparity_load_frame_kernel	status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_66"<<flush;
	status = clReleaseKernel(set_warp_new_image_kernel);		if (status != CL_SUCCESS)	{ cout << "\nset_warp_new_image_kernel		status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_66"<<flush;
	status = clReleaseKernel(warp_image_kernel);				if (status != CL_SUCCESS)	{ cout << "\nwarp_image_kernel				status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_66"<<flush;

	status = clReleaseKernel(correlation_one_step_kernel);		if (status != CL_SUCCESS)	{ cout << "\ncorrelation_one_step_kernel	status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_66"<<flush;
	status = clReleaseKernel(correlation_2nd_step_kernel);		if (status != CL_SUCCESS)	{ cout << "\ncorrelation_2nd_step_kernel	status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_66"<<flush;

	status = clReleaseKernel(mean_sq_3rows_kernel);				if (status != CL_SUCCESS)	{ cout << "\nmean_sq_3rows_kernel			status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_66"<<flush;
	status = clReleaseKernel(mean_sq_cols_kernel);				if (status != CL_SUCCESS)	{ cout << "\nmean_sq_cols_kernel			status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_66"<<flush;
	status = clReleaseKernel(sigma_3rows_kernel);				if (status != CL_SUCCESS)	{ cout << "\nsigma_3rows_kernel				status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_66"<<flush;
	status = clReleaseKernel(sigma_3cols_kernel);				if (status != CL_SUCCESS)	{ cout << "\nsigma_3cols_kernel				status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_66"<<flush;
	status = clReleaseKernel(covariance_3rows_kernel);			if (status != CL_SUCCESS)	{ cout << "\ncovariance_3rows_kernel		status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_66"<<flush;
	status = clReleaseKernel(covariance_cols_kernel);			if (status != CL_SUCCESS)	{ cout << "\ncovariance_cols_kernel			status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_66"<<flush;
	status = clReleaseKernel(blur_volume_kernel);				if (status != CL_SUCCESS)	{ cout << "\nblur_volume_kernel				status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_66"<<flush;

	status = clReleaseKernel(compute_warp_kernel);				if (status != CL_SUCCESS)	{ cout << "\ncompute_warp_kernel			status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_66"<<flush;
	status = clReleaseKernel(regularize_warp_kernel);			if (status != CL_SUCCESS)	{ cout << "\nregularize_warp_kernel			status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_66"<<flush;
	status = clReleaseKernel(propagate_warp_kernel);			if (status != CL_SUCCESS)	{ cout << "\npropagate_warp_kernel			status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_66"<<flush;

	status = clReleaseKernel(warp_and_depth_error_kernel);		if (status != CL_SUCCESS)	{ cout << "\nwarp_and_depth_error_kernel	status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_66"<<flush;
*/

	// release command queues
	status = clReleaseCommandQueue(m_queue);                   if (status != CL_SUCCESS)	{ cout << "\nm_queue                        status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_67"<<flush;
	status = clReleaseCommandQueue(uload_queue);               if (status != CL_SUCCESS)	{ cout << "\nuload_queue 	                status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_68"<<flush;
	status = clReleaseCommandQueue(dload_queue);               if (status != CL_SUCCESS)	{ cout << "\ndload_queue 	                status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_69"<<flush;
	status = clReleaseCommandQueue(track_queue);               if (status != CL_SUCCESS)	{ cout << "\ntrack_queue 	                status = " << checkerror(status) <<"\n"<<flush; }		if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_70"<<flush;

	// release Program
	clReleaseProgram(m_program);	if (status != CL_SUCCESS)	{ cout << "\nm_program 	status = " << checkerror(status) <<"\n"<<flush; }	if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_71"<<flush;

	// release context
	clReleaseContext(m_context);	if (status != CL_SUCCESS)	{ cout << "\nm_context 	status = " << checkerror(status) <<"\n"<<flush; }	if(verbosity>local_verbosity_threshold) cout<<"\nRunCL::~RunCL_chk_72"<<flush;
																																			cout<<"\nRunCL::~RunCL_chk1_finished"<<flush;
}

void RunCL::exit_(int res)   // TODO convert all uses to exit_(res); Will call RunCL::~RunCL() automatically.
{
	cout <<endl<< flush;
	cerr <<endl<< flush;
	exit(res);
}
