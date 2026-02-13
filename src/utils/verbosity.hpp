    #pragma once
// LOCAL VERBOSITY FOR EACH FUNCTION  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// NB NEED TO INCLUDE CLASSNAME. TO PREVENT FUNCTION NAME CLASHES BETWEEN CLASSES.
// NB ALSO OVERLADED FUNCTION NAMES WITHIN A CLASS.

	// Global verbosity is set in conf.json
	// "verbosity": 1,									// -1= none, 0=errors only, 1=basic, 2=lots.

    // UTILS/ ###############################################################
    /////////////////////////////////////// CONVERTAHANDAPOVRAYTOSTANDARD.CPP
	#define	V_CONVERTAHANDAPOVRAYTOSTANDARD				2
	#define	V_LOADDEPTHAHANDA							2

	/////////////////////////////////////// CONVERT_TRANSFORMS.CPP
	#define	V_CONVERT_TRANSFORMS						1

	/////////////////////////////////////// CONF_PARAMS.CPP
	#define	V_CONF_PARAMS_CONF_PARAMS					1
	#define	V_CONF_PARAMS_DISPLAY_PARAMS				1

	/////////////////////////////////////// PRINT_FUNCTIONS.CPP
	#define	V_PRINT_MATX33F								2
	#define	V_PRINT_MATX44F								2
	#define	V_PRINT_MATX61F								2
	#define	V_PRINT_FLOAT_6								2
	#define	V_PRINT_FLOAT_9								2
	#define	V_PRINT_FLOAT_16							2
	#define	V_PRINT_JSON_FLOAT_9						2
	#define	V_PRINT_MATF								2

    // DYNAIC_SLAM/ ##############################################
    /////////////////////////////////////// DYNAMIC_SLAM_CLASS.CPP
	#define	V_DYNAMIC_SLAM_DYNAMIC_SLAM					1
	#define V_DYNAMIC_GENERATE_DELTAS					1
	#define	V_DYNAMIC_SLAM_INITIALIZE_CAMERA			2
	#define	V_DYNAMIC_SLAM_NEXTFRAME					2
	#define	V_DYNAMIC_SLAM_GETFRAME						2
	#define	V_DYNAMIC_SLAM_GETPOSE						1
	#define	V_DYNAMIC_SLAM_GETINVPOSE					1
	#define	V_DYNAMIC_SLAM_GETFRAMEDATA					1
	#define	V_DYNAMIC_SLAM_USE_GT_POSE					1
    ////
	#define	V_DYNAMIC_SLAM_ESTIMATECALIBRATION			0
	#define	V_DYNAMIC_SLAM_SPATIALCOSTFNS				0
	#define	V_DYNAMIC_SLAM_PARSIMONYCOSTFNS				0
	#define	V_DYNAMIC_SLAM_EXHAUSTIVESEARCH				0
	#define	V_DYNAMIC_SLAM_GETRESULT					1

    /////////////////////////////////////// DYNAMIC_SLAM_TRACKING.CPP
	#define	V_DYNAMIC_SLAM_ARTIFICIAL_POSE_ERROR		1
	#define	V_DYNAMIC_SLAM_PREDICTFRAME					1
	#define	V_DYNAMIC_SLAM_GENERATE_INVK_				1
	#define	V_DYNAMIC_SLAM_GENERATE_SE3_K2K				1
	#define	V_DYNAMIC_SLAM_UPDATE_K2K					2
	#define	V_DYNAMIC_SLAM_COMPUTE_OPTIMUM				2
	#define	V_DYNAMIC_SLAM_ESTIMATESE3					2 //

	// DYNAIC_SLAM/testing/ ##############################################
	/////////////////////////////////////// DYNAMIC_SLAM_GROUND_TRUTH.CPP

	/////////////////////////////////////// DYNAMIC_SLAM_RESULTS.CPP
	#define	V_DYNAMIC_SLAM_REPORT_GT_POSE_ERROR			2

	// RUNCL/ ##############################################
	/////////////////////////////////////// RUNCL_CLASS.CPP
	#define	V_RUNCL_RUNCL								0
	#define	V_RUNCL_TESTOPENCL							0
	#define	V_RUNCL_GETDEVICEINFOOPENCL					0
	#define	V_RUNCL_CREATEQUEUES						0
	#define	V_RUNCL_CREATEANDBULIDPROGRAMFROMSOURCE		0
	#define	V_RUNCL_CREATEKERNELS						0

	#define	V_RUNCL_CONVERTTOSTRING						0
	#define	V_RUNCL_INITIALIZE_FP32_PARAMS				0
	#define	V_RUNCL_INITIALIZE_RUNCL					1
	#define	V_RUNCL_MIPMAP_CALL_KERNEL					1
	#define V_RUNCL_LAYER_CALL_KERNEL					1
	#define	V_RUNCL_WAITFOREVENTANDRELEASE				0
	#define	V_RUNCL_SET_CAM_BUFS						1
	#define	V_RUNCL_ALLOCATEMEM							0
	#define	V__RUNCL									0
	#define	V_EXIT_										0

	/////////////////////////////////////// RUNCL_DEPTH.CPP
	#define V_RUNCL_UPDATE_DEPTH						-2


	/////////////////////////////////////// RUNCL_DOWNLOADANDSAVE.CPP
	#define	V_RUNCL_CREATEFOLDERS						2
	#define	V_RUNCL_READOUTPUT							2
	#define	V_RUNCL_SAVECOSTVOLS						2
	#define	V_RUNCL_STORE_KEYFRAME						2
	#define	V_RUNCL_SAVE_VTK							2

	#define	V_RUNCL_DOWNLOADANDSAVE						2
	#define	V_RUNCL_DOWNLOADANDSAVE_2CHANNEL_VOLUME		2

	#define	V_RUNCL_DOWNLOADANDSAVE_3CHANNEL			2
	#define	V_RUNCL_DOWNLOADANDSAVE_3CHANNEL_VOLUME		2

	#define	V_RUNCL_PREPARERESULTS_3CHANNEL				2
	#define	V_RUNCL_PREPARERESULTS_3CHANNEL_VOLUME		2

	#define	V_RUNCL_DOWNLOADANDSAVE_6CHANNEL			2
	#define	V_RUNCL_WRITETORESULTSMAT					2

	#define	V_RUNCL_DOWNLOADANDSAVE_HSV_GRAD			2
	#define	V_RUNCL_SAVEMAT								2
	#define	V_RUNCL_SAVEMAT_1CHAN						2

	#define	V_RUNCL_DOWNLOADANDSAVE_6CHANNEL_VOLUME		2

	#define	V_RUNCL_DOWNLOADANDSAVE_8CHANNEL			2
	#define	V_RUNCL_DOWNLOADANDSAVE_8CHANNEL_VOLUME		2

	#define	V_RUNCL_DOWNLOADANDSAVEVOLUME				2

	////////////////////////////////////// RUNCL_LOAD_IMAGE.CPP
	#define	V_RUNCL_LOADFRAME							1
	#define	V_RUNCL_CVT_COLOR_SPACE						0
	#define	V_RUNCL_SUM_IMAGE_VARIANCE					1
	#define	V_RUNCL_SAMPLE_IMAGE_VARIANCE				0

	#define	V_RUNCL_MIPMAP_LINEAR						1
	#define	V_RUNCL_IMG_GRADIENTS						1

	#define	V_RUNCL_LOAD_GT_DEPTH						1
	#define	V_RUNCL_CONVERT_DEPTH						1
	#define	V_RUNCL_MIPMAP_DEPTHMAP						2

	/////////////////////////////////////// RUNCL_MAPPING.CPP

	#define	V_RUNCL_SPATIALCOSTFNS						1
	#define	V_RUNCL_PARSIMONYCOSTFNS					1
	#define	V_RUNCL_EXHAUSTIVESEARCH					1

	/////////////////////////////////////// RUNCL_PATCH_SLAM.CPP
	#define V_RUNCL_INITIALIZE_PATCH_PARAMS				1
	#define	V_RUNCL_COMPUTE_PATCH_LOOKUP_TABLE			2
	#define	V_RUNCL_PATCH_IMG_GRADIENTS					2

	/////////////////////////////////////// RUNCL_PATCH_TRACKING.CPP
	#define V_RUNCL_REDUCE_IMG							2
	#define V_RUNCL_BLUR_IMG							2


	/////////////////////////////////////// RUNCL_SETUP_KEYFRAME.CPP
	#define	V_RUNCL_ESTIMATECALIBRATION					1


	/////////////////////////////////////// RUNCL_TRACKING.CPP
	#define	V_RUNCL_PRECOM_PARAM_MAPS					1


	#define	V_RUNCL_UPDATE_K2K_BUF						2
	#define	V_RUNCL_RHO_SQ								2 //
	#define	V_RUNCL_REDUCE_PATCH_RHO					2
	#define	V_RUNCL_UPDATE_K2K							2

	#define	V_RELATIVEVEL_MAP							1


