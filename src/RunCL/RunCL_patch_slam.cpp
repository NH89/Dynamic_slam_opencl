#include "RunCL.hpp"

void RunCL::initialize_patch_params(){
	string fname					= "RunCL::initialize_patch_params()";
	int local_verbosity_threshold	= V_RUNCL_INITIALIZE_PATCH_PARAMS;

	//size_t  device_max_workitem_sizes[3];															i.e. max num threads per "symmetric multi-processor"
	cl_int device_info_1 = clGetDeviceInfo(
							deviceId,							//cl_device_id device,
							CL_DEVICE_MAX_WORK_ITEM_SIZES,		//cl_device_info param_name,
							sizeof(device_max_workitem_sizes),	//size_t param_value_size,
							device_max_workitem_sizes,			//void* param_value,
							NULL								//size_t* param_value_size_ret
	);
	//cl_uint  device_max_compute_units;  															i.e. the number of "symmetric multi-processors"
	cl_int device_info_2 = clGetDeviceInfo(
							deviceId,							//cl_device_id device,
							CL_DEVICE_MAX_COMPUTE_UNITS,		//cl_device_info param_name,
							sizeof(device_max_compute_units),	//size_t param_value_size,
							&device_max_compute_units,			//void* param_value,
							NULL								//size_t* param_value_size_ret
	);
	//cl_ulong  device_local_mem_size																i.e. max local memory per work_group
	cl_int device_info_3 = clGetDeviceInfo(
							deviceId,							//cl_device_id device,
							CL_DEVICE_LOCAL_MEM_SIZE,			//cl_device_info param_name,
							sizeof(device_local_mem_size),		//size_t param_value_size,
							&device_local_mem_size,				//void* param_value,
							NULL								//size_t* param_value_size_ret
	);
}


void RunCL::compute_patch_lookup_table( uint start, uint stop){
	string fname = "RunCL::compute_patch_lookup_table( )";
	int local_verbosity_threshold = V_RUNCL_COMPUTE_PATCH_LOOKUP_TABLE;
	cl_kernel kernel = compute_patch_lookup_table_kernel;
																																if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::compute_patch_lookup_table( ..)_chk1 #############################################################"<<flush;
																																	cout << "\n local_work_size = " << local_work_size
																																	<< ",  start = " << start << ",  stop = " << stop
																																	<< flush;
																																}
	size_t kernel_workgroup_size;
	cl_int k_wg_info =  clGetKernelWorkGroupInfo(
							kernel,								//cl_kernel kernel,
							deviceId,							//cl_device_id device,
							CL_KERNEL_WORK_GROUP_SIZE,			//cl_kernel_work_group_info param_name,
							sizeof(kernel_workgroup_size),		//size_t param_value_size,
							&kernel_workgroup_size,				//void* param_value,
							NULL								//size_t* param_value_size_ret
						);
	patch_kernel_workgroup_size	= kernel_workgroup_size;
	size_t	max_workgroup_size	= min(kernel_workgroup_size, device_max_workitem_sizes[0] );
																																if( verbosity>local_verbosity_threshold+1) {cout<<"\n\nRunCL::compute_patch_lookup_table( ..)_chk_2 "<<flush;
																																	cout <<"\n kernel_workgroup_size = "			<<kernel_workgroup_size<<flush;
																																	cout <<"\n device_max_workitem_sizes[0] = "		<<device_max_workitem_sizes[0]<<flush;
																																}
	_clSetKernelArg( kernel,  3, sizeof( cl_mem), &mipmap_buf,				fname);												// __constant	uint8*		mipmap_params,			//3
	_clSetKernelArg( kernel,  4, sizeof( cl_mem), &uint_param_buf,			fname);												// __constant	uint*		uint_params,			//4
	_clSetKernelArg( kernel,  5, sizeof( cl_mem), &fp32_param_buf,			fname);												// __constant	float*		fp32_params,			//5
	// output
	_clSetKernelArg( kernel,  6, sizeof( cl_mem), &patch_lookup_table_buf,	fname);												// __global		float4*		lookup_table			//6
																																if(verbosity>local_verbosity_threshold+1) {
																																	cout<<"\nRunCL::compute_patch_lookup_table( )_chk3"
																																	<<",  start="				<<start
																																	<<",  stop="				<<stop
																																	<<",  local_work_size="		<<local_work_size
																																	<<flush;
																																}
	cl_event		ev;
	cl_int			res, status;
	patch_lookup_table_offset[0]	= 0; 																						// NB 'start' may not be set to zero

	for(uint layer = 0; layer <= stop; layer++) {																				// NB processes largest layer first.
																																if(verbosity>local_verbosity_threshold+1) { cout<<"\nRunCL::compute_patch_lookup_table( )_chk4,  reduction="\
																																	<<layer<<",  patch_num_threads[reduction]="<<patch_num_threads[layer]<<"  local_work_size="<<local_work_size<<flush;
																																}
		uint				read_rows					= MipMap[layer * 8 + MiM_READ_ROWS] ;
		uint				read_cols					= MipMap[layer * 8 + MiM_READ_COLS] ;

		uint				rows_blocks					= ceil( (float)  read_rows / patch_size );
		uint				cols_blocks					= ceil( (float)  read_cols / patch_size );
		uint				cols_per_row				= cols_blocks  * patch_size;

		uint				patches_required			= cols_blocks  * rows_blocks;
		uint				patches_per_compute_uint	= ceil( (float)patches_required / device_max_compute_units ) ;
		uint				blocks_per_k_wg_size		= max_workgroup_size			/ device_work_size_multiple;
							patches_per_compute_uint	= min( patches_per_compute_uint,  blocks_per_k_wg_size );

		uint				blocks_required				= ceil( (float)patches_required / patches_per_compute_uint );
		size_t				local_work_size_[1] 		= { patches_per_compute_uint	* patch_size };
		size_t				threads_to_launch 			= blocks_required 				* local_work_size_[0];
																																if( verbosity>local_verbosity_threshold+1){
																																	cout<<"\nread_rows= 				"<<read_rows;
																																	cout<<"\nread_cols= 				"<<read_cols;

																																	cout<<"\nrows_blocks= 				"<<rows_blocks;
																																	cout<<"\ncols_blocks= 				"<<cols_blocks;
																																	cout<<"\ncols_per_row= 				"<<cols_per_row				<<"		= cols_blocks  * patch_size";

																																	cout<<"\npatches_required= 			"<<patches_required			<<"		= cols_blocks  * rows_blocks";
																																	cout<<"\npatches_per_compute_uint=	"<<patches_per_compute_uint	<<"		= ceil( (float)patches_required / device_max_compute_units )";
																																	cout<<"\nblocks_per_k_wg_size= 		"<<blocks_per_k_wg_size		<<"		= max_workgroup_size			/ device_work_size_multiple";
																																	cout<<"\npatches_per_compute_uint=	"<<patches_per_compute_uint	<<"		= max( patches_per_compute_uint,  blocks_per_k_wg_size )";

																																	cout<<"\nblocks_required= 			"<<blocks_required			<<"		= ceil( (float)patches_required / patches_per_compute_uint )";
																																	cout<<"\nlocal_work_size_= 			"<<local_work_size_[0]		<<"		= { patches_per_compute_uint	* patch_size }";
																																	cout<<"\nthreads_to_launch= 		"<<threads_to_launch		<<"		= blocks_required 				* local_work_size_[0]";		// TODO precompute an array for this function. ? where to store
																																}
		patch_local_work_size[layer]					= local_work_size_[0];
		patch_num_threads[layer]						= threads_to_launch;
		patch_cols_per_row[layer]						= cols_per_row;
																																	// ? Have a subclass and object for each kernel ?
																																if( verbosity>local_verbosity_threshold+1) {cout<<"\n\nRunCL::compute_patch_lookup_table( )_chk_5 "<<flush;
																																	cout <<"\n"
																																	<<",  patches_required="						<<patches_required
																																	<<",  patches_per_compute_uint="				<<patches_per_compute_uint
																																	<<",  blocks_per_k_wg_size="					<<blocks_per_k_wg_size
																																	<<",  blocks_required="							<<blocks_required
																																	<<",  threads_to_launch="						<<threads_to_launch
																																	<<",  local_work_size="							<<local_work_size
																																	<<"},  local_work_size_[1]="					<<local_work_size_[0]
																																	<< flush;
																																	for (uint reduction = 0; reduction < 8  ; reduction ++){
																																		cout << "\n reduction = "					<< reduction
																																		<<"  patch_num_threads[reduction] = "		<< patch_num_threads[reduction]
																																		<< flush;
																																	}
																																	cout<<"\ntracking_num_samples*2*mm_size_bytes_C4="<<tracking_num_samples*2*mm_size_bytes_C4
																																		<<"     24 * mm_size_bytes_C1="<<24 * mm_size_bytes_C1<<flush;
																																}
		uint 	lookup_table_offset_uint 				= patch_lookup_table_offset[layer];
		res 	= clSetKernelArg(kernel, 0, sizeof(int), &layer );																// __private	uint		layer,					//0
		res 	= clSetKernelArg(kernel, 1, sizeof(int), &lookup_table_offset_uint );											// __private	uint		lookup_table_offset,	//1
		res 	= clSetKernelArg(kernel, 2, sizeof(int), &cols_per_row);														// __private	uint		cols_per_row,			//2
																															if (res    !=CL_SUCCESS)	{ cout <<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	;

		res 	= clEnqueueNDRangeKernel(m_queue,		kernel, 1, 0, &threads_to_launch, local_work_size_, 0, NULL, &ev); // run mipmap_float4_kernel, NB wait for own previous iteration.
																															if (res    != CL_SUCCESS)	{ cout << "\nres = " << checkerror(res) <<"\n"<<flush; exit_(res);}
		status	= clFlush(m_queue);																							if (status != CL_SUCCESS)	{ cout << "\nRunCL::compute_patch_lookup_table( ),  clFlush(m_queue) status  = "<<status<<" "<< checkerror(status) <<"\n"<<flush; exit_(status);}
		status	= clWaitForEvents (1, &ev);																					if (status != CL_SUCCESS)	{ cout << "\nRunCL::compute_patch_lookup_table( ),  clWaitForEventsh(1, &ev) ="	<<status<<" "<<checkerror(status)  <<"\n"<<flush; exit_(status);}

		patch_lookup_table_offset[layer +1]				=	patch_lookup_table_offset[layer] + threads_to_launch;				// NB this pads the lookup table, so that local work groups will not be shared betwen layers.
																																// It also  means that this offset should be used to launch layers from the lookup table.
	}
																																if( verbosity>local_verbosity_threshold+1) {
																																	for(uint reduction = 0; reduction <= stop; reduction++) {
																																		cout << "\npatch_num_threads["<<reduction<<"] = "<<patch_num_threads[reduction]<<",   MipMap[reduction*8 +MiM_PIXELS] = "<<MipMap[reduction*8 +MiM_PIXELS]<<flush;
																																	}
																																	cout <<"\nNB there will only be a gap in the lookuptable when:  patch_num_threads[reduction] > MipMap[reduction*8 +MiM_PIXELS]  , which depends on the image dimensions and local_work_size"<<flush;
																																}
																																if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::compute_patch_lookup_table( )_chk6 ."<<flush;	// Save buffers to file ###########
																																	stringstream ss;
																																	ss << "compute_patch_lookup_table_" << save_index ;
																																	bool show 		= false;
																																	bool old_tiff 	= tiff;
																																	tiff 			= true;
																																	float max_range	= 1;
																																	cv::Mat bufImg;
																																	_cl_flush_finish(m_queue, fname);
																																	//void DownloadAndSave_3Channel(cl_mem buffer, std::string count, std::filesystem::path folder_tiff, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, cv::Mat *bufImg, float max_range=1, uint offset=0, bool exception_tiff=false );
																																	DownloadAndSave_3Channel( 	patch_lookup_table_buf,	ss.str( ), paths.at( "lookup_table_buf"),  		mm_size_bytes_C4,   mm_Image_size,   CV_32FC4, 	show, &bufImg, max_range, 0, false);
																																	// NB the tiff file holds the int32 values as float32. This is okay because they fit in the mantissa.
																																	// BGRA format, B=u, G=v, R=read_index, A=alpha.
																																	tiff 			= old_tiff;
																																}
																																if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::compute_patch_lookup_table( )_finished #############################################################"<<flush;
																																}
}


void RunCL::patch_img_gradients_set_params( uint out_block_size ){	// Uses patch lookup table
	string fname = "RunCL::patch_img_gradients_set_params()";
	int local_verbosity_threshold = V_RUNCL_PATCH_IMG_GRADIENTS;																if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::patch_img_gradients_set_params()_chk1 #############################################################"<<flush;}
	cl_kernel kernel		= patch_img_grad_kernel;																			// To be launched with 1 thread per col for 32x32 patches, and an integer multiple of 32 threads.
																																// Needs 16 elements of local mem per 32x32 patch, to pass data between threads in recursive square reduction.
																																// Needs 32 elem array of private mem per thread.
	size_t kernel_workgroup_size;
	cl_int k_wg_info =  clGetKernelWorkGroupInfo(
							kernel,								//cl_kernel kernel,
							deviceId,							//cl_device_id device,
							CL_KERNEL_WORK_GROUP_SIZE,			//cl_kernel_work_group_info param_name,
							sizeof(kernel_workgroup_size),		//size_t param_value_size,
							&kernel_workgroup_size,				//void* param_value,
							NULL								//size_t* param_value_size_ret
						);
																																if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::patch_img_gradients_set_params()_chk2 #############################################################"<<flush;
																																	cout <<"\n kernel_workgroup_size = "<<kernel_workgroup_size<<endl<<flush;
																																}
																															if( patch_kernel_workgroup_size > kernel_workgroup_size ) {	// #####  Error !
																																cout<<"\n\nRunCL::patch_img_gradients_set_params()  ( patch_kernel_workgroup_size="<<patch_kernel_workgroup_size<<" > kernel_workgroup_size="<<kernel_workgroup_size<<" )."
																																	<<"Need code to handle workgroup size for this kernel."<<endl<<flush;

																																cerr<<"\n\nRunCL::patch_img_gradients_set_params()  ( patch_kernel_workgroup_size="<<patch_kernel_workgroup_size<<" > kernel_workgroup_size="<<kernel_workgroup_size<<" )."
																																	<<"Need code to handle workgroup size for this kernel."<<endl<<flush;
																																exit_(1);
																															}
	size_t		one_patch_local_Hessian_size		= sizeof(cl_float4)				*num_SE3_DoF *num_SE3_DoF		*patch_size;
																																if( verbosity>local_verbosity_threshold+1 ) {
																																	cout <<"\none_patch_local_Hessian_size		= "<<one_patch_local_Hessian_size<<flush;
																																	cout <<"\ndevice_local_mem_size				= "<<device_local_mem_size<<flush;
																																}
	uint		patches_per_workgroup_local_mem		= floor ( device_local_mem_size	/ one_patch_local_Hessian_size);
																																if( verbosity>local_verbosity_threshold+1 ) {
																																	cout <<"\npatches_per_workgroup_local_mem 	= "<<patches_per_workgroup_local_mem<<flush;
																																}
	for (uint layer_=0; layer_<max_mipmap_layers; layer_++){
		patch_img_gradients_workgroup_size[layer_]	= min( patch_local_work_size[layer_],  patches_per_workgroup_local_mem * patch_size );
																																if( verbosity>local_verbosity_threshold+1 ) {
																																	cout <<"\npatch_img_gradients_workgroup_size["<<layer_<<"] = "<<patch_img_gradients_workgroup_size[layer_]<<flush;
																																}
	}
	//Inputs:
	//__private
	//_clSetKernelArg( kernel,	0, sizeof(int), 		&layer,							fname );								// __private	uint		layer,					//0
	_clSetKernelArg( kernel,	2, sizeof(int), 		&out_block_size,				fname );								// __private	uint		out_block_size,			//1
	//__constant
	_clSetKernelArg( kernel,	5, sizeof( cl_mem), 	&mipmap_buf,					fname);									// __constant	uint8*		mipmap_params,			//2
	_clSetKernelArg( kernel,	6, sizeof( cl_mem), 	&uint_param_buf,				fname);									// __constant	uint*		uint_params,			//3
	_clSetKernelArg( kernel,	7, sizeof( cl_mem), 	&SE3_map_mem,					fname);									// __constant 	float2*		SE3_map,				//4
	//__global
	_clSetKernelArg( kernel,	8, sizeof( cl_mem), 	&patch_lookup_table_buf,		fname);									// __global 	float4*		lookup_table,			//5
	//_clSetKernelArg( kernel,	9, sizeof( cl_mem), 	&imgmem_,						fname);									// __global 	float4*		img,					//6
	//Outputs:
	//__global
	_clSetKernelArg( kernel,	10, sizeof( cl_mem), 	&SE3_grad_map_mem,				fname);									// __global 	float8*		SE3_grad_map,			//7		// We keep hsv sepate at this stage, so 6*4*2=24, but float16 is the largest type, so 6*float8.
	_clSetKernelArg( kernel,	11, sizeof( cl_mem), 	&SE3_hessian_pinv_map_mem,		fname);									// __global 	float4*		SE3_Hessian_pinv_map,		//8		// HSV (6x6) matrix so 36*float8
	_clSetKernelArg( kernel,	13, sizeof( cl_mem), 	&HSV_grad_mem,					fname);									// __global 	float8*		HSV_grad				//10

	// For the SE3 Hessian patches, and their reduction.	/////////////
																																if( verbosity>local_verbosity_threshold+1 ) {
																																	cout<<"\nvoid RunCL::patch_img_gradients_set_params(  ):  mm_start="<<mm_start<<"   mm_stop="<<mm_stop<<flush;
																																}
	uint	hessian_layer_offset		=	5	+		mm_width;
	uint	st3_hessian_layer_offset	=	5	+		( 4+( MipMap[	 0*8 + MiM_READ_ROWS]		/block_size) )*mm_width   * (num_SE3_DoF + 1);

	for (uint layer =0; layer<mm_stop; layer++){																				// NB must match where the SE3 Hessian is written in SE3_hessian_map_mem.
																																// i.e. 6x6 elems in img pyramid horizontally across the top of the buffer.
		patch_hessian_cols[		layer]	= ceil( (float)		  MipMap[layer*8 + MiM_READ_COLS]		/block_size );
		patch_hessian_rows[		layer]	= ceil( (float)		  MipMap[layer*8 + MiM_READ_ROWS]		/block_size );

		uint	hessian_elem_step		=				  4 + MipMap[layer*8 + MiM_READ_COLS]		/block_size;
		uint	hessian_row_step		=				 (4 + MipMap[layer*8 + MiM_READ_ROWS]		/block_size )		*  mm_width;

		uint	st3_hessian_elem_step	=				  4 + MipMap[layer*8 + MiM_READ_COLS]		/out_block_size;
		uint	st3_hessian_row_step	=	(8 + ceil( (float)MipMap[layer*8 + MiM_READ_ROWS]		/out_block_size) )	*  mm_width;

		for( uint row = 0; row<6; row++){
			for( uint col = 0; col<6; col++){
				patch_hessian_start_idx[		layer][row][col]	=	hessian_layer_offset		+ hessian_elem_step*col			+ hessian_row_step*row;
			}
		}
		for( uint row = 0; row<3; row++){
			for( uint col = 0; col<3; col++){
				patch_ST3_hessian_start_idx[	layer][row][col]	=	st3_hessian_layer_offset	+ st3_hessian_elem_step*col		+ st3_hessian_row_step*row;
			}
		}
																																if( verbosity>local_verbosity_threshold ) {
																																	cout<<"\npatch_hessian_cols["<<layer<<"]="			<<patch_hessian_cols[layer]
																																	<<",    MipMap[layer*8 + MiM_READ_COLS]="			<<MipMap[layer*8 + MiM_READ_COLS]
																																	<<",    MipMap[layer*8 + MiM_READ_OFFSET]="			<<MipMap[layer*8 + MiM_READ_OFFSET]
																																	<<",    hessian_elem_step="							<<hessian_elem_step
																																	<<",    hessian_layer_offset - mm_width ="			<<hessian_layer_offset - mm_width
																																	<<"\t\t"
																																	<<",    (st3_hessian_layer_offset -5)/ mm_width ="	<<(float)(st3_hessian_layer_offset -5) / mm_width
																																	<<",    st3_hessian_elem_step ="					<<st3_hessian_elem_step
																																	<<",    st3_hessian_row_step ="						<<st3_hessian_row_step
																																	<<",    out_block_size ="							<<out_block_size
																																	<<"\n"<<flush;
																																}
		hessian_layer_offset			+=		6* hessian_elem_step;					//MipMap[layer*8 + MiM_READ_OFFSET]	/mm_width;
		st3_hessian_layer_offset		+=		3* st3_hessian_row_step;
	}
}


void RunCL::patch_img_gradients( uint layer ){
	string fname = "RunCL::patch_img_gradients()";
	int local_verbosity_threshold = V_RUNCL_PATCH_IMG_GRADIENTS;																if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::patch_img_gradients()_chk1 #############################################################"<<flush;}
	cl_kernel		kernel						= patch_img_grad_kernel;
	const cl_mem 	imgmem_						= current_frames[ 						current_frames_idx[0] ].img_buf;
	size_t			local_work_size_			= patch_img_gradients_workgroup_size[	layer];											// patch_local_work_size[	layer];						//patch_img_gradients_workgroup_size; // patch_img_gradients_local_work_size_;
	size_t			threads_to_launch			= patch_num_threads[					layer];
	size_t			local_Hessian_size			= sizeof(cl_float4)						*num_SE3_DoF *num_SE3_DoF	*local_work_size_;	//*patch_img_gradients_workgroup_size;
	uint			lookup_table_offset_uint 	= patch_lookup_table_offset[			layer];

	uint			SE3_h_offset				= patch_hessian_start_idx[layer][0][0];
	uint			ST3_h_offset				= patch_ST3_hessian_start_idx[layer][0][0];
	cl_uint3		SE3_hessian_offset			= {{ SE3_h_offset,	(patch_hessian_start_idx[layer][0][1]     - SE3_h_offset) ,	(patch_hessian_start_idx[layer][1][0]     - SE3_h_offset)  }};
	cl_uint3		ST3_out_offset				= {{ ST3_h_offset,	(patch_ST3_hessian_start_idx[layer][0][1] - ST3_h_offset) ,	(patch_ST3_hessian_start_idx[layer][1][0] - ST3_h_offset)  }};

	_clSetKernelArg( kernel,	0, sizeof(int),			&layer,							fname);									// __private	uint		layer,						//0
	_clSetKernelArg( kernel,	1, sizeof(int),			&lookup_table_offset_uint,		fname);									// __private	uint		lookup_table_offset_uint,	//1
	//_clSetKernelArg( kernel,	2, sizeof(int),			&out_block_size,				fname);									// __private	uint		out_block_size,				//2
	_clSetKernelArg( kernel,	3, sizeof(cl_uint3),	&SE3_hessian_offset,			fname);									// __private	uint		SE3_hessian_offset,			//3
	_clSetKernelArg( kernel,	4, sizeof(cl_uint3),	&ST3_out_offset,				fname);									// __private	uint		SE3_hessian_offset,			//3

	_clSetKernelArg( kernel,	9, sizeof( cl_mem),		&imgmem_,						fname);									// __global 	float4*		img,					//6		//	"current_frames[idx].img_buf	= imgmem[idx];", NB changes every new frame.
	_clSetKernelArg( kernel,	12,local_Hessian_size,	NULL,							fname);									// __local		float4*		local_Hessian,			//9		// local_Hessian[ sizeof(float4) *6*6 *local_size]

	/* // Debugging kernel arg setting
	// size_t		param_value_size		= 0;
	// char		param_value[32]			= {' '};
 //
	// cl_int ret = clGetKernelArgInfo(
	// 	kernel,								//cl_kernel kernel,
	// 	(cl_uint)2,							//cl_uint arg_index,
	// 	CL_KERNEL_ARG_TYPE_NAME,			//cl_kernel_arg_info param_name,
	// 	param_value_size,					//size_t  param_value_size,
	// 	param_value,						//void*   param_value,
	// 	NULL								//size_t* param_value_size_ret
	// );
 //
	// cout << "\n\nRunCL::patch_img_gradients(..) ret = "<<ret<<",  CL_KERNEL_ARG_TYPE_NAME = "<< string(param_value, param_value_size) << "\n" << flush;
	*/
	/*
	cout <<"\nRunCL::patch_img_gradients()_chk1.5   threads_to_launch="<<threads_to_launch<<",		local_work_size_="<<local_work_size_<<flush;
	for (uint layer_=0; layer_<max_mipmap_layers; layer_++){
		cout<<"\npatch_num_threads["<<layer_<<"] = "											<<patch_num_threads[					layer_]
			<<",		patch_cols_per_row[layer_] = "											<<patch_cols_per_row[					layer_]
			<<",		patch_num_threads[] = "													<<patch_num_threads[					layer_]
			<<",		patch_img_gradients_workgroup_size[	layer_] = "							<<patch_img_gradients_workgroup_size[	layer_]
			<<",		patch_num_threads[] / patch_img_gradients_workgroup_size[ layer_] = "	<<(float)patch_num_threads[				layer_] / patch_img_gradients_workgroup_size[ layer_]
			<<flush;
	}

	cout<<"\n\nlocal_Hessian_size 		= "	<<local_Hessian_size
			<<" = sizeof(cl_float4)"<<sizeof(cl_float4)<<"   * num_SE3_DoF "<<num_SE3_DoF<<"    * num_SE3_DoF "<<num_SE3_DoF<<"    *local_work_size_ "<<local_work_size_
			<<"\ndevice_local_mem_size 	= "<< device_local_mem_size
			<<flush;
	*/
	cl_event	ev;
	cl_int		res, status;

	res 	= clEnqueueNDRangeKernel(m_queue,		kernel, 1, 0, &threads_to_launch, &local_work_size_, 0, NULL, &ev);
																	if (res    != CL_SUCCESS)	{ cout << "\nres = " << checkerror(res) <<"\n"<<flush; exit_(res);}
	status	= clFlush(m_queue);										if (status != CL_SUCCESS)	{ cout << "\nRunCL::patch_img_gradients( ),  clFlush(m_queue) status  = "<<status<<" "<< checkerror(status) <<"\n"<<flush; exit_(status);}
	status	= clWaitForEvents (1, &ev);								if (status != CL_SUCCESS)	{ cout << "\nRunCL::patch_img_gradients( ),  clWaitForEventsh(1, &ev) ="	<<status<<" "<<checkerror(status)  <<"\n"<<flush; exit_(status);}

																																if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::patch_img_gradients()_chk2 ."<<flush;	// Save buffers to file ###########
																																	stringstream ss;
																																	ss << "patch_img_gradients_" << save_index ;
																																	bool show 		= false;
																																	bool old_tiff 	= tiff;
																																	tiff 			= true;
																																	float max_range	= 1;
																																	cv::Mat bufImg;
																																	_cl_flush_finish(m_queue, fname);
																																	//DownloadAndSave_3Channel( 	SE3_hessian_map_mem,	ss.str( ), paths.at( "hessian"),  		mm_size_bytes_C4,   mm_Image_size,   CV_32FC4, 	show);
																																	DownloadAndSave_3Channel( 	SE3_hessian_pinv_map_mem,	ss.str( ), paths.at( "hessian"),  		mm_size_bytes_C4,   mm_Image_size,   CV_32FC4, 	show, &bufImg, max_range,  0,			false);
																																	DownloadAndSave_3Channel( 	SE3_hessian_pinv_map_mem,	ss.str( ), paths.at( "jacobian"),  		mm_size_bytes_C4,   mm_Image_size,   CV_32FC4, 	show, &bufImg, max_range,  mm_size_bytes_C4/*mm_layerstep*/, false);
																																	// NB the tiff file holda the int32 values as float32. This is okay because they fit in the mantissa.
																																	// BGRA format, B=u, G=v, R=read_index, A=alpha.
																																	tiff 			= old_tiff;
																																}
																																if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::patch_img_gradients()_finished #############################################################"<<flush;
																																}
}


void  RunCL::patch_hessian_reduce(uint layer){
	string 		fname	= "RunCL::patch_global_hessian_reduce()";
	int local_verbosity_threshold = V_RUNCL_PATCH_IMG_GRADIENTS;																if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::patch_hessian_reduce()_chk1 layer="<<layer<<" #############################################################"<<flush;}
	cl_kernel	kernel	= patch_hessian_reduce_kernel;

	uint	cols		=		patch_hessian_cols[ layer];
	uint	rows		=		patch_hessian_rows[ layer];
	uint	mm_cols		=		mm_width;

	_clSetKernelArg( kernel,	1, sizeof(int),			&layer,						fname);									// __private	uint	cols		//1
	_clSetKernelArg( kernel,	2, sizeof(int),			&cols,						fname);									// __private	uint	cols		//2
	_clSetKernelArg( kernel,	3, sizeof(int),			&rows,						fname);									// __private	uint	rows		//3
	_clSetKernelArg( kernel,	4, sizeof(int),			&mm_cols,					fname);									// __private	uint	mm_cols		//4

	_clSetKernelArg( kernel,	6, sizeof(int),			&mm_layerstep,				fname);									// __private	uint	mm_pixels	//6
	_clSetKernelArg( kernel,	7, sizeof(cl_mem),		&SE3_hessian_pinv_map_mem,	fname);									// __private	uint				//7

	cl_int		status	= CL_SUCCESS;
	cl_event	ev		= 0;
	cl_int		res		= 0;
	size_t		threads_to_launch	= block_size;	// NB could be a problem on AMD GPUs with minmum 64 threads, not 32.
	size_t		local_work_size_	= block_size;

	for( uint row = 0; row<6; row++){
		for( uint col = 0; col<6; col++){
			uint start_idx	=	patch_hessian_start_idx[layer][row][col];
			uint elem 		=	row * 6 + col;
			_clSetKernelArg( kernel,	0, sizeof(int),	&start_idx,					fname);									// __private	uint	start_idx	//0
			_clSetKernelArg( kernel,	5, sizeof(int),	&elem,						fname);									// __private	uint	elem		//5

			// launch workgroup for this elem.
			res = clEnqueueNDRangeKernel(m_queue,	kernel, 1, 0, &threads_to_launch, &local_work_size_, 0, NULL, &ev);		if (res    != CL_SUCCESS)	{ cout << "\nres = " << checkerror(res) <<"\n"<<flush; exit_(res);}
		}
	}
	status	= clFlush(m_queue);							if (status != CL_SUCCESS)	{ cout << "\nRunCL::patch_hessian_reduce( ),  clFlush(m_queue) status  = "<<status<<" "<<checkerror(status)  <<"\n"<<flush; exit_(status);}
	status	= clWaitForEvents (1, &ev);					if (status != CL_SUCCESS)	{ cout << "\nRunCL::patch_hessian_reduce( ),  clWaitForEventsh(1, &ev) = "<<status<<" "<<checkerror(status)  <<"\n"<<flush; exit_(status);}
	status	= clFinish(m_queue);						if (status != CL_SUCCESS)	{ cout << "\nRunCL::patch_hessian_reduce( ),  clFinish(m_queue) status = "<<status<<" "<<checkerror(status)  <<"\n"<<flush; exit_(status);}

	Matx16f		jacobian;
	Matx66f		Hessian;
	Mat			hessian_Mat(	(num_SE3_DoF+1),	num_SE3_DoF,	CV_32FC4);
	size_t		data_size	=	(num_SE3_DoF+1) *	num_SE3_DoF *	sizeof(cl_float4);
	size_t		offset		=	layer*8*6 ;																					cout<<"\noffset="<<offset<<flush;

	ReadOutput( hessian_Mat.data, SE3_hessian_pinv_map_mem, data_size, offset*sizeof(cl_float4) );
/*
	Mat			test_Mat(	42,	15,	CV_32FC4);
	ReadOutput( test_Mat.data, SE3_hessian_pinv_map_mem, 42*15*sizeof(cl_float4), 0);
	cout<<"\ntest_mat=\n"<<test_Mat<<endl<<endl<<flush;
*/

	Mat J					= Mat( hessian_Mat, Rect(0,0,6,1)	);

	for(int row=0; row<1; row++){																									// per_pixel division currently done in kernel, TODO which is better ?
		for(int col=0; col<num_SE3_DoF; col++){
			jacobian.operator()(row,col)		= J.at<cl_float4>( row,col ).x / J.at<cl_float4>( row,col ).w;						// NB choose colour channel of Hessian
		}
	}

	// NB we have one Hessian per color channel. HSV=>4,  HSV_grad => 8, likewise for the Jacobian.

	//Mat H					= Mat( hessian_Mat, Rect(0,1,6,6)	); 																	//hessian.inv()  NB computed in kernel: sum of pixelwise pseudo-inverse of the Hessian.

	for(int row=0; row<num_SE3_DoF; row++){																							// per_pixel division currently done in kernel, TODO which is better ?
		for(int col=0; col<num_SE3_DoF; col++){
			Hessian.operator()(row,col)		= hessian_Mat.at<cl_float4>( row+1,col ).x / hessian_Mat.at<cl_float4>( row+1,col ).w;	// NB choose colour channel of Hessian
		}
	}

	current_frames[ current_frames_idx[0] ].Jacobian[layer]			= jacobian;
	current_frames[ current_frames_idx[0] ].invHessian[layer]		= Hessian.inv();							//inv_Hessian.inv();
																																if( verbosity>local_verbosity_threshold) {
																																	cout <<"\nJacobian \n" << current_frames[ current_frames_idx[0] ].Jacobian[layer]	<< endl << endl <<flush;
																																	cout <<"\nHessian  \n" << current_frames[ current_frames_idx[0] ].invHessian[layer]	<< endl << endl <<flush;
																																	cout <<"\n\nRunCL::patch_hessian_reduce()_finished #############################################################"<<flush;
																																}
}

/*
void RunCL::IC_LK_SE3_tracking_update(uint layer){	// Inverse compositional Lucas-Kanade optimization of camera pose.


	Matx66f invHessian = current_frames[ current_frames_idx[0] ].invHessian[layer];
	float	sum_Rho, num_pixels;	// NB num_pixels should be only for img overlap => update each iteration.
	Matx61f	sum_Rho_J;				// pixelwise:  Rho * J
	Matx61f	sum_J;					// J = SE3_grad_map * img_grad

	Matx61f	pose_update	= invHessian * ( sum_Rho_J  - (sum_Rho * sum_J) )/ num_pixels;

}
*/

//void RunCL::patch_hessian_reduce( uint layer, uint out_block_size ){

	// launch kernel



	// compute Hessians for this layer


	// reload Hessians to GPU


//}


/* void RunCL::rho_sq(..)
//  example of using patches, but without lookuptable.
//  Also need to separate and save patch launch variables.
void RunCL::rho_sq(uint out_block_size, uint iter, uint layer, float delta_theta, float delta   ){	// To be launched with 1 thread per col for 32x32 patches, and an integer multiple of 32 threads.
																	// Needs 16 elements of local mem per 32x32 patch, to pass data between threads in recursive square reduction.
																	// Needs 32 elem array of private mem per thread.
																	// Writes answer to SE3_rho_map_mem, BUT as float2
	string fname					= "RunCL::rho_sq( ..)";
	int local_verbosity_threshold	= V_RUNCL_RHO_SQ;
	cl_kernel	kernel 				= rho_sq_kernel;
	//const int se3_dof				= 6;
	cl_float2	delta_SE3			= {{delta_theta, delta}};
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::rho_sq( ..)_chk0 .##################################################################"<<flush;
																																				cout << "\nRunCL::rho_sq( ..)__chk_1: K2K= ";
																																				for ( int i=0; i<16; i++){ cout << ",  "<< fp32_k2keyframe[i];  }	cout << flush;
																																				cout<<"\n\nRunCL::rho_sq( ..)_chk_2 ,  dataset_frame_num="<<dataset_frame_num<<",   out_block_size="<<out_block_size<<" , iter="<<iter<<" , layer="<<layer<<flush;
																																			}
	const uint			patch_size					= 32;																					// TODO set global patch size from device parameters // generally: device_work_size_multiple = patch_size * integer, eg 32, 64, 128
																									if (fmod(device_work_size_multiple, patch_size)!=0)   { cout <<"\nRunCL::rho_sq( ..)  Error: fmod(device_work_size_multiple, patch_size) != 0 \n"<<flush;exit_(0);}
	const float zero  = 0;
	_clEnqueueFillBuffer( uload_queue, SE3_rho_map_mem, 	&zero, sizeof( float), 0, 		  2*mm_size_bytes_C1, 	fname);					//_clEnqueueWriteBuffer( uload_queue, k2kbuf, CL_FALSE, 0, local_num_samples*16*sizeof( float), k2k_3_16_[start_sample_idx], fname);
	_clEnqueueFillBuffer( uload_queue, SE3_weight_map_mem, 	&zero, sizeof( float), 0, se3_dof*2*mm_size_bytes_C1, 	fname);
	_clEnqueueFillBuffer( uload_queue, SE3_incr_map_mem, 	&zero, sizeof( float), 0, se3_dof*2*mm_size_bytes_C1, 	fname);
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::rho_sq( ..)_chk_3 "<<flush;}
	size_t kernel_workgroup_size;
	cl_int k_wg_info =  clGetKernelWorkGroupInfo(
							kernel,								//cl_kernel kernel,
							deviceId,							//cl_device_id device,
							CL_KERNEL_WORK_GROUP_SIZE,			//cl_kernel_work_group_info param_name,
							sizeof(kernel_workgroup_size),		//size_t param_value_size,
							&kernel_workgroup_size,				//void* param_value,
							NULL								//size_t* param_value_size_ret
						);
	size_t  device_max_workitem_sizes[3];
	cl_int device_info_1 = clGetDeviceInfo(
							deviceId,							//cl_device_id device,
							CL_DEVICE_MAX_WORK_ITEM_SIZES,		//cl_device_info param_name,
							sizeof(device_max_workitem_sizes),	//size_t param_value_size,
							device_max_workitem_sizes,			//void* param_value,
							NULL								//size_t* param_value_size_ret
	);
	cl_uint device_max_compute_units;
	cl_int device_info_2 = clGetDeviceInfo(
							deviceId,							//cl_device_id device,
							CL_DEVICE_MAX_COMPUTE_UNITS,		//cl_device_info param_name,
							sizeof(device_max_compute_units),	//size_t param_value_size,
							&device_max_compute_units,			//void* param_value,
							NULL								//size_t* param_value_size_ret
	);
	size_t	max_workgroup_size	= min(kernel_workgroup_size, device_max_workitem_sizes[0] );
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::rho_sq( ..)_chk_4 "<<flush;
																																				// cout << "\n"
																																				// 	<<",  device_max_compute_units="				<<device_max_compute_units
																																				// 	<<",  max_workgroup_size="						<<max_workgroup_size
																																				// 	<<",  kernel_workgroup_size="					<<kernel_workgroup_size
																																				// 	<<",  device_max_workitem_sizes[0,1,2]={"		<<device_max_workitem_sizes[0]<<",  "
																																				// 													<<device_max_workitem_sizes[1]<<",  "
																																				// 													<<device_max_workitem_sizes[2]<<"}"
																																				// 	<< flush;
																																			}
	//uint				reduction 					= layer;
	uint				read_rows					= MipMap[layer * 8 + MiM_READ_ROWS] ;
	uint				read_cols					= MipMap[layer * 8 + MiM_READ_COLS] ;
	uint				rows_blocks					= ceil( (float)  read_rows / patch_size );
	uint				cols_blocks					= ceil( (float)  read_cols / patch_size );
	uint				cols_per_row				= cols_blocks  * patch_size;
	uint				patches_required			= cols_blocks  * rows_blocks;

	uint				patches_per_compute_uint	= ceil( (float)patches_required / device_max_compute_units ) ;
	uint 				blocks_per_k_wg_size		= max_workgroup_size			/ device_work_size_multiple;
						patches_per_compute_uint	= min( patches_per_compute_uint,  blocks_per_k_wg_size );

	uint				blocks_required				= ceil( (float)patches_required / patches_per_compute_uint );
	size_t				local_work_size_[1] 		= { patches_per_compute_uint	* patch_size };
	size_t				threads_to_launch 			= blocks_required 				* local_work_size_[0];									// TODO precompute an array for this function. ? where to store
																																			// ? Have a subclass and object for each kernel ?
																																			if( verbosity>local_verbosity_threshold-3) {cout<<"\n\nRunCL::rho_sq( ..)_chk_5 "<<flush;
																																				cout <<"\n"
																																					//<<",  reduction="<<reduction
																																					//<<",  num_threads[reduction]="					<<num_threads[reduction]
																																					//<<",  num_threads[reduction] / patch_size="		<<num_threads[reduction] / patch_size
																																					<<",  patches_required="						<<patches_required
																																					<<",  patches_per_compute_uint="				<<patches_per_compute_uint
																																					<<",  blocks_required="							<<blocks_required
																																					<<",  threads_to_launch="						<<threads_to_launch
																																					<<",  local_work_size="							<<local_work_size
																																					<<"},  local_work_size_[1]="					<<local_work_size_[0]
																																					<< flush;
																																				for (uint reduction = 0; reduction < 8  ; reduction ++){
																																					cout << "\n reduction = "						<< reduction
																																						<<"  num_threads[reduction] = "				<< num_threads[reduction]
																																						<< flush;
																																				}
																																				cout<<"\ntracking_num_samples*2*mm_size_bytes_C4="<<tracking_num_samples*2*mm_size_bytes_C4
																																					<<"     24 * mm_size_bytes_C1="<<24 * mm_size_bytes_C1<<flush;
																																			}
	//input integers
	_clSetKernelArg( kernel, 0, sizeof( uint),								&layer,	 												fname);		//__private		uint 		layer,					//0
	_clSetKernelArg( kernel, 1, sizeof( uint),								&cols_per_row,											fname);		//__private		uint 		cols_per_row,			//1
	_clSetKernelArg( kernel, 2, sizeof( uint),								&out_block_size,										fname);		//__private		uint 		out_block_size,			//2
	_clSetKernelArg( kernel, 3, sizeof( cl_float2),							&delta_SE3,												fname);							//__private	float2		delta_SE3,				//3

	_clSetKernelArg( kernel, 4, sizeof( cl_mem), 							&mipmap_buf,											fname);		//__constant	uint8*		mipmap_params,			//3
	_clSetKernelArg( kernel, 5, sizeof( cl_mem), 							&uint_param_buf,										fname);		//__constant	uint*		uint_params,			//4
	_clSetKernelArg( kernel, 6, sizeof( cl_mem), 							&fp32_param_buf,										fname);		//__constant	float*		fp32_params,			//5
	_clSetKernelArg( kernel, 7, sizeof( cl_mem), 							&k2kbuf,												fname);		//__constant	float16*	inv_k2k,				//6		// transforms for 4 past frames

	_clSetKernelArg( kernel, 8, sizeof( cl_mem),							&current_frames[current_frames_idx[0]].img_buf,			fname);		//__global		float4*		img_cur,				//7		// multiple past frames. NB retain frames at powers of 2, and vary starting power plus num franes.
	_clSetKernelArg( kernel, 9, sizeof( cl_mem), 							&current_frames[current_frames_idx[1]].img_buf,			fname);		//__global		float4*		img_past_0,				//8
	_clSetKernelArg( kernel,10, sizeof( cl_mem), 							&current_frames[current_frames_idx[2]].img_buf,			fname);		//__global		float4*		img_past_1,				//9
	_clSetKernelArg( kernel,11, sizeof( cl_mem), 							&current_frames[current_frames_idx[3]].img_buf,			fname);		//__global		float4*		img_past_2,				//10
	_clSetKernelArg( kernel,12, sizeof( cl_mem), 							&current_frames[current_frames_idx[4]].img_buf,			fname);		//__global		float4*		img_past_3,				//11
																																																					// NB GT_depth loaded to depth_mem by void RunCL::loadFrameData( ..)
	_clSetKernelArg( kernel,13, sizeof( cl_mem), 							&depth_mem,												fname);		//__global		float* 		depth_map,				//12	// current frame depth, now stored as inv_depth
	_clSetKernelArg( kernel,14, sizeof( cl_mem), 							&g1mem,													fname);		//__global		float8* 	g1p,					//13	// current frame g1mem
	_clSetKernelArg( kernel,15, sizeof( cl_mem), 							&SE3_grad_map_mem,										fname);		//__global 		float8*		SE3_grad_map_cur_frame,	//14

	_clSetKernelArg( kernel,16, sizeof( cl_mem), 							&current_frames[current_frames_idx[0]].r_vel_buf,		fname);		//__global		float4*		img_cur,				//15	// multiple past frames.
	_clSetKernelArg( kernel,17, sizeof( cl_mem), 							&current_frames[current_frames_idx[1]].r_vel_buf,		fname);		//__global		float4*		img_past_0,				//16
	_clSetKernelArg( kernel,18, sizeof( cl_mem), 							&current_frames[current_frames_idx[2]].r_vel_buf,		fname);		//__global		float4*		img_past_1,				//17
	_clSetKernelArg( kernel,19, sizeof( cl_mem), 							&current_frames[current_frames_idx[3]].r_vel_buf,		fname);		//__global		float4*		img_past_2,				//18
	_clSetKernelArg( kernel,20, sizeof( cl_mem), 							&current_frames[current_frames_idx[4]].r_vel_buf,		fname);		//__global		float4*		img_past_3,				//19
	//output
	_clSetKernelArg( kernel,21, sizeof( cl_mem), 							&SE3_rho_map_mem, 										fname);		//__global		float2* 	Rho_,					//20	// { sum rho^2 ,  count of valid pixels used } Writen to dense patches.
	_clSetKernelArg( kernel,22, sizeof( cl_float2)*local_work_size,			NULL, 													fname);		//__local		float2*		local_rho				//21	// float2 local_rho[ local_work_size/2 ]  hence sizeof( float)*local_work_size.

	_clSetKernelArg( kernel,23, sizeof( cl_mem), 							&SE3_incr_map_mem,										fname);		//__global 		float4*		SE3_incr_map_,			//24
	_clSetKernelArg( kernel,24, sizeof( cl_float2)*local_work_size*se3_dof,	NULL,													fname);		//__local 		float4*		local_SE3_incr			//25
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::rho_sq( ..)_chk_6 .  "<<flush;}

	cl_command_queue 	queue_to_call		= m_queue;
	cl_int				res, status;
	cl_event			ev;
																									auto step_0 = high_resolution_clock::now();
	res 	= clEnqueueNDRangeKernel(queue_to_call, kernel, 1, 0, &threads_to_launch, local_work_size_, 0, NULL, &ev);
																									if (res    != CL_SUCCESS)	{ cout << "\nres = " << checkerror(res) <<"\n"<<flush; exit_(res);}
	status 	= clFlush(queue_to_call);																if (status != CL_SUCCESS)	{ cout << "\nRunCL::rho_sq( ..) call_kernel( cl_kernel "<<kernel<<",  clFlush(queue_to_call) status  = "		<<status<<" "<< checkerror(status) <<"\n"<<flush; exit_(status);}
																									auto step_1 = high_resolution_clock::now();
	status 	= clWaitForEvents (1, &ev);																if (status != CL_SUCCESS)	{ cout << "\nRunCL::rho_sq( ..) call_kernel( cl_kernel "<<kernel<<") final,  clWaitForEventsh(1, &ev) ="		<<status<<" "<<checkerror(status)  <<"\n"<<flush; exit_(status);}
																									auto step_2 = high_resolution_clock::now();
	clReleaseEvent(ev);
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::rho_sq( ..)_chk_7 . "<<\
																																				"Execution time = "<<  duration_cast<microseconds>(step_1 - step_0).count() \
																																				<<" , "<<duration_cast<microseconds>(step_1 - step_2).count() <<flush;
																																			}
																																			if( verbosity>local_verbosity_threshold -3) {cout<<"\n\nRunCL::rho_sq( ..)_chk_8 ."<<flush;
																																				stringstream ss;
																																				ss << "_ds-framenum"<<dataset_frame_num<<"_img_layer"<<layer<<"_iter"<<iter<<"_out_bock_size"<<out_block_size<<"_rho_sq()";
																																				stringstream ss_path;
																																				bool show				= false;
																																				float max_range			= -1;
																																				uint vol_layers			= 1;
																																				bool exception_tiff 	= false;
																																				bool display			= false;	cout << "\nRunCL::rho_sq( ..)_chk_9   display="<< display<< endl << flush;
																																				bool old_tiff			= tiff;
																																				tiff					= true;
																																				DownloadAndSave_2Channel_volume(  SE3_rho_map_mem,		ss.str( ), paths.at( "SE3_rho_map_mem"),	2*mm_size_bytes_C1,   mm_Image_size,	CV_32FC2, show, max_range,	1);
																																				DownloadAndSave_2Channel_volume(  SE3_weight_map_mem,	ss.str( ), paths.at( "SE3_weight_map_mem"),	2*mm_size_bytes_C1,   mm_Image_size,	CV_32FC2, show, max_range,	vol_layers);
																																				DownloadAndSave_2Channel_volume(  SE3_incr_map_mem,		ss.str( ), paths.at( "SE3_incr_map_mem"),	2*mm_size_bytes_C1,   mm_Image_size,	CV_32FC2, show, max_range,	vol_layers);
																																				tiff = old_tiff;
																																				cout<<"\n\nRunCL::rho_sq( ..) finished"<< flush;
																																			}
}
*/

/* void RunCL::disparity_load_frame(..)
// example of using lookuptable
void RunCL::disparity_load_frame(cl_mem input_img, cl_mem output_img, std::string folder ){													// Loads an image into layer zero of a padded image pyramid, e.g. from basemem to img_mem or keyframe_img_mem.
	string fname = "RunCL::disparity_load_frame( )";
	int local_verbosity_threshold = V_RUNCL_WARP_IMAGE;
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::disparity_load_frame(..)_chk0 #############################################################"<<flush;
																																				cout << "\n local_work_size = "<<local_work_size<<",  num_threads[0] = "<<num_threads[0]<< flush;
																																			}
	uint baseImagePixels =   baseImage_size.height * baseImage_size.width;																	cout << "\nbaseImagePixels = "<<baseImagePixels<<",  baseImage_size.height = "<<baseImage_size.height<<",  baseImage_size.width = "<<baseImage_size.width <<flush;
																																			cout << "\nimage_size_bytes = "<<image_size_bytes<<flush;
	// __private
	//_clSetKernelArg( warp_image_kernel,			0, sizeof( uint), 	&read_offset,			fname);										// __private	uint	read_offset,			//0
	_clSetKernelArg( disparity_load_frame_kernel,	1, sizeof( uint), 	&baseImagePixels,		fname);										// __private	uint	baseImage_size,			//1
	// __global
	_clSetKernelArg( disparity_load_frame_kernel,  	2, sizeof( cl_mem), &lookup_table_buf,		fname);										// __global		float4*	lookup_table,			//2
	_clSetKernelArg( disparity_load_frame_kernel,  	3, sizeof( cl_mem), &input_img,				fname);										// __global		float* 	depth_map,				//3
	// output
	_clSetKernelArg( disparity_load_frame_kernel,  	4, sizeof( cl_mem), &output_img,			fname);										// __global		float2*	warp,					//4

																																				if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::disparity_load_frame( ..)_chk1 ."<<flush;}
	int layer = 0;																															// NB this kernel is called for layer 0 only.
	layer_call_kernel( disparity_load_frame_kernel, m_queue, layer, local_work_size);
																																			if( verbosity>local_verbosity_threshold / * && layer==1 * / ) {cout<<"\n\nRunCL::disparity_load_frame( ..)_chk2 ."<<flush;								// Save buffers to file ###########
																																				stringstream ss;
																																				ss << "_binoc__disparity_load_frame_" << save_index <<"_layer_"<<layer ;
																																				bool show 		= false;
																																				bool old_tiff 	= tiff;
																																				tiff 			= true;
																																				float max_range = 1; 		// -1 -> gray = zero.
																																				_cl_flush_finish(m_queue, fname);
																																				DownloadAndSave_3Channel(  output_img,  ss.str( ),  paths.at(folder),  mm_size_bytes_C4,  mm_Image_size,  CV_32FC4, show,  max_range);
																																				tiff 			= old_tiff;
																																			}
																																			if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::disparity_load_frame( ..)_finished ."<<flush;}
}
*/

/* void RunCL::patch_img_gradients()
// example of launching image gradients
void RunCL::patch_img_gradients(){ //getFrame();   TODO Need to rewrite for launching as a patch kernel. See RunCL_tracking.cpp   RunCL::rho_sq(...)
									//					Need to store global variables for the launch of patch kernels
	string fname = "RunCL::img_gradients()";
	int local_verbosity_threshold = V_RUNCL_IMG_GRADIENTS;																					if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::img_gradients(..)_chk0"<<flush;}
	size_t num_threads = ceil( (float)(mm_layerstep)/(float)local_work_size ) * local_work_size ;

	const cl_mem imgmem_   = current_frames[ current_frames_idx[0] ].img_buf;
																																			if(verbosity>local_verbosity_threshold) {cout << "\n num_threads = " << num_threads << ",   mm_layerstep = " << mm_layerstep << ",  local_work_size = " << local_work_size  <<endl << flush;}
	//      __private	 uint layer, set in mipmap_call_kernel(..) below                                                                      __private	 uint	    layer,		//0
    _clSetKernelArg(img_grad_kernel,  1, sizeof(cl_mem), &mipmap_buf, fname);																//__constant uint*	mipmap_params,	//1
	_clSetKernelArg(img_grad_kernel,  2, sizeof(cl_mem), &uint_param_buf, fname);															//__constant uint*	uint_params		//2
	_clSetKernelArg(img_grad_kernel,  3, sizeof(cl_mem), &fp32_param_buf, fname);															//__constant float*	fp32_params		//3
	_clSetKernelArg(img_grad_kernel,  4, sizeof(cl_mem), &imgmem_, fname);																	//__global   float4*	img,		//4
	_clSetKernelArg(img_grad_kernel,  5, sizeof(cl_mem), &SE3_map_mem, fname);																//__constant float2*	SE3_map,	//8
	_clSetKernelArg(img_grad_kernel,  6, sizeof(cl_mem), &SE3_grad_map_mem, fname);															//__global 	 float4*	SE3_grad_map//9
	_clSetKernelArg(img_grad_kernel,  7, sizeof(cl_mem), &HSV_grad_mem, fname);																//__global 	 float4*	HSV_grad_mem//10
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::img_gradients(..)_chk2"<<flush;}
	mipmap_call_kernel( img_grad_kernel, m_queue );
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::img_gradients(..)_chk3 Finished all loops. Saving gxmem, gymem."<<flush;  // , g1mem
																																				stringstream ss;	ss << dataset_frame_num << "__img_grad_kernel";
																																				stringstream ss_path;
																																				///
																																				ss_path.str(std::string()); // reset ss_path
																																				ss_path << "SE3_grad_map_mem"<<flush;
																																				cout << "\n" << ss_path.str() <<flush;
																																				cout << "\n" <<  paths.at(ss_path.str()) <<flush;
																																				DownloadAndSave_6Channel_volume(  SE3_grad_map_mem, ss.str(), paths.at(ss_path.str()), mm_size_bytes_C4, mm_Image_size, CV_32FC4, false, -1, 6 );
																																				///
																																				ss_path.str(std::string()); // reset ss_path
																																				ss_path << "HSV_grad_mem"<<flush;
																																				cout << "\n" << ss_path.str() <<flush;
																																				cout << "\n" <<  paths.at(ss_path.str()) <<flush;
																																				DownloadAndSave_HSV_grad(  HSV_grad_mem, ss.str(), paths.at(ss_path.str()), mm_size_bytes_C8, mm_Image_size, CV_32FC(8), false, -1, 0 );

																																				cout << "\n\n SE3_grad_map_mem = SE3_grad_map_mem = "<<SE3_grad_map_mem;
																																			}
																																			if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::img_gradients(..)_chk4 Finished."<<flush;}
}
*/

/* Notes on the Hessian:

For the Inverse Compositional variant of the Lucas-Kanade algorithm, we use the Gauss-Newton approximation of the Hessian matrix:

H = Sum_x {  [ Img_grad * d_Warp/d_pose ]^T  *  [ Img_grad * d_Warp/d_pose ]^T  }

So H is a 6x6 matrix per pixel, then summed over the image.

NB we will also want

1) Patches of ST3 for computing depth and relative velocity

2) Jacobians and Hessians wrt
(a) camera intrinsic matrix,  (b) lens distorsion

3) Jacobians & Hessians wrt
(a) reflectance,  (b) illumination,
(c) curvature ?
(d) orientation ?

(3) mechanical properties ?
(4) other optical properties ? transparency, refractive index, disersion...


Notes on the application of the Inverse of the Hessian:

1) computed when the image is loaded
2)

delta_parameter  =  H^(-1)  *  Sum_pixels {  [ Image_gradient * d_warp/d_pose ] * Rho(pixel)  }



*/






