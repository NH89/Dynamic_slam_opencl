#include "RunCL.hpp"

// called by Dynamic_slam::Dynamic_slam //////////////////////////////////////////////////////////////////////////////////////////////////

void RunCL::initialize_patch_params(){																// called by Dynamic_slam::Dynamic_slam
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


void RunCL::compute_patch_lookup_table(){										// called by Dynamic_slam::Dynamic_slam
	string fname = "RunCL::compute_patch_lookup_table( )";
	int local_verbosity_threshold = V_RUNCL_COMPUTE_PATCH_LOOKUP_TABLE;
	cl_kernel kernel = compute_patch_lookup_table_kernel;
																																if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::compute_patch_lookup_table( ..)_chk1 #############################################################"<<flush;
																																	cout << "\n local_work_size = " << local_work_size
																																	<< ",  mm_stop = " << mm_stop
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
																																	<<",  mm_stop="				<<mm_stop
																																	<<",  local_work_size="		<<local_work_size
																																	<<flush;
																																}
	cl_event		ev;
	cl_int			res, status;
	patch_lookup_table_offset[0]	= 0; 																						// NB 'start' may not be set to zero

	for(uint layer = 0; layer <= mm_stop; layer++) {																				// NB processes largest layer first.
																																if(verbosity>local_verbosity_threshold+1) { cout<<"\n\n\nRunCL::compute_patch_lookup_table( )_chk4,  reduction="\
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
		blocks_required									= fmax(blocks_required, 1);
		size_t				local_work_size_[1] 		= { patches_per_compute_uint	* patch_size };							// NB "patch_size" must be an integer fraction of the minimum workgroup for the particular GPU.
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
																																	cout <<"\n layer = "							<<layer
																																	<<"\n,  patches_required="						<<patches_required
																																	<<"\n,  patches_per_compute_uint="				<<patches_per_compute_uint
																																	<<"\n,  blocks_per_k_wg_size="					<<blocks_per_k_wg_size
																																	<<"\n,  blocks_required="						<<blocks_required
																																	<<"\n,  threads_to_launch="						<<threads_to_launch
																																	<<"\n,  local_work_size="						<<local_work_size
																																	<<"\n,  local_work_size_[0]="					<<local_work_size_[0]
																																	<<"\n,  patch_local_work_size["<<layer<<"]="	<<patch_local_work_size[layer]
																																	<< flush;
																																	for (uint reduction = 0; reduction < max_mipmap_layers; reduction ++){
																																		cout << "\n reduction = "					<< reduction
																																		<<"  patch_num_threads[reduction] = "		<< patch_num_threads[reduction]
																																		<< flush;
																																	}
																																	cout<<"\ntracking_num_samples*2*mm_size_bytes_C4="<<tracking_num_samples*2*mm_size_bytes_C4
																																		<<"     24 * mm_size_bytes_C1="<<24 * mm_size_bytes_C1<<"\n"<<flush;

																																	// for(uint layer = 0; layer <= stop; layer++) {
																																	// 	cout <<"\npatch_local_work_size["<<layer<<"] = "<<patch_local_work_size[layer]<< flush;
																																	// }
																																}
		uint 	lookup_table_offset_uint 				= patch_lookup_table_offset[layer];
		res 	= clSetKernelArg(kernel, 0, sizeof(int), &layer );																// __private	uint		layer,					//0
		res 	= clSetKernelArg(kernel, 1, sizeof(int), &lookup_table_offset_uint );											// __private	uint		lookup_table_offset,	//1
		res 	= clSetKernelArg(kernel, 2, sizeof(int), &cols_per_row);														// __private	uint		cols_per_row,			//2
																															if (res    !=CL_SUCCESS)	{ cout <<"\nres = "<<checkerror(res)<<"\n"<<flush;exit_(res);}	;
																																if( verbosity>local_verbosity_threshold+1) {cout<<"\n\nRunCL::compute_patch_lookup_table( )_chk_6"
																																	<<",  threads_to_launch="						<<threads_to_launch
																																	<<",    local_work_size_="						<< local_work_size_
																																	<<flush;
																																}

		res 	= clEnqueueNDRangeKernel(m_queue,		kernel, 1, 0, &threads_to_launch, local_work_size_, 0, NULL, &ev); // run mipmap_float4_kernel, NB wait for own previous iteration.
																															if (res    != CL_SUCCESS)	{ cout << "\nres = " << checkerror(res) <<"\n"<<flush; exit_(res);}
		status	= clFlush(m_queue);																							if (status != CL_SUCCESS)	{ cout << "\nRunCL::compute_patch_lookup_table( ),  clFlush(m_queue) status  = "<<status<<" "<< checkerror(status) <<"\n"<<flush; exit_(status);}
		status	= clWaitForEvents (1, &ev);																					if (status != CL_SUCCESS)	{ cout << "\nRunCL::compute_patch_lookup_table( ),  clWaitForEventsh(1, &ev) ="	<<status<<" "<<checkerror(status)  <<"\n"<<flush; exit_(status);}
		if (layer < mm_stop ){
			patch_lookup_table_offset[layer +1]				=	patch_lookup_table_offset[layer] + threads_to_launch;			// NB this pads the lookup table, so that local work groups will not be shared betwen layers.
																																// It also  means that this offset should be used to launch layers from the lookup table.
		}
																																if( verbosity>local_verbosity_threshold+1) {cout<<"\n\nRunCL::compute_patch_lookup_table( )_chk_7"<<flush;}
	}
																																if( verbosity>local_verbosity_threshold) {
																																	for(uint reduction = 0; reduction <= mm_stop; reduction++) {
																																		cout << "\npatch_num_threads["<<reduction<<"] = "<<patch_num_threads[reduction]<<",   MipMap[reduction*8 +MiM_PIXELS] = "<<MipMap[reduction*8 +MiM_PIXELS]<<flush;
																																	}
																																	cout <<"\nNB there will only be a gap in the lookuptable when:  patch_num_threads[reduction] > MipMap[reduction*8 +MiM_PIXELS]  , which depends on the image dimensions and local_work_size"<<flush;
																																}
																																if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::compute_patch_lookup_table( )_chk6 ."<<flush;	// Save buffers to file ###########
																																	stringstream ss;
																																	ss << "compute_patch_lookup_table_";// << save_index ;
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
																																	// for(uint layer = 0; layer <= stop; layer++) {
																																	// 	cout <<"\npatch_local_work_size["<<layer<<"] = "<<patch_local_work_size[layer]<< flush;
																																	// }
																																}

}


void RunCL::patch_img_gradients_set_params(){	// Uses patch lookup table		// called by Dynamic_slam::Dynamic_slam
	string fname = "RunCL::patch_img_gradients_set_params()";
	int local_verbosity_threshold = V_RUNCL_PATCH_IMG_GRADIENTS;																if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::patch_img_gradients_set_params()_chk1 #############################################################"<<flush;
																																	for(uint layer = 0; layer < max_mipmap_layers; layer++) {
																																		cout <<"\npatch_local_work_size["<<layer<<"] = "<<patch_local_work_size[layer]<< flush;
																																	}
																																}
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
																																	cout <<"\npatch_size = "<<patch_size<<flush;
																																	// for(uint layer = 0; layer < max_mipmap_layers; layer++) {
																																	// 	cout <<"\npatch_local_work_size["<<layer<<"] = "<<patch_local_work_size[layer]<< flush;
																																	// }
																																}
	for (uint layer_=0; layer_<max_mipmap_layers; layer_++){
		patch_img_gradients_workgroup_size[layer_]	= min( patch_local_work_size[layer_],  patches_per_workgroup_local_mem * patch_size );
																																if( verbosity>local_verbosity_threshold+1 ) {
																																	cout <<"\npatch_local_work_size["<<layer_<<"] = "<<patch_local_work_size[layer_];
																																	cout <<"\tpatch_img_gradients_workgroup_size["<<layer_<<"] = "<<patch_img_gradients_workgroup_size[layer_]<<flush;
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
	//Outputs:
	//__global
	_clSetKernelArg( kernel,	11, sizeof( cl_mem), 	&SE3_grad_map_mem,				fname);									// __global 	float8*		SE3_grad_map,			//7		// We keep hsv sepate at this stage, so 6*4*2=24, but float16 is the largest type, so 6*float8.
	_clSetKernelArg( kernel,	12, sizeof( cl_mem), 	&SE3_hessian_pinv_map_mem,		fname);									// __global 	float4*		SE3_Hessian_pinv_map,	//8		// HSV (6x6) matrix so 36*float8
	_clSetKernelArg( kernel,	14, sizeof( cl_mem), 	&HSV_grad_mem,					fname);									// __global 	float8*		HSV_grad				//10

	// For the SE3 Hessian patches, and their reduction.	/////////////
																																if( verbosity>local_verbosity_threshold+1 ) {
																																	cout<<"\nvoid RunCL::patch_img_gradients_set_params(  ):  mm_start="<<mm_start<<"   mm_stop="<<mm_stop<<flush;
																																}
	uint	hessian_layer_offset		=	5	+		5*mm_width;
	uint	st3_hessian_layer_offset	=	5	+		( 4+( MipMap[	 0*8 + MiM_READ_ROWS]		/block_size) )*mm_width   * (num_SE3_DoF + 1);

	for (uint layer =0; layer<=mm_stop; layer++){																				// NB must match where the SE3 Hessian is written in SE3_hessian_map_mem.
																																// i.e. 6x6 elems in img pyramid horizontally across the top of the buffer.
		patch_hessian_cols[		layer]	= ceil( ((float)		  MipMap[layer*8 + MiM_READ_COLS]	)	/block_size );
		patch_hessian_rows[		layer]	= ceil( ((float)		  MipMap[layer*8 + MiM_READ_ROWS]	)	/block_size );

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
																																	// <<".\n    patch_hessian_cols[	"<<layer<<"]	= "		<<patch_hessian_cols[ layer]<<",    MipMap[layer*8 + MiM_READ_COLS] = "<<MipMap[layer*8 + MiM_READ_COLS]
																																	// <<",\n    patch_hessian_rows[	"<<layer<<"]	= "		<<patch_hessian_rows[ layer]<<",    MipMap[layer*8 + MiM_READ_ROWS] = "<<MipMap[layer*8 + MiM_READ_ROWS]
																																	// <<",\n    block_size = "								<<block_size
																																	<<flush;


																																}
		hessian_layer_offset			+=		6* hessian_elem_step;					//MipMap[layer*8 + MiM_READ_OFFSET]	/mm_width;
		st3_hessian_layer_offset		+=		3* st3_hessian_row_step;
	}
																																if( verbosity>local_verbosity_threshold ) {
																																	for (uint layer =0; layer<mm_stop; layer++){
																																		cout<<"\npatch_ST3_hessian_start_idx[	layer="<<layer<<"][row][col] = "<<patch_hessian_start_idx[layer][0][0]<<flush;
																																	}
																																}
}

// called by Dynamic_slam::getFrame //////////////////////////////////////////////////////////////////////////////////////////////////

void RunCL::patch_img_gradients( uint layer ){														// called by Dynamic_slam::getFrame
	string fname = "RunCL::patch_img_gradients()";
	int local_verbosity_threshold = V_RUNCL_PATCH_IMG_GRADIENTS;																if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::patch_img_gradients()_chk1 #############################################################"<<flush;}
	cl_kernel		kernel						= patch_img_grad_kernel;
																																cout<<"  chk_1 , layer = "<<layer<<flush;
	const cl_mem 	imgmem_						= current_frames[ 						current_frames_idx[0] ].img_buf;

																																cout<<"  chk_2 "<<flush;
	size_t			local_work_size_			= patch_img_gradients_workgroup_size[	layer];											// patch_local_work_size[	layer];						//patch_img_gradients_workgroup_size; // patch_img_gradients_local_work_size_;
	size_t			threads_to_launch			= patch_num_threads[					layer];
	size_t			local_Hessian_size			= sizeof(cl_float4)						*num_SE3_DoF *num_SE3_DoF	*local_work_size_;	//*patch_img_gradients_workgroup_size;
	uint			lookup_table_offset_uint 	= patch_lookup_table_offset[			layer];
																																if( verbosity>local_verbosity_threshold ) {
																																	for (uint layer_ =0; layer_<mm_stop; layer_++){
																																		cout<<"\npatch_ST3_hessian_start_idx[	layer="<<layer_<<"][row][col] = "<<patch_hessian_start_idx[layer_][0][0]<<flush;
																																	}
																																}
																																cout<<"  chk_3 "<<flush;
	uint			SE3_h_offset				= patch_hessian_start_idx[layer][0][0];
	uint			ST3_h_offset				= patch_ST3_hessian_start_idx[layer][0][0];
	cl_uint3		SE3_hessian_offset			= {{ SE3_h_offset,	(patch_hessian_start_idx[layer][0][1]     - SE3_h_offset) ,	(patch_hessian_start_idx[layer][1][0]     - SE3_h_offset)  }};
	cl_uint3		ST3_out_offset				= {{ ST3_h_offset,	(patch_ST3_hessian_start_idx[layer][0][1] - ST3_h_offset) ,	(patch_ST3_hessian_start_idx[layer][1][0] - ST3_h_offset)  }};
																																cout<<"  chk_4, local_Hessian_size = "<<local_Hessian_size<<"  "<<flush;

																																cout<<" out_block_size = "<<out_block_size
																																<<"\n SE3_h_offset = "<<SE3_h_offset
																																<<"\n ST3_h_offset = "<<ST3_h_offset
																																<<"\n SE3_hessian_offset = ("<<SE3_hessian_offset.x <<", "<<SE3_hessian_offset.y <<", "<<SE3_hessian_offset.z <<")"
																																<<"\n ST3_out_offset = ("<<ST3_out_offset.x <<", "<<ST3_out_offset.y <<", "<<ST3_out_offset.z <<")"
																																<<flush;

	_clSetKernelArg( kernel,	0, sizeof(int),			&layer,							fname);									// __private	uint		layer,						//0
	_clSetKernelArg( kernel,	1, sizeof(int),			&lookup_table_offset_uint,		fname);									// __private	uint		lookup_table_offset_uint,	//1
	//_clSetKernelArg( kernel,	2, sizeof(int),			&out_block_size,				fname);									// __private	uint		out_block_size,				//2
	_clSetKernelArg( kernel,	3, sizeof(cl_uint3),	&SE3_hessian_offset,			fname);									// __private	uint		SE3_hessian_offset,			//3
	_clSetKernelArg( kernel,	4, sizeof(cl_uint3),	&ST3_out_offset,				fname);									// __private	uint		SE3_hessian_offset,			//3

	_clSetKernelArg( kernel,	9, sizeof( cl_mem),		&imgmem_,						fname);									// __global 	float4*		img,					//6		//	"current_frames[idx].img_buf	= imgmem[idx];", NB changes every new frame.
	_clSetKernelArg( kernel,	10, sizeof( cl_mem), 	&depth_mem,						fname);									// __global		float* 		depth_map,				//12	// current frame depth, now stored as inv_depth


	_clSetKernelArg( kernel,	13,local_Hessian_size,	NULL,							fname);									// __local		float4*		local_Hessian,			//9		// local_Hessian[ sizeof(float4) *6*6 *local_size]

/*
cl_int clGetMemObjectInfo(
cl_mem memobj,
cl_mem_info param_name,   CL_MEM_SIZE
size_t param_value_size,
void* param_value,
size_t* param_value_size_ret);
*/

																																cout<<"\nRunCL::patch_img_gradients()_chk_5 "<<flush;
	cl_event	ev;
	cl_int		res, status;

	res 	= clEnqueueNDRangeKernel(m_queue,		kernel, 1, 0, &threads_to_launch, &local_work_size_, 0, NULL, &ev);
																	if (res    != CL_SUCCESS)	{ cout << "\nres = " << checkerror(res) <<"\n"<<flush; exit_(res);}
	status	= clFlush(m_queue);										if (status != CL_SUCCESS)	{ cout << "\nRunCL::patch_img_gradients( ),  clFlush(m_queue) status  = "<<status<<" "<<checkerror(status) <<"\n"<<flush; exit_(status);}
	status	= clWaitForEvents (1, &ev);								if (status != CL_SUCCESS)	{ cout << "\nRunCL::patch_img_gradients( ),  clWaitForEventsh(1, &ev) =" <<status<<" "<<checkerror(status) <<"\n"<<flush; exit_(status);}
																																/*
																																// if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::patch_img_gradients()_chk2 ."<<flush;	// Save buffers to file ###########
																																// 	stringstream ss;
																																// 	ss << "patch_img_gradients_";// << save_index ;
																																// 	bool show 		= false;
																																// 	bool old_tiff 	= tiff;
																																// 	tiff 			= true;
																																// 	float max_range	= 1;
																																// 	cv::Mat bufImg;
																																// 	_cl_flush_finish(m_queue, fname);
																																// 	//DownloadAndSave_3Channel( 	SE3_hessian_map_mem,	ss.str( ), paths.at( "hessian"),  		mm_size_bytes_C4,   mm_Image_size,   CV_32FC4, 	show);
																																// 	DownloadAndSave_3Channel( 	SE3_hessian_pinv_map_mem,	ss.str( ), paths.at( "hessian"),  		mm_size_bytes_C4,   mm_Image_size,   CV_32FC4, 	show, &bufImg, max_range,  0,			false);
																																// 	DownloadAndSave_3Channel( 	SE3_hessian_pinv_map_mem,	ss.str( ), paths.at( "jacobian"),  		mm_size_bytes_C4,   mm_Image_size,   CV_32FC4, 	show, &bufImg, max_range,  mm_size_bytes_C4 / * mm_layerstep * / , false);
																																// 	// NB the tiff file holda the int32 values as float32. This is okay because they fit in the mantissa.
																																// 	// BGRA format, B=u, G=v, R=read_index, A=alpha.
																																// 	////////////////
																																// 	if( layer==0){
																																// 		stringstream 	ss_path;
																																// 		ss_path.str(std::string()); // reset ss_path
																																// 		ss_path 		<< "SE3_grad_map_mem"<<flush;
																																// 		cout 			<< "\n" << ss_path.str() <<flush;
																																// 		cout 			<< "\n" << paths.at(ss_path.str()) <<flush;
																																// 		DownloadAndSave_6Channel_volume(  SE3_grad_map_mem, ss.str(), paths.at(ss_path.str()), mm_size_bytes_C4, mm_Image_size, CV_32FC4, false, -1, 6 );
																																// 	}
																																// 	//////////
																																// 	tiff 			= old_tiff;
																																// }
																																*/
																																if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::patch_img_gradients()_finished #############################################################"<<flush;

// #define MiM_PIXELS			0	// for mipmap_buf, 				when launching one kernel per layer. 	Updated for each layer.
// #define MiM_READ_OFFSET		1	// for ths layer, 				start of image data
// #define MiM_WRITE_OFFSET	2
// #define MiM_READ_COLS		3	// cols without margins
// #define MiM_WRITE_COLS		4
// // #define MiM_GAUSSIAN_SIZE	5	// filter box size
// #define MiM_READ_ROWS		6	// rows without margins
// #define MiM_WRITE_ROWS		7
																																	int offset	=	MipMap[layer*8 +  MiM_READ_OFFSET   ];
																																	int rows	=	MipMap[layer*8 +  MiM_READ_ROWS   ];
																																	int size_bytes	= rows * mm_width * 4*sizeof(float) ;

																																	cv::Mat temp_mat = cv::Mat::zeros (rows, mm_width, CV_32FC4);
																																	cout<<"\n chk 1, offset = "<<offset<<",  rows ="<<rows<<flush;

																																	ReadOutput(temp_mat.data, SE3_grad_map_mem, size_bytes, offset*4*sizeof(float)   );// , 0/*offset*/	// read 1st elem of Jacobian to verify kernel summation.

																																	// cout<<"\n chk 2"<<flush;
																																	// cv::imshow( "SE3_grad_map_mem", temp_mat);
																																	// cv::waitKey(-1);

																																	cout<<"\n chk 3"<<flush;
																																	cl_float4 sum_J1	= {{0.0f}};
																																	cl_float4 sum_H11	= {{0.0f}};

																																	for(int row=0; row<temp_mat.rows; row++){
																																		for(int col=0; col<temp_mat.cols; col++){
																																			sum_J1.w += temp_mat.at<cl_float4>(row,col).w;
																																			sum_J1.x += temp_mat.at<cl_float4>(row,col).x;
																																			sum_J1.y += temp_mat.at<cl_float4>(row,col).y;
																																			sum_J1.z += temp_mat.at<cl_float4>(row,col).z;

																																			sum_H11.w += pow(temp_mat.at<cl_float4>(row,col).w, 2);
																																			sum_H11.x += pow(temp_mat.at<cl_float4>(row,col).x, 2);
																																			sum_H11.y += pow(temp_mat.at<cl_float4>(row,col).y, 2);
																																			sum_H11.z += pow(temp_mat.at<cl_float4>(row,col).z, 2);
																																		}
																																	}
																																	cout<<"\n\n##### layer = "<<layer<<", SE3_grad_map_mem sum_J1 = "<<sum_J1.w<<", "<<sum_J1.x<<", "<<sum_J1.y<<", "<<sum_J1.z
																																													<<",    sum_H11 = "<< sum_H11.w<<", "<<sum_H11.x<<", "<<sum_H11.y<<", "<<sum_H11.z
																																	<<endl<<endl<<flush;
																																}
}


void  RunCL::patch_hessian_reduce(uint layer){														// called by Dynamic_slam::getFrame
	string 		fname	= "RunCL::patch_hessian_reduce()";
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
			uint start_idx	=	patch_hessian_start_idx[layer][row][col];														if( verbosity>local_verbosity_threshold) {cout<<"\nRunCL::patch_hessian_reduce()_chk2  patch_hessian_start_idx["<<layer<<"]["<<row<<"]["<<col<<"] = "<<patch_hessian_start_idx[layer][row][col]<<flush; }
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

	Matx16f		Jacobian;
	Matx66f		Hessian;
	Mat			hessian_Mat(	(num_SE3_DoF+1),	num_SE3_DoF,	CV_32FC4);
	size_t		data_size	=	(num_SE3_DoF+1) *	num_SE3_DoF *	sizeof(cl_float4);
	size_t		offset		=	layer*8*6 ;																						cout<<"\noffset="<<offset<<flush;

	ReadOutput( hessian_Mat.data, SE3_hessian_pinv_map_mem, data_size, offset*sizeof(cl_float4) );
																																cout<<"\nhessian_Mat = \n"<<hessian_Mat<<flush;
	Mat J									= Mat( hessian_Mat, Rect(0,0,6,1)	);

	for(int row=0; row<1; row++){																									// per_pixel division currently done in kernel, TODO which is better ?
		for(int col=0; col<num_SE3_DoF; col++){
			Jacobian.operator()(row,col)	= J.at<cl_float4>( row,col ).x; // / J.at<cl_float4>( row,col ).w;						// NB choose colour channel of Hessian
		}
	}
/*
	// NB we have one Hessian per color channel. HSV=>4,  HSV_grad => 8, likewise for the Jacobian.

	//Mat H					= Mat( hessian_Mat, Rect(0,1,6,6)	); 																	//hessian.inv()  NB computed in kernel: sum of pixelwise pseudo-inverse of the Hessian.
*/
	for(int row=0; row<num_SE3_DoF; row++){																							// per_pixel division currently done in kernel, TODO which is better ?
		for(int col=0; col<num_SE3_DoF; col++){
			Hessian.operator()(row,col)		= hessian_Mat.at<cl_float4>( row+1,col ).x; // / hessian_Mat.at<cl_float4>( row+1,col ).w;	// NB choose colour channel of Hessian
		}
	}
	current_frames[ current_frames_idx[0] ].Jacobian[layer]			= Jacobian;
/*
	//current_frames[ current_frames_idx[0] ].invHessian[layer]		= Hessian.inv();							//inv_Hessian.inv();
*/
	//Matx66f GN_Hessian 						=  Jacobian.t()  * Jacobian ; /// #### TODO wrong !  need GN_H  = sum( elementwise J.t * J )
	//invert(GN_Hessian,	current_frames[ current_frames_idx[0] ].invHessian[layer]	);

	//  Eigen pseudo-inverse
	Eigen::MatrixXd GN_H(6,6);																					// TODO replace Eigen with a kernel for 6x6 matrix pseudo-inverse or inverse.
	for (int i=0;i<6;i++){																						// Hard code efficient computation of 6x6 inversion, & Det.
		for (int j=0;j<6;j++){
			GN_H(i,j) 						= Hessian.operator()(i,j);	// GN_Hessian.operator()(i,j);
		}
	}
	Eigen::MatrixXd pinv 					= GN_H.completeOrthogonalDecomposition().pseudoInverse();
	Matx66f pinv_H;
	for (int i=0;i<6;i++){
		for (int j=0;j<6;j++){
			pinv_H.operator()(i,j)			= pinv(i,j) ;
		}
	}
	current_frames[ current_frames_idx[0] ].invHessian[layer]		= pinv_H;
/*
	//  Eigen matrix inverse ////////////////////////////////////////////////
	typedef Eigen::Matrix<double,3,3> 		Matrix3x3;
	Matrix3x3	m = Matrix3x3::Random();

	Eigen::Matrix3f A ;
	A <<1,2,3,4,5,6,7,8,9;

	Eigen::FullPivLU<Eigen::Matrix3f>		lu(  A  ) ;
*/
	////Prove inv = pinv when invertible. NB pinv is numerically safer.
	typedef Eigen::Matrix<double,6,6> Matrix6x6d;
	Matrix6x6d GN_H2;

	for (int i=0;i<6;i++){
		for (int j=0;j<6;j++){
			GN_H2(i,j) 						= Hessian.operator()(i,j);
		}
	}

	Eigen::FullPivLU<Matrix6x6d> lu_GN_H2(GN_H2);
	bool invertible							= lu_GN_H2.isInvertible();
	Matrix6x6d 			inv 				= lu_GN_H2.inverse();

																																if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::patch_hessian_reduce()_chk3 ."<<flush;	// Save buffers to file ###########
																																	stringstream ss;
																																	ss << "patch_hessian_reduce_";// << save_index ;
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
																																	////////////////
																																	// cv::Mat temp_mat = cv::Mat::zeros (mm_height, mm_width, CV_32FC4);
																																	// ReadOutput(temp_mat.data, SE3_grad_map_mem, mm_size_bytes_C4);// , 0/*offset*/	// read 1st elem of Jacobian to verify kernel summation.
																																	// cl_float4 sum = {{0.0f}};
                                 //
																																	// for(int row=0; row<temp_mat.rows; row++){
																																	// 	for(int col=0; col<temp_mat.cols; col++){
																																	// 		sum.w += temp_mat.at<cl_float4>(row,col).w;
																																	// 		sum.x += temp_mat.at<cl_float4>(row,col).x;
																																	// 		sum.y += temp_mat.at<cl_float4>(row,col).y;
																																	// 		sum.z += temp_mat.at<cl_float4>(row,col).z;
																																	// 	}
																																	// }
																																	// cout<<"\n\n##### SE3_grad_map_mem sum.x = "<<sum.w<<", "<<sum.x<<", "<<sum.y<<", "<<sum.z<<endl<<endl<<flush;

																																	if( layer==0){
																																		stringstream 	ss_path;
																																		ss_path.str(std::string()); // reset ss_path
																																		ss_path 		<< "SE3_grad_map_mem"<<flush;
																																		cout 			<< "\n" << ss_path.str() <<flush;
																																		cout 			<< "\n" << paths.at(ss_path.str()) <<flush;
																																		DownloadAndSave_6Channel_volume(  SE3_grad_map_mem, ss.str(), paths.at(ss_path.str()), mm_size_bytes_C4, mm_Image_size, CV_32FC4, false, 1, 6 );
																																	}
																																	//////////
																																	tiff 			= old_tiff;
																																}
																																if( verbosity>local_verbosity_threshold) {
																																	const auto old_precision{ cout.precision() };
																																	cout << setprecision(15);
																																	cout <<"\nJacobian \n"				<< current_frames[ current_frames_idx[0] ].Jacobian[layer]		<< endl << endl <<flush;
																																	cout <<"\nEigen GN_H \n" 			<< GN_H															<< endl << endl <<flush;
																																	cout <<"\nEigen pinv \n" 			<< pinv															<< endl << endl <<flush;

																																	// cout <<"\nGN_Hessian \n"			<< GN_Hessian													<< endl << endl <<flush;
																																	// cout <<"\nGN_Hessian.inv() \n"	<< GN_Hessian.inv()												<< endl << endl <<flush;
																																	// cout <<"\nocv invHessian \n"		<< current_frames[ current_frames_idx[0] ].invHessian[layer]	<< endl << endl <<flush;

																																	cout <<"\nEigen FullPivLU GN_H2 \n" << GN_H2															<< endl << endl <<flush;
																																	cout <<"\nEigen FullPivLU isInvertible = "<< invertible <<endl<<flush;
																																	cout <<"\nEigen FullPivLU inv \n" 	<< inv															<< endl << endl <<flush;

																																	cout << setprecision( old_precision );

																																	cout <<"\n\nRunCL::patch_hessian_reduce()_finished #############################################################"<<flush;


																																	//
// 																													/*__private*/	uint		layer;							//,		//0
// 																													/*__private*/	uint		lookup_table_offset;			//,		//1
// 																													/*__private*/	cl_uint3	SE3_offset3;					//,		//3
// 																													/*__global*/ 	cl_float4*	lookup_table;					//,		//8
//
// 																																	const uint	SE3_offset					= SE3_offset3.s0;
// 																																	cl_float4	lookup_ref					= lookup_table[		 lookup_table_offset	];						// global_id_uint=0 +
//
// 																																	uint		u							= lookup_ref.x;														// read_column
// 																																	uint		v							= lookup_ref.y;														// read_row
//
// 																																	uint		mm_pixels					= uint_params[MM_PIXELS];
// 																																	uint		write_index_2				= u/block_size		+ (v/block_size)*mm_cols	+ SE3_offset;
// 																																	int 		offset_2 					= write_index_2		+ mm_pixels;									// + i*SE3_v_step	i=0 for first se3 dof
																																	if( layer ==0){
																																		cv::Mat temp_mat = cv::Mat::zeros (mm_height, mm_width, CV_32FC4);
																																		cout<<"\n chk 1, offset = "<<offset<<",  rows ="<<rows<<flush;

																																		ReadOutput(temp_mat.data, SE3_hessian_pinv_map_mem, mm_size_bytes_C4,    mm_size_bytes_C4   );// read Jacobian buffer into Mat

																																		// cout<<"\n chk 2"<<flush;
																																		// cv::imshow( "SE3_grad_map_mem", temp_mat);
																																		// cv::waitKey(-1);

																																		cout<<"\n chk 3"<<flush;
																																		cl_float4 sum = {{0.0f}};

																																		for(int row=1; row<=15; row++){
																																			for(int col=5; col<=24; col++){
																																				sum.w += temp_mat.at<cl_float4>(row,col).w;
																																				sum.x += temp_mat.at<cl_float4>(row,col).x;
																																				sum.y += temp_mat.at<cl_float4>(row,col).y;
																																				sum.z += temp_mat.at<cl_float4>(row,col).z;
																																			}
																																		}
																																		cout<<"\n\n##### layer = "<<layer<<", layer 0, 1st Jacobian sum = "<<sum.w<<", "<<sum.x<<", "<<sum.y<<", "<<sum.z<<endl<<endl<<flush;
																																	}
																																}
}


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






