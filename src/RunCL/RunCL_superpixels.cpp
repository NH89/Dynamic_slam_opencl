#include "RunCL.hpp"
void RunCL::compute_superpx_params(){
	int			local_verbosity_threshold				= V_RUNCL_COMPUTE_SUPERPIX_PARAMS;
	superpx_params[0].cluster_layer_offset				= 0;
	uint cluster_dim									= obj["cluster_dim"].asUInt();

	for(uint layer=0; layer<mm_num_reductions; layer++){
		superpx_params[layer].cluster_dim				= cluster_dim	;
		superpx_params[layer].cols_of_clusters			= MipMap[layer*8 + MiM_READ_COLS]				/ cluster_dim;
		superpx_params[layer].rows_of_clusters			= MipMap[layer*8 + MiM_READ_ROWS] 				/ cluster_dim;
		superpx_params[layer].num_clusters				= superpx_params[layer].cols_of_clusters		* superpx_params[layer].rows_of_clusters;
		superpx_params[layer+1].cluster_layer_offset	= superpx_params[layer].cluster_layer_offset	+ superpx_params[layer].num_clusters;


		// 9 neighbours + final sum.
// (superpx_params[0].cols_of_clusters +2)	*	(superpx_params[0].rows_of_clusters	+2)
																													if( verbosity>local_verbosity_threshold ){
																														Superpixel_params p = superpx_params[layer];
																														cout<<"\n RunCL::compute_superpx_params()"
																															<<",	layer="					<<layer
																															<<",	cluster_dim="			<<p.cluster_dim
																															<<",	cols_of_clusters="		<<p.cols_of_clusters
																															<<",	num_clusters="			<<p.num_clusters
																															<<flush;
																													}
	}
	uint layer=0;
	superpix_img_bytes									= sizeof(cl_float2)	* superpx_params[layer].num_clusters;						//lowest_multiple(   superpx_params[layer].num_clusters 						, local_work_size );
	superpix_vol_bytes									= sizeof(cl_float2) * superpx_params[layer].num_clusters * costVolLayers * 10;	//lowest_multiple( ( superpx_params[layer].num_clusters * costVolLayers * 10 )	, local_work_size ); //* sizeof(float) *

																													if( verbosity>local_verbosity_threshold ){
																														Superpixel_params p = superpx_params[layer];
																														cout<<"\n RunCL::compute_superpx_params()"
																															<<",	layer="					<<layer
																															<<",	num_clusters="			<<p.num_clusters
																															<<",	local_work_size="		<<local_work_size
																															<<",	sizeof(cl_float2)="		<<sizeof(cl_float2)
																														//	<<",	lowest_multiple(   superpx_params[layer].num_clusters, local_work_size )="	<<lowest_multiple(   superpx_params[layer].num_clusters, local_work_size )
																															<<",	superpix_img_bytes="	<<superpix_img_bytes
																															<<",	superpix_vol_bytes="	<<superpix_vol_bytes
																															<<flush;
																													}
}


void RunCL::initiate_cluster_centres(uint layer){
	string		fname						= "RunCL::initiate_cluster_centres( )";
	int			local_verbosity_threshold	= V_RUNCL_INITIATE_CLUSTER_CETRES;
	cl_kernel	kernel						= initiate_cluster_centres_kernel;
																													if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::initiate_cluster_centres( ..)_chk0 #############################################################"<<flush;
																													}
	const int	cluster_dim					= superpx_params[layer].cluster_dim;

	size_t		threads_to_launch			= ceil_( MipMap[layer*8 + ROWS], cluster_dim ) * ceil_( MipMap[layer*8 + COLS], cluster_dim );//lowest_multiple( MipMap[layer*8 + ROWS], cluster_dim ) * lowest_multiple( MipMap[layer*8 + COLS], cluster_dim );
	threads_to_launch						= lowest_multiple( threads_to_launch, local_work_size );
																													if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::initiate_cluster_centres( ..)_chk1 "<<flush;
																														cout<<"\nthreads_to_launch="				<<threads_to_launch\
																															<<"\npatch_num_threads["<< layer<<"]="	<<patch_num_threads[ layer]\
																															<<"\nlocal_work_size="					<<local_work_size\
																															<<"\nlowest_multiple( MipMap[layer*8 + ROWS], cluster_dim )="	<<lowest_multiple( MipMap[layer*8 + ROWS], cluster_dim )\
																															<<"\nlowest_multiple( MipMap[layer*8 + COLS], cluster_dim )="	<<lowest_multiple( MipMap[layer*8 + COLS], cluster_dim )\
																															<<"\nlowest_multiple( threads_to_launch, local_work_size )="	<<lowest_multiple( threads_to_launch, local_work_size )\
																															<<flush;
																													}
	//Inputs:
	_clSetKernelArg( kernel,  0, sizeof( uint),		&layer,					fname);									//  __private	uint	layer,					//0
	_clSetKernelArg( kernel,  1, sizeof( int),		&cluster_dim,			fname);									//  __private	uint	cluster_dim,			//1

	_clSetKernelArg( kernel,  2, sizeof( cl_mem),	&mipmap_buf,			fname);									//  __constant	uint8*	mipmap_params,			//2
	_clSetKernelArg( kernel,  3, sizeof( cl_mem),	&uint_param_buf,		fname);									//  __constant	uint*	uint_params,			//3

	_clSetKernelArg( kernel,  4, sizeof( cl_mem),	&img_grad_mem,			fname);									//  __global	float2*	gradient_map,			//4		img_size * sizeof(float2)

	//Output
	_clSetKernelArg( kernel,  5, sizeof( cl_mem),	&cluster_centers_mem,	fname);									//  __global	uint4*	cluster_centers			//5		(img size / cluster_dim^2) * sizeof(uint4)

	_clEnqueueNDRangeKernel(m_queue, kernel, 1, 0, &threads_to_launch, &local_work_size, fname);
																													if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::initiate_cluster_centres( )_chk2 ."<<flush;	// Save buffers to file ###########
																														stringstream ss;
																														ss << "initiate_cluster_centres_";// << save_index ;
																														bool show 		= false;
																														bool old_tiff 	= tiff;
																														tiff 			= true;
																														float max_range	= 0;
																														cv::Mat bufImg;
																														_cl_flush_finish(m_queue, fname);
																														//void DownloadAndSave_3Channel(cl_mem buffer, std::string count, std::filesystem::path folder_tiff, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, cv::Mat *bufImg, float max_range=1, uint offset=0, bool exception_tiff=false );
																														DownloadAndSave_3Channel( 	cluster_centers_mem,	ss.str( ), paths.at( "cluster_centers_mem"),  		mm_size_bytes_C1,   mm_Image_size,   CV_32SC4, 	show, &bufImg, max_range, 0, false);
																														// NB the tiff file holds the int32 values as float32. This is okay because they fit in the mantissa.
																														// BGRA format, B=u, G=v, R=read_index, A=alpha.
																														tiff 			= old_tiff;
																													}
}


void RunCL::associate_pixels(uint layer){
	string			fname						= "RunCL::associate_pixels( )";
	int				local_verbosity_threshold	= V_RUNCL_ASSOCIATE_PIXELS;
																													if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::associate_pixels( ..)_chk0 #############################################################"<<flush;}
	cl_kernel		kernel						= associate_pixels_kernel;
	const frame		*frame_0					= &current_frames[				current_frames_idx[0] ];
	const cl_mem 	imgmem_						= frame_0->img_buf;
	uint			mm_cols						= uint_params[MM_COLS];
	uint			lookup_table_offset			= patch_lookup_table_offset[	layer];
	size_t			threads_to_launch			= patch_num_threads[			layer];
																													if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::associate_pixels( ..)_chk1"<<flush;
																														cout<<"\nlayer					="	<<	layer\
																															<<"\nmm_cols				="	<<	mm_cols	\
																															<<"\nlookup_table_offset	="	<<	lookup_table_offset\
																															<<"\nthreads_to_launch		="	<<	threads_to_launch\
																															<<"\ncurrent_frames_idx[0]	="	<<	current_frames_idx[0]\
																															<<"\nimgmem_				="	<<	imgmem_\
																															<<flush;
																													}
	//Inputs:
	_clSetKernelArg( kernel,  0, sizeof( uint),		&superpx_params[layer].cluster_layer_offset,fname);				//	__private	uint	cluster_layer_offset,	//0
	_clSetKernelArg( kernel,  1, sizeof( uint),		&superpx_params[layer].num_clusters,		fname);				//	__private	uint	num_clusters,			//1
	_clSetKernelArg( kernel,  2, sizeof( uint),		&lookup_table_offset,						fname);				//	__private	uint	lookup_table_offset,	//1
	_clSetKernelArg( kernel,  3, sizeof( uint),		&superpx_params[layer].cluster_dim,			fname);				//	__private	uint	cluster_dim,			//3
	_clSetKernelArg( kernel,  4, sizeof( uint),		&superpx_params[layer].cols_of_clusters,	fname);				//	__private	uint	cols_of_clusters,		//4
	_clSetKernelArg( kernel,  5, sizeof( uint),		&mm_cols,									fname);				//	__private	uint	mm_cols,				//5

	_clSetKernelArg( kernel,  6, sizeof( cl_mem),	&patch_lookup_table_buf,					fname);				//	__global	uint4*	lookup_table,			//6
	_clSetKernelArg( kernel,  7, sizeof( cl_mem),	&imgmem_,									fname);				//	__global	float4*	img,					//7		img_size * sizeof(float4)
	_clSetKernelArg( kernel,  8, sizeof( cl_mem),	&cluster_centers_mem,						fname);				//	__global	uint4*	cluster_centers,		//8		(img size / cluster_dim^2) * sizeof(uint4)

	//Output
	_clSetKernelArg( kernel,  9, sizeof( cl_mem),	&cluster_map_mem,							fname);				//	__global	float*	cluster_map				//9		img_size * sizeof(float)   densely packed for one layer.  Need a layer offset.

	_clEnqueueNDRangeKernel(m_queue, kernel, 1, 0, &threads_to_launch, &local_work_size, fname);
																													if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::associate_pixels( )_chk2 ."<<flush;	// Save buffers to file ###########
																														stringstream ss;
																														ss << "associate_pixels_";// << save_index ;
																														bool show 		= false;
																														bool old_tiff 	= tiff;
																														tiff 			= true;
																														float max_range	= 1;
																														cv::Mat bufImg;
																														_cl_flush_finish(m_queue, fname);
																														//void RunCL::DownloadAndSave(cl_mem buffer, std::string count, std::filesystem::path folder_tiff, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, float max_range )
																														DownloadAndSave( 	cluster_map_mem,	ss.str( ), paths.at( "cluster_map_memm"),  		mm_size_bytes_C1,   mm_Image_size,   CV_32FC1, 	show , max_range );
																														tiff 			= old_tiff;
																													}
}


void RunCL::superpixel_depth(uint layer ){
	string		fname						= "RunCL::superpixel_depth(..)";
	int			local_verbosity_threshold	= V_RUNCL_SUPERPIXEL_DEPTH;												if(verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::superpixel_depth(..)_chk0"<<
																																",   layer = "<<layer<<flush; }
	cl_kernel	kernel						= superpixel_depth_kernel;

	const uint	num_clusters				= superpx_params[layer].num_clusters;
	const uint	vol_size					= num_clusters * NUM_DEPTH_STEPS;
	const uint 	cluster_cols				= superpx_params[layer].cols_of_clusters;

	float		inv_depth_step				= fp32_params[INV_DEPTH_STEP];//fp32_params[MAX_INV_DEPTH] / ((float)uint_params[COSTVOL_LAYERS]  );	cout<<"\ninv_depth_step="<<inv_depth_step<<" =  fp32_params[MAX_INV_DEPTH] "<<fp32_params[MAX_INV_DEPTH]<<" / ((float)uint_params[COSTVOL_LAYERS] "<<uint_params[COSTVOL_LAYERS]<<flush;  // NUM_DEPTH_STEPS

	size_t		local_work_size_			= block_size;									// Could be changed to an integer multiple, i.e. use "RunCL::local_work_size", beware numbers not multiples of out_block_size.
	size_t		threads_to_launch			= num_clusters;
	threads_to_launch						= lowest_multiple( threads_to_launch, local_work_size );

	// private
	_clSetKernelArg( kernel, 0, sizeof(uint), 						&num_clusters,											fname);		// __private	uint num_clusters	//0
	_clSetKernelArg( kernel, 1, sizeof(uint), 						&vol_size,												fname);		// __private	uint vol_size,		//1
	_clSetKernelArg( kernel, 2, sizeof(uint), 						&cluster_cols,											fname);		// __private	uint cluster_cols
	_clSetKernelArg( kernel, 3, sizeof(float), 						&inv_depth_step,										fname);		// __private	uint inv_depth_step

	// global
	_clSetKernelArg( kernel, 4, sizeof(cl_mem), 					&cluster_costvol_mem,									fname);		// __global	float* cluster_costvol	//2
	_clSetKernelArg( kernel, 5, sizeof(cl_mem), 					&cluster_depth_mem,										fname);		// __global	float* superpixel_depth	//2

	_clEnqueueNDRangeKernel(										// NB depth iteration is internal to the kernel within the layer.  Regularization and propagation to next layer requires further kernels.
		m_queue,				//cl_command_queue _queue,
		kernel,					//cl_kernel        kernel,
		1,						//cl_uint          work_dim,
		0,						//const size_t *   global_work_offset,
		&threads_to_launch,		//const size_t *   global_work_size,
		&local_work_size_,		//const size_t *   local_work_size,
		fname					//string           fname
	);
																													if( verbosity>local_verbosity_threshold) {
																																	cout<<"\n\nRunCL::superpixel_depth() chk_1   "<<flush;
																																	bool show = false;
																																	stringstream ss;
																																	ss << "superpixel_depth" ;
		uint 		offset				=	9 * (superpix_vol_bytes/10);
		float		max_range			=	1.0f;
		cv::Size	superpix_img_size	=	cv::Size( superpx_params[layer].cols_of_clusters , superpx_params[layer].rows_of_clusters );

																													cout<<"\nRunCL::superpixel_depth() chk_2,  superpix_img_bytes="<<superpix_img_bytes<<",  superpix_img_size="<<superpix_img_size<<flush;

//		DownloadAndSave_2Channel_volume( cluster_costvol_mem,	ss.str( ), paths.at( "cluster_costvol_mem"), superpix_vol_bytes,  superpix_img_size,	CV_32FC2, show, max_range,	costVolLayers, offset);		// superpx_params[layer].num_clusters*2*sizeof(float)
		DownloadAndSave_2Channel(			cluster_depth_mem,	ss.str( ), paths.at( "cluster_depth_mem"),   superpix_img_bytes,  superpix_img_size,	CV_32FC2, show, max_range,	0 );

// void RunCL::DownloadAndSave_2Channel(cl_mem buffer, 	std::string count, 	std::filesystem::path folder_tiff, 	size_t image_size_bytes, 	cv::Size size_mat, 	int type_mat, bool show, float max_range, uint offset )
																													}
}


/*
void RunCL::check_superpixel_continuity(){ // May not be desired. Discontinuous superpixels represent partially occluded background, or other repeated areas.
	string		fname						= "RunCL::check_superpixel_continuity( )";
	int			local_verbosity_threshold	= V_RUNCL_CHECK_SUPERPIXEL_CONTINUITY;
	cl_kernel	kernel						= check_superpixel_continuity_kernel;
																																if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::initiate_cluster_centres( ..)_chk1 #############################################################"<<flush;
																																}




}


void RunCL::update_cluster_centres_pvt(){ // Not needed if not iterating superpixel refinement.
	string		fname						= "RunCL::update_cluster_centres_pvt( )";
	int			local_verbosity_threshold	= V_RUNCL_UPDATE_CLUSTER_CENTRES;
	cl_kernel	kernel						= update_cluster_centres_pvt_kernel;
																																if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::initiate_cluster_centres( ..)_chk1 #############################################################"<<flush;
																																}





}


void RunCL::superpixel_depth_est(){




}
*/
