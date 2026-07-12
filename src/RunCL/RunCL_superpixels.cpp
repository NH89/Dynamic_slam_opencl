
#include "RunCL.hpp"

void RunCL::initiate_cluster_centres(uint layer){
	string		fname						= "RunCL::initiate_cluster_centres( )";
	int			local_verbosity_threshold	= V_RUNCL_INITIATE_CLUSTER_CETRES;
	cl_kernel	kernel						= initiate_cluster_centres_kernel;
																													if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::initiate_cluster_centres( ..)_chk1 #############################################################"<<flush;
																													}
	const uint	cluster_dim					= 16;

	//Inputs:
	_clSetKernelArg( kernel,  0, sizeof( uint),		&layer,					fname);									//  __private	uint	layer,					//0
	_clSetKernelArg( kernel,  1, sizeof( uint),		&cluster_dim,			fname);									//	__private	uint	cluster_dim,			//1

	_clSetKernelArg( kernel,  2, sizeof( cl_mem),	&mipmap_params,			fname);									//  __constant	uint8*	mipmap_params,			//2
	_clSetKernelArg( kernel,  3, sizeof( cl_mem),	&uint_params,			fname);									//  __constant	uint*	uint_params,			//3

	_clSetKernelArg( kernel,  4, sizeof( cl_mem),	&img_grad_mem,			fname);									//  __global	float2*	gradient_map,			//4		img_size * sizeof(float2)

	//Output
	_clSetKernelArg( kernel,  5, sizeof( cl_mem),	&cluster_centers_mem,	fname);									//  __global	uint4*	cluster_centers			//5		(img size / cluster_dim^2) * sizeof(uint4)

	res 	= clEnqueueNDRangeKernel(m_queue,		kernel, 1, 0, &threads_to_launch, &local_work_size_, 0, NULL, &ev);
																	if (res    != CL_SUCCESS)	{ cout << "\nres = " << checkerror(res) <<"\n"<<flush; exit_(res);}
	status	= clFlush(m_queue);										if (status != CL_SUCCESS)	{ cout << "\nRunCL::patch_img_gradients( ),  clFlush(m_queue) status  = "<<status<<" "<<checkerror(status) <<"\n"<<flush; exit_(status);}
	status	= clWaitForEvents (1, &ev);								if (status != CL_SUCCESS)	{ cout << "\nRunCL::patch_img_gradients( ),  clWaitForEventsh(1, &ev) =" <<status<<" "<<checkerror(status) <<"\n"<<flush; exit_(status);}


																													if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::compute_patch_lookup_table( )_chk6 ."<<flush;	// Save buffers to file ###########
																														stringstream ss;
																														ss << "compute_patch_lookup_table_";// << save_index ;
																														bool show 		= false;
																														bool old_tiff 	= tiff;
																														tiff 			= true;
																														float max_range	= 0;
																														cv::Mat bufImg;
																														_cl_flush_finish(m_queue, fname);
																														//void DownloadAndSave_3Channel(cl_mem buffer, std::string count, std::filesystem::path folder_tiff, size_t image_size_bytes, cv::Size size_mat, int type_mat, bool show, cv::Mat *bufImg, float max_range=1, uint offset=0, bool exception_tiff=false );
																														DownloadAndSave_3Channel( 	cluster_centers_mem,	ss.str( ), paths.at( "cluster_centers_mem"),  		mm_size_bytes_C4,   mm_Image_size,   CV_32UC4, 	show, &bufImg, max_range, 0, false);
																														// NB the tiff file holds the int32 values as float32. This is okay because they fit in the mantissa.
																														// BGRA format, B=u, G=v, R=read_index, A=alpha.
																														tiff 			= old_tiff;
																													}



}


void RunCL::associate_pixels(){
	string		fname						= "RunCL::associate_pixels( )";
	int			local_verbosity_threshold	= V_RUNCL_ASSOCIATE_PIXELS;
	cl_kernel	kernel						= associate_pixels_kernel;
																																if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::initiate_cluster_centres( ..)_chk1 #############################################################"<<flush;
																																}




    //Inputs:
	_clSetKernelArg( kernel,  0, sizeof( uint), &layer,				fname);									//	__private	uint	cluster_layer_offset,	//0
	_clSetKernelArg( kernel,  1, sizeof( uint), &layer,				fname);									//	__private	uint	num_clusters,			//1
	_clSetKernelArg( kernel,  2, sizeof( uint), &layer,				fname);									//	__private	uint	lookup_table_offset,	//1
	_clSetKernelArg( kernel,  3, sizeof( uint), &layer,				fname);									//	__private	uint	block_size,				//2
	_clSetKernelArg( kernel,  4, sizeof( uint), &layer,				fname);									//	__private	uint	cluster_dim,			//3
	_clSetKernelArg( kernel,  5, sizeof( uint), &layer,				fname);									//	__private	uint	cols_of_clusters,		//4
	_clSetKernelArg( kernel,  6, sizeof( uint), &layer,				fname);									//	__private	uint	mm_cols,				//5

	_clSetKernelArg( kernel,  7, sizeof( cl_mem), &layer,				fname);									//	__global	uint4*	lookup_table,			//6
	_clSetKernelArg( kernel,  8, sizeof( cl_mem), &layer,				fname);									//	__global	float4*	img,					//7		img_size * sizeof(float4)
	_clSetKernelArg( kernel,  9, sizeof( cl_mem), &layer,				fname);									//	__global	uint4*	cluster_centers,		//8		(img size / cluster_dim^2) * sizeof(uint4)

	//Output
	_clSetKernelArg( kernel,  10, sizeof( cl_mem), &layer,			fname);									//	__global	float4*	cluster_map				//9		img_size * sizeof(float4)   densely packed for one layer.  Need a layer offset.


}



void RunCL::check_superpixel_continuity(){
	string		fname						= "RunCL::check_superpixel_continuity( )";
	int			local_verbosity_threshold	= V_RUNCL_CHECK_SUPERPIXEL_CONTINUITY;
	cl_kernel	kernel						= check_superpixel_continuity_kernel;
																																if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::initiate_cluster_centres( ..)_chk1 #############################################################"<<flush;
																																}




}


void RunCL::update_cluster_centres_pvt(){
	string		fname						= "RunCL::update_cluster_centres_pvt( )";
	int			local_verbosity_threshold	= V_RUNCL_UPDATE_CLUSTER_CENTRES;
	cl_kernel	kernel						= update_cluster_centres_pvt_kernel;
																																if( verbosity>local_verbosity_threshold) {cout<<"\n\nRunCL::initiate_cluster_centres( ..)_chk1 #############################################################"<<flush;
																																}





}
