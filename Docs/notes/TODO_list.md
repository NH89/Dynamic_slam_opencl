int main(int argc, char *argv[])	
		// TODO write new depthmap transformation based on bin sort from fluids_v3 & Morphogenesis.

void RunCL::rho_sq(			
		// TODO set global patch size from device parameters // generally: device_work_size_multiple = patch_size * integer, eg 32, 64, 128
		// TODO precompute an array for this function. ? where to store
					
void Dynamic_slam::patch_slam(){	
		//TODO NB currently runcl.img_stats[..] only for layer"0"
		// TODO break out of layer loop if gradient nears zero....  OR change technique.  e.g. use optimim from 3rd iter.
		// TODO 1) change layers 2) use patch kernel

		// TODO Problem, need to undo previous update.				// Generate two sample steps
		// TODO also need to update relative to the other reference frames
					
					
void RunCL::_clEnqueueNDRangeKernel(	
		// TODO will become obsolete when all kernels use newest patch system.					
					
RunCL_utils.cpp		
		// TODO
		// Need to put all buffers, kernels and command queues into a set of c++ dictionaries.
		// Use loops to release them.
		// Use a local alias cl_kernel kern = kDict[..kernelname..] to set arguments etc..
		// Have name lists to instantiate  kernels and command queues.
		

kernels.h  (previously kernels_patch_slam.cl)
		TODO Declare constants at top of the device prgram file.
		
		

__kernel void compute_param_maps(		
		// TODO // Create a 'reproject' & 'img_grad_sum' kernels


__kernel void cvt_color_space_linear(		
		// TODO shift "/M_PI_F" to CPU data saving ?
		
		
__kernel void sum_image_variance(
		float4 variance = powr( (img[read_index]-img_stats[layer*2 + IMG_MEAN]), 2);
		// TODO why does this cause NaNs ?  // img_stats[8*4*2]	= {0};	// 8 layers, 4 channels,
		
		
void RunCL::initialize_fp32_params(){	
		// TODO remove most, ie DTAM pararms		### done. could reduce the array & buffer, or reuse for other params. 
		
		
void RunCL::initialize_RunCL(cv::Mat baseImage_){
		costVolLayers 		=( 1 + obj["layers"].asUInt() ); // TODO  2;    ### need to search for & remove redundant variables
		
		
void RunCL::initialize_RunCL(cv::Mat baseImage_){		
		patch_img_gradients_set_params();		
		//TODO set in .conf file,  uint out_block_sizefor ST3_hessian		// will need a runcl.set_patch_kernels_params() function
		
		
void RunCL::set_mimpmap_offsets(){
		// TODO compute required reduction and blur depending on img size


void RunCL::set_mimpmap_offsets(){
		// ## set array of arrays for workgoup offsets, for patch kernels on mipmaps ########################################################### 
		TODO Replace with fixed arrays capable of 10K images


void RunCL::mipmap_call_kernel(cl_kernel kernel_to_call, cl_command_queue queue_to_call, uint start, uint stop, bool layers_sequential, const size_t local_work_size){
		// TODO execute layers in asynchronous parallel. i.e. relax clWaitForEvents.


RunCL::~RunCL(){  
		// TODO  ? Replace individual buffer clearance with the large array method from Morphogenesis &  fluids_v3 ? OR a C++ vector ?
		

void RunCL::exit_(int res)   
		// TODO convert all uses to exit_(res); Will call RunCL::~RunCL() automatically.
		
		
void RunCL::precomp_param_maps ( float SE3_k2k[  max_mipmap_layers*num_SE3_DoF*16  ]){ 
		// float mid_depth 	= ( fp32_params[MAX_INV_DEPTH] + fp32_params[MIN_INV_DEPTH])/2.0;
		// TODO fix : depthmap not used as a kernel arg. NB want to match scale of depth range, but ? parallax may vary.


void RunCL::createFolders(){
		/* TODO update list of buffers
		
		
void RunCL::DownloadAndSave_3Channel_volume(
		// writeToResultsMat(&bufImg , iter , row_of_images+i );
		// Add patch from bufImg to resultsMat  
		TODO this is a bad idea, tangled code.
		// DownloadAndSave_3Channel_volume(..) is called for several differnt buffers. !
		// Onlly valid when called by RunCL::tracking_result
		
		
RunCL_DownloadAndSave.cpp
		/////////////////////////////////////////////////////// 
		# TODO New way: generic RunCL::DownloadAndSave_buffer(...) , and specialized calling functions.
		
		
int Dynamic_slam::nextFrame() {		
		// TODO replace with patch_slam and multi-frame depth+motion+accel maps,  together with vel, accel, jolt of camera,   and later reflectance & illum etc...
		
		
void Dynamic_slam::getFrameData_vec(){  // Dynamic_slam::initialize_camera_vec(),  Dynmaic_slam::nextFrame()
		frame_data.back().frame_data_GT			= datum; 
		// TODO entirely remove Dynamic_slam::frame_data.
		

void Dynamic_slam::set_artif_pose_error(){	
		/* ..
		frame_data.back().frame_data_GT			= datum;						
		// TODO if(use GT),  but move it out to Dynamic_slam::next_frame()
		.. */
		
		
/* old artificial_pose_error_vec()
void Dynamic_slam::artificial_pose_error_vec(){	
		// TODO if(GT_available==true){}else{}
		...
		frame_data.back().frame_data.pose	= poseStep	*	frame_data.back().frame_data.pose;	
		// TODO which side to multiply from ?
		
		
Dynamic_slam.hpp	
		#define Rx	0	
		// TODO  correct to match Python code and convention ( st(3), so(3) )
		
		//    std::vector<keyframe_datum>  keyframe_data;		
		// TODO remove keyframes ?
		
		cv::Mat image, depth_GT, cameraMatrix;				
		// TODO should these be Matx ?   , projection   NB cameraMatrix => K_GT
		
		void generate_SE3_k2k_vec( float _SE3_k2k[  max_mipmap_layers* num_SE3_DoF *16  ] );		
		// TODO chk all maths vs Python version. Also ensure compatibility with new code.
		
		
void Dynamic_slam::estimateSLAM(){
		float		num_pixels	=	runcl.se3_rho_result.SE3_incr_arry[1];
		// TODO move numpixels to SE3_incr.w   & reduce SE3_incr_map_mem from float8 tro float4


__kernel void Rho_sq(
		__global	float4*		vel_past_0,				//16	
		// TODO, relative velocity not used yet. Will use it to modify depth map with timestep for past frames.	    
		
		// PATCH KERNEL 
		//  TODO need to transfer computation of Huber Norm weighting, Jacobian and Hessian here,
		// because Hessian must include weights and therefore be updated if weights change.
		
		/*..
		// TODO, relative velocity not used yet. Will use it to modify depth map with timestep for past frames.
		..*/
		
		uint past_frame_idx =0; 
		// TODO remove and restore long outer loop.
		
		
__kernel void reduce_patch_Rho(	
		// Sum pixels in the row. 
		// TODO summing weights seems wrong.
		
		
__kernel void update_k2k(	
		// TODO need new kernel, for global synchronization between workgroups.  // Only one workgroup needed
		
		/* ..
		// TODO step size needs to decrease with layer.
		.. */
		
		
		__local float local_pose[SE3_elems];
		// update the stored pose. TODO tuck this into 2nd stage of void update_k2_kdev_fn(..)


RunCL.hpp
		cl_kernel			rho_sq_kernel, reduce_patch_Rho_kernel, update_k2k_kernel;	
		// TODO declare, create, release kernel in Run_cl.h etc.
		
	
__kernel void compute_patch_lookup_table(					
		// computed once at start of program	
		// TODO when is it possible to roll the layers together ?  i.e. when local mem is not used.
		
	
__kernel void  patch_img_grad(		
		Hessian_pinv_pvt_arr[row_in_block][i][j]= Jacobian[i] * Jacobian[j];	// Gauss-Newton approx H = J.transpose * J  
		// TODO compute and sum lower triangle only.

		uint past_frame_idx =0; 
		// TODO remove and restore long outer loop.

		local_Hessian[		lid-step + (i*6 + j)*local_size]		= Hessian_pinv_pvt_arr[	block_row ][i][j];	//+ se3_dim*block_size ];  
		TODO correct size and indexing of local_Hessiasn
	
 		barrier(CLK_GLOBAL_MEM_FENCE );			
 		// TODO is this needed?
 		
 		barrier(CLK_GLOBAL_MEM_FENCE );		
 		// TODO is this needed?
 		
 	
__kernel void  patch_hessian_reduce(
		SE3_Hessian_map[idx]					=	H_pvt_arr[0] ;											
		//TODO will need atomic fn for larger images.
		
		SE3_Hessian_map[idx]					=	J_pvt_arr[0] ;											
		//TODO will need atomic fn for larger images.		
		//	Write result to the first 36 pixels of SE3_Hessian_map, directly above this layer of hessian pyramid,  because this will be fastest to read to CPU.


void GN_Hessian_pseudo_inv_3x3( float J[3], float H_pinv[9]
		float denom = J[0]*J[0] + J[1]*J[1] + J[2]*J[2];  				
		//TODO check is it sum of sqares of J or sum of squares of diagonal of H ?


void RunCL::compute_patch_lookup_table(){
		cout<<"\nthreads_to_launch= 		"<<threads_to_launch		<<"		= blocks_required 				* local_work_size_[0]";		
		// TODO precompute an array for this function. ? where to store
		
		
void  RunCL::patch_hessian_reduce(uint layer){	
		for(int row=0; row<1; row++){	
		// per_pixel division currently done in kernel, 
		TODO which is better ?
		
		for(int row=0; row<num_SE3_DoF; row++){																							
		// per_pixel division currently done in kernel, 
		TODO which is better ?
		
		
		//  Eigen pseudo-inverse
		Eigen::MatrixXd GN_H(6,6);																					
		// TODO replace Eigen with a kernel for 6x6 matrix pseudo-inverse or inverse.
		
		
void RunCL::cvt_color_space(){ 
		// TODO NB it would be faster to find the mean from the smallest layer, BUT only if there are no bugs e.g. the black bottom edge.
		// Variance however must be computed for each layer, because blurring may reduce contrast &=> variance.


void RunCL::sum_image_variance(){
		// TODO ? create a class for data, holding buffer, CPU data, stats about the data object, functions for write, read, save, display, & set_kernel_arg ?
		_clEnqueueNDRangeKernel(m_queue,  kernel, 1, 0, &global_work_size, &local_work_size, fname); 											
		// run img_variance _kernel  aka img_variance(..) ##### 
		TODO which CommandQueue to use ? What events to check ?
		
		uint layer = 0; 
		// TODO convert to mimpap version.
		

void RunCL::mipmap_linear(cl_mem image_buf, std::string folder){
		mipmap_call_kernel( mipmap_float4_kernel, m_queue, true );   
		// TODO Start at first reduction, rehash __kernel void mipmap_linear_flt(..) and call only the num threads required. NB currently uses 4x as many threads as needed.
		
	
void RunCL::load_GT_depth(cv::Mat GT_depth, bool invert){
		float max_range_ = 0.0f;	
		// 0.0f => (temp_mat / maxVal) * 256*256 for .png; 
		TODO move this to conf.json
		
		
void RunCL::mipmap_depthmap(cl_mem depthmap_){
		mipmap_call_kernel( mipmap_float_kernel, m_queue, true);
		// TODO Start at first reduction, rehash __kernel void mipmap_linear_flt(..) and call only the num threads required. NB currently uses 4x as many threads as needed.
		
		
		
