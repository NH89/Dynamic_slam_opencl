#include "RunCL.hpp"



/* Matx44f  RunCL::update_pose_bufs_cur_frames( Matx44f new_pose_0to1 ){								// To be called after tracking and before depth and other optimisations.
// 		string		fname = "RunCL::update_pose_bufs_cur_frames(..)";
// 		int			local_verbosity_threshold = V_RUNCL_UPDATE_POSE_BUFS_CUR_FRAMES;
//
// 		float		cur_frames_k2k[num_current_frames*16];
// 		cl_float4	cur_frames_st3[num_current_frames];
//
// 		Matx44f		inv_pose_gt		=  getInvPose(current_frames[	current_frames_idx[0]	].pose_gt);
//
// 		//Matx44f		pose;
// 		//float16arry_To_Matx44f( &current_frames[	current_frames_idx[0]	].pose[0],  pose );
// 																																			if(verbosity>local_verbosity_threshold) {
// 																																				cout<<"\n\nRunCL::update_pose_bufs_cur_frames(..) chk_1\n"<<flush;
// 																																				//PRINT_MATX44F( pose , "pose" );
// 																																				for( int frame=0;	frame<num_current_frames;	frame++){
// 																																					cout<<"\nframe = "<< frame;
// 																																					cout<<"\ncurrent_frames_idx["	<<frame<<"] = "<<current_frames_idx[frame]<<flush;
// 																																					cout<<"\ndataset_frame_num = "	<<current_frames[ current_frames_idx[frame] ].dataset_frame_num<<flush;
// 																																					cout<<"\nframe_count = "		<<current_frames[ current_frames_idx[frame] ].frame_count<<flush;
// 																																					cout<<"\nframe_data_index = "	<<current_frames[ current_frames_idx[frame] ].frame_data_index<<flush;
// 																																					PRINT_MATX44F( current_frames[ current_frames_idx[frame] ].pose_from_0,  );
// 																																					PRINT_MATX44F( current_frames[ current_frames_idx[frame] ].pose_gt,  );
// 																																				}
// 																																			}
//
//
// 		for( int frame=0;	frame<num_current_frames;	frame++){ // TODO need if(use_gt_pose)
//
// 			Matx44f										gt_pose_0_to_this_frame		= current_frames[ current_frames_idx[frame] ].pose_gt * inv_pose_gt;
//
// 			//gt_pose_0_to_this_frame ; //current_frames[ current_frames_idx[frame] ].pose_0_to_this_frame  *  pose;
// 			Matx44f 									new_frame_pose				= current_frames[ current_frames_idx[frame] ].pose_from_0  *  new_pose_0to1;
//
// 			cout<<"\n\nRunCL::update_pose_bufs_cur_frames(..) chk_2   frame="<<frame<<flush;
// 			PRINT_MATX44F(gt_pose_0_to_this_frame,);
// 			PRINT_MATX44F(new_frame_pose,)
//
// 			current_frames[ current_frames_idx[frame] ].pose_from_0 		= new_frame_pose;
//
// 			Matx44f 									new_k2k						= current_frames[ current_frames_idx[frame] ].K	* new_frame_pose * current_frames[ current_frames_idx[frame] ].inv_K;
//
// 			Matx44f_To_float16arry(						new_k2k,					&cur_frames_k2k[frame*16] );
// 			cur_frames_st3[								frame]						= {{	new_frame_pose(0,3),	new_frame_pose(1,3),	new_frame_pose(2,3),	0.0f	}};
//
// 																																			// if(verbosity>local_verbosity_threshold) {
// 																																			// 	cout<<"\n\nRunCL::update_pose_bufs_cur_frames(..) chk_3\n"<<flush;
// 																																			// 	cout<<"\nframe = "<<frame<<flush;
// 																																			// 	cout<<"\ncurrent_frames_idx["<<frame<<"] = "<<current_frames_idx[frame]<<flush;
// 																																			// 	//PRINT_MATX44F( gt_pose_0_to_this_frame, );
// 																																			// 	PRINT_MATX44F( new_frame_pose, );
// 																																			// 	PRINT_MATX44F( current_frames[ current_frames_idx[frame] ].K, );
// 																																			// 	PRINT_MATX44F( current_frames[ current_frames_idx[frame] ].inv_K, );
// 																																			// 	PRINT_MATX44F( (current_frames[ current_frames_idx[frame] ].K * current_frames[ current_frames_idx[frame] ].inv_K) , );
//                                    //
// 																																			// 	PRINT_MATX44F( new_k2k, );
// 																																			// 	PRINT_FLOAT_16( (cur_frames_k2k + (frame*16) ) , );
// 																																			// }
// 		}
//
// 		_clEnqueueWriteBuffer(
// 			uload_queue,							//cl_command_queue 	command_queue,
// 			cur_frames_k2kbuf,						//cl_mem 			buffer,
// 			CL_FALSE,								//cl_bool 			blocking_write,
// 			0,										//size_t 			offset,
// 			num_current_frames*16*sizeof(float),	//size_t 			size,
// 			cur_frames_k2k,							//const void* 		ptr,
// 			fname									//string 			fname
// 		);
//
// 		_clEnqueueWriteBuffer(
// 			uload_queue,							//cl_command_queue 	command_queue,
// 			cur_frames_st3buf,						//cl_mem 			buffer,
// 			CL_FALSE,								//cl_bool 			blocking_write,
// 			0,										//size_t 			offset,
// 			num_current_frames*4*sizeof(float),		//size_t 			size,
// 			cur_frames_st3,							//const void* 		ptr,
// 			fname									//string 			fname
// 		);
// 																																			if(verbosity>local_verbosity_threshold) {
// 																																				cout<<"\n\n///RunCL::update_pose_bufs_cur_frames(..) finished //////////////////\n"<<flush;
// 																																			}
// 		return pose;
// }
*/

void RunCL::initialize_current_frame( int idx ){									// Used at statem initialization.
	current_frames[idx].dataset_frame_num		= -1;
	current_frames[idx].frame_count				= 0;
	current_frames[idx].frame_data_index		= idx;								// Initialized with cl_mem buffers in order. This will change with update_current_frames_idx().
	////////////////////////////////////////////////////////////
	current_frames[idx].img_buf					= imgmem[			idx];
	current_frames[idx].depth_buf				= depth_mem[		idx];
	current_frames[idx].r_vel_buf				= velmap[			idx];			// velocity _relative_ to the camera.

	current_frames[idx].pose_buf				= pose_buf[			idx];
	current_frames[idx].k2k_buf					= k2kbuf[			idx];			// used from new frame for depth, cam & lens calib, relative to current depth map
	current_frames[idx].k2k_buf_to_0			= cur_frames_k2kbuf[idx];			// used in tracking new frame, relative to depth map of ??
	////////////////////////////////////////////////////////////
	current_frames[idx].pose_gt					= Matx44f::eye();
	current_frames[idx].pose_from_start			= Matx44f::eye();
	current_frames[idx].pose_from_0 			= Matx44f::eye();
	current_frames[idx].pose_to_0 				= Matx44f::eye();

	current_frames[idx].K			 			= Matx44f::eye();
	current_frames[idx].inv_K		 			= Matx44f::eye();
	current_frames[idx].k2k_from_0	 			= Matx44f::eye();
	current_frames[idx].k2k_to_0	 			= Matx44f::eye();
	///////////////////////////////////////////////////////////
	for(uint layer=0; layer<max_mipmap_layers; layer++){
		current_frames[idx].inv_SE3_Hessian[			layer]	= Matx66f::eye();
		current_frames[idx].inv_camera_matrix_Hessian[	layer]	= Matx55d::eye();
		current_frames[idx].inv_lens_distortion_Hessian[layer]	= Matx55d::eye();
	}
}

void RunCL::initialize_current_frames(){
	for (uint idx = 0; idx < num_current_frames; idx++){
		initialize_current_frame( idx);
	}
}

void RunCL::initialize_new_frame(){													// Used to set 1st estimate of new frame.
	int idx		= current_frames_idx[0];
	int idx2	= current_frames_idx[1];
																					cout<<"\n\nRunCL::initialize_new_frame()"<<flush;
																					cout<<"\ncurrent_frames_idx[0] = "<<current_frames_idx[0]<<flush;
																					cout<<"\ncurrent_frames_idx[1] = "<<current_frames_idx[1]<<endl<<flush;
	current_frames[ idx ].dataset_frame_num		= current_frames[ idx2 ].dataset_frame_num + 1;
	current_frames[ idx ].frame_count			= current_frames[ idx2 ].frame_count;
	current_frames[ idx ].frame_data_index		= current_frames[ idx2 ].frame_data_index;
	////////////////////////////////////////////////////////////
	/*	GPU buffers to be Initialized by the kernels that use them.
	//current_frames[idx].img_buf			= imgmem[idx];			// needs to load new frame - done where ?
	//current_frames[idx].depth_buf			= depth_mem[idx];		// TODO needs to sample & interpolate previous
	//current_frames[idx].r_vel_buf			= velmap[idx];			// TODO needs to sample & interpolate previous 		// velocity _relative_ to the camera.
	*/
	////////////////////////////////////////////////////////////
	current_frames[ idx ].pose_gt				= Matx44f::eye();
	current_frames[ idx ].pose_from_start		= current_frames[ idx2 ].pose_from_start * current_frames[ idx2 ].pose_to_0;
	current_frames[ idx ].pose_from_0			= current_frames[ idx2 ].pose_from_0;
	current_frames[ idx ].pose_to_0				= current_frames[ idx2 ].pose_to_0;

	current_frames[ idx ].K						= current_frames[ idx2 ].K;
	current_frames[ idx ].inv_K					= current_frames[ idx2 ].inv_K;
	current_frames[ idx ].k2k_from_0			= current_frames[ idx2 ].k2k_from_0;
	current_frames[ idx ].k2k_to_0				= current_frames[ idx2 ].k2k_to_0;
	////////////////////////////////////////////////////////////
	for(uint layer=0; layer<max_mipmap_layers; layer++){
		current_frames[idx].inv_SE3_Hessian[			layer]	= Matx66f::eye();
		current_frames[idx].inv_camera_matrix_Hessian[	layer]	= Matx55d::eye();
		current_frames[idx].inv_lens_distortion_Hessian[layer]	= Matx55d::eye();
	}
}

void RunCL::update_current_frames_idx(){											// Call immediately _before_ loading new frame.
		uint mod_16		= fmod(frame_count,16); // NB fastest way would be a nested if sequence, using bit shift to test the last bit.
		uint mod_8  	= fmod(mod_16,8);
		uint mod_4		= fmod(mod_8,4);
		uint mod_2		= fmod(mod_4,2);
																					cout<<"\n\nRunCL::update_current_frames_idx()"<<flush;
																					for(int idx = 0; idx< num_current_frames; idx++){
																						cout<<"\nidx="<<idx<<",  current_frames_idx["<<idx<<"] = "<<current_frames_idx[idx]<<flush;
																					}
		if ( (mod_8==0) || (frame_count<num_current_frames) ){						//cout<<"\n(mod_16==0) "; NB in first 4 frames keeps every frame until the array is full.
			new_current_frames_idx[0] = current_frames_idx[4];
			new_current_frames_idx[1] = current_frames_idx[0];
			new_current_frames_idx[2] = current_frames_idx[1];
			new_current_frames_idx[3] = current_frames_idx[2];
			new_current_frames_idx[4] = current_frames_idx[3];
		}else if (mod_4==0){														//cout<<"\n(mod_8==0) ";
			new_current_frames_idx[0] = current_frames_idx[3];
			new_current_frames_idx[1] = current_frames_idx[0];
			new_current_frames_idx[2] = current_frames_idx[1];
			new_current_frames_idx[3] = current_frames_idx[2];
			new_current_frames_idx[4] = current_frames_idx[4];
		}else if (mod_2==0){														//cout<<"\n(mod_4==0) ";
			new_current_frames_idx[0] = current_frames_idx[2];
			new_current_frames_idx[1] = current_frames_idx[0];
			new_current_frames_idx[2] = current_frames_idx[1];
			new_current_frames_idx[3] = current_frames_idx[3];
			new_current_frames_idx[4] = current_frames_idx[4];
		}else{																		//cout<<"\n(mod_2==0) ";
			new_current_frames_idx[0] = current_frames_idx[1];
			new_current_frames_idx[1] = current_frames_idx[0];
			new_current_frames_idx[2] = current_frames_idx[2];
			new_current_frames_idx[3] = current_frames_idx[3];
			new_current_frames_idx[4] = current_frames_idx[4];
		}
		swap( new_current_frames_idx, current_frames_idx);

		initialize_new_frame();	// Re-initializes the new frame.
		return;
	};

/*void RunCL::test_update_current_frames_idx(uint num_iter){
	// 	cout << "\n\n RunCL::test_update_current_frames_idx(uint "<<num_iter<<")";
	// 	for (uint iter=0; iter<=num_iter; iter++){
	// 		dataset_frame_num++;
	// 		frame_count++;
	// 		update_current_frames_idx();
	// 		current_frames[current_frames_idx[0]].frame_data_index = dataset_frame_num;			//iter;
	// 																																	cout<<"\niter="<<iter;
	// 																																	for (uint idx=0; idx<5; idx++){
	// 																																		cout<<"\t\t current_frames_idx["<<idx<<"]="<<current_frames_idx[idx]
	// 																																		<<", frame="<< current_frames[current_frames_idx[idx]].frame_data_index<<",";
	// 																																	}
	// 																																	cout << flush;
	// 	}
	// }
*/
