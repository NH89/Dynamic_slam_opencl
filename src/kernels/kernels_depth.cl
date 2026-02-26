#include "kernels__macros.h"
#include "kernels.h"

__kernel void update_depth(							// To be launched with 1 thread per col for 32x32 patches, and an integer multiple of 32 threads.
													// Needs 16 elements of local mem per 32x32 patch, to pass data between threads in recursive square reduction.
													// Needs 32 elem array of private mem per thread.
	//Inputs:
	__private	const uint	frame_count,			//0
	__private	const float	reduction,				//1		//	i.e. 2^layer		= base_cols/read_cols_;		//  NB these __private args couldbe a single __constant uint* buffer, uploaded at the start of the loop. //
	__private	const uint	lookup_table_offset,	//2															//  Likewise could list the order of img_ and vel_ buffers with a __constant uint* buffer				 //
	__private	const uint	out_block_size,			//3

	__private	const uint	read_offset_,			//4					= mipmap_params_[MiM_READ_OFFSET];
	__private	const uint	stop_offset,			//5					= layer_offset + (read_rows_ -1) * mm_cols + read_cols_	;	// bottom right corner of source image layer
	__private	const uint	layer_pixels,			//6					= mipmap_params_[MiM_PIXELS];
	__private	const uint	read_cols_,				//7					= mipmap_params_[MiM_READ_COLS];
	__private	const uint	read_rows_,				//8					= mipmap_params_[MiM_READ_ROWS];

	__private	const uint	mm_cols,				//9					= uint_params[MM_COLS];
	__private	const uint	mm_rows,				//10				= uint_params[MM_ROWS];
	__private	const uint	mm_pixels,				//11				= uint_params[MM_PIXELS];
																	//layer_offset/mm_cols;
	__private	const uint	ST3_offset,				//12				= ST3_offset3.s0;	//SE3_out_step_1 * (num_SE3_DoF + 1);// + layer_offset; __private	uint3		ST3_offset3,			//4
	__private	const uint	ST3_u_step,				//13				= ST3_offset3.s1;	// step between elements of the Hessian matrix
	__private	const uint	ST3_v_step,				//14				= ST3_offset3.s2;

	__constant	float16*	inv_k2k,				//15		// transforms for 4 past frames,  k2k_buf
	__constant	float4*		st3,					//16		// array of pose transforms to the set previous frames
	__constant 	float4*		lookup_table,			//17		// should ideally be a constant.
	__constant 	float4*		SE3_grad_map,			//18		// _cur_frame

	__global	float4*		img_cur,				//19		// multiple past frames. NB retain frames at powers of 2, and vary starting power plus num franes.
	__global	float4*		img_past_0,				//20
	__global	float4*		img_past_1,				//21
	__global	float4*		img_past_2,				//22
	__global	float4*		img_past_3,				//23

	__global	float*		depth_map,				//24	// current frame depth, now stored as inv_depth

	__global	float4*		vel_cur,				//25	// multiple past frames.
	__global	float4*		vel_past_0,				//26	// TO DO, relative velocity not used yet. Will use it to modify depth map with timestep for past frames.
	__global	float4*		vel_past_1,				//27
	__global	float4*		vel_past_2,				//28
	__global	float4*		vel_past_3,				//29

	//outputs
	__global	float2*		Rho_,					//30	// { sum rho^2 ,  count of valid pixels used } Writen to dense patches.
	__local		float2*		local_rho,				//31	// float2 local_rho[ local_work_size/2 ]  hence sizeof( float)*local_work_size.

	__global	float2*		inv_depth_incr,			//32
	__local		float*		local_depth_incr,		//33
	__local		float2*		local_J_inv_d			//34
	)
{


	__global float4*	img_past[num_past_frames]		= { img_past_0, img_past_1, img_past_2, img_past_3 };
	__global float4*	vel_past[num_past_frames]		= { vel_past_0, vel_past_1, vel_past_2, vel_past_3 };

	const	uint	max_frames							= min(frame_count-1, num_past_frames);

	const	uint	global_id_uint						= get_global_id(0);
	const	uint	lid									= get_local_id(0);
	const	uint	group_id							= get_group_id(0);
	const	uint	local_size							= get_local_size(0);

	if(global_id_uint==0){
		printf("\n__kernel void update_depth(..) st3=[0]=(%f,	%f,	%f,	%f),  \ninv_k2k[0]=\n(%f,	%f,	%f,	%f) \n(%f,	%f,	%f,	%f) \n(%f,	%f,	%f,	%f) \n(%f,	%f,	%f,	%f)",\
				st3[0].s0,      st3[0].s1,      st3[0].s2,      st3[0].s3,\
				inv_k2k[0].s0,  inv_k2k[0].s1,  inv_k2k[0].s2,  inv_k2k[0].s3,\
				inv_k2k[0].s4,  inv_k2k[0].s5,  inv_k2k[0].s6,  inv_k2k[0].s7,\
				inv_k2k[0].s8,  inv_k2k[0].s9,  inv_k2k[0].sA,  inv_k2k[0].sB,\
				inv_k2k[0].sC,  inv_k2k[0].sD,  inv_k2k[0].sE,  inv_k2k[0].sF\
			);
	}

	const	float4	lookup_ref							= lookup_table[global_id_uint + lookup_table_offset];
			uint	read_index							= floor(lookup_ref.z);
	const	uint	u									= lookup_ref.x;								// read_column
	uint	v											= lookup_ref.y;								// read_row, NB _not_ constant

	uint	write_index									= u/out_block_size	+ (v/out_block_size)*mm_cols	+ ST3_offset;
	uint	thread_lidi_offset							= (lid / out_block_size) * (block_size	/ out_block_size);

	float	inv_depth_incr_arr[	block_size]				= {0.0f};
	float4	Jacobian_pvt_arr[	block_size][6]			= {{zero_f4}};								// pvt variable for values in this column.
	float4	Hessian_pvt_arr[	block_size][6][6]		= {{{zero_f4}}};

	float2	rho_pvt_arr[		block_size]				= {zero_f2};								// pvt variable for values in this column.
	float4	rho_pvt_flt4								= zero_f4;
	float2	rho_pvt_flt2								= zero_f2;

	float2 	J_inv_d[			block_size]				= {0};
	float	J_inv_d_pvt									= 0;

	float4	img_cur_pvt[		block_size]				= {zero_f4};								// pvt variable for values in this column.
	float4	old_px										= zero_f4;

	bool	intersection								= false;
	bool	print_ 										= false;
	local_rho[					lid]					= zero_f2;
	local_J_inv_d[				lid]					= zero_f2;

	if( (lid	%	out_block_size) ==0 ){
		for(int i=0; i<(block_size/out_block_size); i++){ local_depth_incr[	thread_lidi_offset + i]				= 0.0f; }
	}
	if(  u==(read_cols_/2) && v==(read_rows_/2) /*global_id_uint==0*/){ printf("\n__kernel void update_depth(..) frame_count=%u,  max_frames=%u,  reduction=%f,  read_index=%u, write_index=%u  ", \
		frame_count, max_frames, reduction, read_index,  write_index ); }

	barrier(CLK_LOCAL_MEM_FENCE );
	////////////////////////////////////////////////////////////////////////////
	uint depth_iter_per_layer	 = 	3;
	for (uint iter=0; iter<depth_iter_per_layer ; iter++){
		for (uint row_in_block=0; (row_in_block<block_size)&&(read_index<=stop_offset&&read_index>0); row_in_block++, v++,  read_index +=mm_cols){		// stop offset prevents bottom row patches from overrunning the bottom of the image layer. // NB readindex may be 0 if not in range according to lookup table.
			if(  u==(read_cols_/2) && v==(read_rows_/2) /*global_id_uint==0*/){
				printf("\n\n__kernel void update_depth(..) iter=%u, row_in_block=%u, read_index=%u, \n inv_k2k[0]=\n(%f	,%f,	%f,	%f) \n(%f,	%f,	%f,	%f) \n(%f,	%f,	%f,	%f) \n(%f,	%f,	%f,	%f)",\
														iter,		row_in_block, read_index, \
					inv_k2k[0].s0,  	inv_k2k[0].s1,  	inv_k2k[0].s2,  	inv_k2k[0].s3,\
					inv_k2k[0].s4,  	inv_k2k[0].s5,  	inv_k2k[0].s6,  	inv_k2k[0].s7,\
					inv_k2k[0].s8,  	inv_k2k[0].s9,  	inv_k2k[0].sA,  	inv_k2k[0].sB,\
					inv_k2k[0].sC,  	inv_k2k[0].sD,  	inv_k2k[0].sE,  	inv_k2k[0].sF\
				);
			}
																																																	// step through rows of the patch, ////////
			uint	offset_2								= thread_lidi_offset		+ (row_in_block	/	out_block_size);
			float	depth_incr								= local_depth_incr[	offset_2];
			img_cur_pvt[		row_in_block]				= img_cur[		read_index];
			float				inv_depth 					= depth_map[	read_index];	//1.0f; //

			for (uint past_frame_idx=0; past_frame_idx < max_frames; past_frame_idx++){																										// step though past frames //////
				float				u2f,	v2f;																																					// current frame

				px_k2k( inv_k2k[past_frame_idx],  reduction,  v,  u,  inv_depth, &u2f,  &v2f, print_ );																								// Where to sample the past image frame //////

				if(  u==(read_cols_/2) && v==(read_rows_/2) /*global_id_uint==0*/){
					printf("\n__kernel void update_depth(..) past_frame_idx=%u, iter=%u, row_in_block=%u, read_index=%u, inv_depth=%f, u=%u, v=%u, u2f=%f,  v2f=%f,  read_cols_/2=%u,  read_rows_/2=%u ",\
															 past_frame_idx, 	iter, 	 row_in_block, 	  read_index, 	 inv_depth,    u,    v,  	u2f,  	v2f,	 read_cols_/2,		read_rows_/2  );
				}

				const uint margin							= 4;
				intersection 								=	(u>margin)		&& (u<=read_cols_-margin)		&& (v>margin)		&& (v<=read_rows_-margin)	&& \
																(u2f>margin)	&& (u2f<=read_cols_-margin)		&& (v2f>margin)		&& (v2f<=read_rows_-margin)	&& (global_id_uint<=layer_pixels);	// if images overlap

				if( u==(read_cols_/2) && v==(read_rows_/2) ){printf("\n__kernel void update_depth(..) intersection=%d, ",intersection);}

				if (intersection){
					rho_pvt_flt4							= zero_f4;
					old_px									= bilinear_flt4( img_past[past_frame_idx],  u2f,  v2f,  mm_cols,  read_offset_ );
					rho_pvt_flt4							= (img_cur_pvt[row_in_block] - old_px) ;																								// Photometric error rho ///////
					rho_pvt_flt4.w							= 1.0f;																																	// rho.w holds pixel count.

					// Gradient of pixel value wrt ST3, taking account of current depth map //////
					J_inv_d_pvt								=  st3[	past_frame_idx].x	* SE3_grad_map[ 	read_index + (0 * mm_pixels) ].x;			// Here for value channel only. Could weight the chroma and cos_hue, sins_hue channels.
					J_inv_d_pvt								+= st3[ past_frame_idx].y	* SE3_grad_map[ 	read_index + (1 * mm_pixels) ].x;
					J_inv_d_pvt								+= st3[ past_frame_idx].z	* SE3_grad_map[ 	read_index + (2 * mm_pixels) ].x;

					J_inv_d[		row_in_block].x			+= rho_pvt_flt4.x			* J_inv_d_pvt;													// Here for value channel only. Could weight the chroma and cos_hue, sins_hue channels.
					J_inv_d[		row_in_block].y			+= J_inv_d_pvt				* J_inv_d_pvt;

					rho_pvt_flt2.x							=  rho_pvt_flt4.x;																														// Sum Rho
					rho_pvt_flt2.y							=  rho_pvt_flt4.x			* rho_pvt_flt4.x;																							// Sum Rho_squared
					rho_pvt_arr[row_in_block]				+= rho_pvt_flt2;																														// save to pvt mem for this column
				}
				barrier( CLK_GLOBAL_MEM_FENCE );
			}
		}

// 		// Sum-reduce image, /////////////  Save intermediate size ST3 patches for depth map updates, and maximally reduced SE3 patches for pose updates. Second reduce_patch_Rho(..) kernel required for SE3 from lareger image pyramid layers, before update_k2k(..) kernel.
// 		uint past_frame_idx =0; // TO DO remove and restore long outer loop.
// 		uint step;
// 		for ( step=1; step<block_size; step *=2){																																					// for each step size, (multiples of 2)
// 			for (uint block_row=0; block_row<block_size ; block_row += step){																														// step through rows in column
// 																							rho_pvt_arr[		block_row ]			+=rho_pvt_arr[		block_row + step ];							// sum pair of values in col,
// 																							J_inv_d[			block_row ]			+=J_inv_d[			block_row + step ];
// 				if( !(fmod((float)lid,(step*2))==0) &&  (fmod((float)lid,step)==0)    ){																											// selects 2nd column, sends data
// 																							local_rho[			lid-step ]			= rho_pvt_arr[		block_row];
// 																							local_J_inv_d[		lid-step ]			= J_inv_d[			block_row];
// 				}
// 				barrier(CLK_LOCAL_MEM_FENCE );																																						// Using barrier as a semaphore, for local mem messages between threads. This minimizes local_mem req, while allowing 2 patch sizes in output, full & ST3 map at out_block_size.
//
// 				if( (fmod((float)lid,(step*2))==0)  ){																																				// selects 1st column, adds data. Sum of patch now held in top left element of patch.
// 																							rho_pvt_arr[		block_row]			+= local_rho[		lid ];
// 																							J_inv_d[			block_row]			+= local_J_inv_d[	lid ];
// 				}
// 				barrier(CLK_LOCAL_MEM_FENCE );
// 			}
// 			// Save intermediate size ST3 patches for depth map updates, //////////
// 			if (step==out_block_size/2){																																							// save ST3 map at out_block_size, to use for updating depth_map and rel_vel_map
// 				uint frame_offset 		= write_index + past_frame_idx * 100 + 25 ;			// NB 100 works for current img size . // stacks frame ST3 maps in adjacent columns..
// 				uint write_block_row	= 0;
// 				if( fmod((float)lid,out_block_size) == 0 ){																																			// selects columns i.e. threads within the workgroup
//
// 					for (uint block_row=0; block_row < block_size ; block_row += step*2, write_block_row++){																						// per iteration results
// 																							uint offset_1 							= frame_offset		+ write_block_row*mm_cols	+ iter*mm_cols* block_size ;			// not correct iter step
// 																							Rho_[				offset_1]			= rho_pvt_arr[		block_row ];
//
// 																							uint offset_2							= thread_lidi_offset		+ (block_row	/	out_block_size);
// 																							local_depth_incr[	offset_2]			+= J_inv_d[			block_row ].x	/	J_inv_d[	block_row ].y;
//
// 						if(iter==depth_iter_per_layer-1){																																			// final results ofkernel
// 																							float2 incr								= { local_depth_incr[	offset_2],		J_inv_d[	block_row ].y  };
// 																							inv_depth_incr[ 	offset_1]			= incr;
// 						}
// 					}
// 				}
// 			}//////////////////////////////////////////////////////////////////////
// 		}
	}
}


__kernel void enlarge_layer_float(
	__private	const uint	lookup_table_read_offset,	//0
	__private	const uint	write_offset,				//1
	__private	uint		buf_width,					//2			mm_cols, i.e. width of the buffer holding the image pyramid
	__private	uint		patch_height,				//3
	__private	uint		stop_offset,				//4
	__constant 	float4*		lookup_table,				//5
	__global 	float*		img							//6
	)
{
	int global_id_u 					= (int)get_global_id(0);
	float4	lookup_ref					= lookup_table[	global_id_u + lookup_table_read_offset];
	uint read_idx						= floor(	lookup_ref.z	);
	uint	u							= lookup_ref.x;														// read_column
	uint	v							= lookup_ref.y;														// read_row
	uint write_idx						= write_offset + u*2 + (v * 2 * buf_width);

	if(global_id_u==0){ printf("\n__kernel void enlarge_layer_float(..)  lookup_table_read_offset=%u ", lookup_table_read_offset ); }

	for (int i=0; i<patch_height; i++){
		if (write_idx > stop_offset) 	return;
		float value						= img[read_idx ];
		img[ write_idx ]				= value;
		img[ write_idx +1 ]				= value;
		img[ write_idx + buf_width ]	= value;
		img[ write_idx + buf_width +1 ]	= value;

		read_idx 						+= buf_width;
		write_idx 						+= buf_width*2;
	}
}
