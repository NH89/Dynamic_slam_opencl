#include "kernels__macros.h"
#include "kernels.h"

													// This version creates a cost vol of multiple depths for each pixel.
													// float2 local_rho holds (pixel count, rho^2).
													// Computes optimum depth from thee depths around best in cost vol.
__kernel void update_depth_2(							// To be launched with 1 thread per col for 32x32 patches, and an integer multiple of 32 threads.
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

	__private	const uint	write_offset,			//10				= patch_depthmap_offset_[layer];
	__private	const uint	dm_win_cols,			//11
	__private	const uint	dm_data_rows,			//12
	__private	const uint	dm_data_stop,			//13

	__private	const float inv_depth_step,			//11

	__global	float16*	inv_k2k_1,				//12		// transforms for 4 past frames,  k2k_buf
	__global	float16*	inv_k2k_2,				//12		// transforms for 4 past frames,  k2k_buf
	__global	float16*	inv_k2k_3,				//12		// transforms for 4 past frames,  k2k_buf
	__global	float16*	inv_k2k_4,				//12		// transforms for 4 past frames,  k2k_buf

	__constant	uint4*		lookup_table,			//13		// should ideally be a constant.

	__global	float4*		img_cur,				//14		// multiple past frames. NB retain frames at powers of 2, and vary starting power plus num franes.
	__global	float4*		img_past_1,				//15
	__global	float4*		img_past_2,				//16
	__global	float4*		img_past_3,				//17
	__global	float4*		img_past_4,				//18

//	__global	float*		depth_map,				//19	// current frame depth, now stored as inv_depth

	__global	float4*		vel_cur,				//20	// multiple past frames.
	__global	float4*		vel_past_1,				//21	// TO DO, relative velocity not used yet. Will use it to modify depth map with timestep for past frames.
	__global	float4*		vel_past_2,				//22
	__global	float4*		vel_past_3,				//23
	__global	float4*		vel_past_4,				//24

	//outputs
	__global	float2*		Rho_,					//25	// { sum rho^2 ,  count of valid pixels used } Writen to dense patches.
	__local		float2*		local_rho,				//26	// float2 local_rho[ num_depth_steps * local_work_size/2 ]  hence sizeof(cl_float2)*local_mem_size*num_depth_steps,

	__global	float2*		inv_depth_incr			//27
	)
{
	const		float16		inv_k2k[num_current_frames]		= { inv_k2k_1[0],	inv_k2k_1[0],	inv_k2k_2[0],	inv_k2k_3[0],	inv_k2k_4[0] };

	__global	float4*		img_past[num_current_frames]	= { img_cur,		img_past_1,		img_past_2,		img_past_3, 	img_past_4 };
	__global	float4*		vel_past[num_current_frames]	= { vel_cur,		vel_past_1,		vel_past_2,		vel_past_3, 	vel_past_4 };


	const	uint	max_frames							= min(frame_count-1, num_current_frames);
	const	uint	global_id_uint						= get_global_id(0);
	const	uint	lid									= get_local_id(0);
	const	uint	group_id							= get_group_id(0);
	const	uint	local_size							= get_local_size(0);
																																					//st3=[%d]=(%f,	%f,	%f,	%f),     past_frame_idx,
																																					//	st3[past_frame_idx].s0,      st3[past_frame_idx].s1,      st3[past_frame_idx].s2,      st3[past_frame_idx].s3,
																																					if(global_id_uint==0){
																																						printf("\n__kernel void update_depth_2(..) frame_count=%d,  max_frames = %d", frame_count, max_frames );
																																						for (uint		past_frame_idx=1; past_frame_idx <= max_frames; past_frame_idx++){
																																							printf("\n__kernel void update_depth_2(..)   \ninv_k2k[%d]=\n(%f,	%f,	%f,	%f) \n(%f,	%f,	%f,	%f) \n(%f,	%f,	%f,	%f) \n(%f,	%f,	%f,	%f)",\
																																								past_frame_idx,\
																																								inv_k2k[past_frame_idx].s0,  inv_k2k[past_frame_idx].s1,  inv_k2k[past_frame_idx].s2,  inv_k2k[past_frame_idx].s3,\
																																								inv_k2k[past_frame_idx].s4,  inv_k2k[past_frame_idx].s5,  inv_k2k[past_frame_idx].s6,  inv_k2k[past_frame_idx].s7,\
																																								inv_k2k[past_frame_idx].s8,  inv_k2k[past_frame_idx].s9,  inv_k2k[past_frame_idx].sA,  inv_k2k[past_frame_idx].sB,\
																																								inv_k2k[past_frame_idx].sC,  inv_k2k[past_frame_idx].sD,  inv_k2k[past_frame_idx].sE,  inv_k2k[past_frame_idx].sF\
																																							);
																																						}
																																					}
																																					barrier(CLK_GLOBAL_MEM_FENCE );
	const	uint4	lookup_ref							= lookup_table[global_id_uint + lookup_table_offset];
	const	uint	read_index_start					= lookup_ref.z;
			uint	read_index							= read_index_start;
	const	uint	u									= lookup_ref.x;								// read_column
	const	uint	v_start								= lookup_ref.y;								// read_row, NB _not_ constant
			uint	v									= v_start;

	const	uint	out_cols							= dm_win_cols;		//(read_cols_/out_block_size) +2; //TODO  need values from dm_params  // + 6; // + margin ?
	const	uint	out_rows							= dm_data_rows;		//read_rows_/out_block_size;

	const	uint	write_layer_pixels					= (out_rows + 1) * out_cols;																				// (layer_pixels / (out_block_size * out_block_size) ) + out_cols;
			uint	write_index							= write_offset +  (u/out_block_size)	+ (v/out_block_size)*out_cols;										// NB write_offset allow each layer of the depthmap to be saved, without overwriting.

	uint	thread_lidi_offset							= (lid / out_block_size) * (block_size	/ out_block_size);

	float	inv_depth_incr_arr[	block_size]				= {0.0f};

	float2	rho_pvt_arr[ NUM_DEPTH_STEPS*block_size]	= {zero_f2};								// pvt variable for values in this column.
	float4	rho_pvt_flt4								= zero_f4;
	float2	rho_sq_pvt_flt2								= zero_f2;

	float4	img_cur_pvt[		block_size]				= {zero_f4};								// pvt variable for values in this column.
	float4	old_px										=  zero_f4;

	bool	intersection								= false;
	bool	print_ 										= false;
	local_rho[					lid]					= zero_f2;

	if( lookup_ref.w != global_id_uint){	printf("\n__kernel void update_depth_2(..) chk_1 lookup_ref.w %u != global_id_uint %u", lookup_ref.w, global_id_uint);	// NB return cols tha are outside img_cur, BUT only after initializing local mem.
											return;
	}
 /*
																																						//if( group_id==1 / *lid==0* / ){printf("\n__kernel void update_depth(..) chk 0 ####### global_id_uint=%d,  lid=%d,  group_id=%d,  ", global_id_uint, lid, group_id );}
// 																	if(global_id_uint==0){ printf("\n__kernel void update_depth(..) write_layer_pixels %d	= (layer_pixels %d / (out_block_size %d ^2) ) + out_cols %d,  write_index	%d		= u %d	/out_block_size %d	+ (v %d /out_block_size %d	) *out_cols %d  ",\
// 																																	write_layer_pixels,		   layer_pixels,	  out_block_size,			out_cols,     write_index			, u		,out_block_size		,  v	,out_block_size		,  out_cols);
// 																	}
	//if( (lid	%	out_block_size) ==0 ){
	//	for(int i=0; i<(block_size/out_block_size); i++){	local_depth_incr[	thread_lidi_offset + i]				= 0.0f; }
	//}
// 																																						if( u==(read_cols_/2) && v==(read_rows_/2) / *global_id_uint==0* /){
// 																																							printf("\n__kernel void update_depth(..) chk 0.5,  frame_count=%u,  max_frames=%u,  reduction=%f,  read_index=%u, write_index=%u,  global_id_uint=%d", \
// 																																							 													frame_count, 	  max_frames, 	  reduction, 	 read_index,  	write_index,  global_id_uint );
// 																																						}
*/
	barrier(CLK_LOCAL_MEM_FENCE );
	////////////////////////////////////////////////////////////////////////////
//	uint depth_iter_per_layer						= 	1;/*max_frames;*/ //3;
	//float 	inv_depth_step							=       ((float)MAX_INV_DEPTH) / ((float)NUM_DEPTH_STEPS);   // ### MAX_INV_DEPTH macro is an array index, not a value !

//	for (uint iter=0; iter<depth_iter_per_layer;  iter++,  write_index +=write_layer_pixels  ){
		for (int i=0; i<block_size; i++){
			inv_depth_incr_arr[		i]				= 0.0f;
			rho_pvt_arr[			i]				= zero_f2;										// pvt variable for values in this column.
			img_cur_pvt[			i]				= zero_f4;										// pvt variable for values in this column.
		}
		read_index									= read_index_start;
		v											= v_start;
		rho_pvt_flt4								= zero_f4;
		rho_sq_pvt_flt2								= zero_f2;
		old_px										= zero_f4;

		intersection								= false;
		print_ 										= false;
		local_rho[					lid]			= zero_f2;

		for (uint row_in_block=0; (row_in_block<block_size)&&(read_index<=stop_offset&&read_index>0); row_in_block++, v++,  read_index +=mm_cols){		// step through rows of the patch, ////////
																																						// stop offset prevents bottom row patches from overrunning the bottom of the image layer.
																																						// NB readindex may be 0 if not in range according to lookup table.
/*
// 																																						if(  u==(read_cols_/2) && v==(read_rows_/2) ){ // global_id_uint==0
// 																															printf("\n\n__kernel void update_depth(..) chk 1,  iter=%u, row_in_block=%u, read_index=%u, \n inv_k2k[0]=\n(%f	,%f,	%f,	%f) \n(%f,	%f,	%f,	%f) \n(%f,	%f,	%f,	%f) \n(%f,	%f,	%f,	%f)\n",\
// 																																																		iter,		row_in_block, read_index, \
// 																																								inv_k2k[0].s0,  	inv_k2k[0].s1,  	inv_k2k[0].s2,  	inv_k2k[0].s3,\
// 																																								inv_k2k[0].s4,  	inv_k2k[0].s5,  	inv_k2k[0].s6,  	inv_k2k[0].s7,\
// 																																								inv_k2k[0].s8,  	inv_k2k[0].s9,  	inv_k2k[0].sA,  	inv_k2k[0].sB,\
// 																																								inv_k2k[0].sC,  	inv_k2k[0].sD,  	inv_k2k[0].sE,  	inv_k2k[0].sF\
// 																																							);
// 																																						}
*/
			uint	offset_2						= thread_lidi_offset	+ (row_in_block	/	out_block_size);
					img_cur_pvt[	row_in_block]	= img_cur[				read_index];


			for (uint 	past_frame_idx=1; past_frame_idx <= max_frames; past_frame_idx++){																// step through past frames //////  /*iter+2*/

				float	 inv_depth		 = inv_depth_step;	//0.0f;
				for (int inv_depth_layer = 0;    inv_depth_layer<NUM_DEPTH_STEPS; inv_depth_layer++ ){							// step through depth layers

					inv_depth	+= inv_depth_step;

// TODO (1) build cost vol for patch, (2) then sum each layer of cost vol for the patch, (3) select the best depth layer, (4) compute optimum from 3 neighbouring layers
//		(5) start with frame 1, then repeat for each layer, narrowing the search range.

		// consider  (a) 2D movement including rel vel & accel.  (b) regularization  by (i) neighbours, (ii) edges, (iii) confidence

		// Is simple rho a bad choice, do I need covariance ?  Do I need more colour channels ? or a way to vary weighting between them ?

					float		u2f,	v2f;																												// current frame
					px_k2k(		inv_k2k[past_frame_idx],  reduction,  v,  u,  inv_depth,  &u2f,  &v2f, print_ );											// Where to sample the past image frame //////
/*
// 																							if( global_id_uint ==10 && row_in_block==0 ){		//  / *lid* / / *group_id * /  / *u==(read_cols_/2) && v==(read_rows_/2)* / / *global_id_uint==0* /
// 																							printf("\n__kernel void update_depth_2(..) chk_2,  reduction=%f,  past_frame_idx=%u,   group_id=%d,  row_in_block=%u, read_index=%u, inv_depth=%f,    u=%u, v=%u, u2f=%f,  v2f=%f,  read_cols_=%u,  read_rows_=%u, global_id_uint=%d, inv_depth_step= %f",\
// 																																			   reduction,     past_frame_idx,      group_id,     row_in_block,    read_index,    inv_depth,       u,    v,    u2f,     v2f,     read_cols_,     read_rows_ ,   global_id_uint,    inv_depth_step );
// 																							}
*/
					const uint margin							= 0;
					intersection 								=	(u>margin)		&& (u<=read_cols_-margin)		&& (v>margin)		&& (v<=read_rows_-margin)	&& \
																	(u2f>margin)	&& (u2f<=read_cols_-margin)		&& (v2f>margin)		&& (v2f<=read_rows_-margin)	&& (global_id_uint<=layer_pixels);	// if images overlap

																																						//if(  group_id==1 /*u==(read_cols_/2) && v==(read_rows_/2)*/ ){printf("\n__kernel void update_depth(..) chk 3,  intersection=%d, global_id_uint=%d,  lid=%d,  group_id=%d,  ", intersection, global_id_uint, lid, group_id );}
					if (intersection){

						old_px									= bilinear_flt4( img_past[past_frame_idx],  u2f,  v2f,  mm_cols,  read_offset_ );
						rho_pvt_flt4							= (img_cur_pvt[ row_in_block ] - old_px) ;													// Photometric error rho ///////
						rho_pvt_flt4.w							= 1.0f;																						// rho.w holds pixel count.
/*
// 						// Gradient of pixel value wrt ST3, cancelling tyhe effect of current depth map //////												// NB could use a variable blend of HSV, in place of just Value.
// 						J_inv_d_pvt								=  st3[	past_frame_idx].x	* SE3_grad_map[ 	read_index + (3 * mm_pixels) ].x;			// Here for value channel only. Could weight the chroma and cos_hue, sins_hue channels.
// 						J_inv_d_pvt								+= st3[ past_frame_idx].y	* SE3_grad_map[ 	read_index + (4 * mm_pixels) ].x;			// SE3_grad_map float4 {HSV,alpha} for each SE3 DoF,  given the existing depth map.
// 						J_inv_d_pvt								+= st3[ past_frame_idx].z	* SE3_grad_map[ 	read_index + (5 * mm_pixels) ].x;			// SE3_grad_map = (gx*SE3_map.x + gy*SE3_map.y) * inv_fepth
// 						J_inv_d_pvt								/= inv_depth;																				// Cancel the effect of existing inv_depth.
//
// 						J_inv_d[		row_in_block].x			+= rho_pvt_flt4.x			* J_inv_d_pvt;													// Here for value channel only. Could weight the chroma and cos_hue, sins_hue channels.
// 						J_inv_d[		row_in_block].y			+= J_inv_d_pvt				* J_inv_d_pvt;
*/
						rho_sq_pvt_flt2.x						=  rho_pvt_flt4.w;																			// pixel count
						rho_sq_pvt_flt2.y						=  pown(rho_pvt_flt4.x, 2) +  pown(rho_pvt_flt4.y, 2) +  pown(rho_pvt_flt4.z, 2) ;	//	* rho_pvt_flt4.x;												// Sum hsv Rho_squared

						rho_pvt_arr[ row_in_block*NUM_DEPTH_STEPS  + inv_depth_layer]		+= rho_sq_pvt_flt2;												// save to pvt mem for this column & depth layer
					}
/*
// 																																						if(  group_id==0 ){		// / *u==(read_cols_/2) && v==(read_rows_/2)* /
// 																																						printf("\n__kernel void update_depth(..) chk 2.2 ####### global_id_uint=%d,  lid=%d,  group_id=%d,  rho_pvt_arr[ row_in_block %u *NUM_DEPTH_STEPS %u + inv_depth_layer %d][ %u ] = %f, %f ",\
// 																																																				 global_id_uint, 	 lid, 	  group_id,					 row_in_block,	  NUM_DEPTH_STEPS,	   inv_depth_layer, \
// 																																							(row_in_block *NUM_DEPTH_STEPS + inv_depth_layer), \
// 																																							rho_pvt_arr[ row_in_block*NUM_DEPTH_STEPS  + inv_depth_layer].x,  rho_pvt_arr[ row_in_block*NUM_DEPTH_STEPS  + inv_depth_layer].y );  }
*/
				}
			}
		}
		barrier( CLK_GLOBAL_MEM_FENCE );
																																						//if(  group_id==1 /*lid==0*/ ){printf("\n__kernel void update_depth(..) chk 4.5 ####### global_id_uint=%d,  lid=%d,  group_id=%d,  ", global_id_uint, lid, group_id );}

		// Sum-reduce image, /////////////  Save intermediate size ST3 patches for depth map updates, and maximally reduced SE3 patches for pose updates. Second reduce_patch_Rho(..) kernel required for SE3 from lareger image pyramid layers, before update_k2k(..) kernel.
		uint past_frame_idx =0; // TO DO remove and restore long outer loop.
		uint step;
		for ( step=1; step<out_block_size; step *=2){																															// for each step size, (multiples of 2)

			for (uint block_row=0; block_row<block_size ; block_row += step){																									// step through rows in column

				float	inv_depth		= 0.0f;
				for (int inv_depth_layer = 0;  inv_depth_layer<NUM_DEPTH_STEPS;  inv_depth += inv_depth_step, inv_depth_layer++ ){												// step through depth layers
																																																										//if( group_id==1/*lid==0*/ ){printf("\n__kernel void update_depth(..) chk 5,  step=%d, block_row=%d,  global_id_uint=%d", step, block_row, global_id_uint );}
																							rho_pvt_arr[	block_row*NUM_DEPTH_STEPS + inv_depth_layer ]	+=rho_pvt_arr[	(block_row + step)*NUM_DEPTH_STEPS  + inv_depth_layer ];	// sum pair of values in col,
					if( !(fmod((float)lid,(step*2))==0) &&  (fmod((float)lid,step)==0)    ){																																			// selects 2nd column, sends data
																							local_rho[		lid-step ]										= rho_pvt_arr[		block_row*NUM_DEPTH_STEPS + inv_depth_layer];
					}
					barrier(CLK_LOCAL_MEM_FENCE );																																			// Using barrier as a semaphore, for local mem messages between threads.
																																															// This minimizes local_mem req, while allowing 2 patch sizes in output, full & ST3 map at out_block_size.

					if( (fmod((float)lid,(step*2))==0)  ){																																	// selects 1st column, adds data. Sum of patch now held in top left element of patch.
																							rho_pvt_arr[	block_row*NUM_DEPTH_STEPS + inv_depth_layer]	+= local_rho[		lid ];
/*
																																														if(  group_id==0 ){
																																															printf("\n__kernel void update_depth_2(..) chk_2.3,   global_id_uint=%d,  lid=%d,  group_id=%d, rho_pvt_arr[ block_row %u *NUM_DEPTH_STEPS %u + inv_depth_layer %d][ %u ] = %f, %f ",\
																																																													global_id_uint, 	 lid, 	  group_id,				 block_row,	   NUM_DEPTH_STEPS,	    inv_depth_layer, \
													  																																			(block_row *NUM_DEPTH_STEPS + inv_depth_layer), \
																																																rho_pvt_arr[ block_row*NUM_DEPTH_STEPS  + inv_depth_layer].x,  rho_pvt_arr[ block_row*NUM_DEPTH_STEPS  + inv_depth_layer].y \
																																															);
																																														}
*/
					}
					barrier(CLK_LOCAL_MEM_FENCE );
				}
			}
		}
		// Save ST3 patches for depth map updates, //////////

		if( fmod((float)lid,out_block_size) == 0 ){																																// selects columns i.e. threads within the workgroup

			for (uint block_row=0, write_block_row	= 0; block_row < block_size; block_row += step, write_block_row++){														// step through rows in column
																							uint	offset_1 								= write_index	+ write_block_row*out_cols;
																							if( offset_1>=dm_data_stop ) return;
																							float	inv_depth								= 0.0f;
																							float	min_rho_sq								= FLT_MAX;
																							int		opt_depth_layer[3]						= {-1};
																							float	depth_layer_rho_sq[ NUM_DEPTH_STEPS]	= {FLT_MAX};

																							float	d2ydx2									= 0.0f;
																							float	confidence								= 0.0f;
																							float	pixels_sampled							= 0.0f;
																							float	brightness								= 0.0f;
				// Find min rho depth layer
				for (int inv_depth_layer = 0;  inv_depth_layer<NUM_DEPTH_STEPS;  inv_depth += inv_depth_step, inv_depth_layer++ ){												// step through depth layers to pick the best fit layer.
/*
																							printf("\n__kernel void update_depth_2(..), chk_2.8,  rho_pvt_arr[	block_row*NUM_DEPTH_STEPS + inv_depth_layer ].y = %f   /  rho_pvt_arr[	block_row %u *NUM_DEPTH_STEPS %u + inv_depth_layer %u ].x  = %f, global_id_uint= %u , frame_count= %u ",\
																								rho_pvt_arr[	block_row*NUM_DEPTH_STEPS + inv_depth_layer ].y  ,\
																								block_row, NUM_DEPTH_STEPS , inv_depth_layer ,\
																								rho_pvt_arr[	block_row*NUM_DEPTH_STEPS + inv_depth_layer ].x , global_id_uint, frame_count );
*/
					if( rho_pvt_arr[  block_row*NUM_DEPTH_STEPS + inv_depth_layer ].x  > 0 ){	//	where .x = pixel count.
																							depth_layer_rho_sq[ inv_depth_layer]	= rho_pvt_arr[	block_row*NUM_DEPTH_STEPS + inv_depth_layer ].y  /  rho_pvt_arr[	block_row*NUM_DEPTH_STEPS + inv_depth_layer ].x ;
/*
																																			//    __kernel void update_depth_2, chk_4  global_id_uint=%u, block_row=%u, frame_count=%u,
																																		printf("\n__kernel void update_depth_2, chk_3, global_id_uint=%u, block_row=%u, frame_count=%u, inv_depth_layer=%u,	depth_layer_rho_sq[ inv_depth_layer ]=%f,  rho_pvt_arr[	block_row*NUM_DEPTH_STEPS + inv_depth_layer ].y = %f   /  rho_pvt_arr[	block_row *NUM_DEPTH_STEPS  + inv_depth_layer ].x  = %f,     ",\
																																														global_id_uint, 	block_row, 		frame_count,	 inv_depth_layer,		depth_layer_rho_sq[ inv_depth_layer ],		rho_pvt_arr[ block_row*NUM_DEPTH_STEPS + inv_depth_layer ].y  , 		rho_pvt_arr[	block_row*NUM_DEPTH_STEPS + inv_depth_layer ].x  );
*/
						if( min_rho_sq >= depth_layer_rho_sq[ inv_depth_layer] ){
																							min_rho_sq 								= depth_layer_rho_sq[	inv_depth_layer];
																							opt_depth_layer[1]						= inv_depth_layer;
						}
					}
				}

				if (opt_depth_layer[1] == 0) {
																							opt_depth_layer[0]	= 0;
																							opt_depth_layer[1]	= 1;
																							opt_depth_layer[2]	= 2;
				}else if (opt_depth_layer[1] == NUM_DEPTH_STEPS -1 ){
																							opt_depth_layer[0]	= NUM_DEPTH_STEPS -3;
																							opt_depth_layer[1]	= NUM_DEPTH_STEPS -2;
																							opt_depth_layer[2]	= NUM_DEPTH_STEPS -1;
				} else {
																							opt_depth_layer[0]	= opt_depth_layer[1] -1;
																							opt_depth_layer[2]	= opt_depth_layer[1] +1;
				}
				// Compute optimum depth for this patch
																							float prediction, optimum;
																							compute_minimum(	depth_layer_rho_sq[ opt_depth_layer[0] ],	depth_layer_rho_sq[ opt_depth_layer[1] ],	depth_layer_rho_sq[ opt_depth_layer[2] ],		opt_depth_layer[0]*inv_depth_step ,		opt_depth_layer[1]*inv_depth_step ,		opt_depth_layer[2]*inv_depth_step,	&prediction,	&optimum,	&d2ydx2,	global_id_uint, block_row );
				// Compute confidence estimate
																							brightness			= img_cur_pvt[ block_row ].x;
																							pixels_sampled		= rho_pvt_arr[ opt_depth_layer[1] ].x;
																							confidence			= d2ydx2  * pixels_sampled	*	brightness ;	//	/ prediction   causes giant values.
																							if(confidence<0.0f || prediction<0.0f){	confidence = FLT_MIN;}		// TODO could clamp  0<confidence<1
				// save depth, confidence and predicted rho.
																							Rho_[	offset_1]	= (float2) { prediction,	prediction/ depth_layer_rho_sq[ opt_depth_layer[1] ]	};	//{ 0.3f, prediction }; //
// 																				float U_;
// 																				if (global_id_uint==0){ U_ = 4000 + u;} else { U_ = u;}
																				inv_depth_incr[ 	offset_1]	= (float2) { optimum,		confidence };		// pixels_sampled		//{ 0.3f, confidence }; // (float2) { U_/4.0f, write_block_row };  //

//TODO needd areturn if offset_1 goes beyond dm_data
//# 																																			printf("\n__kernel void update_depth_2, chk_4, global_id_uint=%u, block_row=%u, frame_count=%u, lid= %u, optimum= %f, min_rho_sq= %f, opt_depth_layer[0]= %u, opt_depth_layer[1]= %u, opt_depth_layer[2]= %u, inv_depth_step=%f, depth_layer_rho_sq[ opt_depth_layer[0] ]=%f, depth_layer_rho_sq[ opt_depth_layer[1] ]=%f, depth_layer_rho_sq[ opt_depth_layer[2] ]=%f ", \
//# 																																															global_id_uint, 	block_row, 	frame_count, 	lid,	optimum, 	  min_rho_sq, 	  opt_depth_layer[0], 	   opt_depth_layer[1],		opt_depth_layer[2],  	inv_depth_step,  depth_layer_rho_sq[ opt_depth_layer[0] ],		depth_layer_rho_sq[ opt_depth_layer[1] ],	depth_layer_rho_sq[ opt_depth_layer[2] ] );
/*
				// confidence measure ?  curvature of fit ?

				// next kernels regularize, optimize,


				// test depth ?

				// complete patch

				// regularize depth

// 																							Rho_[				offset_1]			= rho_pvt_arr[			block_row ];
//
// 																							uint offset_2							= thread_lidi_offset	+ (block_row	/	out_block_size);
//
//
// 									/ * for computation * /									float pvt_depth_incr					= ;//J_inv_d[			block_row ].x	/	J_inv_d[	block_row ].y;
// 															if( isnan( pvt_depth_incr ))	{pvt_depth_incr 						= 0.0f;}
// 																							local_depth_incr[	offset_2]			-= pvt_depth_incr;	// J_inv_d[			block_row ].x	/	J_inv_d[	block_row ].y;				// __local float*  sizeof(cl_float2)*local_work_size,
//
// 																							float2 incr								= { local_depth_incr[	offset_2],	/ *  J_inv_d[	block_row ].y  * /  }; // { (float)lid, (float)group_id }; // offset_1 , iter  //
//									/ * for debugging * /										inv_depth_incr[ 	offset_1]			= incr;																			// __global float*  mm_size_bytes_C1,    depth_mem_temp,
// 																																						if(global_id_uint==0){ printf("\n__kernel void update_depth(..) offset_1 %d	= write_index %d		+ write_block_row %d  *out_cols %d   ",\
// 																																																						offset_1 ,	  write_index			, write_block_row	  ,out_cols	); }

*/
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE );
		barrier(CLK_GLOBAL_MEM_FENCE );
/*			//}//////////////////////////////////////////////////////////////////////
		//}
//	}
	// After iterations, need to apply result to depth map, and propagate to the next layer of depth map.  Host code must call kernel again for the next layer of the depth img pyramid.

	//if( group_id==1 / *lid==0* / ){printf("\n__kernel void update_depth(..) finished ####### global_id_uint=%d,  lid=%d,  group_id=%d,  ", global_id_uint, lid, group_id );}
*/
}

__kernel void regularize_depth(
	__private	const uint	lookup_table_read_offset,	//0
	__private	const uint	depth_read_offset,			//1
	__private	const uint	write_offset,				//2
	__private	const uint	depth_width,				//3

	__private	const uint	buf_width,					//4			i.e. width of the buffer holding the image pyramid
	__private	const uint	patch_height,				//5
	__private	const uint	stop_offset,				//6
	__constant	uint4*		lookup_table,				//7
	__global	float2*		img_grad,					//8			(d_val/du, d_val/dv)	### TODO Need to be reading a patch condensed version for the img pyr layer.
	__global	float2*		depth,						//9			{ optimum,	confidence = d2ydx2 * pixels_sampled * brightness / prediction; }	### TODO need to (i) write regularized version, to the right of 1st version. (ii) take account of previous layer, => close null zone gaps eg screen
														//																											(iii) possibly iterate regularization ?  (iv) scale for use in tracking.  (v) ? reduce range and refine step size of depth search.
	__private	const uint	prev_layer_dm_offset,		//10
	__private	const uint	prev_layer_depth_width		//11
	//__private	const uint	regularized_dm_offset		//10
)
{
	uint global_id_uint 				= get_global_id(0);
	uint4	lookup_ref					= lookup_table[	global_id_uint + lookup_table_read_offset];
	if( lookup_ref.w != global_id_uint){	printf("\n__kernel void regularize_depth(..) lookup_ref.w %u != global_id_uint %u", lookup_ref.w, global_id_uint);	// NB return cols tha are outside img_cur, BUT only after initializing local mem.
											return;
	}
	uint read_idx_imgrad				= lookup_ref.z; 																					// - buf_width - 1;	// up one row and left one column.
	uint	u							= lookup_ref.x;														// read_column
	uint	v							= lookup_ref.y;														// read_row

	uint pixel_offset					= u + (v * depth_width);
	uint read_idx_depth					= depth_read_offset + pixel_offset;
	uint write_idx						= write_offset		+ pixel_offset;

	uint prev_layer_u					= u/2;
	uint prev_layer_v					= v/2;
	uint prev_layer_index				= prev_layer_dm_offset + prev_layer_u + (prev_layer_v * prev_layer_depth_width);
/*
											if(global_id_uint==0){ printf("\n__kernel void regularize_depth(..) chk_0  lookup_table_read_offset=%u,	 read_idx_depth=%u, depth_read_offset=%u,	u=%u, v=%u,	depth_width=%u,	pixel_offset=%u,	write_idx=%u, 	write_offset=%u,  buf_width=%u,  ", \
																													   lookup_table_read_offset, 	 read_idx_depth,    depth_read_offset,		u,		v,	depth_width,	pixel_offset,		write_idx,		write_offset,     buf_width     ); }
*/
	barrier( CLK_GLOBAL_MEM_FENCE );

	float2 pvt_depth[3][3];				// 3x3 arrays of global data.
	uint read_idx_layer_step			= buf_width -3;
	uint depth_idx_layer_step			= depth_width -3;

	for(int j=0; j<3; j++){
		for(int i=0; i<3; i++){															// NB image must be surrounded by empty (zero valued) margin in the mipmap. This avoids the need to check image bounds.
			pvt_depth[j][i]				= depth[read_idx_depth];
			read_idx_depth++;
		}
/*
																																				if(global_id_uint==0) printf("\n__kernel void regularize_depth() chk_1.5  global_id_uint=%u,	read_idx_depth=%u,   j=%d", \
																																																	                      global_id_uint,		read_idx_depth,      j );
*/
		barrier( CLK_GLOBAL_MEM_FENCE );
		read_idx_depth					+= depth_idx_layer_step;
	}
/*
																																				if(global_id_uint==0) printf("\n__kernel void regularize_depth() chk_2  global_id_uint=%u,  depth_idx_layer_step=%u,   depth_width=%u \n\n", \
																																																					    global_id_uint,     depth_idx_layer_step,      depth_width );
*/
	barrier( CLK_GLOBAL_MEM_FENCE );
	uint arr_idx[3]						= {0,1,2};
	float lambda_sq						= pown( 0.25, 2);								// typical lambda should be 0.0 to 0.25. => lambda_sq 0.0 to 0.0625

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	for (int patch_row=0; patch_row<patch_height; patch_row++){																					// for row in patch
		float sum_depth					= 0.0f;
		float sum_weights				= 0.0f;
																						//float sum_aniostropy			= 0.0f;
		float temp_anisotropy;
		float depth_flt;
		float confidence;
		float weight;

		float2 pvt_img_grad				= img_grad[ read_idx_imgrad];					// lookup_ref.z];

		float anisotropy[3][3];
		anisotropy[0][0]				=	M_SQRT2_F * fabs(pvt_img_grad.x		+pvt_img_grad.y	);												// magnitude of scalar img grad =  sqrt(1/2) * fabs( imgrad_x + imgrad_y)   i.e cos(45)=sin(45)
		anisotropy[0][1]				=				fabs(					 pvt_img_grad.y	);
		anisotropy[0][2]				=	M_SQRT2_F * fabs(pvt_img_grad.x		-pvt_img_grad.y	);
		anisotropy[1][0]				=				fabs(pvt_img_grad.x						);
		for (int i=0; i<4; i++){
			anisotropy[i/3][i%3]		= anisotropy[i/3][i%3] / ( 1 + ( pown(anisotropy[i/3][i%3],2) / lambda_sq) );							// Perona-Malik non-linear fn.
		}
		anisotropy[1][1]				=	( anisotropy[0][0] +anisotropy[0][1] +anisotropy[0][2] +anisotropy[1][0] )/ 4.0f;
		anisotropy[1][2]				=	anisotropy[1][0];
		anisotropy[2][0]				=	anisotropy[0][2];
		anisotropy[2][1]				=	anisotropy[0][1];
		anisotropy[2][2]				=	anisotropy[0][0];

		for (int j=0; j<3; j++){
			for (int k=0; k<3; k++){
				depth_flt				= pvt_depth[ arr_idx[j] ][k].x;
				confidence				= pvt_depth[ arr_idx[j] ][k].y;
				temp_anisotropy			= anisotropy[j][k];

				weight					= clamp( (confidence / temp_anisotropy), 0.0f, FLT_MAX );					//  1.0f 												// confidence * img grad in dir of pixel.		// use mad( ,  ,  )  multiply add
																						//sum_aniostropy				+= temp_anisotropy;
				sum_weights				+= weight;
				sum_depth				+= depth_flt * weight;
/*
				if(global_id_uint==5) printf("\n__kernel void regularize_depth() chk_2.1 global_id_uint=%u,  u=%u, patch_row=%u, (j,k)=(%u,%u),  depth_flt=%f,  confidence = %f,  temp_anisotropy=%f  weight=%f", \
																						 global_id_uint,     u,    patch_row,    j, k,           depth_flt,     confidence,       temp_anisotropy,    weight);
*/
			}
		}
		// previous layer
		float2 prev_layer				= depth[prev_layer_index];
		if( prev_layer.y >0.0f && prev_layer.x>0.0f ){															// i.e. If not 1st layer to be processed.
			weight						= 0.1 ;								// NB this is regularized confidence.
			sum_weights					+= weight;
			sum_depth					+= prev_layer.x * weight;
		}
/*
		if(global_id_uint<32) printf("\n__kernel void regularize_depth() chk_2.1 global_id_uint=%u,  u=%u, patch_row=%u,  prev_layer = ( %f, %f ),     sum_depth=%f,  sum_weights=%f,  weight=%f,  pvt_img_grad=(%f, %f),          read_idx_imgrad=%u", \
																				 global_id_uint,     u,    patch_row,     prev_layer.x, prev_layer.y,  sum_depth,     sum_weights,     weight,     pvt_img_grad.x, pvt_img_grad.y, read_idx_imgrad	);

		                                                                                                                                        if(global_id_uint==0) printf("\n__kernel void regularize_depth() chk_2.1  global_id_uint=%u,  read_idx_depth=%u,  write_idx=%u,   write_idx -read_idx_depth -regluarized_dm_offset=%d", \
			                                                                                                                                                                                                              global_id_uint,     read_idx_depth,     write_idx,     ((int)write_idx -(int)read_idx_depth -(int)regularized_dm_offset) );
*/
		float regularized_depth			= sum_depth / sum_weights;
		float2 out;
		//if (global_id_uint==0){	out		= (float2){ u, patch_row }; }
		//else				  {	out		= (float2){ regularized_depth, sum_weights }; }
		depth[write_idx]				= (float2){ regularized_depth, sum_weights };		// pvt_depth[ arr_idx[1] ][1];    //  out;  //  (float2){ u, patch_row };						//depth[read_idx_depth];	//
		barrier( CLK_GLOBAL_MEM_FENCE );

		for (int j=0; j<3; j++){																											// fill next row of rolling arrays from __global buffers.
			arr_idx[j]					+=1;
			arr_idx[j]					= arr_idx[j]%3;
		}
/*
		                                                                                                                                        if(global_id_uint==0){
			                                                                                                                                    	printf("\n__kernel void regularize_depth() chk_2.2  global_id_uint=%u,  read_idx_layer_step=%u,  read_idx_depth=%u,  write_idx=%u,  regluarized_dm_offset=%u,  write_idx -read_idx_depth -regluarized_dm_offset=%d,               depth_idx_layer_step=%u,  depth_width=%u,   arr_idx[j]=%u,%u,%u",\
																                                                                                                                                        global_id_uint, 	read_idx_layer_step,     read_idx_depth,     write_idx,     regularized_dm_offset,    ((int)write_idx -(int)read_idx_depth -(int)regularized_dm_offset),  depth_idx_layer_step,     depth_width,      arr_idx[0],arr_idx[1],arr_idx[2] );
		}
*/
		for(int i=0; i<3; i++){
/*
			                                                                                                                                    if(global_id_uint==0) printf("\n__kernel void regularize_depth() chk_2.3  global_id_uint=%u,   write_idx -read_idx_depth -regluarized_dm_offset=%d",\
				                                                                                                                                                                                                          global_id_uint,     ((int)write_idx -(int)read_idx_depth -(int)regularized_dm_offset) );
*/
			pvt_depth[ arr_idx[2] ][i]	= depth[read_idx_depth];						// Load next row of raw depth data.
			barrier( CLK_GLOBAL_MEM_FENCE );
			read_idx_depth++;
		}
/*
																																				if(global_id_uint==0) printf("\n__kernel void regularize_depth() chk_3  global_id_uint=%u,  patch_row=%u,   write_idx -read_idx_depth -regluarized_dm_offset=%d \n",\
			                                                                                                                                                                                                            global_id_uint,     patch_row,     ((int)write_idx -(int)read_idx_depth -(int)regularized_dm_offset)    );
*/
		read_idx_depth					+= depth_idx_layer_step;
		read_idx_imgrad					+= buf_width;
		write_idx 						+= depth_width;
		prev_layer_index				+= prev_layer_depth_width * patch_row%2;	// i.e. increment every second loop iteration.
		if (write_idx >= stop_offset) 	return;
	}
}


__kernel void enlarge_layer_float(
	__private	const uint	lookup_table_read_offset,	//0
	__private	const uint	write_offset,				//1
	__private	uint		buf_width,					//2			mm_cols, i.e. width of the buffer holding the image pyramid
	__private	uint		patch_height,				//3
	__private	uint		stop_offset,				//4
	__constant 	uint4*		lookup_table,				//5
	__global 	float2*		img							//6
	)
{
	uint global_id_uint 				= get_global_id(0);
	uint4	lookup_ref					= lookup_table[	global_id_uint + lookup_table_read_offset];
	if( lookup_ref.w != global_id_uint){	printf("\n__kernel void enlarge_layer_float(..) lookup_ref.w %u != global_id_uint %u", lookup_ref.w, global_id_uint);	// NB return cols tha are outside img_cur, BUT only after initializing local mem.
											return;
	}
	uint read_idx						= lookup_ref.z;
	uint	u							= lookup_ref.x;														// read_column
	uint	v							= lookup_ref.y;														// read_row
	uint write_idx						= write_offset + u*2 + (v * 2 * buf_width);
/*
	if(global_id_uint==0){ printf("\n__kernel void enlarge_layer_float(..)  lookup_table_read_offset=%u ", lookup_table_read_offset ); }
*/
	for (int i=0; i<patch_height; i++){
		if (write_idx > stop_offset) 	return;
		float2 value					= img[read_idx ];
		img[ write_idx ]				= value;
		img[ write_idx +1 ]				= value;
		img[ write_idx + buf_width ]	= value;
		img[ write_idx + buf_width +1 ]	= value;

		read_idx 						+= buf_width;
		write_idx 						+= buf_width*2;
	}
}


__kernel void use_inferred_depthmap(
	__private	const uint	lookup_table_read_offset,	//0
	__private	const uint	read_offset,				//1
	__private	uint		depth_in_width,				//2

	__private	uint		write_offset,				//3
	__private	uint		depth_width_out,			//4

	__private	uint		patch_height,				//5
	__private	uint		stop_offset,				//6

	__constant 	uint4*		lookup_table,				//7

	__global 	float2*		temp_depth,					//8
	__global	float2*		depth_map					//9
	)
{
	uint global_id_uint 						= get_global_id(0);
	uint4	lookup_ref							= lookup_table[	global_id_uint + lookup_table_read_offset];
	if( lookup_ref.w != global_id_uint){		printf("\n__kernel void use_inferred_depthmap(..) lookup_ref.w %u != global_id_uint %u", lookup_ref.w, global_id_uint);	// NB return cols tha are outside img_cur, BUT only after initializing local mem.
												return;
	}
	uint	u									= lookup_ref.x;														// read_column
	uint	v									= lookup_ref.y;														// read_row
	uint pixel_offset							= u + (v * depth_in_width);
	uint read_idx								= read_offset		+ pixel_offset - depth_in_width - 1;
/*
	if ( global_id_uint==0) printf("\n__kernel void use_inferred_depthmap()  read_idx(%u)	= read_offset(%u)	+ pixel_offset(%u) - depth_in_width(%u),		lookup_table_read_offset=%u",\
																			 read_idx,		read_offset,		pixel_offset,		depth_in_width,				lookup_table_read_offset  );
*/
	uint	arr_idx[3]							= {0,1,2};
	float2	depth_in[3][3];
	__attribute__((opencl_unroll_hint))																			// depends on opencl >=2.0
	for (int i=1; i<3; i++){
		for (int j=0; j<3; j++){
			depth_in[i][j]						= temp_depth[read_idx];
			read_idx++;
		}
		read_idx 								+= depth_in_width - 3;
	}
	int		write_index_start 					= write_offset + u*4 + (v * 4 * depth_width_out);
/*
	if ( global_id_uint==0) printf("\n__kernel void use_inferred_depthmap()  write_index_start(%u)	= write_offset(%u) + u(%u)*4 + (v(%u) * 4 * depth_width_out(%u))",\
																			write_index_start,		write_offset,		u,			v,			depth_width_out);
*/
	const float weights_u[16] = {  3.0f/8.0f,  1.0f/8.0f,  3.0f/8.0f,  1.0f/8.0f,  7.0f/8.0f,  5.0f/8.0f,  7.0f/8.0f,  5.0f/8.0f,  3.0f/8.0f,  1.0f/8.0f,  3.0f/8.0f,  1.0f/8.0f,  7.0f/8.0f,  5.0f/8.0f,  7.0f/8.0f,  5.0f/8.0f  };
	const float weights_v[16] = {  3.0f/8.0f,  3.0f/8.0f,  1.0f/8.0f,  1.0f/8.0f,  3.0f/8.0f,  3.0f/8.0f,  1.0f/8.0f,  1.0f/8.0f,  7.0f/8.0f,  7.0f/8.0f,  5.0f/8.0f,  5.0f/8.0f,  7.0f/8.0f,  7.0f/8.0f,  5.0f/8.0f,  5.0f/8.0f  };
//									0			1			2			3			4			5			6			7			8			9			10			11			12			13			14			15

	for (int patch_row=0; patch_row<patch_height; patch_row++){
		int	  write_index						= write_index_start + 4*patch_row*depth_width_out;

		__attribute__((opencl_unroll_hint))
		for (int j=0; j<3; j++){																				// fill next row of rolling arrays from __global buffers.
			arr_idx[j]							+=1;
			arr_idx[j]							= arr_idx[j]%3;
		}
		__attribute__((opencl_unroll_hint))
		for(int i=0; i<3; i++){
			depth_in[ arr_idx[2] ][i]			= temp_depth[read_idx];
			read_idx++;
		}
		read_idx 								+= depth_in_width - 3;

		int m = 0;
		__attribute__((opencl_unroll_hint))																		// depends on opencl >=2.0
		for (int i=0; i<2; i++){
			for (int j=0; j<2; j++){
				for (int k=0; k<2; k++){
					for (int l=0; l<2; l++){
						float2 depth_conf		=  weights_v[m]*(  weights_u[m]*depth_in[arr_idx[i]][j]	+	(1-weights_u[m])*depth_in[arr_idx[i]][j+1] )	+	(1-weights_v[m])*(  weights_u[m]*depth_in[arr_idx[i+1]][j]	+	(1-weights_u[m])*depth_in[arr_idx[i+1]][j+1]	);
						if (depth_conf.x>1.0f		|| depth_conf.x<0.0f	|| isnan(depth_conf.x) )	depth_conf.x =0.0f;
						if (depth_conf.y>FLT_MAX	|| depth_conf.y<0.0f	|| isnan(depth_conf.y) )	depth_conf.y =0.0f;
						depth_map[write_index]	=	depth_conf;													// ### TODO (1) move this check to the regularize depth kernel. (2) apply confidence & anisotropy in enlargement.
						m++;
						write_index				+= 1;
					}
					write_index					+=   depth_width_out -2;
				}
				write_index						-= 2*depth_width_out -2;
			}
			write_index							+= 2*depth_width_out -4;
		}
/*
		if ( global_id_uint==10) {
			float2	depth_in_	=	depth_in[ arr_idx[1] ][1];
			float2	depth_out	=	depth_map[write_index];
			printf("\n__kernel void use_inferred_depthmap()  read_idx(%u),	write_index(%u),	patch_row=%d,	depth_in[ arr_idx[1] ][1]=( %f, %f ),	temp_depth[read_idx]=( %f, %f ),				depth_map[write_index]=( %f, %f )",\
															read_idx,		write_index,		patch_row,		depth_in_.x,	depth_in_.y,			temp_depth[read_idx].x, temp_depth[read_idx].y,	depth_out.x,	depth_out.y		 );
		}
*/
		if (read_idx >= stop_offset) 			return;
	}
}


__kernel void use_GT_depthmap(
	__private	const uint	lookup_table_read_offset,	//0
	__private	const uint	write_offset,				//1

	__private	uint		depth_width_out,			//2
	__private	uint		patch_height,				//3
	__private	uint		stop_offset,				//4

	__constant 	uint4*		lookup_table,				//5

	__global 	float*		GT_depth,					//6
	__global	float2*		depth_map					//7
	)
{
	uint global_id_uint 					= get_global_id(0);
	uint4	lookup_ref						= lookup_table[	global_id_uint + lookup_table_read_offset];
	if( lookup_ref.w != global_id_uint){	printf("\n__kernel void use_inferred_depthmap(..) lookup_ref.w %u != global_id_uint %u", lookup_ref.w, global_id_uint);	// NB return cols tha are outside img_cur, BUT only after initializing local mem.
											return;
	}
	uint	u								= lookup_ref.x;														// read_column
	uint	v								= lookup_ref.y;														// read_row
	int		write_index 					= write_offset + u + (v * depth_width_out);

	for (int i=0; i<patch_height; i++){
		if (write_index >= stop_offset) 	return;
		float2 depth						= {GT_depth[write_index], 1.0f};
		depth_map[write_index]				= depth;
		write_index							+= depth_width_out;
	}
}

