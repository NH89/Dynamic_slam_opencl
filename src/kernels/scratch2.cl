#include "kernels_macros.h"
#include "kernels.h"

__kernel void Rho_sq(
	__private	uint		layer,					//0
	__private	uint 		cols_per_row,			//1

	__constant	uint8*		mipmap_params,			//3
	__constant	uint*		uint_params,			//4
	//output
	__global	float2*		Rho_,					//20	// { sum rho^2 ,  count of valid pixels used } Writen to dense patches.
	__local		float2*		local_rho,				//21	// float2 local_rho[ local_work_size/2 ]  hence sizeof( float)*local_work_size.

	__global	float2*		weights_map,			//22
	__local		float2*		local_weights,			//23

	__global	float2*		SE3_incr_map_,			//24
	__local		float2*		local_SE3_incr			//25
){
	const uint block_size	= 32;					// or send as __private arg ? BUT as hardcoded "const uint" it can be used to size arrays etc.
	const uint se3_dof		= 6;
	uint  global_id_u 		= get_global_id(0);
	uint  lid 				= get_local_id(0);
	uint  group_id			= get_group_id(0);
	const uint local_size 	= get_local_size(0);

	uint8 mipmap_params_ 	= mipmap_params[layer];
	uint read_offset_ 		= mipmap_params_[MiM_READ_OFFSET];
	uint read_cols_ 		= mipmap_params_[MiM_READ_COLS];
	uint read_rows_ 		= mipmap_params_[MiM_READ_ROWS];

	uint mm_cols			= uint_params[MM_COLS];
	uint mm_pixels			= uint_params[MM_PIXELS];

	uint row_length			= cols_per_row; 													// blocks_cols * block_size;
	uint row_col			= fmod((float)global_id_u, row_length);
	uint block_row			= global_id_u / row_length;
	uint read_index			= read_offset_ + row_col + block_row*block_size*mm_cols;
	uint write_index_2 		= row_col/block_size + block_row*mm_cols; 							//fmod((float)global_id_u, blocks_cols * out_block_size)	+ (global_id_u / (uint)(blocks_cols * out_block_size)) * out_block_size;;

	float2 rho[block_size]				= {0.0f};												// pvt variable for values in this column.
	float2 weights[block_size*se3_dof]	= {0.0f};												// pvt variable for values in this column.
	float2 SE3_incr[block_size*se3_dof]	= {0.0f};												// pvt variable for values in this column.

	bool intersection;
	uint step;
	for ( step=2; step<block_size; step *=2){				// out_block_size																													// for each step size, (multiples of 2)
																																																// selects pairs of columns to sum  i.e. threads within the workgroup
		for (uint block_row=0; block_row<block_size ; block_row += step){																														// step through rows in column
			if (fmod((float)lid,step/2)==0) {																																					// selects threads separated by 1/2 step, i.e results of previous iteration of patch reduction.
																						rho[	  block_row ]							+= rho[		 block_row + step/2 ];						// sum pair of values in col,
				for (uint se3_dim=0; se3_dim<se3_dof; se3_dim++) {
																						SE3_incr[ block_row + se3_dim*block_size ]		+= SE3_incr[ block_row + step/2 + se3_dim*block_size ];
																						weights[  block_row + se3_dim*block_size ]		+= weights[  block_row + step/2 + se3_dim*block_size ];
				}
			}
			if( !(fmod((float)lid,step)==0) &&  (fmod((float)lid,step/2)==0)    ){																												// selects 2nd column, sends data
																						local_rho[		lid/step ] 						= rho[		 block_row];
				for (uint se3_dim=0; se3_dim<se3_dof; se3_dim++) {
																						local_SE3_incr[ lid/step + se3_dim*local_size ]	= SE3_incr[  block_row + se3_dim*block_size ];			// NB integer division. Hence both threads use the same index to local memory.
																						local_weights[  lid/step + se3_dim*local_size ]	= weights[   block_row + se3_dim*block_size ];
				}
			}
			barrier(CLK_LOCAL_MEM_FENCE );																																						// Using barrier as a semaphore, for local mem messages between threads.
			if( (fmod((float)lid,step)==0)  ){																																					// selects 1st column, adds data. Sum of patch now held in top left element of patch.
																						rho[block_row] 									+= local_rho[ lid/step ];
				for (uint se3_dim=0; se3_dim<se3_dof; se3_dim++) {
																						SE3_incr[  block_row + se3_dim*block_size ]		+= local_SE3_incr[ lid/step + se3_dim*local_size ];
																						weights[   block_row + se3_dim*block_size ]		+= local_weights[  lid/step + se3_dim*local_size ];
				}
			}
			barrier(CLK_LOCAL_MEM_FENCE );
		}
	}
	float2 temp2a											= { (float)/*block_row*/group_id, (float)/*block_col*/lid };
	if ( read_index < mm_pixels )	{	Rho_[read_index ] 	= temp2a;	}																														// Marks the area where the img buf is read, lines show top row of each patch.
																																																// Breaks show bondaries of patches.
	barrier(CLK_GLOBAL_MEM_FENCE );
	uint write_block_row=0;
	if( fmod((float)lid,block_size) ==0 ){		//out_block_size						// for each 32x32 block of the input image : 															// selects columns i.e. threads within the workgroup
		uint frame_offset_1 = write_index_2;	// stacks frame SE3 results vertically.
		step = block_size/2;

		for (uint block_row=0; block_row < block_size ; block_row += step, write_block_row++){
																						uint offset_2 				= frame_offset_1 + write_block_row*mm_cols;
																						Rho_[			offset_2  ]	= rho[		 block_row ];
			for (uint se3_dim=0; se3_dim<se3_dof; se3_dim++) {																																	// All 6 DoF of SE3
																						uint offset_3 				= offset_2 		+ se3_dim*( 4 + (read_rows_/block_size) )*mm_cols;
																						uint offset_4				= block_row 	+ se3_dim*block_size;
																						SE3_incr_map_[	offset_3 ]	= SE3_incr[  offset_4 ];
																						weights_map[	offset_3 ]	= weights[   offset_4 ];
			}
		}
	}
	barrier(CLK_GLOBAL_MEM_FENCE );
/////////////////////////////////////////////////////////////////////////////////////////////
	if( fmod((float)lid,block_size) ==0 ){												// for each 32x32 block of the input image :
		// const uint block_size		= 32

		row_length		= cols_per_row; 												// for this image size
		row_col			= fmod((float)global_id_u, row_length);							// which pixel column of this row
		block_row		= global_id_u / row_length;										// which row of blocks NB num threads is set to match: image rows/blocksize

		uint frame_offset_1	= row_col/block_size + block_row*mm_cols;					// pixel to which to write this block's result

		step = block_size/2;

		for (uint block_row=0; block_row < block_size ; block_row += step, write_block_row++){  // will repeat twice
			for (uint se3_dim=0; se3_dim<se3_dof; se3_dim++) {
				uint offset_3		= (frame_offset_1  + write_block_row*mm_cols) 		+ se3_dim*( 4 + (read_rows_/block_size) )*mm_cols;
			}
		}
	}
}

__kernel void update_SE3(									// call just one workgroup to sum the whole image maps from the patch kernel.
	__private	uint		cols,					//0
	__private	uint 		rows,					//1
	__private	uint 		row_offset,				//2		// index of 1st pixel of the 2nd patch, i.e. spacing between patches
	__private	uint		thread_offset,			//3		// smallest 2^n > rows * cols NB rows=3, cols=4, -> 12 ->16 for layer 1.  6*8=48 -> 64 for layer 0, where base image has 640*480 pixels. NB for larger images may need a patch approach to update_SE3, to kep each SE3 DoF within
	__private	uint		mm_cols,				//4

	__global	float2*		Rho_,					//7		// { sum rho^2 ,  count of valid pixels used } Writen to dense patches.
	__global	float2*		weights_map,			//8
	__global	float2*		SE3_incr_map_			//9
){
	const uint	SE3_DoF		= 6;
	uint   global_id_u 		= get_global_id(0);
	float  global_id_f 		= global_id_u;
	uint   lid 				= get_local_id(0);
																								// read in global data : Rho, weights, SE3_incr
																								// NB 10x8 pactch for each SE3.
																								// Read & sum pixels in column, NB img overlap pixel count
	uint	SE3				= global_id_u / thread_offset;
	bool	in_range		= fmod( global_id_f, thread_offset) < cols  &&  (SE3 < SE3_DoF);	// NB integer division.

	float2 pvt_rho 			= {0.0f,0.0f};
	float2 pvt_weights 		= {0.0f,0.0f};
	float2 pvt_incr			= {0.0f,0.0f};

	if (in_range){
		row_offset 			*=SE3;

		for(uint idx = 0; idx<rows; idx ++){
			uint idx_2		= idx * mm_cols;
			pvt_rho			+= Rho_[idx_2];														// NB only one Rho[], but 6 DoF for weights_map[] & SE3_incr_map_[]
			idx_2 			+= row_offset;
			pvt_weights		+= weights_map[idx_2];
			pvt_incr		+= SE3_incr_map_[idx_2];											// pvt variable will hold sum for column in Rho_, weights_map, SE3_incr_map_  // TODO should these be combined BEFORE bering summed ?
		}
	}
}

//////////////////////////////////////////


// for ( step=1; step<=block_size; step *=2){																																					// for each step size, (multiples of 2)
// 		for (uint block_row=0; block_row<block_size ; block_row += step){																														// step through rows in column
// 																					rho_pvt_arr[	block_row]		+= rho_pvt_arr[	block_row + step ];
//
// 			if( !(fmod((float)lid,(step*2))==0) &&  (fmod((float)lid,step)==0)	){	local_rho[		lid-step ]		= rho_pvt_arr[	block_row]	;}
// 			if(  (fmod((float)lid,(step*2))==0)									){	rho_pvt_arr[	block_row]		+= local_rho[	lid		 ]	;}
// 		}
// 	}

////////////////////////////////////////////





void test_fn(
	__private	uint		layer,					//0
	__private	uint 		cols_per_row,			//1
	__private	uint 		out_block_size,			//2

	__constant	uint8*		mipmap_params,			//3
	__constant	uint*		uint_params,			//4
	__constant	float*		fp32_params,			//5
	__constant	float16*	inv_k2k,				//6		// transforms for 4 past frames,  k2k_buf

	__global	float4*		img_cur,				//7		// multiple past frames. NB retain frames at powers of 2, and vary starting power plus num franes.
	__global	float4*		img_past_0,				//8
	__global	float4*		img_past_1,				//9
	__global	float4*		img_past_2,				//10
	__global	float4*		img_past_3,				//11

	__global	float*		depth_map,				//12	// current frame depth, now stored as inv_depth
	__global	float8*		g1p,					//13	// current frame g1mem
	__global 	float8*		SE3_grad_map_cur_frame,	//14

	__global	float4*		vel_cur,				//15	// multiple past frames.
	__global	float4*		vel_past_0,				//16	// TODO, relative velocity not used yet. Will use it to modify depth map with timestep for past frames.
	__global	float4*		vel_past_1,				//17
	__global	float4*		vel_past_2,				//18
	__global	float4*		vel_past_3,				//19

	//output
	__global	float2*		Rho_,					//20	// { sum rho^2 ,  count of valid pixels used } Writen to dense patches.
	__local		float2*		local_rho,				//21	// float2 local_rho[ local_work_size/2 ]  hence sizeof( float)*local_work_size.

	//__global	float2*		weights_map,			//22
	//__local	float2*		local_weights,			//23

	__global	float2*		SE3_incr_map_,			//24
	__local		float2*		local_SE3_incr			//25
){
	const uint block_size		= 32;						// or send as __private arg ? BUT as hardcoded "const uint" it can be used to size arrays etc.
	const uint se3_dof			= 6;
	const uint num_past_frames	= 4;						// 1,2,4,8,16,32,64 // variable select window of 4 frames.
	const float4 zero_f4		= {0.0f,0.0f,0.0f,0.0f};
	const float2 zero_f2		= {0.0f,0.0f};

	__global float4*	img_past[num_past_frames]		= { img_past_0, img_past_1, img_past_2, img_past_3 };
	__global float4*	vel_past[num_past_frames]		= { vel_past_0, vel_past_1, vel_past_2, vel_past_3 };

	uint  global_id_u 			= get_global_id(0);
	//float global_id_flt 		= global_id_u;
	uint  lid 					= get_local_id(0);
	uint  group_id				= get_group_id(0);
	const uint local_size 		= get_local_size(0);

	const uint8 mipmap_params_ 	= mipmap_params[layer];
	uint read_offset_ 			= mipmap_params_[MiM_READ_OFFSET];
	uint read_cols_ 			= mipmap_params_[MiM_READ_COLS];
	uint read_rows_ 			= mipmap_params_[MiM_READ_ROWS];
	uint layer_pixels			= mipmap_params_[MiM_PIXELS];

	uint mm_cols				= uint_params[MM_COLS];
	uint mm_pixels				= uint_params[MM_PIXELS];

	float min_inv_depth			= fp32_params[MIN_INV_DEPTH];									//+ inv_d_step;
	float max_inv_depth			= fp32_params[MAX_INV_DEPTH];									//- inv_d_step;

	float reduction				= mm_cols/read_cols_;
	uint row_length				= cols_per_row;													// blocks_cols * block_size;
	uint row_col				= fmod((float)global_id_u, row_length);
	uint block_row				= global_id_u / row_length;
	uint read_index				= read_offset_ + row_col + block_row*block_size*mm_cols;
	uint row_offset				= read_offset_/mm_cols;

	uint write_spacing			= block_size/out_block_size;
	uint write_index 			= row_col/out_block_size + block_row*write_spacing*mm_cols;
	uint write_index_2 			= row_col/block_size + block_row*mm_cols;

	float2 rho_pvt_arr[block_size]				= {zero_f2};									// pvt variable for values in this column.
	float4 rho_pvt_flt4							= zero_f4;
	float2 rho_pvt_flt2							= zero_f2;

	float2 grad_pvt_arr[block_size*se3_dof]		= {zero_f2};									// pvt variable for values in this column.
	float4 grad_pvt_flt4						= zero_f4;
	float2 grad_pvt_flt2						= zero_f2;

	float2 SE3_incr_pvt_arr[block_size*se3_dof]	= {zero_f2};									// pvt variable for values in this column.
	float4 SE3_incr_pvt_flt4					= zero_f4;
	float2 SE3_incr_pvt_flt2					= zero_f2;

	float4 img_cur_pvt[block_size];																// pvt variable for values in this column.
	float8 g1p_pvt[block_size];
	float4 new_px;
	bool   intersection;

	local_rho[lid]								= zero_f2;
	//////////////////////////////////////...////////
	uint past_frame_idx =0;
	uint step;
	for ( step=1; step<=block_size; step *=2){																																					// for each step size, (multiples of 2)
		for (uint block_row=0; block_row<block_size ; block_row += step){																														// step through rows in column
			barrier(CLK_LOCAL_MEM_FENCE );
																						rho_pvt_arr[		block_row ]							+=rho_pvt_arr[		block_row + step ];			// sum pair of values in col,
			for (uint se3_dim=0; se3_dim<se3_dof; se3_dim++) {
																						SE3_incr_pvt_arr[	block_row + se3_dim*block_size ]	+=SE3_incr_pvt_arr[	block_row + step + se3_dim*block_size ];
			}
			barrier(CLK_LOCAL_MEM_FENCE );

			if( !(fmod((float)lid,(step*2))==0) &&  (fmod((float)lid,step)==0)    ){																											// selects 2nd column, sends data
																						local_rho[			lid-step ]							= rho_pvt_arr[		block_row];
				for (uint se3_dim=0; se3_dim<se3_dof; se3_dim++) {																																// NB integer division. Hence both threads use the same index to local memory.
																						local_SE3_incr[		lid-step + se3_dim*local_size ]		= SE3_incr_pvt_arr[	block_row + se3_dim*block_size ];
				}
			}
			barrier(CLK_LOCAL_MEM_FENCE );																																						// Using barrier as a semaphore, for local mem messages between threads. This minimizes local_mem req, while allowing 2 patch sizes in output, full & ST3 map at out_block_size.

			if( (fmod((float)lid,(step*2))==0)  ){																																				// selects 1st column, adds data. Sum of patch now held in top left element of patch.
																						rho_pvt_arr[		block_row] 							+= local_rho[		lid ];
				for (uint se3_dim=0; se3_dim<se3_dof; se3_dim++) {
																						SE3_incr_pvt_arr[	block_row + se3_dim*block_size ]	+= local_SE3_incr[	lid		 + se3_dim*local_size ];
				}
			}
			barrier(CLK_LOCAL_MEM_FENCE );
		}
		if (step==out_block_size/2){																																							// save ST3 map at out_block_size, to use for updating depth_map and rel_vel_map
			uint frame_offset 		= write_index + past_frame_idx * 100 + 25 ; // NB 100 works for current img size . // stacks frame ST3 maps in adjacent collumns..
			uint write_block_row	= 0;
			if( fmod((float)lid,out_block_size) == 0 ){																																			// selects columns i.e. threads within the workgroup
				for (uint block_row=0; block_row < block_size ; block_row += step*2, write_block_row++){
																						uint offset_1 				= frame_offset		+ write_block_row*mm_cols;
																						Rho_[			offset_1]	= rho_pvt_arr[		block_row ];
					for (uint se3_dim=3; se3_dim<se3_dof; se3_dim++) {																															// select only ST3
																						uint offset_2 				= offset_1			+ (se3_dim-3)*( 4+ (read_rows_/out_block_size) )*mm_cols;
																						uint offset_3 				= block_row			+ se3_dim*block_size;									// NB read_rows_/out_block_size = writre_rows
																						SE3_incr_map_[	offset_2 ]	= SE3_incr_pvt_arr[	offset_3 ];
					}
				}
			}
		}
	}


}


//===============================================


// if (layer>2){
// 						uint tmp_offset = block_row + se3_dim*block_size;
// 						printf("\n__kernel void Rho_sq(..)1	group_id=,%u,	lid=,%u,	block_row=,%u,		+ se3_dim=,%u,	block_size=,%u,	SE3_incr_pvt_arr[ tmp_offset ]=,%f,%f,							(lid + se3_dim*local_size)=,%u,	local_SE3_incr[lid+se3_dim*local_size]=,%f,%f",\
// 															group_id,		lid,		block_row,				se3_dim,	block_size,		SE3_incr_pvt_arr[tmp_offset].x,SE3_incr_pvt_arr[tmp_offset].y,	(lid + se3_dim*local_size),		local_SE3_incr[lid+se3_dim*local_size].x, local_SE3_incr[lid+se3_dim*local_size].y );
// 					}

// 					if (layer>2 && (se3_dim==5 || se3_dim==4) ){
// 						uint tmp_offset 	= block_row + se3_dim*block_size;
// 						printf("\n__kernel void Rho_sq(..)-2	global_id_u=,%u,	group_id=,%u,	lid=,%u,	block_row=,%u,		+ se3_dim=,%u,	block_size=,%u,	SE3_incr_pvt_arr[ tmp_offset ]=,%f,%f,	",\
// 																global_id_u,		group_id,		lid,		block_row,				se3_dim,	block_size,		SE3_incr_pvt_arr[tmp_offset].x,SE3_incr_pvt_arr[tmp_offset].y	  );
// 					}

// 				if (layer>2 && (se3_dim==5 || se3_dim==4) ){
// 						uint tmp_offset 	= block_row + se3_dim*block_size;
// 						uint tmp_offset_2 	= block_row + step + se3_dim*block_size;
// 						printf("\n__kernel void Rho_sq(..)-1	global_id_u=,%u,	group_id=,%u,	lid=,%u,	block_row=,%u,		+ se3_dim=,%u,	block_size=,%u,	SE3_incr_pvt_arr[ tmp_offset ]=,%f,%f,							(lid - step + se3_dim*local_size)=,%u,	local_SE3_incr[lid+se3_dim*local_size]=,%f,%f,	step=%u",\
// 															global_id_u,		group_id,		lid,		block_row,				se3_dim,	block_size,		SE3_incr_pvt_arr[tmp_offset].x,SE3_incr_pvt_arr[tmp_offset].y,	(lid - step + se3_dim*local_size),		local_SE3_incr[lid+se3_dim*local_size].x, local_SE3_incr[lid+se3_dim*local_size].y, step );
// 				}

// 					if (layer>2 && (se3_dim==5 || se3_dim==4) ){
// 						uint tmp_offset = block_row + se3_dim*block_size;
// 						printf("\n__kernel void Rho_sq(..)0	global_id_u=,%u,	group_id=,%u,	lid=,%u,	block_row=,%u,		+ se3_dim=,%u,	block_size=,%u,	SE3_incr_pvt_arr[ tmp_offset ]=,%f,%f,							(lid - step + se3_dim*local_size)=,%u,	local_SE3_incr[lid+se3_dim*local_size]=,%f,%f,	step=%u",\
// 															global_id_u,		group_id,		lid,		block_row,				se3_dim,	block_size,		SE3_incr_pvt_arr[tmp_offset].x,SE3_incr_pvt_arr[tmp_offset].y,	(lid - step + se3_dim*local_size),		local_SE3_incr[lid+se3_dim*local_size].x, local_SE3_incr[lid+se3_dim*local_size].y, step );
// 					}

// 					if (layer>2 && (se3_dim==5 || se3_dim==4) ){
// 						uint tmp_offset = block_row + se3_dim*block_size;
// 						printf("\n__kernel void Rho_sq(..)1	global_id_u=,%u,	group_id=,%u,	lid=,%u,	block_row=,%u,		+ se3_dim=,%u,	block_size=,%u,	SE3_incr_pvt_arr[ tmp_offset ]=,%f,%f,							(lid + se3_dim*local_size)=,%u,	local_SE3_incr[lid+se3_dim*local_size]=,%f,%f",\
// 															global_id_u,		group_id,		lid,		block_row,				se3_dim,	block_size,		SE3_incr_pvt_arr[tmp_offset].x,SE3_incr_pvt_arr[tmp_offset].y,	(lid + se3_dim*local_size),		local_SE3_incr[lid+se3_dim*local_size].x, local_SE3_incr[lid+se3_dim*local_size].y );
// 					}

// 					if (layer==0 && out_block_size==32){
// 						printf("\n__kernel void Rho_sq(..)1	global_id_u=,%u,	group_id=,%u,	lid=,%u,	step=%u,	block_row=,%u,	block_size=,%u,	out_block_size=%u,	offset_1=,%u,	rho_pvt_arr[ block_row ]=,%f,%f,",\
// 															global_id_u,		group_id,		lid,		step,		block_row,		block_size,		out_block_size,		offset_1,		rho_pvt_arr[block_row].x,	rho_pvt_arr[block_row ].y );
// 					}

																						//weights_map[	offset_2 ]	= grad_pvt_arr[		offset_3 ]

	/*
	float2 temp2a											= { (float)group_id, (float)lid }; // block_row, block_col
	if ( read_index < mm_pixels )	{	Rho_[read_index ] 	= temp2a;	}																														// Marks the area where the img buf is read, lines show top row of each patch. Breaks show bondaries of patches.
	barrier(CLK_GLOBAL_MEM_FENCE );
	*/


// 		if (layer==0 && out_block_size==32){
// 				printf("\n__kernel void Rho_sq(..)2	global_id_u=,%u,	group_id=,%u,	lid=,%u,	step=%u,	block_row=,%u,	block_size=,%u,	out_block_size=%u,	offset_2=,%u,	rho_pvt_arr[ block_row ]=,%f,%f,",\
// 													global_id_u,		group_id,		lid,		step,		block_row,		block_size,		out_block_size,		offset_2,		rho_pvt_arr[block_row].x,	rho_pvt_arr[block_row ].y );
// 		}

// 			if (layer>2 && (se3_dim==5 || se3_dim==4) ){
// 				printf("\n__kernel void Rho_sq(..)2	global_id_u=,%u,	group_id=,%u,	lid=,%u,	step=%u,	block_row=,%u,		+ se3_dim=,%u,	block_size=,%u,	= offset_4,%u,	SE3_incr_pvt_arr[ offset_4 ]=,%f,%f,",\
// 													global_id_u,		group_id,		lid,		step,		block_row,				se3_dim,	block_size,			offset_4,	SE3_incr_pvt_arr[ offset_4 ].x,	SE3_incr_pvt_arr[  offset_4 ].y );
// 			}

 /*&& (step==block_size/2)*/

		//step 						=  block_size/2;



																					//if(read_col==0 && idx==0 && SE3==5) printf("\nlid=%u, old_row_offset=%u,  SE3=%u, idx(%u),  idx_2(%u),  idx_3(%u)",\
																						lid, old_row_offset, SE3, idx, idx_2, idx_3);
			/*
// 																						printf("\nlid=%u, old_row_offset=%u,  SE3=%u,  mm_cols=%u,  read_col(%u) +  (idx(%u) *  mm_cols(%u)) = idx_2(%u),  + row_offset(%u) = idx_3(%u),  pvt_incr={%f, %f}	 weights_map[idx_3]{%f, %f},   Rho_[idx_2]={%f, %f}   ,  SE3_incr_map_[idx_3]={%f,  %f}",\
// 																							lid, old_row_offset, SE3, mm_cols, read_col, idx, mm_cols,  idx_2, row_offset, idx_3, \
// 																							pvt_incr.x, pvt_incr.y,          weights_map[idx_3].x,     weights_map[idx_3].y,      \
// 																							Rho_[idx_2].x,   Rho_[idx_2].y,  SE3_incr_map_[idx_3].x,   SE3_incr_map_[idx_3].y  );
			*/

/*
// 	if(lid==0){printf("\n\n__kernel void update_k2k(..) local_pose_inv_K = \n");
// 		for (uint i=0; i< 2 ; i++){ for (uint j=0; j< 4 ; j++){ for (uint k=0; k< 4 ; k++){		printf(",	%f",local_pose_inv_K[	i*16 +j*4 +k]);	} printf("\n"); } printf("\n\n"); }
// 	}
*/



																					//if (global_id_u==0) printf("\n__kernel void reduce_patch_Rho(..)   read_col(%u) < cols(%u)  &&  (SE3(%u) < SE3_DoF(%u),  thread_offset(%u) in_range(%u)  ",\
																						read_col, cols, SE3, SE3_DoF, thread_offset, in_range  );
