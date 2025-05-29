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
	const uint block_size		= 32;						// or send as __private arg ? BUT as hardcoded "const uint" it can be used to size arrays etc.
	const uint se3_dof			= 6;
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
	if( fmod((float)lid,block_size) ==0 ){		//out_block_size																																// selects columns i.e. threads within the workgroup
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
