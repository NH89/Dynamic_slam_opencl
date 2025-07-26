#include "kernels_macros.h"
#include "kernels.h"

 __kernel void compute_patch_lookup_table(					// computed once at start of program	// TODO when is it possible to roll the layers together ?  i.e. when local mem is not used.
	// inputs
	__private	uint		layer,					//0
	__private	uint		lookup_table_offset,	//1
	__private	uint		cols_per_row,			//2

	__constant 	uint8*		mipmap_params,			//3
	__constant 	uint*		uint_params,			//4
	__constant  float*  	fp32_params,			//5

	// output
	__global 	float4*		lookup_table			//6
)
{
	uint  global_id_u 								= get_global_id(0);

	const uint8 mipmap_params_						= mipmap_params[layer];
	uint read_offset_ 								= mipmap_params_[MiM_READ_OFFSET];
	uint read_cols_ 								= mipmap_params_[MiM_READ_COLS];
	uint read_rows_ 								= mipmap_params_[MiM_READ_ROWS];

	uint mm_cols									= uint_params[MM_COLS];
	uint mm_pixels									= uint_params[MM_PIXELS];

	int v 											= global_id_u / cols_per_row;
	int u 											= fmod( (float)global_id_u, cols_per_row );
	uint read_index									= read_offset_ + u + v*block_size*mm_cols;
	uint row_offset									= read_offset_ / mm_cols;
	float4 lookup 									= zero_f4;

	if ( read_index < mm_pixels  &&  u< read_cols_  &&  v < read_rows_)	{
		lookup										= (float4)(u, v, read_index, layer /*row_offset*/);
	}
	lookup_table[global_id_u + lookup_table_offset]	= lookup;
}


__kernel void  patch_img_grad(						// To be launched with 1 thread per col for 32x32 patches, and an integer multiple of 32 threads.
													// Needs 16 elements of local mem per 32x32 patch, to pass data between threads in recursive square reduction.
													// Needs 32 elem array of private mem per thread.
	//Inputs:
	__private	uint		layer,					//0
	__private	uint		out_block_size,			//1

	__constant	uint8*		mipmap_params,			//2
	__constant	uint*		uint_params,			//3
	__constant 	float2*		SE3_map,				//4

	__global 	float4*		lookup_table,			//5
	__global 	float4*		img,					//6

	//Outputs:
	__global 	float8*		SE3_grad_map,			//7												// We keep hsv sepate at this stage, so 6*4*2=24, but float16 is the largest type, so 6*float8.
	__global 	float4*		SE3_Hessian_map,		//8												// HSV (6x6) matrix so 36*float8
	__local		float4*		local_Hessian,			//9												//	local_Hessian[ sizeof(float4) *6*6 *local_size]

	__global 	float8*		HSV_grad				//10
){
	uint	global_id_uint								= get_global_id(0);
	uint	lid											= get_local_id(0);
	uint	group_id									= get_group_id(0);
	const	uint local_size								= get_local_size(0);

	float4	lookup_ref									= lookup_table[global_id_uint];
	uint	read_index									= floor(lookup_ref.z);
	uint	v											= lookup_ref.x;														// read_row
	uint	u											= lookup_ref.y;														// read_column

	uint	write_spacing								= block_size/out_block_size;
	uint	write_index									= u/out_block_size;													// + block_row*write_spacing*mm_cols;
	uint	write_index_2								= u/block_size;														// + block_row*mm_cols;

	uint8	mipmap_params_ 								= mipmap_params[layer];
	uint	read_cols_									= mipmap_params_[MiM_READ_COLS];
	uint	read_rows_									= mipmap_params_[MiM_READ_ROWS];

	uint	mm_cols										= uint_params[MM_COLS];
	uint	mm_pixels									= uint_params[MM_PIXELS];

	int		lfoff										= -(u >1);															//-(read_column != 0);
	int		rtoff										=  (u < read_cols_-2);												// (read_column < mm_cols-1);

	float4 	Hessian_pvt_arr[block_size][6][6]			= {{{zero_f4}}};													// pvt variable for values in this column.

	for (uint row_in_block=0; row_in_block<block_size; row_in_block++, v++,  read_index +=mm_cols,  write_index +=write_spacing*mm_cols,  write_index_2 +=mm_cols ){

		int upoff										= -(v  >1 )*mm_cols;												//-(read_row  != 0)*mm_cols;	// up, down, left, right offsets, by boolean logic.
		int dnoff										=  (v  < read_rows_-2) * mm_cols;									// (read_row  < read_rows_-1) * mm_cols;

		float4 pu, pd, pl, pr;
		pr												=  img[read_index + rtoff];
		pl												=  img[read_index + lfoff];
		pu												=  img[read_index + upoff];
		pd												=  img[read_index + dnoff];

		float4 gx										= { (pr.x - pl.x), (pr.y - pl.y), (pr.z - pl.z), 1.0f };			// Signed img gradient in hsv
		float4 gy										= { (pd.x - pu.x), (pd.y - pu.y), (pd.z - pu.z), 1.0f };

		float4 Jacobian[6]								=  {0};

		for (uint i=0; i<6; i++) {
			float2 SE3_px								= SE3_map[read_index + i* mm_pixels];								// SE3_map[read_index + i* uint_params[MM_PIXELS]  ] = partial_gradient;
																															// float2 partial_gradient={u_flt-u2 , v_flt-v2}; // Find movement of pixel
			float8 SE3_grad_px							= {gx*SE3_px[0]  , gy*SE3_px[1] };									// NB float4 gx, gy => float8
			SE3_grad_map[read_index + i* mm_pixels]		= SE3_grad_px ;
			Jacobian[i]									= SE3_grad_px.lo + SE3_grad_px.hi;
		}
		for (uint i=0; i<6; i++) {
			for (uint j=0; j<6; j++) {
				Hessian_pvt_arr[row_in_block][i][j]		= Jacobian[i] * Jacobian[j];
			}
		}
		float H 										= img[read_index][0] * 2*M_PI_F;
		float S 										= img[read_index][1];
		float V 										= img[read_index][2];
		float8 temp_float8								= { sin(H) , cos(H), S, V, gx[1], gy[1], gx[2], gy[2] };			// HSV_grad = { sin(H) , cos(H), S, V, gx[1], gy[1], gx[2], gy[2] };
		HSV_grad[read_index]						= temp_float8;

	} // end of column of this patch.

	// Sum-reduce image, /////////////  Save intermediate size ST3 patches for depth map updates, and maximally reduced SE3 patches for pose updates. Second reduce_patch_Rho(..) kernel required for SE3 from lareger image pyramid layers, before update_k2k(..) kernel.
	// make this a device function ?

	uint past_frame_idx =0; // TODO remove and restore long outer loop.
	uint step;
	uint SE3_out_step_1			= ( 4 + (read_rows_/block_size) )*mm_cols;
	uint SE3_out_step_2			= SE3_out_step_1 * 6;
	uint SE3_out_step_5			= 4+ (read_cols_/block_size);


	for ( step=1; step<block_size; step *=2){																																					// for each step size, (multiples of 2)
		for (uint block_row=0; block_row<block_size ; block_row += step){																														// step through rows in column
			for (uint i=0; i<6; i++) {
				for (uint j=0; j<6; j++) {
																						Hessian_pvt_arr[	block_row][i][j]					+=Hessian_pvt_arr[	block_row + step ][i][j];  //+ se3_dim*block_size ];
				}
			}

			if( !(fmod((float)lid,(step*2))==0) &&  (fmod((float)lid,step)==0)    ){																											// selects 2nd column, sends data
				for (uint i=0; i<6; i++) {
					for (uint j=0; j<6; j++) {
																						local_Hessian[		lid-step + (i*6 + j)*local_size]	= Hessian_pvt_arr[	block_row ][i][j];			//+ se3_dim*block_size ];  TODO correct size and indexing of local_Hessiasn
					}
				}
			}
			barrier(CLK_LOCAL_MEM_FENCE );																																						// Using barrier as a semaphore, for local mem messages between threads. This minimizes local_mem req, while allowing 2 patch sizes in output, full & ST3 map at out_block_size.

			if( (fmod((float)lid,(step*2))==0)  ){																																				// selects 1st column, adds data. Sum of patch now held in top left element of patch.
				for (uint i=0; i<6; i++) {
					for (uint j=0; j<6; j++) {
																						Hessian_pvt_arr[	block_row][i][j]					+= local_Hessian[	lid + (i*6 + j)*local_size ];   // + se3_dim*block_size ]
					}
				}
			}
			barrier(CLK_LOCAL_MEM_FENCE );
		}
		// Save intermediate size ST3 Hessian patches for depth map updates, //////////
		if (step==out_block_size/2){																																							// save ST3 map at out_block_size, to use for updating depth_map and rel_vel_map
			uint frame_offset 		= write_index + past_frame_idx * 100 + 25 ;																													// NB 100 works for current img size . // stacks frame ST3 maps in adjacent collumns..
			uint write_block_row	= 0;
			uint SE3_out_step_3		= ( 4 + (read_rows_/block_size) )*mm_cols;

			if( fmod((float)lid,out_block_size) == 0 ){																																			// selects columns i.e. threads within the workgroup
				for (uint block_row=0; block_row < block_size ; block_row += step*2, write_block_row++){
																						uint offset_1 						= frame_offset		+ write_block_row*mm_cols	+ SE3_out_step_2;
					for (uint i=3; i<se3_dof; i++) {																																			// select only ST3
						for (uint j=0; j<6; j++) {
																						offset_1 							+= (i-3)*SE3_out_step_3;
																																																//uint offset_3 = block_row	+ i*block_size;	// NB read_rows_/out_block_size = writre_rows
																						SE3_Hessian_map[	offset_1 ]		= Hessian_pvt_arr[	block_row ][i][j];
						}
					}
				}
			}
		}
	}
	/// Save maximally reduced SE3 Hessian 32x32 patches for pose updates ////////////////////																									// Writes dense blocks. Reduces required transfer to host.
	uint write_block_row			=  0;
	if( fmod((float)lid,block_size) == 0 ){																																						// selects columns i.e. threads within the workgroup
		uint frame_offset_1 		=  write_index_2;																																			// stacks frame SE3 results vertically.
		uint block_row				=  0;
																						uint offset_2 						= frame_offset_1	+ write_block_row*mm_cols;
		for (uint i=0; i<se3_dof; i++) {																																						// All 6 DoF of SE3
			for (uint j=0; j<6; j++) {
																						offset_2 							+= i*SE3_out_step_1	+ j*SE3_out_step_5;								//se3_dim*( 4 + (read_rows_/block_size) )*mm_cols;
																																																//uint offset_4		= block_row			+ i*block_size;
																						SE3_Hessian_map[	offset_2 ]		= Hessian_pvt_arr[	block_row ][i][j];
			}
		}
	}

}


__kernel void  patch_global_hessian_reduce(

){

}


__kernel void  patch_Gauss_Jordan_elimination( // invert hessians for (1) ST3+rot (4x4) depth and rel_vel_map,  (2) SO3 (6x6) global camera  (3) camera intrinsic matrix (4) lens distortion

){

}


__kernel void  patch_Inverse_Compositional_update(  // (1) for depth & rel_vel_map,  (2) SO3 camera pose,  (3) camera intrinsic matrix (4) lens distortion

){

}


