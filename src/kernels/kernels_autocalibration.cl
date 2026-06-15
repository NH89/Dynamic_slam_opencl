#include "kernels__macros.h"
#include "kernels.h"

__kernel void comp_cam_and_lens_maps(
	__private	uint		layer,			//0
														//	__private	float		inv_depth,		//1		// may need real inv_depth map ###
	__private	uint		num_vars,		//1

	__constant	uint8*		mipmap_params,	//2
	__constant	uint*		uint_params,	//3
	__constant	float16*	param_k2k,		//4

	__global 	float2*		depth_map,		//5

	__global 	float2*		SE3_map			//6
		 )
{
	uint global_id_u 	= get_global_id(0);
	float global_id_flt = global_id_u;
	uint8 mipmap_params_= mipmap_params[layer];
	uint read_offset_ 	= mipmap_params_[MiM_READ_OFFSET];
	uint read_cols_ 	= mipmap_params_[MiM_READ_COLS];
	uint read_rows_ 	= mipmap_params_[MiM_READ_ROWS];
	if (global_id_u 	>= mipmap_params_[MiM_PIXELS]) return;

	uint lid 			= get_local_id(0);
	uint group_size 	= get_local_size(0);

	uint margin			= uint_params[MARGIN];
	uint mm_cols		= uint_params[MM_COLS];
	uint base_cols		= uint_params[COLS];
	float reduction		= base_cols/read_cols_;
	uint v				= global_id_u / read_cols_;													// read_row
	uint u				= fmod(global_id_flt, read_cols_);											// read_column

	if (lid<num_vars ) { printf("\n layer=%u,	gid=%u, (u,v)=(%u,%u),	param_k2k[%u]={{ %f,	%f,	%f,	%f},{ %f,	%f,	%f,	%f},{ %f,	%f,	%f,	%f},{ %f,	%f,	%f,	%f}}",\
		layer, global_id_u, u,v, lid, \
		param_k2k[lid].s0, param_k2k[lid].s1, param_k2k[lid].s2, param_k2k[lid].s3,\
		param_k2k[lid].s4, param_k2k[lid].s5, param_k2k[lid].s6, param_k2k[lid].s7,\
		param_k2k[lid].s8, param_k2k[lid].s9, param_k2k[lid].sa, param_k2k[lid].sb,\
		param_k2k[lid].sc, param_k2k[lid].sd, param_k2k[lid].se, param_k2k[lid].sf);
	}

	float u2, v2, u_ref, v_ref;
	uint read_index 	= read_offset_  +  v  * mm_cols  + u ;
	bool print			= false;	//if( (u==10)&&(v==10) ){ print=true; }

	float  inv_depth	= depth_map[ read_index ].s0;
																														// computes u_ref and v_ref, i.e. pixel reprojection of existing k2k.
	px_k2k( 							param_k2k[num_vars],	reduction,  v,  u,  inv_depth, &u_ref,	&v_ref,	print );
	for (uint i=0; i<num_vars; i++) {																					// for each param DoF, find new pixel position, h=homogeneous coords.
		px_k2k( 						param_k2k[i],			reduction,  v,  u,  inv_depth, &u2,		&v2,	print );
		float2 partial_gradient								=	{ u_ref - u2,	v_ref - v2 }; 							// Find movement of pixel
		SE3_map[read_index + i* uint_params[MM_PIXELS]  ]	=	partial_gradient;

		if((u%100)==0 & (v%100)==0)printf("\n__kernel void comp_cam_and_lens_maps(..) i=%u, (read_index + i* uint_params[MM_PIXELS]) = %u,  partial_gradient=(%f, %f), 		u_ref=%f, u2=%f,		 v_ref=%f, v2=%f, u=%u, v=%u",\
																						i,  (read_index + i* uint_params[MM_PIXELS]), 	partial_gradient.x, partial_gradient.y, u_ref, u2,	 v_ref, v2, u, v );
		barrier(CLK_GLOBAL_MEM_FENCE );
	}

	if(print==true && global_id_u==0){ printf("\n");}

	// TO DO // Create a 'reproject' & 'img_grad_sum' kernels
}



__kernel void  patch_cam_and_lens_Hessian(			// To be launched with 1 thread per col for 32x32 patches, and an integer multiple of 32 threads.
													// Needs 16 elements of local mem per 32x32 patch, to pass data between threads in recursive square reduction.
													// Needs 32 elem array of private mem per thread.
	//Inputs:
	__private	uint		layer,					//0
	__private	uint		lookup_table_offset,	//1
	__private	uint		out_block_size,			//2
	__private	uint3		SE3_offset3,			//3
	__private	uint3		ST3_offset3,			//4
//	__private	float16		cam_param_weights,		//

	__constant	uint8*		mipmap_params,			//5
	__constant	uint*		uint_params,			//6

	__global	float2*		depth_map,				//7	// current frame depth, now stored as inv_depth
	__global	float2*		param_map,				//8
	__global	uint4*		lookup_table,			//9
	__global	float8*		img_grad_uv,			//10

	//Outputs:
	__global 	float4*		param_grad_map,			//11											// We keep hsv sepate at this stage, so 6*4*2=24, but float16 is the largest type, so 6*float8.
	__global 	float4*		cam_Hessian_map,		//12											// HSV (5x5) matrix so 25*float4. 2nd half holds Jacobian maps, req for IC-LK algorithm. Size 2xmm_pixels.
	__local		float4*		local_Hessian			//13											// local_Hessian_pseudo_inverse[ sizeof(float4) *5*5 *local_size]
){
	uint	global_id_uint								= get_global_id(0);
	uint	lid											= get_local_id(0);
	uint	group_id									= get_group_id(0);
	const	uint local_size								= get_local_size(0);

	local_Hessian[lid]									= zero_f4;									// NB must zero local memory _before_ closing unused threads.

	float	null_factor = 1.0f;
	uint4	lookup_ref									= lookup_table[global_id_uint + lookup_table_offset];
	if( lookup_ref.w != global_id_uint){	printf("\n__kernel void patch_img_grad(..) lookup_ref.w %u != global_id_uint %u", lookup_ref.w, global_id_uint);	// NB nullify cols tha are outside img_cur.  TODO (1) can threads be returned when not used? (2) can if statements be reduced / made more efficient ?
											null_factor = 0.0f;
	}
	uint	read_index									= lookup_ref.z;
	uint	u											= lookup_ref.x;																							// read_column
	uint	v											= lookup_ref.y;																							// read_row

	uint4	lookup_ref_layer							= lookup_table[lookup_table_offset].z;	//0;
	uint	layer_offset								= lookup_ref_layer.z;

	uint8	mipmap_params_ 								= mipmap_params[layer];
	uint	read_cols_									= mipmap_params_[MiM_READ_COLS];
	uint	read_rows_									= mipmap_params_[MiM_READ_ROWS];

	uint	mm_cols										= uint_params[MM_COLS];
	uint	mm_rows										= uint_params[MM_ROWS];
	uint	mm_pixels									= uint_params[MM_PIXELS];

	uint	stop_offset									= layer_offset + (read_rows_ -1) * mm_cols + read_cols_	;												// bottom right corner of source image layer

	int		rtoff										=  (u < read_cols_-2);						// +1														// (read_column < mm_cols-1);
	int		lfoff										= -(u >1);									// -1														//-(read_column != 0);

	float4	Jacobian_pvt_arr[block_size][num_cam_matx_DoF]						=  {{zero_f4}};
	float4	Hessian_pvt_arr[block_size][num_cam_matx_DoF][num_cam_matx_DoF]		= {{{zero_f4}}};																// pvt variable for values in this column.

	const uint SE3_offset								= SE3_offset3.s0;
	const uint ST3_offset								= ST3_offset3.s0;

	const uint SE3_u_step								= SE3_offset3.s1;																						// step between elements of the Hessian matrix
	const uint ST3_u_step								= ST3_offset3.s1;

	const uint SE3_v_step								= SE3_offset3.s2;
	const uint ST3_v_step								= ST3_offset3.s2;

	uint	write_index									= u/out_block_size			 + (v/out_block_size)*mm_cols	+ ST3_offset;
	uint	write_index_2								= u/block_size				 + (v/block_size)*mm_cols		+ SE3_offset;
	uint 	offset_1_1_max								= mm_cols * read_rows_ / out_block_size 		 			+ ST3_offset;
/*
// 	if(global_id_uint==0){printf("\n__kernel_patch_cam_and_lens_Hessian, cam_param_weights=%f,  %f,  %f,  %f,  %f,  %f,  %f,  %f,  %f,  %f ",\
// 		cam_param_weights.s0, cam_param_weights.s1, cam_param_weights.s2, cam_param_weights.s3, cam_param_weights.s4,\
// 		cam_param_weights.s5, cam_param_weights.s6, cam_param_weights.s7, cam_param_weights.s8, cam_param_weights.s9 );}

	float	SE3_weights[5]								= { cam_param_weights.s0, cam_param_weights.s2, cam_param_weights.s4, cam_param_weights.s6, cam_param_weights.s8 };
	float	ST3_weights[5]								= { cam_param_weights.s1, cam_param_weights.s3, cam_param_weights.s5, cam_param_weights.s7, cam_param_weights.s9 };

// 	if(global_id_uint==0){printf("\n__kernel_patch_cam_and_lens_Hessian, SE3_weights[5] = %f,  %f,  %f,  %f,  %f,     ST3_weights = %f,  %f,  %f,  %f,  %f ",
// 		SE3_weights[0], SE3_weights[1], SE3_weights[2], SE3_weights[3], SE3_weights[4], \
// 		ST3_weights[0], ST3_weights[1], ST3_weights[2], ST3_weights[3], ST3_weights[4]  ); }
*/
	for (uint row_in_block=0; (row_in_block<block_size)&&(read_index<=stop_offset&&read_index>0); row_in_block++, v++,  read_index +=mm_cols){					// stop offset prevents bottom row patches from overrunning the bottom of the image layer. // NB readindex may be 0 if not in range according to lookup table.
		int dnoff										=  (v  < read_rows_-2) * mm_cols;			// +1														// (read_row  < read_rows_-1) * mm_cols;
		int upoff										= -(v  >1 )*mm_cols;						// -1														//-(read_row  != 0)*mm_cols;	// up, down, left, right offsets, by boolean logic.

		float  inv_depth								= depth_map[ read_index ].s0;
		float8 pvt_img_grad								= img_grad_uv[ read_index ];
		float4 gu										= pvt_img_grad.s0123;
		float4 gv										= pvt_img_grad.s4567;
		float4	Jacobian[num_cam_matx_DoF]				=  {0};

		for (uint i=0; i<num_cam_matx_DoF; i++) {
			float2	param_px							= param_map[read_index + i* mm_pixels];																	// SE3_map[read_index + i* uint_params[MM_PIXELS]  ] = partial_gradient;  // float2 partial_gradient={u_flt-u2 , v_flt-v2}; // Find movement of pixel
			float4	gxSE3								= gu*param_px[0];																						// J_SE3 * img gradient i.e. edges
			float4	gySE3								= gv*param_px[1];
			Jacobian[i]									= (gxSE3 + gySE3)  * null_factor;	//* ( SE3_weights[i]  +  ST3_weights[i] * inv_depth );
			Jacobian[i].w								= 1.0f;
			param_grad_map[read_index + i* mm_pixels]	= Jacobian[i];
		}
		for (uint i=0; i<num_cam_matx_DoF; i++) {
			Jacobian_pvt_arr[row_in_block][i]			= Jacobian[i];

			for (uint j=0; j<num_cam_matx_DoF; j++) {
				Hessian_pvt_arr[row_in_block][i][j]		= Jacobian[i] * Jacobian[j];																			// Gauss-Newton approx H = J.transpose * J  // TO DO compute and sum lower triangle only.
			}
		}
	} // end of column of this patch.

	// Sum-reduce image, /////////////  Save intermediate size ST3 patches for depth map updates, and maximally reduced SE3 patches for pose updates. Second reduce_patch_Rho(..) kernel required for SE3 from lareger image pyramid layers, before update_k2k(..) kernel.
	// make this a device function ?

	uint past_frame_idx = 0; // TO DO remove and restore long outer loop.
	uint step;

	for ( step=1; step<block_size; step *=2){																																						// for each step size, (multiples of 2)
		for (uint block_row=0; block_row<block_size ; block_row += step){																															// step through rows in column

			for (uint i=0; i<num_cam_matx_DoF; i++) {
																						Jacobian_pvt_arr[	block_row][i]						+=Jacobian_pvt_arr[		block_row + step ][i];
				for (uint j=0; j<num_cam_matx_DoF; j++) {
																						Hessian_pvt_arr[	block_row][i][j]					+=Hessian_pvt_arr[		block_row + step ][i][j];	//+ se3_dim*block_size ];
				}
			}

			// First reduce the Jacobian, using the Hessian local memory.
			if( !(fmod((float)lid,(step*2))==0) &&  (fmod((float)lid,step)==0)    ){																												// selects 2nd column, sends data
				for (uint i=0; i<num_cam_matx_DoF; i++) {
																						local_Hessian[		lid-step + i*local_size]			= Jacobian_pvt_arr[		block_row][i];
				}
			}
			barrier(CLK_LOCAL_MEM_FENCE );																																							// Using barrier as a semaphore, for local mem messages between threads. This minimizes local_mem req, while allowing 2 patch sizes in output, full & ST3 map at out_block_size.

			if( (fmod((float)lid,(step*2))==0)  ){																																					// selects 1st column, adds data. Sum of patch now held in top left element of patch.
				for (uint i=0; i<num_cam_matx_DoF; i++) {
																						Jacobian_pvt_arr[	block_row][i]						+= local_Hessian[		lid + i*local_size];		//if(/*lid==0 &&*/ i==0) printf("\nstep=%d, J block_row=%d, lid=%d .x=%f, .w=%f",\
																																																	//					step, block_row, lid, Jacobian_pvt_arr[ block_row][i].x, Jacobian_pvt_arr[	block_row][i].w);
				}
			}
			barrier(CLK_LOCAL_MEM_FENCE );																																							// Using barrier as a semaphore, for local mem messages between threads. This minimizes local_mem req, while allowing 2 patch sizes in output, full & ST3 map at out_block_size.

			// Now Reduce Hessian
			if( !(fmod((float)lid,(step*2))==0) &&  (fmod((float)lid,step)==0)    ){																												// selects 2nd column, sends data
				for (uint i=0; i<num_cam_matx_DoF; i++) {
					for (uint j=0; j<num_cam_matx_DoF; j++) {
																						local_Hessian[		lid-step + (i*6 + j)*local_size]	= Hessian_pvt_arr[		block_row ][i][j];			//+ se3_dim*block_size ];  TO DO correct size and indexing of local_Hessiasn
					}
				}
			}
			barrier(CLK_LOCAL_MEM_FENCE );																																							// Using barrier as a semaphore, for local mem messages between threads. This minimizes local_mem req, while allowing 2 patch sizes in output, full & ST3 map at out_block_size.

			if( (fmod((float)lid,(step*2))==0)  ){																																					// selects 1st column, adds data. Sum of patch now held in top left element of patch.
				for (uint i=0; i<num_cam_matx_DoF; i++) {
					for (uint j=0; j<num_cam_matx_DoF; j++) {
																						Hessian_pvt_arr[	block_row][i][j]					+= local_Hessian[		lid + (i*6 + j)*local_size ];	// + se3_dim*block_size ]
					}
				}
			}
			barrier(CLK_LOCAL_MEM_FENCE );
		}
		// Save intermediate size ST3 Hessian patches for depth map updates, ////////// ### TODO comment this out if not debugging.
		if ( step==out_block_size/2 ){																																							// save ST3 map at out_block_size, to use for updating depth_map and rel_vel_map
			uint frame_offset 		= write_index + past_frame_idx * 100 + 25 ;																													// NB 100 works for current img size . // stacks frame ST3 maps in adjacent collumns..
			uint write_block_row	= 0;
			bool inbounds			= false;
			uint offset_1_1			= 0;

			for (uint block_row=0; block_row < block_size ; block_row += step*2, write_block_row++){
				inbounds																= false;
				if ( (write_index + write_block_row*mm_cols) < offset_1_1_max) inbounds	= true;

				for (uint i=0; i<3; i++) {																																						// select only ST3
					if( fmod((float)lid,out_block_size) == 0  ){																																// write Jacobian to 2nd page of SE3_Hessian_pinv_map buffer.
																						offset_1_1 								= write_index		+ i*ST3_v_step	+ mm_pixels + write_block_row*mm_cols;
																						float4	pvt_Jacobian 					= Jacobian_pvt_arr[		block_row][i/*+3*/];
																						if( inbounds == true ){
																							cam_Hessian_map[	offset_1_1 ]	= pvt_Jacobian;
																						}
					}
					barrier(CLK_GLOBAL_MEM_FENCE );			// TO DO is this needed?
					for (uint j=0; j<3; j++) {																																					// write Hessian to 1st page of SE3_Hessian_pinv_map buffer.
						if( /*inbounds == true*/ fmod((float)lid,out_block_size) == 0 ){																										// selects columns i.e. threads within the workgroup
																						offset_1_1								= write_index		+ i*ST3_v_step	+ j*ST3_u_step + write_block_row*mm_cols;
																						float4	pvt_Hessian 					= Hessian_pvt_arr[	block_row ][i/*+3*/][j/*+3*/];
																						if( inbounds == true ){
																							cam_Hessian_map[	offset_1_1 ]	= pvt_Hessian;
																						}
						}
						barrier(CLK_GLOBAL_MEM_FENCE );		// TO DO is this needed?
					}
				}
			}
		}
	}
	/// Save maximally reduced SE3 Hessian 32x32 patches for pose updates ////////////////////																									// Writes dense blocks. Reduces required transfer to host.
	uint frame_offset_1 			=  0;
	uint offset_2					=  0;
	uint block_row					=  0;

	for (uint i=0; i<num_cam_matx_DoF; i++) {																																						// write Jacobian to 2nd page of SE3_Hessian_pinv_map buffer.		// All 6 DoF of SE3
		if( fmod((float)lid,block_size) == 0 ){																																					// selects columns i.e. threads within the workgroup
																						offset_2 								= write_index_2		+ i*SE3_v_step	+ mm_pixels;
																						float4	pvt_Jacobian 					= Jacobian_pvt_arr[		block_row][i];
																						cam_Hessian_map[	offset_2 ]			= pvt_Jacobian;
		}
		barrier(CLK_GLOBAL_MEM_FENCE );

		for (uint j=0; j<num_cam_matx_DoF; j++) {																																					// write Hessian to 1st page of SE3_Hessian_pinv_map buffer.
			if( fmod((float)lid,block_size) == 0 ){
																						offset_2 								= write_index_2		+ i*SE3_v_step	+ j*SE3_u_step;
																						float4	pvt_Hessian 					= Hessian_pvt_arr[	block_row ][i][j];
																						cam_Hessian_map[	offset_2 ]			= pvt_Hessian;

																					//	float4		debug 						= {(float)lid, global_id_uint, group_id, 1.0f};					// NB Must comment out to compute correct Hessian.
																					//	cam_Hessian_map[read_index]				= debug;														// marks top left corner of where original patches are read from.
			}
			barrier(CLK_GLOBAL_MEM_FENCE );
		}
	}

}
