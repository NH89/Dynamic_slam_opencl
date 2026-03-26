#include "kernels__macros.h"
#include "kernels.h"

 __kernel void compute_patch_lookup_table(					// computed once at start of program	// TO DO when is it possible to roll the layers together ?  i.e. when local mem is not used.
	// inputs
	__private	uint		layer,					//0
	__private	uint		lookup_table_offset,	//1
	__private	uint		cols_per_row,			//2

	__constant 	uint8*		mipmap_params,			//3
	__constant 	uint*		uint_params,			//4
	__constant  float*  	fp32_params,			//5

	// output
	__global 	uint4*		lookup_table			//6
)
{
	uint  global_id_u 								= get_global_id(0);

	const uint8 mipmap_params_						= mipmap_params[layer];
	uint read_offset_ 								= mipmap_params_[MiM_READ_OFFSET];
	uint read_cols_ 								= mipmap_params_[MiM_READ_COLS];
	uint read_rows_ 								= mipmap_params_[MiM_READ_ROWS];

	uint mm_cols									= uint_params[MM_COLS];
	uint mm_pixels									= uint_params[MM_PIXELS];

	int v 											= global_id_u / cols_per_row;				// NB integer division.
	v												*= block_size;
	int u 											= fmod( (float)global_id_u, cols_per_row );
	uint read_index									= read_offset_ + u + v*mm_cols;
	uint row_offset									= read_offset_ / mm_cols;
	uint4 lookup 									= {0,0,0,0};

	if ( read_index < mm_pixels  &&  u< read_cols_  &&  v < read_rows_)	{
		lookup										= (uint4)(u, v, read_index, global_id_u /*row_offset*/);  /* layer */ // NB global_id_u is used to check bounds.
	}
	//lookup_table[read_index] = lookup;  //debugging, NB Must be written first, otherwise it overwrites and corrupts the lookup. Can only be used with oner layer.

	lookup_table[global_id_u + lookup_table_offset]	= lookup;
}


__kernel void  patch_img_grad(						// To be launched with 1 thread per col for 32x32 patches, and an integer multiple of 32 threads.
													// Needs 16 elements of local mem per 32x32 patch, to pass data between threads in recursive square reduction.
													// Needs 32 elem array of private mem per thread.
	//Inputs:
	__private	uint		layer,					//0
	__private	uint		lookup_table_offset,	//1
	__private	uint		out_block_size,			//2

	__private	uint3		SE3_offset3,			//3
	__private	uint3		ST3_offset3,			//4

	__constant	uint8*		mipmap_params,			//5
	__constant	uint*		uint_params,			//6
	__constant	float2*		SE3_map,				//7

	__global	uint4*		lookup_table,			//8
	__global	float4*		img,					//9
	__global	float*		depth_map,				//10	// current frame depth, now stored as inv_depth

	//Outputs:
	__global 	float4*		SE3_grad_map,			//11											// We keep hsv sepate at this stage, so 6*4*2=24, but float16 is the largest type, so 6*float8.
	__global 	float4*		SE3_Hessian_map,		//12											// HSV (6x6) matrix so 36*float4. 2nd half holds Jacobian maps, req for IC-LK algorithm. Size 2xmm_pixels.
	__local		float4*		local_Hessian,			//13											// local_Hessian_pseudo_inverse[ sizeof(float4) *6*6 *local_size]
	__global 	float4*		ST3_img_grad			//14	// collecting only value channel. Adapt if full HSV required.
	//__global 	float8*		HSV_grad				//14
){
	uint	global_id_uint								= get_global_id(0);
	uint	lid											= get_local_id(0);
	uint	group_id									= get_group_id(0);
	const	uint local_size								= get_local_size(0);

	uint4	lookup_ref									= lookup_table[global_id_uint + lookup_table_offset];
	if( lookup_ref.w != global_id_uint){	printf("\n__kernel void patch_img_grad(..) lookup_ref.w %u != global_id_uint %u", lookup_ref.w, global_id_uint);	// NB return cols tha are outside img_cur, BUT only after initializing local mem.
											return;
	}
	uint	read_index									= lookup_ref.z;
	uint	u											= lookup_ref.x;														// read_column
	uint	v											= lookup_ref.y;														// read_row

	uint4	lookup_ref_layer							= lookup_table[lookup_table_offset].z;	//0;
	uint	layer_offset								= lookup_ref_layer.z;

	uint8	mipmap_params_ 								= mipmap_params[layer];
	uint	read_cols_									= mipmap_params_[MiM_READ_COLS];
	uint	read_rows_									= mipmap_params_[MiM_READ_ROWS];

	uint	mm_cols										= uint_params[MM_COLS];
	uint	mm_rows										= uint_params[MM_ROWS];
	uint	mm_pixels									= uint_params[MM_PIXELS];

	uint	stop_offset									= layer_offset + (read_rows_ -1) * mm_cols + read_cols_	;			// bottom right corner of source image layer
/*
// 	if(/ *global_id_uint* /lid==30)printf("\n\n__kernel void  patch_img_grad():  mm_cols=%u,  mm_rows=%u,  mm_pixels=%u   ST3_offset3=%u, %u, %u    stop_offset=%d,    read_index=%d \n", \
// 																				mm_cols,  	mm_rows, 		mm_pixels,  ST3_offset3.x, ST3_offset3.y, ST3_offset3.z, stop_offset, read_index );
// 	float4	blue		= {1,0,0,1};
// 	float4 	green		= {0,1,0,1};
// 	float4	red			= {0,0,1,1};
*/
	int		lfoff										= -(u >1);															//-(read_column != 0);
	int		rtoff										=  (u < read_cols_-2);												// (read_column < mm_cols-1);

	float4	Jacobian_pvt_arr[block_size][6]				= {{zero_f4}};
	float4	Hessian_pvt_arr[block_size][6][6]			= {{{zero_f4}}};													// pvt variable for values in this column.
	local_Hessian[lid]									= zero_f4;

	const uint SE3_offset		= SE3_offset3.s0;		//layer_offset/mm_cols;
	const uint ST3_offset		= ST3_offset3.s0;		//SE3_out_step_1 * (num_SE3_DoF + 1);// + layer_offset;

	const uint SE3_u_step		= SE3_offset3.s1;		// step between elements of the Hessian matrix
	const uint ST3_u_step		= ST3_offset3.s1;

	const uint SE3_v_step		= SE3_offset3.s2;
	const uint ST3_v_step		= ST3_offset3.s2;

	uint	write_index									= u/out_block_size			 + (v/out_block_size)*mm_cols	+ ST3_offset;
	uint	write_index_2								= u/block_size				 + (v/block_size)*mm_cols		+ SE3_offset;
	uint 	offset_1_1_max								= mm_cols * read_rows_ / out_block_size 		 			+ ST3_offset;

	for (uint row_in_block=0; (row_in_block<block_size)&&(read_index<=stop_offset&&read_index>0); row_in_block++, v++,  read_index +=mm_cols){	// stop offset prevents bottom row patches from overrunning the bottom of the image layer. // NB readindex may be 0 if not in range according to lookup table.
		int upoff										= -(v  >1 )*mm_cols;												//-(read_row  != 0)*mm_cols;	// up, down, left, right offsets, by boolean logic.
		int dnoff										=  (v  < read_rows_-2) * mm_cols;									// (read_row  < read_rows_-1) * mm_cols;

		float4 pu, pd, pl, pr;
		pr												=  img[read_index + rtoff];
		pl												=  img[read_index + lfoff];
		pu												=  img[read_index + upoff];
		pd												=  img[read_index + dnoff];

		float4 gx										= { (pr.x - pl.x)/2.0f,  (pr.y - pl.y)/2.0f,  (pr.z - pl.z)/2.0f,   0.5f };			// Signed img gradient in hsv
		float4 gy										= { (pd.x - pu.x)/2.0f,  (pd.y - pu.y)/2.0f,  (pd.z - pu.z)/2.0f,   0.5f };

		float	inv_depth								=  depth_map[read_index];
		float4	Jacobian[6]								=  {0};
/*
		// float4 edge_weight				= (1.0f - g1p_pvt[row_in_block].s3);  // NB any weighting needs to be folded into the Jacobian and Hessian.  // edge weighting may be better done by havig an edge list of pixels.to reduce computation.
*/
		for (uint i=0; i<6; i++) {
			float2	SE3_px								= SE3_map[read_index + i* mm_pixels];								// SE3_map[read_index + i* uint_params[MM_PIXELS]  ] = partial_gradient;  // float2 partial_gradient={u_flt-u2 , v_flt-v2}; // Find movement of pixel
			float4	gxSE3								= gx*SE3_px[0];
			float4	gySE3								= gy*SE3_px[1];
			Jacobian[i]									= gxSE3 + gySE3;	//gx*SE3_px[0]  + gy*SE3_px[1];					// J_SE3 * img gradient i.e. edges
			if(i>2){
				ST3_img_grad[read_index + (i-3)* mm_pixels] = Jacobian[i];  //(float2){gxSE3.x, gySE3.x};	// collecting full HSV.

				int4 J_notnan = !isnan(Jacobian[i]);
				if(  !isnan( inv_depth ) && J_notnan.x && J_notnan.y && J_notnan.z && J_notnan.w ){
					Jacobian[i]							*= inv_depth;														// ST3 depends on inv_depth
				}else{
					Jacobian[i]							= zero_f4;
				}
			}
			Jacobian[i].w								= 1.0f;
			SE3_grad_map[read_index + i* mm_pixels]	= Jacobian[i] ;														// float4

/*
// 																															if (lid==0 && row_in_block==0 && group_id==0 ){		// ### Debugging ###
// 																																printf("\ndebug  i=%d SE3_px =(%f,%f),  gx,gy=(%f,%f),  read_index=%d  row=%d,  col=%d",\
// 																																i,SE3_px.x,SE3_px.y, gx.x,gy.x,  read_index,  read_index/mm_cols,  read_index%mm_cols );
// 																															}
*/
		}
		for (uint i=0; i<6; i++) {
			Jacobian_pvt_arr[row_in_block][i]			= Jacobian[i];
/*
// 																															if (lid==0 && row_in_block==0 && group_id==0 ){		// ### Debugging ###
// 																																printf("\npixel 0, Jacobian[i%d]=%2.10f,   %f,   ",i, Jacobian[i].x, Jacobian[i].w);
// 																															}
*/
			for (uint j=0; j<6; j++) {
				Hessian_pvt_arr[row_in_block][i][j]		= Jacobian[i] * Jacobian[j];	// Gauss-Newton approx H = J.transpose * J  // TO DO compute and sum lower triangle only.
				Hessian_pvt_arr[row_in_block][i][j].w 	=1.0f;
/*
// 																															if (lid==0 && row_in_block==0 && group_id==0 ){		// ### Debugging ###
// 																																printf("   Hessian[i%d][j%d]=%2.10f,   ",i, j, Hessian_pinv_pvt_arr[row_in_block][i][j].x );
// 																															}
*/
			}
		}
		/*	HSV_grad[] not currently in use. Would provide 8chan colourspace.
			float H 										= 0; img[read_index][0] * 2*M_PI_F;
			float S 										= 0; img[read_index][1];
			float V 										= 0; img[read_index][2];
			float8 temp_float8								= { sin(H) , cos(H), S, V, gx[1], gy[1], gx[2], gy[2] };			// HSV_grad = { sin(H) , cos(H), S, V, gx[1], gy[1], gx[2], gy[2] };
			HSV_grad[read_index]							= temp_float8;
		*/
	} // end of column of this patch.

	// Sum-reduce image, /////////////  Save intermediate size ST3 patches for depth map updates, and maximally reduced SE3 patches for pose updates. Second reduce_patch_Rho(..) kernel required for SE3 from lareger image pyramid layers, before update_k2k(..) kernel.
	// make this a device function ?

	uint past_frame_idx = 0; // TO DO remove and restore long outer loop.
	uint step;

	for ( step=1; step<block_size; step *=2){																																					// for each step size, (multiples of 2)
		for (uint block_row=0; block_row<block_size ; block_row += step){																														// step through rows in column
			for (uint i=0; i<6; i++) {
																						Jacobian_pvt_arr[		block_row][i]						+=Jacobian_pvt_arr[		block_row + step ][i];
				for (uint j=0; j<6; j++) {
																						Hessian_pvt_arr[	block_row][i][j]					+=Hessian_pvt_arr[	block_row + step ][i][j];  //+ se3_dim*block_size ];
				}
			}
			// First reduce the Jacobian, using the Hessian local memory.
			if( !(fmod((float)lid,(step*2))==0) &&  (fmod((float)lid,step)==0)    ){																											// selects 2nd column, sends data
				for (uint i=0; i<6; i++) {
																						local_Hessian[		lid-step + i*local_size]				= Jacobian_pvt_arr[		block_row][i];
				}
			}
			barrier(CLK_LOCAL_MEM_FENCE );																																						// Using barrier as a semaphore, for local mem messages between threads. This minimizes local_mem req, while allowing 2 patch sizes in output, full & ST3 map at out_block_size.

			if( (fmod((float)lid,(step*2))==0)  ){																																				// selects 1st column, adds data. Sum of patch now held in top left element of patch.
				for (uint i=0; i<6; i++) {
																						Jacobian_pvt_arr[		block_row][i]						+= local_Hessian[		lid + i*local_size];	//if(/*lid==0 &&*/ i==0) printf("\nstep=%d, J block_row=%d, lid=%d .x=%f, .w=%f",\
																																																	//					step, block_row, lid, Jacobian_pvt_arr[ block_row][i].x, Jacobian_pvt_arr[	block_row][i].w);
				}
			}
			barrier(CLK_LOCAL_MEM_FENCE );																																						// Using barrier as a semaphore, for local mem messages between threads. This minimizes local_mem req, while allowing 2 patch sizes in output, full & ST3 map at out_block_size.

			// Now Reduce Hessian
			if( !(fmod((float)lid,(step*2))==0) &&  (fmod((float)lid,step)==0)    ){																											// selects 2nd column, sends data
				for (uint i=0; i<6; i++) {
					for (uint j=0; j<6; j++) {
																						local_Hessian[		lid-step + (i*6 + j)*local_size]		= Hessian_pvt_arr[	block_row ][i][j];	//+ se3_dim*block_size ];  TO DO correct size and indexing of local_Hessiasn
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
			bool inbounds			= false;
			uint offset_1_1			= 0;

			for (uint block_row=0; block_row < block_size ; block_row += step*2, write_block_row++){
				inbounds																= false;
				if ( (write_index + write_block_row*mm_cols) < offset_1_1_max) inbounds	= true;
				for (uint i=0; i<3; i++) {																																						// select only ST3
					if( fmod((float)lid,out_block_size) == 0  ){																																// write Jacobian to 2nd page of SE3_Hessian_pinv_map buffer.
																						offset_1_1 									= write_index		+ i*ST3_v_step	+ mm_pixels + write_block_row*mm_cols;
																						float4	pvt_Jacobian 						= Jacobian_pvt_arr[		block_row][i];
																						if( inbounds == true ){
																							SE3_Hessian_map[	offset_1_1 ]	= pvt_Jacobian;
																							//if ( layer >4) printf("\n__kernel void  patch_img_grad(..) layer=%u,  i=%u,  global_id_uint=%u,  block_row=%u,  offset_1_1=%u", layer, i, global_id_uint, block_row, offset_1_1 );
																						}
					}
					barrier(CLK_GLOBAL_MEM_FENCE );			// TO DO is this needed?
					for (uint j=0; j<3; j++) {																																					//float4	debug						= {(float)(i)/3, (float)(j)/3, global_id_uint, 1 };
						if( /*inbounds == true*/ fmod((float)lid,out_block_size) == 0 ){																															// write Hessian to 1st page of SE3_Hessian_pinv_map buffer.		// selects columns i.e. threads within the workgroup
																						offset_1_1									= write_index		+ i*ST3_v_step	+ j*ST3_u_step + write_block_row*mm_cols;
																						float4	pvt_Hessian 						= Hessian_pvt_arr[	block_row ][i][j];
																						if( inbounds == true ){
																							SE3_Hessian_map[	offset_1_1 ]	= pvt_Hessian;													// Hessian_pinv_pvt_arr[	block_row ][i][j] / Hessian_pinv_pvt_arr[	block_row ][i][j].w;
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

	if( fmod((float)lid,block_size) == 0 ){																																						// selects columns i.e. threads within the workgroup
		for (uint i=0; i<num_SE3_DoF; i++) {																																					// write Jacobian to 2nd page of SE3_Hessian_pinv_map buffer.		// All 6 DoF of SE3
																						offset_2 								= write_index_2		+ i*SE3_v_step	+ mm_pixels;				// if(global_id_uint==0){printf("\n J offset_2=%d, SE3_DoF = %d  row=%d, col=%d,   .x=%f    .w=%f",\
																																																// offset_2, i, offset_2/mm_cols,  offset_2-mm_cols*(offset_2/mm_cols),   Jacobian_pvt_arr[	block_row][i].x, Jacobian_pvt_arr[	block_row][i].w  );}
																						float4	pvt_Jacobian 					= Jacobian_pvt_arr[		block_row][i];
																						SE3_Hessian_map[	offset_2 ]			= pvt_Jacobian;
			for (uint j=0; j<num_SE3_DoF; j++) {																																				// write Hessian to 1st page of SE3_Hessian_pinv_map buffer.
																						offset_2 								= write_index_2		+ i*SE3_v_step	+ j*SE3_u_step;				// if(global_id_uint==0){printf("\n H offset_2=%d, SE3_DoF = %d,%d  row=%d, col=%d ",offset_2, i,j, offset_2/mm_cols,  offset_2-mm_cols*(offset_2/mm_cols)    );}
																						float4	pvt_Hessian 					= Hessian_pvt_arr[	block_row ][i][j];
																						SE3_Hessian_map[	offset_2 ]			= pvt_Hessian;													// Hessian_pinv_pvt_arr[	block_row ][i][j] / Hessian_pinv_pvt_arr[	block_row ][i][j].w;

																						float4		debug 						= {(float)lid, global_id_uint, group_id, 1.0f};
																						SE3_Hessian_map[read_index]				= debug;
			}
		}
	}
/*
	// debugging...
	//barrier(CLK_GLOBAL_MEM_FENCE );

	//SE3_Hessian_map[old_read_index + 502*mm_cols]			= debug;

	//if( fmod((float)lid,block_size) == 0 ){
		//SE3_Hessian_map[	frame_offset_1	+ 400  / *write_block_row*mm_cols* /	]	= green;
		//SE3_Hessian_map[	offset_2		+ 100						]	= blue; //tag4;
		//SE3_Hessian_map[	write_index_2 	+ 200						]	= red;

		//printf("\n__kernel void  patch_img_grad, global_id_u=%u,	lid=%u,	read_index=%u,	step=%u,	SE3_out_step_1=%u,	SE3_out_step_2=%u,	SE3_out_step_5=%u,	offset_2=%u,	write_index_2=%u,	read_cols_=%u,	read_rows_=%u",\
												 global_id_uint,	lid, 	read_index, 	step, 		SE3_out_step_1, 	SE3_out_step_2, 	SE3_out_step_3, 	offset_2,		write_index_2,		read_cols_,		read_rows_ );
	//}
	//SE3_Hessian_map[	write_index_2 	+ 300						]	= red;
	//barrier(CLK_GLOBAL_MEM_FENCE );
*/
}


__kernel void  patch_hessian_reduce(						// one workgroup per element.
	//Inputs:
	__private	uint		start_idx,						//0
	__private	uint		layer,							//1
	__private	uint		cols,							//2
	__private	uint		rows,							//3
	__private	uint		mm_cols,						//4
	__private	uint		elem,							//5
	__private	uint		mm_pixels,						//6

	__global 	float4*		SE3_Hessian_map					//7		// NB mean of pixel-wise Hessian pseudo-inverse.
){
	uint					lid						= get_local_id(0);
	uint 					index					= lid + start_idx;
	float4					H_pvt_arr[ block_size]	= {zero_f4};											// pvt variable for values in this column.
	float4					J_pvt_arr[ block_size]	= {zero_f4};											// pvt variable for values in this column.

	__local float4			local_msg[32];			local_msg[lid]	= zero_f4;
	if(lid>cols) return;																					// NB if(lid>cols) return;  in case there are partial patches. Will produce a column extra, which will be 0.000 otherwise.

	// Hesssian reduction	//////////////////////////////////////////////////////////////////////
	// load pvt array
	for( int i=0; i<=rows; i++ ){	H_pvt_arr[i]		+= SE3_Hessian_map[index + i*mm_cols]; }				// NB (i<=rows) and  in case there are partial patches. Will produce a row extra, which will be 0.000 otherwise.

	// Reduction
	uint step				= 2;
	uint old_step			= 1;
	const uint iter			= ceil( log2((float)cols) );
																											//if(get_global_id(0)==0){ printf("A      __kernel void  patch_hessian_reduce()  start_idx=%d, layer=%d, cols=%d, rows=%d, iter=%d,  2^iter=%f", start_idx, layer, cols, rows, iter, pown(2.0f,(int)iter) ); }
	for ( step=1; step<cols; step *=2){																																				// for each step size, (multiples of 2)
		for (uint block_row=0; block_row<block_size ; block_row += step){																											// step through rows in column
																						H_pvt_arr[	block_row]		+=H_pvt_arr[	block_row + step ];

			if( !(fmod((float)lid,(step*2))==0) &&  (fmod((float)lid,step)==0)	){		local_msg[		lid-step ]	= H_pvt_arr[	block_row ];	}									// selects 2nd column, sends data
			barrier(CLK_LOCAL_MEM_FENCE );																																			// Using barrier as a semaphore, for local mem messages between threads. This minimizes local_mem req, while allowing 2 patch sizes in output, full & ST3 map at out_block_size.

			if( (fmod((float)lid,(step*2))==0)  ){										H_pvt_arr[	block_row]		+= local_msg[	lid ];	}										// selects 1st column, adds data. Sum of patch now held in top left element of patch.
			barrier(CLK_LOCAL_MEM_FENCE );
		}
	}
	// Write result
	if( /*fmod((float)lid, step)*/				lid	==0 ) {
		int idx = elem + 6 +(layer*8*6);																	//	Each layer's results spaced by 8*6=48 pixels.
		SE3_Hessian_map[idx]					=	H_pvt_arr[0] ;											//TO DO will need atomic fn for larger images.
																											//printf("\n__kernel void  patch_hessian_reduce()  layer=%u		SE3_Hessian  %u [%i]		=%f, %f, %f, %f",  layer,  elem,	idx,	pvt_Hessian.s0,  pvt_Hessian.s1,  pvt_Hessian.s2,  pvt_Hessian.s3 );
	}

	// Jacobian reduction	//////////////////////////////////////////////////////////////////////////
																											// i = st3,  ST3_v_step = ST3_offset3.s2,
																											// offset_1_1	= write_index		+ i*ST3_v_step	+ mm_pixels + write_block_row*mm_cols;
	if ( !(elem==0 || elem==6 || elem==12 || elem==18 || elem==24 || elem==30 ) ) return;					// Only these elements remain.
	local_msg[lid]	= zero_f4;
	// load pvt array
	for( int i=0; i<=rows; i++ ){
		J_pvt_arr[i]		+= SE3_Hessian_map[index + i*mm_cols + mm_pixels];
																																													// if(i<4 && lid<4){printf("\nB        __kernel void  patch_hessian_reduce()  layer=%u	   SE3_Jacobian  elem/6 = %u 	[i=%d] =	%f, %f, %f, %f	",
																																													//		layer,  					elem/6,		i,		J_pvt_arr[i].s0,  J_pvt_arr[i].s1,  J_pvt_arr[i].s2,  J_pvt_arr[i].s3 );
																																													//	}
	}																										// NB (i<=rows) and  in case there are partial patches. Will produce a row extra, which will be 0.000 otherwise.

	// Reduction
	step 		= 2;
	old_step 	= 1;
																																													//if( lid==0){	printf("\nC ");}//step = %d,   ", step); }
	for ( step=1; step<cols; step *=2){																														// for each step size, (multiples of 2)
																																													//if(lid==0){	printf("D ");}//step = %d,   ", step); }
		for (uint block_row=0; block_row<block_size ; block_row += step){																					// step through rows in column
																						J_pvt_arr[	block_row]		+=J_pvt_arr[	block_row + step ];
			if( !(fmod((float)lid,(step*2))==0) &&  (fmod((float)lid,step)==0)	){		local_msg[		lid-step ]	= J_pvt_arr[	block_row ];	}		// selects 2nd column, sends data
			barrier(CLK_LOCAL_MEM_FENCE );																													// Using barrier as a semaphore, for local mem messages between threads. This minimizes local_mem req, while allowing 2 patch sizes in output, full & ST3 map at out_block_size.

			if( (fmod((float)lid,(step*2))==0)  ){										J_pvt_arr[	block_row]		+= local_msg[	lid ];	}				// selects 1st column, adds data. Sum of patch now held in top left element of patch.
			barrier(CLK_LOCAL_MEM_FENCE );
																																													// if(block_row==0 && lid==0){	printf("\nD        __kernel void  patch_hessian_reduce() step = %d,  block_row=%d,   layer=%u	SE3_Jacobian  elem/6 = %u [block_row=%d]	=	%f, %f, %f, %f	",
																																													//	step,  block_row, 			layer,  				elem/6,		block_row,  	J_pvt_arr[block_row].s0,  J_pvt_arr[block_row].s1,  J_pvt_arr[block_row].s2,  J_pvt_arr[block_row].s3 );
																																													// }
		}
	}
	// Write result
	if( /*fmod((float)lid, step)*/				lid	==0 ) {													//	Each layer's results spaced by 8*6=48 pixels.
		int idx = (elem/6) + (layer*8*6);
		SE3_Hessian_map[idx]					=	J_pvt_arr[0] ;											//TO DO will need atomic fn for larger images.							//	Write result to the first 36 pixels of SE3_Hessian_map, directly above this layer of hessian pyramid,  because this will be fastest to read to CPU.
																																													// printf("\nE        __kernel void  patch_hessian_reduce()  layer=%u	SE3_Jacobian  elem/6 = %u [idx = %i]	=	%f, %f, %f, %f		lid=%d	gid=%lu",
																																													//		layer,  				elem/6,		idx,  				J_pvt_arr[0].s0,  J_pvt_arr[0].s1,  J_pvt_arr[0].s2,  J_pvt_arr[0].s3,	lid,  get_global_id(0) );
	}
}

/*__kernel void  compute_SO3_Hessian_lookup_table(

){


}
*/


// 	// Sample ST3
// 	float4 ST3[3][6]	= {{0}};
// 	ST3[0][3]			= 1;
// 	ST3[1][4]			= 1;
// 	ST3[2][5]			= 1;

/*  // NB  Gauss-Newton approx  H = J.T * J  has det=0, so is NOT invertible. Must use H.pinv()

void Gauss_Jordan_elimination_3x3( // invert hessians for (1) ST3+rot (4x4) depth and rel_vel_map,  (2) SO3 (6x6) global camera  (3) camera intrinsic matrix (4) lens distortion
	float ST3[3][6],
	bool symmetrical[3]
){
														uint row_list[3]	= {0,1,2};
	for (uint i=0; i<3; i++)								symmetrical[i]	= false;
	for (uint i=0; i<3; i++){
															uint row 		= row_list[i];
		for (uint j = i+1; j<3; j++){																// select the row with the largest value on the diagonal.
			if ( fabs(ST3[row][row]) < fabs(ST3[row_list[j]][row_list[j]] ) ) {
															uint temp 		= row_list[j];
															row_list[j] 	= row_list[i];
															row_list[i]		= temp;
			};
		}
		if ( !isnormal(ST3[row][row]) ) 			symmetrical[row] 		= true;					// can't fit wrt parameter where the image is symmetrical

		for (uint row_j = i+1; row_j<3; row_j++){
													float multiplier 		=  ST3[row_list[row_j]][i] / ST3[row][i];
			for (uint k = i; k<3  ; k++){
													ST3[row_list[row_j]][k]	-=  multiplier * ST3[row][k];
			}
		}
	}
}

void Gauss_Jordan_elimination_6x6( // invert hessians for (1) ST3+rot (4x4) depth and rel_vel_map,  (2) SO3 (6x6) global camera  (3) camera intrinsic matrix (4) lens distortion
	float ST3[6][12],
	bool symmetrical[6]
){
									uint stop				= 6;
									uint row_list[6]		=  {0,1,2,3,4,5};
	for (uint i=0; i<6; i++){													// NB given the Gauss-Newton approximation:  H = J^t * J,  the expected cause of non-invertible H is 0 values in J.
										stop--;
										symmetrical[i]		=  false;			// This causes 0 val col & row for the affected parameter.
		if (ST3[0][i]==0){														// Solution: move 0 val rows to the bottom of the row_list -> sub-H of the non-zero rows & cols.
										symmetrical[i]		=  true;			// NB Must use row_list for accessing both rows and cols

			for (uint j=i; j<5; j++){											// symmetrical[] records which params have J[param]=0, ie the image is smmetrical wrt that parameter.
										row_list[j] 		= row_list[j+1];
			}
										row_list[5] 		= i;
		}
	}

	for (uint i=0; i<stop ; i++){
													uint row 							=  row_list[i];

// 		for (uint j = i+1; j<6; j++){																// select the row with the largest value on the diagonal.
// 			if ( fabs(ST3[row][row]) < fabs(ST3[row_list[j]][row_list[j]] ) ) {
// 															uint temp 		= row_list[j];
// 															row_list[j] 	= row_list[i];
// 															row_list[i]		= temp;
// 			};
// 		}
// 		if ( !isnormal(ST3[row][row]) ) 			symmetrical[row] 		= true;					// can't fit wrt parameter where the image is symmetrical

		for (uint row_j = i+1; row_j<stop; row_j++){
													float multiplier 					=  ST3[row_list[row_j]][row] / ST3[row][row];
			for (uint k = i; k<stop  ; k++){
													ST3[row_list[row_j]][row_list[k]]	-=  multiplier * ST3[row][row_list[k]];
			}
		}
	}
}
*/

void Hessian_inv_Cayley_Hamilton_3x3( float Hessian[9], float Hinv[9] ){
#define a	0
#define b	1
#define c	2
#define d	3
#define e	4
#define f	5
#define g	6
#define h	7
//#define i	8	// NB "i" is used asa variable.

#define A	0
#define D	1
#define G	2
#define B	3
#define E	4
#define H	5
#define C	6
#define F	7
#define I	8

	Hinv[A]	=	( Hessian[e]*Hessian[I] - Hessian[f]*Hessian[h] );
	Hinv[D]	=  -( Hessian[d]*Hessian[I] - Hessian[f]*Hessian[g] );
	Hinv[G]	=	( Hessian[d]*Hessian[h] - Hessian[e]*Hessian[g] );

	Hinv[B]	=  -( Hessian[b]*Hessian[I] - Hessian[c]*Hessian[h] );
	Hinv[E]	=	( Hessian[a]*Hessian[I] - Hessian[b]*Hessian[g] );
	Hinv[H]	=  -( Hessian[a]*Hessian[h] - Hessian[b]*Hessian[g] );

	Hinv[C]	=	( Hessian[b]*Hessian[f] - Hessian[c]*Hessian[e] );
	Hinv[F]	=  -( Hessian[a]*Hessian[f] - Hessian[c]*Hessian[d] );
	Hinv[I]	=	( Hessian[a]*Hessian[e] - Hessian[b]*Hessian[d] );

	float Det							= Hessian[a]*Hinv[A]		+ Hessian[b]*Hinv[D]		+ Hessian[c]*Hinv[G];

	if (fabs(Det) < 0.01f){	Det			= 0.01f;	}				// Prevent divison by zero, as per DISOpticalFlow.

	for(int i=0; i<9; i++){	Hinv[i]		/=Det;		}

#undef a
#undef b
#undef c
#undef d
#undef e
#undef f
#undef g
#undef h

#undef A
#undef B
#undef C
#undef D
#undef E
#undef F
#undef G
#undef H
#undef I

}


void GN_Hessian_pseudo_inv_3x3( float J[3], float H_pinv[9]			// NB only valid for Real valued H = J^T * J,  NB Not for a sum of multiple Hessians.
){
	float denom = J[0]*J[0] + J[1]*J[1] + J[2]*J[2];  				//TO DO check is it sum of sqares of J or sum of squares of diagonal of H ?
	denom		*= denom;

	if ( isnormal(denom) ){ denom = 1/denom; } else { denom = 0; }	// prevent div by zero error

	for (uint i=0; i<3; i++){
		for (uint j=0; j<3; j++){
			H_pinv[i*3 + j] 	= denom * J[i] * J[j];
		}
	}
}


void GN_Hessian_pseudo_inv_6x6( float J[6], float H_pinv[36]		// NB only valid for Real valued H = J^T * J,  NB Not for a sum of multiple Hessians.
){
	float denom = 0;
	for (uint i=0; i<6; i++){ denom += J[i]*J[i]; }
	denom		*= denom;

	if ( isnormal(denom) ){ denom = 1/denom; } else { denom = 0; }	// prevent div by zero error

	for (uint i=0; i<6; i++){
		for (uint j=0; j<6; j++){
			H_pinv[i*6 + j] 	= denom * J[i] * J[j];
		}
	}
}



__kernel void  patch_Inverse_Compositional_update(  // (1) for depth & rel_vel_map,  (2) SO3 camera pose,  (3) camera intrinsic matrix (4) lens distortion

){

}


