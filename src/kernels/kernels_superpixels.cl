#include "kernels__macros.h"
#include "kernels.h"


// find lowst gradient in 3x3 region for initial cluster centre
__kernel void initiate_cluster_centres(
	//Inputs:
	__private	uint	layer,					//0
	__private	uint	cluster_dim,			//1

	__constant	uint8*	mipmap_params,			//2
	__constant	uint*	uint_params,			//3

	__global	float2*	gradient_map,			//

	//Output
	__global	uint4*	cluster_centers			//

){
	uint 	global_id_u				= get_global_id(0);
	float	global_id_flt			= global_id_u;

	uint8	mipmap_params_			= mipmap_params[layer];
	if (global_id_u	* cluster_dim * cluster_dim	>= mipmap_params_[MiM_PIXELS]) return;

	uint	mm_cols					= uint_params[MM_COLS];
	uint	read_offset_			= mipmap_params_[MiM_READ_OFFSET];
	uint	read_cols_				= mipmap_params_[MiM_READ_COLS];
	uint	sample_cols				= read_cols_ / cluster_dim;
	uint	read_rows_				= mipmap_params_[MiM_READ_ROWS];

	uint	v						= cluster_dim * (global_id_u / sample_cols )	- (cluster_dim/2)	-1;			// read_row
	uint	u						= cluster_dim * (global_id_u % sample_cols)		- (cluster_dim/2)	-1;			// read_column
	uint	idx						= read_offset_ + u + v * mm_cols	* cluster_dim;
	if (u>read_cols_ ||v>read_rows_ ) return;

	float	min_pix_grad			= FLT_MAX/4;
	uint	centre_idx				= 0;

	for(int i=0; i<3; i++){
		for(int j=0; j<3; j++){
			float	px_grad			= fast_length( gradient_map[idx] );
			if( px_grad < min_pix_grad){
				min_pix_grad		= px_grad;
				centre_idx			= idx;
			}
			idx++;
		}idx += (mm_cols*cluster_dim - 3);
	}
	uint4	cluster_centre;
	cluster_centre.z				= centre_idx;
	centre_idx						-= read_offset_;
	cluster_centre.x				= centre_idx % mm_cols;
	cluster_centre.y				= centre_idx / mm_cols;
	cluster_centre.w				= 0;

	if(u=0)

	cluster_centers[global_id_u]	= cluster_centre;
}


float pixel_distance( float2 centre, float2 px_uv, float4 cluster_colour, float4 px_colour, float geometric_normalizer, float colour_normalizer ){
	float	geometric_dist			= fast_length( px_uv				- centre				);
	float	colour_dist				= fast_length( px_colour.xyz		- cluster_colour.xyz	);
	return							(geometric_dist*geometric_normalizer + colour_dist*colour_normalizer) ;
}



__kernel void associate_pixels(																	// Do one patch. preload centre pixels.
	//Inputs:
	__private	uint	layer,					//0
	__private	uint	block_size,				//1
	__private	uint	cluster_dim,			//2
	__constant	uint8*	mipmap_params,			//3
	__constant	uint*	uint_params,			//4

	__global	float4*	img,					//5
	__global	uint4*	cluster_centers,		//6

	//Output
	__global	float4*	cluster_map				//7
){
	uint 	global_id_u				= get_global_id(0);
	float	global_id_flt			= global_id_u;

	uint8	mipmap_params_			= mipmap_params[layer];
	if (global_id_u	* cluster_dim * cluster_dim	>= mipmap_params_[MiM_PIXELS]) return;

	uint	mm_cols					= uint_params[MM_COLS];
	uint	read_offset_			= mipmap_params_[MiM_READ_OFFSET];
	uint	read_cols_				= mipmap_params_[MiM_READ_COLS];
	uint	sample_cols				= read_cols_ / cluster_dim;
	uint	read_rows_				= mipmap_params_[MiM_READ_ROWS];

	uint	v						= (global_id_u / read_cols_ );									// read_row
	uint	u						= (global_id_u % read_cols_);									// read_column
	uint	idx						= read_offset_ + u + v * mm_cols	* cluster_dim;
	if (u>read_cols_ ||v>read_rows_ ) return;

	uint	cluster_centre_px_idx[9];
	float2	cluster_centres_pvt[9];
	float4	cluster_colour[9];

	float geometric_normalizer;
	float colour_normalizer;

	for(uint i=0; i<3; i++){																		// initialize centre pixels
		for(uint j=0; j<3; j++){
			uint4 centre						=	cluster_centers[ idx + i*mm_cols + j ];
			float2 centre_uv					=	{(float)centre.x, (float)centre.y};
			cluster_centre_px_idx[	i*3 + j]	=	centre.z;
			cluster_centres_pvt[	i*3 + j]	=	centre_uv;
			cluster_colour[			i*3 + j]	=	img[centre.z];
		}
	}

	for(int row=0; row<block_size ; row+=cluster_dim){												// for rows in patch

		for(int cluster_row=0; cluster_row<cluster_dim; cluster_row++, idx+=mm_cols){								// for rows in cluster
			float	closest_centre_dist			=	FLT_MAX/4;
			uint	closest_centre_id			=	0;
			float2	px_uv						=	{ (float)u, (float)(v + row + cluster_row) };
			float4	px_colour					=	img[idx] ;


			for(int i=0; i<3; i++){
				for(int j=0; j<3; j++){
					float dist				= pixel_distance(	cluster_centres_pvt[i*3 + j],	px_uv,	cluster_colour[	i*3 + j],	px_colour, geometric_normalizer, colour_normalizer );


				}
			}
		}

		// update cluster_centres_pvt[9];





	}

}


__kernel void (




}{






}
