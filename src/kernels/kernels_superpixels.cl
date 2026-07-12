#include "kernels__macros.h"
#include "kernels.h"


// find lowst gradient in 3x3 region for initial cluster centre
__kernel void initiate_cluster_centres(
	//Inputs:
	__private	uint	layer,					//0
	__private	uint	cluster_dim,			//1

	__constant	uint8*	mipmap_params,			//2
	__constant	uint*	uint_params,			//3

	__global	float2*	gradient_map,			//4		img_size * sizeof(float2)

	//Output
	__global	uint4*	cluster_centers			//5		(img size / cluster_dim^2) * sizeof(uint4)
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
	cluster_centre.z				= centre_idx;												// pixel index of cluster centre
	centre_idx						-= read_offset_;
	cluster_centre.x				= centre_idx % mm_cols;										// u coord of centre
	cluster_centre.y				= centre_idx / mm_cols;										// v coord of centre
	cluster_centre.w				= global_id_u;												// index of cluster

	cluster_centers[global_id_u]	= cluster_centre;
}


float pixel_distance( float2 centre, float2 px_uv, float4 cluster_colour, float4 px_colour, float geometric_normalizer, float colour_normalizer ){
	float	geometric_dist			= fast_length( px_uv				- centre				);
	float	colour_dist				= fast_length( px_colour.xyz		- cluster_colour.xyz	);
	return							(geometric_dist*geometric_normalizer + colour_dist*colour_normalizer) ;
}


__kernel void associate_pixels(																	// Do one patch. preload centre pixels.
	//Inputs:
	__private	uint	cluster_layer_offset,	//0
	__private	uint	num_clusters,			//1
	__private	uint	lookup_table_offset,	//1
	__private	uint	block_size,				//2
	__private	uint	cluster_dim,			//3
	__private	uint	cols_of_clusters,		//4
	__private	uint	mm_cols,				//5

	__global	uint4*	lookup_table,			//6
	__global	float4*	img,					//7		img_size * sizeof(float4)
	__global	uint4*	cluster_centers,		//8		(img size / cluster_dim^2) * sizeof(uint4)

	//Output
	__global	float4*	cluster_map				//9		img_size * sizeof(float4)   densely packed for one layer.  Need a layer offset.
){
	uint 	global_id_u							=	get_global_id(0);
	float	global_id_flt						=	global_id_u;

	float	null_factor = 1.0f;
	uint4	lookup_ref							=	lookup_table[global_id_u + lookup_table_offset];
	if(lookup_ref.w != global_id_u){null_factor =	0.0f;
	}
	uint	read_index							=	lookup_ref.z;
	uint	u									=	lookup_ref.x;												// read_column
	uint	v									=	lookup_ref.y;												// read_row
	uint	cluster_offset						=	u/cluster_dim	+ cols_of_clusters*(v/cluster_dim);

	uint4	lookup_ref_layer					=	lookup_table[lookup_table_offset].z;						//0;
	uint	layer_offset						=	lookup_ref_layer.z;

	uint	cluster_centre_px_idx[9]			=	{UINT_MAX};
	float2	cluster_centres_pvt[9]				=	{FLT_MAX};
	float4	cluster_colour[9]					=	{FLT_MAX};
	uint	cluster_idx[9]						=	{UINT_MAX};

	float 	geometric_normalizer				=	cluster_dim;
	float 	colour_normalizer;
																												// NB order of integer arrithmetic.

	for(int i=-3; i<6; i+=3){																					// initialize centre pixels
		for(int j=-1; j<2; j++){
			int		i_j							=	i + j;
			int		offset						=	cluster_offset +  i_j;
			if( offset<0 || offset>num_clusters ){
				continue;
			}
			uint4 centre						=	cluster_centers[ offset + cluster_layer_offset ];
			float2 centre_uv					=	{(float)centre.x, (float)centre.y};
			cluster_centre_px_idx[	i_j]		=	centre.z;													// pixel index of centre
			cluster_centres_pvt[	i_j]		=	centre_uv;													//
			cluster_colour[			i_j]		=	img[centre.z];												//
			cluster_idx[			i_j]		=	centre.w;													//
		}
	}

	for(uint i=0; i<9; i+=3){																					// initialize centre pixels
		for(uint j=0; j<3; j++){
			colour_normalizer					=	fast_length(cluster_colour[i + j] - cluster_colour[4] );
		}
	}
	colour_normalizer							/=	8;

	uint row_of_clusters						=	0;
	for(int row=0; row<block_size; row+=cluster_dim, row_of_clusters+=3){										// for rows in patch

		for(int cluster_row=0; cluster_row<cluster_dim; cluster_row++, read_index+=mm_cols){					// for rows in cluster
			float	closest_centre_dist			=	FLT_MAX/4;
			uint	closest_centre_id			=	0;
			float2	px_uv						=	{ (float)u, (float)(v + row + cluster_row) };
			float4	px_colour					=	img[read_index] ;

			for(int i=0; i<9; i+=3){
				for(int j=0; j<3; j++){
					float dist					=	pixel_distance(	cluster_centres_pvt[i + j],	px_uv,	cluster_colour[i + j],	px_colour, geometric_normalizer, colour_normalizer );
					if( dist < closest_centre_dist){
						closest_centre_id		=	i + j;
						dist					= 	closest_centre_dist;
					}
				}
			}
			cluster_map[ read_index ]			=	cluster_idx[ closest_centre_id ];
																												// Could atomic write u,v pix coords + counter to a centres_buffer, to move centres.
		}
		// next row of clusters

		for(uint j=0; j<3; j++){
			uint4	centre										=	cluster_centers[ read_index + /*row_of_clusters**/mm_cols + j ];
			float2	centre_uv									=	{(float)centre.x, (float)centre.y};
			cluster_centre_px_idx[	row_of_clusters + j]		=	centre.z;
			cluster_centres_pvt[	row_of_clusters + j]		=	centre_uv;
			cluster_colour[			row_of_clusters + j]		=	img[centre.z];
		}
	}
}



__kernel void check_superpixel_continuity(



){




}


__kernel void update_cluster_centres_pvt(



){





}
