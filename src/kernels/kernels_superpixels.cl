#include "kernels__macros.h"
#include "kernels.h"


// find lowst gradient in 3x3 region for initial cluster centre
__kernel void initiate_cluster_centres(
	//Inputs:
	__private	uint	layer,					//0
	__private	int		cluster_dim,			//1

	__constant	uint8*	mipmap_params,			//2
	__constant	uint*	uint_params,			//3

	__global	float2*	gradient_map,			//4		img_size * sizeof(float2)

	//Output
	__global	int4*	cluster_centers			//5		(img size / cluster_dim^2) * sizeof(uint4)
){
	int 	global_id_u				= get_global_id(0);
	float	global_id_flt			= global_id_u;

	uint8	mipmap_params_			= mipmap_params[layer];
	if( global_id_u<50 ){
		printf("\n__kernel void initiate_cluster_centres()_1, global_id_u=%u",
																global_id_u );
	}
	if (global_id_u	* cluster_dim * cluster_dim	>= mipmap_params_[MiM_PIXELS]) return;

	int	mm_cols						= uint_params[MM_COLS];
	int	read_offset_				= mipmap_params_[MiM_READ_OFFSET];
	int	read_cols_					= mipmap_params_[MiM_READ_COLS];
	int	sample_cols					= read_cols_ / cluster_dim;
	int	read_rows_					= mipmap_params_[MiM_READ_ROWS];

	int	v							= cluster_dim * (global_id_u / sample_cols )	+ (cluster_dim/2)	-1;			// read_row
	int	u							= cluster_dim * (global_id_u % sample_cols)		+ (cluster_dim/2)	-1;			// read_column
	int	idx							= read_offset_ + u + v * mm_cols	* cluster_dim;
	if( global_id_u<50 ){
		printf("\n__kernel void initiate_cluster_centres()_2, global_id_u=%d, u=%d, v=%d,  cluster_dim=%d,  (global_id_u / sample_cols )=%d,  (cluster_dim/2)=%d,		idx=%d",
																global_id_u,	u,	v,		cluster_dim,	(global_id_u / sample_cols ),		(cluster_dim/2),		idx );
	}
	if (u>read_cols_ ||v>read_rows_ ) return;

	float	min_pix_grad			= FLT_MAX/4;
	int		centre_idx				= 0;

	for(int i=0; i<3; i++){
		for(int j=0; j<3; j++){
			float	px_grad			= fast_length( gradient_map[idx] );
			if( px_grad < min_pix_grad){
				min_pix_grad		= px_grad;
				centre_idx			= idx;
			}
			idx++;
		}idx 						+= (mm_cols*cluster_dim - 3);
	}
	int4	cluster_centre;
	cluster_centre.z				= centre_idx;												// pixel index of cluster centre
	centre_idx						-= read_offset_;
	cluster_centre.x				= centre_idx % mm_cols;										// u coord of centre
	cluster_centre.y				= centre_idx / mm_cols;										// v coord of centre
	cluster_centre.w				= global_id_u;												// index of cluster

	cluster_centers[global_id_u]	= cluster_centre;

// 	if( global_id_u<1000){
// 		printf(" %u,", global_id_u);
// 	}
	if( global_id_u<50 ){
		printf("\n__kernel void initiate_cluster_centres()_3,  global_id_u=%d, cluster_centre=%d, %d,	%d, %d,		idx=%d,	read_offset_=%d,	u=%d, v=%d, mm_cols=%u, cluster_dim=%d,		centre_idx=%d",\
			global_id_u, cluster_centre.x,cluster_centre.y,cluster_centre.z,cluster_centre.w,\
			idx, read_offset_, u, v, mm_cols, cluster_dim, centre_idx	\
		);
	}
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
	__private	uint	lookup_table_offset,	//2
	__private	uint	cluster_dim,			//3
	__private	uint	cols_of_clusters,		//4
	__private	uint	mm_cols,				//5

	__global	uint4*	lookup_table,			//6
	__global	float4*	img,					//7		img_size * sizeof(float4)
	__global	int4*	cluster_centers,		//8		(img size / cluster_dim^2) * sizeof(uint4)

	//Output
	__global	float*	cluster_map				//9	 /float* //	img_size * sizeof(float4)   densely packed for one layer.  Need a layer offset.
){
	uint 	global_id_u							=	get_global_id(0);
	float	global_id_flt						=	global_id_u;

	uint4	lookup_ref							=	lookup_table[global_id_u + lookup_table_offset];
	if(lookup_ref.w != global_id_u){			return;}

	uint	read_index							=	lookup_ref.z;
	uint	u									=	lookup_ref.x;												// read_column
	uint	v									=	lookup_ref.y;												// read_row
	uint	cluster_offset						=	u/cluster_dim	+ cols_of_clusters*(v/cluster_dim);			// cluster offset withn this image layer

	uint4	lookup_ref_layer					=	lookup_table[lookup_table_offset].z;						//0;
	uint	layer_offset						=	lookup_ref_layer.z;

	float2	cluster_centres_pvt[9];
	float4	cluster_colour[9];
	uint	cluster_idx[9];

	for( int i_j=0; i_j<9; i_j++ ){
		cluster_centres_pvt[i_j]				=	max_f2;
		cluster_colour[i_j]						=	max_0_f4;
		cluster_idx[i_j]						=	UINT_MAX/4;
	}

	int		left_edge							= cluster_offset		%cols_of_clusters;						// will be 0 if true, >0 if falae
	int		right_edge							= (cluster_offset+1)	%cols_of_clusters;

	if( global_id_u==0 ){printf("\n\n__kernel void associate_pixels()_0, cluster_layer_offset=%u,	num_clusters=%u,	lookup_table_offset=%u, 	cluster_dim=%u,  	cols_of_clusters=%u,	mm_cols=%u \n", \
																		cluster_layer_offset, 		num_clusters, 		lookup_table_offset,		cluster_dim,		cols_of_clusters,		mm_cols ); \
	}
																												// NB order of integer arrithmetic.
	for(int i=-3; i<6; i+=3){																					// initialize centre pixels
		for(int j=-1; j<2; j++){
			int		i_j							=	i + j + 4;													// i_j is the offset within the pvt arrays [9].
			int		offset						=	cluster_offset +  i*cols_of_clusters + j;					// Where to sample clusters within this image layer
			if( global_id_u==0 ){
					printf("\n__kernel void associate_pixels()_1 i= %d,	j= %d,	i_j= %d,	cluster_offset= %d,		i*cols_of_clusters= %d,	offset= %d,	(offset>=0)= %d,	(offset<num_clusters)= %d,	( (offset>=0) && (offset<num_clusters) )= %d",\
																i,		j,		i_j,		cluster_offset,		 	i*cols_of_clusters,		offset,		(offset>=0),		(offset<num_clusters),		( (offset>=0) && (offset<num_clusters) ) );
			}
			if( (offset>=0) && (offset<(int)num_clusters) && left_edge+j>=0  &&  right_edge+j>0 ) {
				int4	centre					=	cluster_centers[ offset + cluster_layer_offset ];
				float2	centre_uv				=	{ (float)centre.x, (float)centre.y};
				if( global_id_u==0 ){
					printf("\n ofset= %d,	centre= %d, %d, %d, %d,		centre_uv= %f, %f,		i_j= %d,		offset+cluster_layer_offset= %d", \
						offset,	centre.x,centre.y,centre.z,centre.w,	centre_uv.x,centre_uv.y, i_j,			offset+cluster_layer_offset);
				}
				cluster_centres_pvt[	i_j]	=	centre_uv;													// float2	cluster_centres_pvt[9]
				cluster_colour[			i_j]	=	img[centre.z];												// float4	cluster_colour[9]
				cluster_idx[			i_j]	=	centre.w;													// uint		cluster_idx[9]
			}
			if( global_id_u==0 ){ printf("\n cluster_centres_pvt[%d]= %f, %f",
											i_j,	cluster_centres_pvt[i_j].x, cluster_centres_pvt[i_j].y );
			}
		}
	}
	if( global_id_u==0 ){
		printf("\n\n num_clusters=%d", num_clusters);
		for(uint iter=0; iter<9; iter++){
			printf("\n cluster_centres_pvt[%u]= %f, %f,	cluster_colour[]= %f, %f, %f, %f,	cluster_idx[]= %u ",\
				iter, cluster_centres_pvt[iter].x, cluster_centres_pvt[iter].y, \
				cluster_colour[iter].x,cluster_colour[iter].y,cluster_colour[iter].z,cluster_colour[iter].w,\
				cluster_idx[iter] );
		}
	}

	float 	geometric_normalizer				=	cluster_dim;

	uint row_of_clusters						=	0;															// Within pvt arrays [9]
	cluster_offset++;
	for(int row=0; row<block_size; row+=cluster_dim, row_of_clusters++, cluster_offset++){						// for cluster rows in img patch
		if( global_id_u==0 ){printf("\n\n");}

		float 	colour_normalizer				=	0.0f;
		float	counter							=	0.0f;
		for(uint i=0; i<9; i+=3){																				// compute colour normalizer wrt adjacent clusters.
			for(uint j=0; j<3; j++){
				float length					=	fast_length( cluster_colour[i + j] - cluster_colour[4] );
				if (length <= sqrt_3){
					colour_normalizer			+=	length;
					counter						++;
				}
				if( global_id_u==0 ){
					printf("\n__kernel void associate_pixels()_2 row= %u,	i= %u,	j= %u,	colour_normalizer= %f",\
																	row,	i,		j,		colour_normalizer);
				}
			}
		} colour_normalizer						/=	counter;
		if( global_id_u==0 ){
					printf("\n__kernel void associate_pixels()_3	row= %u,	final colour_normalizer= %f,	geometric_normalizer= %f \n",\
																	row,			colour_normalizer,			geometric_normalizer);
		}
		if( global_id_u==0 ){ printf("\n\n"); }

		for(int cluster_row=0; cluster_row<cluster_dim; cluster_row++, read_index+=mm_cols, v++){				// for pixel rows in cluster
			float	closest_centre_dist			=	FLT_MAX/4;
			uint	closest_centre_id			=	0;
			float2	px_uv						=	{ (float)u, (float)(v + row + cluster_row) };
			float4	px_colour					=	img[read_index] ;
			if( global_id_u==0 ){
						printf("\n__kernel void associate_pixels()_4	row= %u,	cluster_row= %u,	read_index= %u, 	px_uv= %f, %f,		px_colour= %f, %f, %f, %f \n",\
																		row,		cluster_row,		read_index,			px_uv.x,px_uv.y,	px_colour.x,px_colour.y,px_colour.z,px_colour.w );
			}
			for(int i=0; i<9; i+=3){																			// for offset i_j in pvt array [9] of adjacent clusters.
				for(int j=0; j<3; j++){
					int	i_j						=	i+j;
					float dist					=	pixel_distance(	cluster_centres_pvt[i_j],	px_uv,	cluster_colour[i_j], px_colour, geometric_normalizer, colour_normalizer );
					if( global_id_u==0 ){
						printf("\n__kernel void associate_pixels()_5	row= %u, cluster_row= %u, read_index= %u,	i= %u, j= %u, cluster_centres_pvt[%d]= %f, %f,	cluster_colour[i_j]= %f, %f, %f, %f,	dist= %f,  closest_centre_dist= %f  ",\
								row, cluster_row, read_index,	i,	j,  i_j, cluster_centres_pvt[i_j].x,cluster_centres_pvt[i_j].y,		cluster_colour[i_j].x,cluster_colour[i_j].y,cluster_colour[i_j].z,cluster_colour[i_j].w, 	dist, closest_centre_dist );
					}

					if( dist < closest_centre_dist){
						closest_centre_id		=	i_j;
						closest_centre_dist		=	dist;
					}
				}
			}
			cluster_map[ read_index	 ]			=	global_id_u;	//(float)cluster_idx[ closest_centre_id ];
			if( global_id_u==0 ){
				printf("\n\n__kernel void associate_pixels()_6	row= %u,	cluster_row= %u,	read_index= %u,	cluster_idx[%u] = %u,	closest_centre_dist= %f \n",\
																row,		cluster_row,		read_index,		closest_centre_id, cluster_idx[ closest_centre_id ],  closest_centre_dist );
			}
																									// Could atomic write u,v pix coords + counter to a centres_buffer, to move centres.
		}
		// prepare pvt arrays with next row of clusters	/////////////////////////////////////////////////////////////
		int i									=	row_of_clusters % 3;
		for(uint j=0; j<3; j++){
			int		i_j							=	i+j;
			int		offset						=	cluster_offset +  i + j*cols_of_clusters;					// Where to sample clusters within this image layer
			if( offset<0 || offset>num_clusters ){	continue;}

			int4 	centre						=	cluster_centers[ offset + cluster_layer_offset ];
			float2	centre_uv					=	{(float)centre.x, (float)centre.y};

			cluster_centres_pvt[	i_j]		=	centre_uv;													// float2	cluster_centres_pvt[9]
			cluster_colour[			i_j]		=	img[centre.z];												// float4	cluster_colour[9]
			cluster_idx[			i_j]		=	centre.w;													// uint		cluster_idx[9]
		}
	}

}



__kernel void check_superpixel_continuity(



){




}


__kernel void update_cluster_centres_pvt(



){





}


