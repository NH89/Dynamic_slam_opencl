#include "kernels_macros.h"
#include "kernels.h"

////////////////////////////////////////////////////////////////////////////////////////////////// Depth mapping //////////////////////////////////////////

__kernel void DepthCostVol(
	__private	uint	mipmap_layer,		//0
	__private	uint	k_verbosity,		//1
	__constant 	uint8*	mipmap_params,		//2
	__constant 	uint*	uint_params,		//3
	__global 	float*  fp32_params,		//4
	__global 	float16*k2k,				//5
	__global 	float8* base,				//6		keyframe_basemem
	__global 	float8* img,				//7		HSV_grad_mem/*imgmem*/ now float8
	__global 	float*  cdata,				//8
	__global 	float*  hdata,				//9
	__global 	float*  lo,					//10
	__global 	float*  hi,					//11
	__global 	float*  a,					//12
	__global 	float*  d,					//13
	__global 	float*  img_sum,			//14
	__global 	float8* cdata_8chan			//15
		 )
{
	uint global_id_u 	= get_global_id(0);
	float global_id_flt = global_id_u;

	uint8 mipmap_params_ = mipmap_params[mipmap_layer];									// choose correct layer of the mipmap
	uint read_offset_ 	= mipmap_params_[MiM_READ_OFFSET];
	uint read_cols_ 	= mipmap_params_[MiM_READ_COLS];
	uint read_rows_		= mipmap_params_[MiM_READ_ROWS];

	uint mm_cols		= uint_params[MM_COLS];
	uint reduction		= mm_cols/read_cols_;
	uint v    			= global_id_u / read_cols_;										// read_row
	uint u 				= fmod(global_id_flt, read_cols_);								// read_column
	float u_flt			= u * reduction;
	float v_flt			= v * reduction;
	uint read_index 	= read_offset_  +  v  * mm_cols  + u ;
	/////////////////////////////////////////////////////////////
	float8 B = base[read_index];	//B.x = base[read_index].x;	B.y = base[read_index].y;	B.z = base[read_index].z;		// pixel from keyframe

	int costvol_layers	= uint_params[COSTVOL_LAYERS];
	//int pixels 			= uint_params[PIXELS];
	uint mm_pixels		= uint_params[MM_PIXELS];
	float inv_d_step 	= fp32_params[INV_DEPTH_STEP];
	float min_inv_depth = fp32_params[MIN_INV_DEPTH];

	float 	u2,	v2, rho,	inv_depth=0.0f,	ns=0.0f,	mini=0.0f,	minv=3.0f,	maxv=0.0f;	// variables for the cost vol
	float8	rho_8chan;
	int 	int_u2, int_v2, coff_00, coff_01, coff_10, coff_11, cv_idx=read_index,	layer = 0;
	float8 	c;																			//, c_00, c_01, c_10, c_11;
	float 	c0 = cdata[cv_idx];															// cost for this elem of cost vol
	float 	w  = hdata[cv_idx];															// count of updates of this costvol element. w = 001 initially

	// layer zero, ////////////////////////////////////////////////////////////////////////////////////////
	// inf depth, rotation without paralax, i.e. reproj without translation.
	// Use depth=1 unit sphere, with rotational-preprojection matrix

	// precalculate depth-independent part of reprojection, h=homogeneous coords.
	float16 k2k_pvt		= k2k[0];
	float uh2 = k2k_pvt[0]*u_flt + k2k_pvt[1]*v_flt + k2k_pvt[2]*1;   // +k2k[3]/z
	float vh2 = k2k_pvt[4]*u_flt + k2k_pvt[5]*v_flt + k2k_pvt[6]*1;   // +k2k[7]/z
	float wh2 = k2k_pvt[8]*u_flt + k2k_pvt[9]*v_flt + k2k_pvt[10]*1;  // +k2k[11]/z
	//float h/z  = k2k[12]*u_flt + k2k_pvt[13]*v_flt + k2k_pvt[14]*1; // +k2k[15]/z
	float uh3, vh3, wh3;

	// cost volume loop  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	#define MAX_LAYERS 256 //64
	float cost[MAX_LAYERS];
	float8 cost_8chan[MAX_LAYERS];
	bool miss = false;
	bool in_image = false;
	if ( global_id_u  < mipmap_params_[MiM_PIXELS] ) in_image = true;
																	VK_MAPPING(\
																	 if(k_verbosity>0 && global_id_u == 1  )   {
																		printf("\n\n\n\n__kernel void DepthCostVol chk 0  (global_id_u == 1 ) :  reduction=%u,  layer=%u, read_offset_=%u, read_cols_=%u, read_rows_=%u,  read_index=%u,  \nk2k_pvt=[\n%f, %f, %f, %f,    \n%f, %f, %f, %f,    \n%f, %f, %f, %f,   \n%f, %f, %f, %f   ]\n", \
																		reduction, layer, read_offset_, read_cols_, read_rows_,  read_index,   k2k_pvt[0],k2k_pvt[1],k2k_pvt[2],k2k_pvt[3],  k2k_pvt[4],k2k_pvt[5],k2k_pvt[6],k2k_pvt[7],    k2k_pvt[8],k2k_pvt[9],k2k_pvt[10],k2k_pvt[11],   k2k_pvt[12],k2k_pvt[13],k2k_pvt[14],k2k_pvt[15]  );\
																	}\
																	)

	for( layer=0;  layer<=costvol_layers; layer++ ){
		inv_depth = (layer * inv_d_step) + min_inv_depth;								// locate pixel to sample from  new image. Depth dependent part.
		uh3  = uh2 + k2k_pvt[3]*inv_depth;
		vh3  = vh2 + k2k_pvt[7]*inv_depth;
		wh3  = wh2 + k2k_pvt[11]*inv_depth;
		u2   = uh3/wh3;
		v2   = vh3/wh3;

		int_u2 = ceil(u2/reduction-0.5f);												// nearest neighbour interpolation
		int_v2 = ceil(v2/reduction-0.5f);												// NB this corrects the sparse sampling to the redued scales.

		if ( !((int_u2<0) || (int_u2>read_cols_ -1) || (int_v2<0) || (int_v2>read_rows_-1)) ) {  	// if (not within new frame) skip     || (in_image == false)
			cv_idx = read_index + layer*mm_pixels;											// Step through costvol layers
			cost[layer] = cdata[cv_idx];													// cost for this elem of cost vol
			w  = hdata[cv_idx];																// count of updates of this costvol element. w = 001 initially

			// c = img[read_index_new];																// nearest neighbour
			c = bilinear(img, u2/reduction, v2/reduction, mm_cols, read_offset_, reduction); 		// bilinear(float8* img, float u_flt, float v_flt, int cols)

			rho						= Tau_HSV_grad(B, c);								// Compute rho photometic cost
			rho_8chan				= Tau_HSV_grad_8chan(B, c);
			cost[layer] 			= (cost[layer]*w + rho) / (w + 1);	 				// Compute update of cost vol element, taking account of 'w  = hdata[cv_idx];' number of hits to this element.
			cost_8chan[layer] 		= (cost_8chan[layer]*w + rho_8chan) / (w + 1);

			cdata[cv_idx] 			= cost[layer];  									// CostVol set here ###########
			cdata_8chan[cv_idx] 	= cost_8chan[layer];
			hdata[cv_idx] 	= w + 1;													// Weightdata, counts updates of this costvol element.
			img_sum[cv_idx] += (c.x + c.y + c.z)/3;
		} else { miss = true; }
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	if ( (miss == false) && (in_image == true) ) {
		for( layer=0;  layer<costvol_layers; layer++ ){
			if (cost[layer] < minv) { 														// Find idx & value of min cost in this ray of cost vol, given this update.
				minv = cost[layer];															// NB Use array private to this thread.
				mini = layer;
			}
			maxv = fmax(cost[layer], maxv);
		}
		lo[read_index] 	= minv; 															// min photometric cost  // rho;//
		a[read_index] 	= mini*inv_d_step + min_inv_depth; //c.x; //uh2; //c.x; // mini*inv_d_step + min_inv_depth;	// inverse distance
		d[read_index] 	= mini*inv_d_step + min_inv_depth; //B.x; //mini*inv_d_step + min_inv_depth; //uh3; //c.y; // mini*inv_d_step + min_inv_depth;
		hi[read_index] 	= maxv; 															// max photometric cost
	}
}

__kernel void UpdateQD(
	__private	uint	mipmap_layer,		//0
	__private	uint	k_verbosity,		//1
	__constant 	uint8*	mipmap_params,		//2
	__constant 	uint*	uint_params,		//3
	__global 	float*  fp32_params,		//4
	__global 	float4* g1pt,				//5		// keyframe_g1mem
	__global 	float* 	qpt,				//6		// qmem,						//	2 * mm_size_bytes_C1
	__global 	float*  apt,				//7		// amem,     auxilliary A
	__global 	float*  dpt					//8		// dmem,     depth D
		 )
{
	uint global_id_u 	= get_global_id(0);
	float global_id_flt = global_id_u;
	uint lid 			= get_local_id(0);
	uint group_size 	= get_local_size(0);

	uint8 mipmap_params_ = mipmap_params[mipmap_layer];					// choose correct layer of the mipmap
	uint mim_pixels		= mipmap_params_[MiM_PIXELS];					// cannot return before the last memory barrier !
	uint read_offset_ 	= mipmap_params_[MiM_READ_OFFSET];
	uint read_cols_ 	= mipmap_params_[MiM_READ_COLS];
	uint read_rows_		= mipmap_params_[MiM_READ_ROWS];
	uint margin 		= uint_params[MARGIN];

	uint mm_cols		= uint_params[MM_COLS];
	uint reduction		= mm_cols/read_cols_;
	uint v    			= global_id_u / read_cols_;						// read_row
	uint u 				= fmod(global_id_flt, read_cols_);				// read_column
	float u_flt			= u * reduction;
	float v_flt			= v * reduction;
	uint read_index 	= read_offset_  +  v  * mm_cols  + u ;

	//uint mm_pixels		= uint_params[MM_PIXELS];
	///////////////////////////////////

	//int g_id 			= get_global_id(0);
	//int rows 			= floor(params[rows_]);
	//int cols 			= floor(params[cols_]);
	int costvol_layers	= uint_params[COSTVOL_LAYERS];
	float epsilon 		= fp32_params[EPSILON];
	float sigma_q 		= fp32_params[SIGMA_Q];
	float sigma_d 		= fp32_params[SIGMA_D];
	float theta 		= fp32_params[THETA];

	int y              = global_id_u / read_cols_;
	int x              = global_id_u % read_cols_;
	unsigned int pt    = read_index ;										//x + y * mm_cols;					// index of this pixel
	const int wh       = uint_params[MM_PIXELS]; 								//(mm_pixels + read_offset_);		//  / *mm_cols*read_rows_* /

	float4 g1_4;
	float g1, qx, qy, d, a;
	float dd_x, dd_y, maxq;

	barrier(CLK_GLOBAL_MEM_FENCE);

	if (global_id_u < mim_pixels) {
		g1_4 = g1pt[pt];
		g1 =  g1_4.x * g1_4.y * g1_4.z ;									// reduce channel count of g1. Here Manhatan norm. bad choice. Hue is not good.
		qx = qpt[pt];														// TODO Later try   g1 = 1-(1-g_saturation)*(1-g_value) , i.e. where sat and val agree: less fooled by shadows.
		qy = qpt[pt+wh];
		d  = dpt[pt];
		a  = apt[pt];

		dd_x = (x==read_cols_-1)? 0.0f : dpt[pt+1]       - d;				// Sample depth gradient in x&y
		dd_y = (y==read_rows_-1)? 0.0f : dpt[pt+mm_cols] - d;				// dd_x, dd_y = 0, if at edge of image, otherwise depth_grad in x,y.

		// qpt2[pt] 		= dd_x;		//dpt[pt+1]			- d;  //x;	//dd_x;		//global_id_u;  //
		// qpt2[pt + wh] 	= dd_y;		//dpt[pt+mm_cols]	- d;		//dd_y;		//pt;			//

		qx = (qx + sigma_q*g1*dd_x) / (1.0f + sigma_q*epsilon);				// DTAM paper, primal-dual update step
		qy = (qy + sigma_q*g1*dd_y) / (1.0f + sigma_q*epsilon);				// sigma_q=0.0559017,  epsilon=0.1,  g1=0.999.. if white, less if visible edge.
		maxq = fmax(1.0f, sqrt(qx*qx + qy*qy));

		//if (k_verbosity>0 && x==100 && y==100) printf("\nKernel UpdateQD_1 mipmap_layer=%u, mim_pixels=%u, mm_cols=%u, wh=%u, pt=%u, d=%f, sigma_q=%f, epsilon=%f, g1=%f, , a=%f, theta=%f, sigma_d=%f, qx=%f, qy=%f, maxq=%f, dd_x=%f, dd_y=%f m x=%i, y=%i", \
		//	mipmap_layer, mim_pixels, mm_cols, wh, pt, d, sigma_q, epsilon, g1,  a, theta, sigma_d, qx, qy, maxq, dd_x, dd_y, x, y );

		qx 			= qx/maxq;
		qy 			= qy/maxq;

		qpt[pt]		= qx;  													//dd_x;//pt2;//wh;//pt;//dd_x;//qx / maxq;
		qpt[pt+wh]	= qy;  													//dd_y;//pt;//;//y;//dd_y;//dpt[pt+1] - d; //dd_y;//qy / maxq;
	}

	barrier(CLK_GLOBAL_MEM_FENCE);											// needs to be after all Q updates.
	if (global_id_u < mim_pixels){
		float dqx_x;														// = div_q_x(q, w, x, i);				// div_q_x(..)
		if (x == 0) dqx_x = qx;
		else if (x == read_cols_-1) dqx_x =  -qpt[pt-1];
		else dqx_x =  qx- qpt[pt-1];

		float dqy_y;														// = div_q_y(q, w, h, wh, y, i);		// div_q_y(..)
		if (y == 0) dqy_y =  qy;											// return q[i];
		else if (y == read_rows_-1) dqy_y = -qpt[pt+wh-mm_cols];			// return -q[i-1];
		else dqy_y =  qy - qpt[pt+wh-mm_cols/*read_cols_*/];				// return q[i]- q[i-w];

		const float div_q = dqx_x + dqy_y;

		dpt[pt] = (d + sigma_d * (g1*div_q + a/theta)) / (1.0f + sigma_d/theta);

		//if (k_verbosity>0 && x==100 && y==100) printf("\nKernel UpdateQD_2 mipmap_layer=%u, mm_cols=%u, wh=%u, dpt[pt]=%f, d=%f, sigma_q=%f, epsilon=%f, g1=%f, div_q=%f, a=%f, theta=%f, sigma_d=%f, qx=%f, qy=%f, maxq=%f, dd_x=%f, dd_y=%f ", \
		//	mipmap_layer, mm_cols, wh, dpt[pt], d, sigma_q, epsilon, g1, div_q , a, theta, sigma_d, qx, qy, maxq, dd_x, dd_y );
	}
}

__kernel void  UpdateG(
	__private	uint		mipmap_layer,	//0
	__private	uint		k_verbosity,	//1
	__constant	uint8*		mipmap_params,	//2
	__constant 	uint*		uint_params,	//3
	__constant 	float*		fp32_params,	//4
	__global 	float8*		img,			//5		// keyframe_imgmem in "HSV_grad" colorspace
	__global 	float8*		g1p				//6     // keyframe_g1mem
		 )
{
	uint global_id_u 	= get_global_id(0);
	float global_id_flt = global_id_u;

	uint8 mipmap_params_ = mipmap_params[mipmap_layer];
	uint read_offset_ 	= mipmap_params_[MiM_READ_OFFSET];
	uint read_cols_ 	= mipmap_params_[MiM_READ_COLS];
	uint read_rows_ 	= mipmap_params_[MiM_READ_ROWS];
	uint mm_cols		= uint_params[MM_COLS];
	uint read_row    	= global_id_u / read_cols_;
	uint read_column 	= fmod(global_id_flt, read_cols_);

	int upoff			= -(read_row  >1 )*mm_cols;													//-(read_row  != 0)*mm_cols;				// up, down, left, right offsets, by boolean logic.
	int dnoff			=  (read_row  < read_rows_-2) * mm_cols;									// (read_row  < read_rows_-1) * mm_cols;
    int lfoff			= -(read_column >1);														//-(read_column != 0);
	int rtoff			=  (read_column < read_cols_-2);											// (read_column < mm_cols-1);
	uint offset			=   read_column + read_row  * mm_cols + read_offset_ ;

	float alphaG		= fp32_params[ALPHA_G];
	float betaG 		= fp32_params[BETA_G];

	float8 pr 	= img[offset + rtoff];
	float8 pl 	= img[offset + lfoff];
	float8 pu 	= img[offset + upoff];
	float8 pd 	= img[offset + dnoff];

	//float4 gx	= { (pr.x - pl.x), (pr.y - pl.y), (pr.z - pl.z), 1.0f };							// Signed img gradient in hsv
	//float4 gy	= { (pd.x - pu.x), (pd.y - pu.y), (pd.z - pu.z), 1.0f };

	float8 gx = pr - pl;																			// Signed img gradient in in HSV_grad colorspace
	float8 gy = pd - pu;
	/*
	float4 g1	= { \
		 exp(-alphaG * pow(sqrt(gx.x*gx.x + gy.x*gy.x), betaG) ), \
		 exp(-alphaG * pow(sqrt(gx.y*gx.y + gy.y*gy.y), betaG) ), \
		 exp(-alphaG * pow(sqrt(gx.z*gx.z + gy.z*gy.z), betaG) ), \
		 1.0f };
	*/
	float8 g1 =  exp(-alphaG * pow(sqrt(gx*gx + gy*gy), betaG) );

	if (global_id_u >= mipmap_params_[MiM_PIXELS]) return;
	g1p[offset]= g1;

																									//if(k_verbosity>0 && offset%100000==0)printf("\n\nkenel UpdateG(..) offset=%u  g1=%f,%f, %f,%f, %f,%f, %f,%f,", offset,  g1.s0, g1.s1,   g1.s2, g1.s3,   g1.s4, g1.s5,   g1.s6, g1.s7 );
}

int set_start_layer(float di, float r, float far, float depthStep, int layers, int x, int y){ //( inverse_depth, r , min_inv_depth, inv_depth_step, num_layers )
    const float d_start = di - r;
    const int start_layer =  floor( (d_start - far)/depthStep );
    return (start_layer<0)? 0 : start_layer;
}

int set_end_layer(float di, float r, float far, float depthStep, int layers, int x, int y){
    const float d_end = di + r;
    const int end_layer = ceil((d_end - far)/depthStep) + 1;
    return  (end_layer>(layers-1))? (layers-1) : end_layer;
}

float get_Eaux(float theta, float di, float aIdx, float far, float depthStep, float lambda, float scale_Eaux, float costval)
{
	const float ai = far + aIdx*depthStep;
	return scale_Eaux*(0.5f/theta)*((di-ai)*(di-ai))  +  lambda*costval;
	/*
	// return (0.5f/theta)*((di-ai)*(di-ai)) + lambda*costval;
	// return 100*(0.5f/theta)*((di-ai)*(di-ai)) + lambda*costval;
	// return 10000*(0.5f/theta)*((di-ai)*(di-ai)) + lambda*costval;
	// return 1000000*(0.5f/theta)*((di-ai)*(di-ai)) + lambda*costval;
	// return 10000000*(0.5f/theta)*((di-ai)*(di-ai)) + lambda*costval;
	*/
}

__kernel void UpdateA(						// pointwise exhaustive search
	__private	uint	mipmap_layer,		//0
	__private	uint	k_verbosity,		//1
	__constant 	uint8*	mipmap_params,		//2
	__constant 	uint*	uint_params,		//3
	__global 	float*  fp32_params,		//4
	__global 	float*  cdata,				//5		// cdatabuf, cost volume
	__global 	float*  lo,					//6
	__global 	float*  hi,					//7
	__global 	float*  apt,				//8		// amem,     auxilliary A
	__global 	float*  dpt,				//9		// dmem,     depth D
	__global 	float*  dbg_data			//10	// dbg_databuf
		 )
{
	uint global_id_u 	= get_global_id(0);
	float global_id_flt = global_id_u;
	uint lid 			= get_local_id(0);
	uint group_size 	= get_local_size(0);

	uint8 mipmap_params_ = mipmap_params[mipmap_layer];
	//if (global_id_u    >= mipmap_params_[MiM_PIXELS]) return;
	uint read_offset_ 	= mipmap_params_[MiM_READ_OFFSET];
	uint read_cols_ 	= mipmap_params_[MiM_READ_COLS];
	uint margin 		= uint_params[MARGIN];
	uint mm_cols		= uint_params[MM_COLS];
	uint reduction		= mm_cols/read_cols_;
	uint v    			= global_id_u / read_cols_;					// read_row
	uint u 				= fmod(global_id_flt, read_cols_);			// read_column
	float u_flt			= u * reduction;
	float v_flt			= v * reduction;
	uint read_index 	= read_offset_  +  v  * mm_cols  + u ;
	//////////////////////////////////////

	int costvol_layers		= uint_params[COSTVOL_LAYERS];			//floor(params[layers_]);
	unsigned int layer_step = uint_params[MM_PIXELS];				//floor(params[pixels_]);
	float lambda			= fp32_params[LAMBDA];					//params[lambda_];
	float theta				= fp32_params[THETA];					//params[theta_];
	float max_d				= fp32_params[MAX_INV_DEPTH];			//params[max_inv_depth_]; //near
	float min_d				= fp32_params[MIN_INV_DEPTH];			//params[min_inv_depth_]; //far
	float scale_Eaux		= fp32_params[SCALE_EAUX];				//params[scale_Eaux_];

	unsigned int cpt = read_index;

	barrier(CLK_GLOBAL_MEM_FENCE);
	if (global_id_u    >= mipmap_params_[MiM_PIXELS]) return;

	float d  		= dpt[read_index];
	float E_a  		= FLT_MAX;
	float min_val	= FLT_MAX;
	int   min_layer	= 0;

	const float depthStep 	= fp32_params[INV_DEPTH_STEP];			//(min_d - max_d) / (costvol_layers - 1);
	const float r 			= sqrt( 2*theta*lambda*(hi[read_index] - lo[read_index]) );
	const int 	start_layer = set_start_layer(d, r, max_d, depthStep, costvol_layers, u, v);  // 0;//
	const int 	end_layer   = set_end_layer  (d, r, max_d, depthStep, costvol_layers, u, v);  // costvol_layers-1; //
	int 		minl 		= 0;
	float 		Eaux_min 	= 1e+30f; 						// set high initial value

	for(int l = start_layer; l <= end_layer; l++) {
		const float cost_total = get_Eaux(theta, d, (float)l, min_d, depthStep, lambda, scale_Eaux, cdata[read_index+l*layer_step]);
/**/	dbg_data[read_index + l*layer_step] = cost_total;  				// DTAM_Mapping collects an Eaux volume, for debugging.
		if(cost_total < Eaux_min) {
			Eaux_min = cost_total;
			minl = l;
		}
	 }
	float a = min_d + minl*depthStep;  								// NB implicit conversion: int minl -> float.

	//refinement step
	if(minl > start_layer && minl < end_layer){ 					//return;// if(minl == 0 || minl == costvol_layers-1) // first or last was best
																	// sublayer sampling as the minimum of the parabola with the 2 points around (minl, Eaux_min)
		const float A = get_Eaux(theta, d, minl-1, max_d, depthStep, lambda, scale_Eaux, cdata[read_index+(minl-1)*layer_step]);
		const float B = Eaux_min;
		const float C = get_Eaux(theta, d, minl+1, max_d, depthStep, lambda, scale_Eaux, cdata[read_index+(minl+1)*layer_step]);
		// float delta = ((A+C)==2*B)? 0.0f : ((A-C)*depthStep)/(2*(A-2*B+C));
		float delta = ((A+C)==2*B)? 0.0f : ((C-A)*depthStep)/(2*(A-2*B+C));
		delta = (fabs(delta) > depthStep)? 0.0f : delta;
		// a[i] += delta;
		a -= delta;
	}
	apt[read_index] = a;
}


__kernel void MeasureDepthFit(						// measure the fit of the depthmap against the groud truth.
	// inputs
	__private	uint	mipmap_layer,				//0
	__private	uint	k_verbosity,				//1
	__constant 	uint8*	mipmap_params,				//2
	__constant 	uint*	uint_params,				//3
	__global 	float*  fp32_params,				//4
	__global 	float*  dpt,						//5		// dmem,     depth D
	__global 	float*  dpt_GT,						//6
	// outputs
	__global 	float4* dpt_disparity,				//7
	__local		float4*	local_sum_dpt_disparity,	//8
	__global	float4*	global_sum_dpt_disparity	//
		 )
{
	uint global_id_u 	= get_global_id(0);
	float global_id_flt = global_id_u;
	uint lid 			= get_local_id(0);
	uint group_size 	= get_local_size(0);

	uint8 mipmap_params_ = mipmap_params[mipmap_layer];
	uint read_offset_ 	= mipmap_params_[MiM_READ_OFFSET];
	uint read_cols_ 	= mipmap_params_[MiM_READ_COLS];
	uint margin 		= uint_params[MARGIN];
	uint mm_cols		= uint_params[MM_COLS];
	uint reduction		= mm_cols/read_cols_;
	uint v    			= global_id_u / read_cols_;					// read_row
	uint u 				= fmod(global_id_flt, read_cols_);			// read_column
	float u_flt			= u * reduction;
	float v_flt			= v * reduction;
	uint read_index 	= read_offset_  +  v  * mm_cols  + u ;

	float depth_disparity = dpt[read_index] - dpt_GT[read_index];

	float sq_depth_disparity = depth_disparity * depth_disparity * 64 * 64;

	float proportional_sdd = sq_depth_disparity / (dpt_GT[read_index] * dpt_GT[read_index] * 64 * 8);

	float4 disparity 	= { depth_disparity , -depth_disparity, proportional_sdd, 1.0f};

						//= {dpt_GT[read_index] * 64, sq_depth_disparity, proportional_sdd , 1.0f };
																							// B = true inverse depth
																							// G = sq_depth_disparity
																							// R = proportional_sdd
						//= {dpt_GT[read_index], depth_disparity , -depth_disparity, 1.0f };
																							// {B, G, R, A} Need x64 to spread in visible range of .tiff .
																							// B = true inverse depth
																							// G = est_inv_depth > true inv_depth, i.e. est too close
																							// R = est_inv_depth < true inv_depth, i.e. est too far
	if (global_id_u    < mipmap_params_[MiM_PIXELS]){
		dpt_disparity[read_index]		= disparity;
		local_sum_dpt_disparity[lid]	= disparity;
	}else{
		local_sum_dpt_disparity[lid]	= 0;
	}

	int max_iter = ilogb((float)(group_size));
	for (uint iter=0; iter<=max_iter ; iter++) {	// for log2(local work group size)					// problem : how to produce one result for each mipmap layer ?
																										// NB kernels launched separately for each layer, but workgroup size varies between GPUs.
		group_size   /= 2;
		barrier(CLK_LOCAL_MEM_FENCE);																	// No 'if->return' before fence between write & read local mem

		if (lid<group_size)  local_sum_dpt_disparity[lid] += local_sum_dpt_disparity[lid+group_size];	// local_sum_pix
	}

	barrier(CLK_LOCAL_MEM_FENCE);  // TODO get summation of depth error working
	if (lid==0) {
		uint group_id 			= get_group_id(0);
		uint global_sum_offset 	= 0; 																	//read_offset_ / local_size ;		// only the base layer		// Compute offset for this layer
		uint num_groups 		= get_num_groups(0);

		float4 layer_data 		= {num_groups, reduction, 0.0f, 0.0f };									// Write layer data to first entry  ## problem float4 into float ##
		if (global_id_u == 0) 	{
			global_sum_dpt_disparity[global_sum_offset] 	= num_groups;
			global_sum_dpt_disparity[global_sum_offset+1] 	= reduction;
			global_sum_dpt_disparity[global_sum_offset+2] 	= 0.0f;
			global_sum_dpt_disparity[global_sum_offset+3] 	= 0.0f;
		}
		global_sum_offset += 4 + group_id;

		if (local_sum_dpt_disparity[0][3] >0){																	// Using alpha channel local_sum_pix[0][3], to count valid pixels being summed.
			global_sum_dpt_disparity[global_sum_offset] 	= local_sum_dpt_disparity[0];				// Save to global_sum_pix // Count hits, and divide group by num hits, without using atomics!
		}else global_sum_dpt_disparity[global_sum_offset] 	= 0;
	}
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

////////// Non-Primal-Dual kernels for cost functions //////////////////////////////////////////////////////////////////////////
/// Priors on Shape ////
__kernel void inv_depth_grad (
	__private	uint	mipmap_layer,		//0
	__constant 	uint8*	mipmap_params,		//1
	__constant 	uint*	uint_params,		//2
	__global 	float*  fp32_params,		//3
	__global 	float*  inv_depth,			//4		// dmem,     depth
	__global 	float2*  grad_inv_depth		//5
)
{
	uint global_id_u 	= get_global_id(0);
	float global_id_flt = global_id_u;
	uint lid 			= get_local_id(0);
	uint group_size 	= get_local_size(0);

	uint8 mipmap_params_ = mipmap_params[mipmap_layer];
	uint read_offset_ 	= mipmap_params_[MiM_READ_OFFSET];
	uint read_cols_ 	= mipmap_params_[MiM_READ_COLS];
	uint read_rows_ 	= mipmap_params_[MiM_READ_ROWS];

	uint margin 		= uint_params[MARGIN];
	uint mm_cols		= uint_params[MM_COLS];
	uint reduction		= mm_cols/read_cols_;
	uint /*v*/  read_row  	= global_id_u / read_cols_;					// read_row
	uint /*u*/  read_column	= fmod(global_id_flt, read_cols_);			// read_column
	uint read_index 	= read_offset_  +  read_row  * mm_cols  + read_column ;
	////
	// From __kernel void  img_grad(..)
	int upoff			= -(read_row  >1 )*mm_cols;													//-(read_row  != 0)*mm_cols;				// up, down, left, right offsets, by boolean logic.
	int dnoff			=  (read_row  < read_rows_-2) * mm_cols;									// (read_row  < read_rows_-1) * mm_cols;
    int lfoff			= -(read_column >1);														//-(read_column != 0);
	int rtoff			=  (read_column < read_cols_-2);											// (read_column < mm_cols-1);
	uint offset			=   read_column + read_row  * mm_cols + read_offset_ ;

	float pr =  inv_depth[offset + rtoff];
	float pl =  inv_depth[offset + lfoff];
	float pu =  inv_depth[offset + upoff];
	float pd =  inv_depth[offset + dnoff];

	float2 grad;
	grad.x = pr - pl;
	grad.y = pd - pu;

	grad_inv_depth[read_index]	= grad;
}


__kernel void div_inv_depth_grad (	// Divergence of gradient of inverse depth i.e. smoothness of gradient of depth
	__private	uint	mipmap_layer,		//0
	__constant 	uint8*	mipmap_params,		//1
	__constant 	uint*	uint_params,		//2
	__global 	float*  fp32_params,		//3
	__global 	float2* grad_inv_depth,		//5
	__global 	float*  div_inv_depth_grad,	//6
	__global 	float2* grad2_inv_depth		//7
)
{
	// from __kernel void UpdateA(..)
	uint global_id_u 	= get_global_id(0);
	float global_id_flt = global_id_u;
	uint lid 			= get_local_id(0);
	uint group_size 	= get_local_size(0);

	uint8 mipmap_params_ = mipmap_params[mipmap_layer];
	uint read_offset_ 	= mipmap_params_[MiM_READ_OFFSET];
	uint read_cols_ 	= mipmap_params_[MiM_READ_COLS];
	uint read_rows_ 	= mipmap_params_[MiM_READ_ROWS];

	uint margin 		= uint_params[MARGIN];
	uint mm_cols		= uint_params[MM_COLS];
	uint reduction		= mm_cols/read_cols_;
	uint /*v*/  read_row   	= global_id_u / read_cols_;					// read_row
	uint /*u*/  read_column = fmod(global_id_flt, read_cols_);			// read_column
	uint read_index 	= read_offset_  +  read_row  * mm_cols  + read_column ;
	///////////////////////////
	// Definition of Divergence of a vector field v:  Del dot v(x,y) = gradient_v(x,z) dot v(x,y)  = delta(V_x) / delta(x)  +  delta(V_y) / delta(y)

	// From __kernel void  img_grad(..)
	int upoff			= -(read_row  >1 )*mm_cols;													//-(read_row  != 0)*mm_cols;				// up, down, left, right offsets, by boolean logic.
	int dnoff			=  (read_row  < read_rows_-2) * mm_cols;									// (read_row  < read_rows_-1) * mm_cols;
    int lfoff			= -(read_column >1);														//-(read_column != 0);
	int rtoff			=  (read_column < read_cols_-2);											// (read_column < mm_cols-1);
	uint offset			=   read_column + read_row  * mm_cols + read_offset_ ;

	float pr =  grad_inv_depth[offset + rtoff].x;
	float pl =  grad_inv_depth[offset + lfoff].x;
	float pu =  grad_inv_depth[offset + upoff].y;
	float pd =  grad_inv_depth[offset + dnoff].y;

	float2 grad;
	grad.x = pr - pl;
	grad.y = pd - pu;

	grad2_inv_depth[read_index]		= grad;
	div_inv_depth_grad[read_index]	= (grad_inv_depth[read_index].x  *  grad.x ) +  (grad_inv_depth[read_index].y   *  grad.y ) ;
}


__kernel void project_point_cloud (				// Possibly useful output, saves CPU work.
	__private	uint		mipmap_layer,		//0
	__constant 	uint8*		mipmap_params,		//1
	__constant 	uint*		uint_params,		//2
	__global 	float16*	inv_k,				//3		// inverse projection matrix
	__global 	float*  	fp32_params,		//4
	__global 	float*  	inv_depth,			//5
	__global 	float3* 	point_cloud			//6
)
{
	// from __kernel void DepthCostVol(..)
	uint global_id_u 	= get_global_id(0);
	float global_id_flt = global_id_u;
	uint8 mipmap_params_= mipmap_params[mipmap_layer];									// choose correct layer of the mipmap

	uint read_offset_ 	= mipmap_params_[MiM_READ_OFFSET];
	uint read_cols_ 	= mipmap_params_[MiM_READ_COLS];
	uint margin 		= uint_params[MARGIN];

	uint mm_cols		= uint_params[MM_COLS];
	uint reduction		= mm_cols/read_cols_;
	uint v    			= global_id_u / read_cols_;										// read_row
	uint u 				= fmod(global_id_flt, read_cols_);								// read_column
	float u_flt			= u * reduction;
	float v_flt			= v * reduction;
	uint read_index 	= read_offset_  +  v  * mm_cols  + u ;
	//////////////

	uint mm_pixels		= uint_params[MM_PIXELS];
	if (global_id_u > mm_pixels) return;

	float inv_d_step 	= fp32_params[INV_DEPTH_STEP];
	float min_inv_depth = fp32_params[MIN_INV_DEPTH];

	float16 inv_k_pvt		= inv_k[0];
	float 	inv_depth_pvt 	= inv_depth[read_index];
	float3 	point;

	point.x = inv_k_pvt[0]*u_flt + inv_k_pvt[1]*v_flt + inv_k_pvt[ 2]*1 + inv_k_pvt[ 3]*inv_depth_pvt;  // TODO check correct matrix computaion for inverse projection. esp 3rd & 4th cols of this equation.
	point.y = inv_k_pvt[4]*u_flt + inv_k_pvt[5]*v_flt + inv_k_pvt[ 6]*1 + inv_k_pvt[ 7]*inv_depth_pvt;
	point.z = inv_k_pvt[8]*u_flt + inv_k_pvt[9]*v_flt + inv_k_pvt[10]*1 + inv_k_pvt[11]*inv_depth_pvt;

	point_cloud[read_index] = point;
}


__kernel void compute_curvature (
	__private	uint	mipmap_layer,		//0
	__constant 	uint8*	mipmap_params,		//1
	__constant 	uint*	uint_params,		//2
	__global 	float*  fp32_params,		//3
	__global 	float*  inv_depth,			//4
	__global 	float*  mean_curvature,		//5
	__global	float8* loss_params			//6
)
{
	uint global_id_u 	= get_global_id(0);
	float global_id_flt = global_id_u;
	uint8 mipmap_params_= mipmap_params[mipmap_layer];									// choose correct layer of the mipmap

	uint read_offset_ 	= mipmap_params_[MiM_READ_OFFSET];
	uint read_rows_ 	= mipmap_params_[MiM_READ_ROWS];
	uint read_cols_ 	= mipmap_params_[MiM_READ_COLS];
	uint margin 		= uint_params[MARGIN];
	uint mm_pixels		= uint_params[MM_PIXELS];
	if (global_id_u > mm_pixels) return;

	uint mm_cols		= uint_params[MM_COLS];
	uint reduction		= mm_cols/read_cols_;
	uint /*v*/  read_row	= global_id_u / read_cols_;									// read_row
	uint /*u*/  read_column	= fmod(global_id_flt, read_cols_);							// read_column
	uint read_index 	= read_offset_  +  read_row  * mm_cols  + read_column ;

	// From SIRFS : Section 5.1 ... Mean curvature is defined as the average of principal curvatures.
	// H = 1/2(k_1 + k_2) , can be approxiated on a surface using filter convolutions that prroximate 1st & 2nd partial derivatives.
	// Eq 15.
	// H(Z) = ((1+Z²_x) z_yy - 2Z_x*Z_y*Z_xy +(1+Z²_y)Z_xx)  / (2(1+Z²_x + Z2_y)⁽3/2) )

	// From SIRFS supplementary data Section 3
	// Mean curvature H(Z) of depthmap Z.

	int upoff			= -(read_row  >1 )*mm_cols;										//-(read_row  != 0)*mm_cols;				// up, down, left, right offsets, by boolean logic.
	int dnoff			=  (read_row  < read_rows_-2) * mm_cols;						// (read_row  < read_rows_-1) * mm_cols;
	int lfoff			= -(read_column >1);											//-(read_column != 0);
	int rtoff			=  (read_column < read_cols_-2);								// (read_column < mm_cols-1);
	uint offset			=   read_column + read_row  * mm_cols + read_offset_ ;

	float a = inv_depth[offset+lfoff+upoff];
	float b = inv_depth[offset		+upoff];
	float c = inv_depth[offset+rtoff+upoff];
	float d = inv_depth[offset+lfoff];
	float e = inv_depth[offset];
	float f = inv_depth[offset+rtoff];
	float g = inv_depth[offset+lfoff+dnoff];
	float h = inv_depth[offset		+dnoff];
	float i = inv_depth[offset+rtoff+dnoff];

	float Z_x  = ( 2*(d - f) + (a - c) + (g - i) )/8;
	float Z_y  = ( 2*(c - h) + (a - g) + (c - i) )/8;
	float Z_xx = ( a -2*b + c + 2*d -4*e + 2*f + g - 2*h + i )/4;
	float Z_yy = ( a +2*b + c - 2*d -4*e - 2*f + g + 2*h + i )/4;
	float Z_xy = ( a - c + g - i )/4;

	float Z_x_sq = pown(Z_x,2);
	float Z_y_sq = pown(Z_y,2);

	float M = sqrt(1 + Z_x_sq + Z_y_sq );
	float N = ((1 + Z_x_sq) * Z_yy) - (2*Z_x*Z_y*Z_xy) + ((1 + Z_y_sq) * Z_xx);
	float D = 2 * pown(M,3);

	mean_curvature[offset] = N/D;

	float M_sq = pown(M,2);
	float F_x  = 2*(Z_x*Z_yy - Z_xy*Z_y) - (3*Z_x*N)/M_sq;
	float F_y  = 2*(Z_xx*Z_y - Z_x*Z_xy) - (3*Z_y*N)/M_sq;
	float F_xx = 1 + Z_y_sq;
	float F_yy = 1 + Z_x_sq;
	float F_xy = -2 *Z_x*Z_y;

	float8 loss_params_pvt = {M, N, D, F_x, F_y, F_xx, F_yy, F_xy};						// M & N included for debugging.

	loss_params[offset] = loss_params_pvt;
}

// NB SIRFS section 5.1 "Our smoothness prior for shapes is a Gaussian scale mixture on the localvariaion of the mean curvature of Z (the depth map).
// Eq 16.  f_k(Z) = SUM_i{ SUM_(j in N(i)){ c(H(Z)_i - H(Z)_j  , alpha_k, sigma_k) } }
// NB this produces a distribution of cost for +ve and -ve curvatures, with minimum cost for a plane surface.
// The learned GSM is very heavy tailed, fig 11, encourages mostly smooth, with occassionally very non-smooth, i.e. bend rarely.

__kernel void curvature_gradient(
	__private	uint	mipmap_layer,		//0
	__constant 	uint8*	mipmap_params,		//1
	__constant 	uint*	uint_params,		//2
	__global 	float*  fp32_params,		//3
	__global 	float*  mean_curvature,		//4
	__global 	float*  grad_curvature		//5
			  )
{

	// TODO ? should we rather have geeric kernel code for grad & div, then in host "createKernel(...)" different instances, with different Arguments set, for inv_depth, curvature, and other maps. //


}

__kernel void curvature_smoothness (
	__private	uint	mipmap_layer,		//0
	__constant 	uint8*	mipmap_params,		//1
	__constant 	uint*	uint_params,		//2
	__global 	float*  fp32_params,		//3
	__global 	float*  mean_curvature,		//4
	__global 	float*  div_curvature		//5
)
{




}
													// Parsimony costs : Need to sort pixels by bin-sort, then attraction between neighbours #################################################
__kernel void plane_parsimony ( 					// 3D orientation of plane + orthogonal distance from camera
	__private	uint	mipmap_layer,		//0
	__constant 	uint8*	mipmap_params,		//1
	__constant 	uint*	uint_params,		//2
	__global 	float*  fp32_params			//3
)
{

}

__kernel void curvature_parsimony ( 				// 2D major,minor curvature
	__private	uint	mipmap_layer,		//0
	__constant 	uint8*	mipmap_params,		//1
	__constant 	uint*	uint_params,		//2
	__global 	float*  fp32_params			//3
)
{

}

/// Priors on Reflectance ///////////////////////////////////////////////////////////////////////////////////////////////////////////

__kernel void reflectance_smoothness ( 				// Modfied Cook-Torrance,  HSV_lambertian, HSV_specular, roughness, metal-glass  (8D reflectance space) NB Plenoxels transmitance....
	__private	uint	mipmap_layer,		//0
	__constant 	uint8*	mipmap_params,		//1
	__constant 	uint*	uint_params,		//2
	__global 	float*  fp32_params			//3
)
{

}

__kernel void reflectance_parsimony (
	__private	uint	mipmap_layer,		//0
	__constant 	uint8*	mipmap_params,		//1
	__constant 	uint*	uint_params,		//2
	__global 	float*  fp32_params			//3
)
{

}

__kernel void reflectance_absolute_value (
	__private	uint	mipmap_layer,		//0
	__constant 	uint8*	mipmap_params,		//1
	__constant 	uint*	uint_params,		//2
	__global 	float*  fp32_params			//3
)
{

}

__kernel void illumination_model (
	__private	uint	mipmap_layer,		//0
	__constant 	uint8*	mipmap_params,		//1
	__constant 	uint*	uint_params,		//2
	__global 	float*  fp32_params			//3
)
{

}
