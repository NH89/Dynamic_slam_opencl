#include "Dynamic_slam.hpp"
#include <fstream>
#include <iomanip>   // for std::setprecision, std::setw

using namespace cv;
using namespace std;

void Dynamic_slam::generate_SE3_k2k_vec( float _SE3_k2k[6*16] ) {																			// Generates a set of 6 k2k to be used to compute the SE3 maps for the current camera intrinsic matrix.
	int local_verbosity_threshold = V_DYNAMIC_SLAM_GENERATE_SE3_K2K;//verbosity_mp["Dynamic_slam::generate_SE3_k2k"];// -2;
																																			if(verbosity>local_verbosity_threshold) cout << "\nDynamic_slam::generate_SE3_k2k( float _SE3_k2k[6*16] ) chk_0" << endl << flush;
	// SE3
	//const float res			= ( obj["cameraMatrix"][2].asFloat() + obj["cameraMatrix"][5].asFloat() ) /2.0;
	// const float f			= ( obj["cameraMatrix"][0].asFloat() + obj["cameraMatrix"][4].asFloat() ) /2.0;									// focal length in pixels.
	// const float delta 	  	= obj["ST3_delta"].asFloat() * obj["min_depth"].asFloat()  / f ;												// ST3_delta * (Translation to cause 1 pixel of parallax at min_depth)  	//1.0;//0.01; //0.001;  //  * obj["min_depth"].asFloat()
	// const float delta_theta = obj["SO3_delta_theta"].asFloat() / f;																			// SO3_delta_theta * (Rotation to cause 1 pixel of rotation flow) //0.01; //0.001;
	// const float cos_theta   = cos(delta_theta);
	// const float sin_theta   = sin(delta_theta);
																																			// Old :  Rotate 0.01 radians i.e 0.573  degrees.  Translate 0.001 'units' of distance
																																			if(verbosity>local_verbosity_threshold){ cout << "\nDynamic_slam::generate_SE3_k2k( ) chk_1,"<<endl << flush;
																																				print_json_float_9(obj, "cameraMatrix");
																																				cout << "  delta_theta = "	<<delta_theta	<< " radians," 								<<endl << flush;
																																				cout << "  delta = "		<<delta			<< " units distance," 						<<endl << flush;
																																				cout << "  f = "			<<f				<< " pixels," 								<<endl << flush;
																																				cout << "  obj[\"ST3_delta\"] =  "            <<obj["ST3_delta"].asFloat()				<<endl << flush;
																																				cout << "  obj[\"min_depth\"] =  "            <<obj["min_depth"].asFloat()				<<endl << flush;
																																				cout << "  obj[\"SO3_delta_theta\"] =  "      <<obj["SO3_delta_theta"].asFloat()		<<endl << flush;
																																			}
	//Identity =				(1,			0,			0,			0,  			0,			1,			0,			0,  			0,			0,			1,			0,  			0,	0,	0,	1);
	cv::Matx44f transform[6];
	transform[Rx] = cv::Matx44f(1,				0,				0,				0,\
								0,				cos_theta,		-sin_theta,		0,\
								0,				sin_theta,		cos_theta,		0,\
								0,				0,				0,				1);

	transform[Ry] = cv::Matx44f(cos_theta,		0,				sin_theta,		0,\
								0,				1,				0,				0,\
								-sin_theta,		0,				cos_theta,		0,\
								0,				0,				0,				1);

	transform[Rz] = cv::Matx44f(cos_theta,		-sin_theta,		0,				0,\
								sin_theta,		cos_theta,		0,				0,\
								0,				0,				1,				0,\
								0,				0,				0,				1);

	transform[Tx] = cv::Matx44f(1,0,0,delta, 	0,1,0,0,		0,0,1,0,		0,0,0,1);
	transform[Ty] = cv::Matx44f(1,0,0,0, 		0,1,0,delta,	0,0,1,0,		0,0,0,1);
	transform[Tz] = cv::Matx44f(1,0,0,0, 		0,1,0,0,		0,0,1,delta,	0,0,0,1);

	cv::Matx44f cam2cam[6];
																																			if(verbosity>local_verbosity_threshold) {
																																				PRINT_MATX44F(frame_data.back().frame_data_GT.K,);
																																				PRINT_MATX44F(frame_data.back().frame_data_GT.inv_K,);
																																			}
	for (int i=0; i<6; i++) {  cam2cam[i] = frame_data.back().frame_data_GT.K  *  transform[i]  *  frame_data.back().frame_data_GT.inv_K;
																																			if(verbosity>local_verbosity_threshold) {
																																				cout << "\ni=" << i << endl;
																																				PRINT_MATX44F(transform[i],);
																																				PRINT_MATX44F(cam2cam[i],);
																																			}
		for (uint row=0; row<4; row++) {
			for (uint col=0; col<4; col++){
				_SE3_k2k[i*16 + row*4 + col] 	= cam2cam[i].operator()(row,col);
			}
		}
	}
																																			if(verbosity>local_verbosity_threshold) {
																																				/*
																																				cout << endl << setprecision(9);
																																				// for (int i=0; i<6; i++) {
																																				// 	cout << "\n _SE3_k2k ["<<i<<"*16 + row*4 + col]=\n";
																																				// 	for (int row=0; row<4; row++) {
																																				// 		for (int col=0; col<4; col++){
																																				// 			cout << setw(6) << _SE3_k2k[i*16 + row*4 + col] <<"\t  ";
																																				// 		}cout<<endl;
																																				// 	}cout<<endl;
																																				// }

																																				// Chk k2k makes sense for each SE3 DoF
																																				// Image size 640 * 480, from K as loaded from .json file.
																																				// Homogeneous image coords : (col, row, depth, w), so ater transformation divide by w to find new (col, row, depth, 1).
																																				// NB if there are points at infinite distance, then w=0.
																																				// For this reason we store inverse depth, and use a minimum depth cut off, to avoid points at the optical centre of the camera.
																																				// See __kernel void compute_param_maps(..) , or hand calculate the elements of the matrix multiplication.
																																				cv::Matx14f topleft = {10,10,1,1},  topright = {630,10,1,1}, centre = {320,240,1,1}, bottomleft = {10,470,1,1}, bottomright = {630,470,1,1} , result;	// (column, row, w, 1/depth)
																																				cv::Matx44f identity = {1,0,0,0,  0,1,0,0,  0,0,1,0,  0,0,0,1};

																																				cout << "\n topleft * 	identity = " << topleft * 		identity << flush;
																																				cout << "\n topright * 	identity = " << topright * 		identity << flush;
																																				cout << "\n bottomleft *  identity = " << bottomleft * 	identity << flush;
																																				cout << "\n bottomright * identity = " << bottomright * identity << flush;
																																				*/
																																				cout << "\n\nDynamic_slam::generate_SE3_k2k( float _SE3_k2k[6*16] )   finished" << endl << flush;
																																			}
}


void Dynamic_slam::compute_optimum( float steps[3], float Rho_sq_results_[tracking_num_samples][max_mipmap_layers][tracking_num_colour_channels], int layer, int channel, float *prediction, float *optimum, float *stepsize  ){
	int local_verbosity_threshold = V_DYNAMIC_SLAM_COMPUTE_OPTIMUM;//verbosity_mp["Dynamic_slam::compute_tracking_increment"];

	float a, b, c,   d, e,   f, g,   h, i,   j, k, d2, f2, h2;																				// compute x value of the optimum of parabola, y= a*x*x + b*x + c
																																			// given samples at x=1,2,4
	d = steps[0];	e = Rho_sq_results_[0][layer][channel] ;
	f = steps[1];	g = Rho_sq_results_[1][layer][channel] ;
	h = steps[2];	i = Rho_sq_results_[2][layer][channel] ;

	d2 = d * d;
	f2 = f * f;
	h2 = h * h;

	j = (f2 - d2)*(f-h) - (h2 - f2)*(d-f) ; // (1*1 - 0*0)*(1-3)  - (3*3 - 1*1)*(0-1)  = 1*-2  - (9-1)*-1 = -2  -8*-1 = -2 +8 =6   //// d=0, f=1, h=3  // d2=0, 	f2=1, 	h2=9, 	j=6, k=0.72331,
	k = (g-i)*(d-f) - (e-g)*(f-h);   		// (0.456964 - 0.4644)*(0-1) - (0.814901 - 0.456964)*(1-3) = −0,007436*-1  - 0,357937*-2 = 0,72331   //// (d,e)=(0,0.814901), 	(f,g)=(1,0.456964), 	(h,i)=(3,0.4644),
	a = k/j;						 		// 0,72331 / 6 = 0,120551667
	b = (e-g +a*(f2-d2))  /  (d-f);  		// (0,814901 - 0,456964  +  0,120551667 * (1*1 - 0*0))  / (0-1)  =  -0,478488667
	c = e - a*d2 - b*d;				 		// 0.814901

	if (a>0){																																// IF concavity leads to a minimum, use it.
		float x 		= -b /(2*a);
		*prediction 	= a*(x*x) + b*x + c;
		*optimum 		= x;																												// dy/dx = 0 = 2*a*x + b   =>  x = -b /(2*a)
/*
																																			if(verbosity>local_verbosity_threshold) {
																																				cout << "\n Dynamic_slam::compute_optimum: if (a>0)"
																																				<< ", \ta=" << a
																																				<< ", \tb=" << b
																																				<< ", \tc=" << c

																																				<< ", \td2=" << d2
																																				<< ", \tf2=" << f2
																																				<< ", \th2=" << h2

																																				<< ", \tj=" << j
																																				<< ", \tk=" << k

																																				<< ", \t(d,e)=("<<d<<","<<e<<")"
																																				<< ", \t(f,g)=("<<f<<","<<g<<")"
																																				<< ", \t(h,i)=("<<h<<","<<i<<")"
																																				<< ", \tprediction="<<*prediction
																																				<< ", \toptimum="<<*optimum
																																				<< ", \tstepsize="<<*stepsize
																																				<< endl << flush;
																																			}
*/
	}else{																																	// IF concavity leads to a maximum, pick the best sample so far.
		if (e>=i){
			*prediction = i;
			*optimum 	= h;
			//*stepsize 	*=2;
		}else{
			*prediction = e;
			*optimum 	= d;
			//*stepsize 	/=2;
		}
	}
																																			if(verbosity>local_verbosity_threshold) {
																																				cout << "\n Dynamic_slam::compute_optimum: "
																																				<< ", \na=" << a
																																				<< ", \tb=" << b
																																				<< ", \tc=" << c
																																				<< ", \n(d,e)=("<<d<<","<<e<<")"<<a*d*d + b*d + c
																																				<< ", \n(f,g)=("<<f<<","<<g<<")"<<a*f*f + b*f + c
																																				<< ", \n(h,i)=("<<h<<","<<i<<")"<<a*h*h + b*h + c
																																				<< ", \nprediction="<<*prediction
																																				<< ", \toptimum="<<*optimum
																																				<< ", \tstepsize="<<*stepsize
																																				<< ", \tlayer="<<layer
																																				<< endl << flush;
																																			}
																																		float prediction_ = *prediction * 0.999f; // prevents rounding error from triggering error.
																																		if( e<prediction_ || g<prediction_ || i<prediction_ ) {
																																			cout <<"\n logic error: prediction > sample." << flush; runcl.exit_(1); }
}
