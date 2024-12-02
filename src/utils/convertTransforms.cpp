#include "convertTransforms.hpp"

using namespace cv;
using namespace std;

Mat  makeGray(Mat image){
    if (image.channels()!=1) {
        cvtColor(image, image, CV_BGR2GRAY);
    }
    return image;
}


Mat make4x4(const Mat& mat){
    if (mat.rows!=4||mat.cols!=4){
        Mat tmp=Mat::eye(4,4,mat.type());
        tmp(Range(0,mat.rows),Range(0,mat.cols))=mat*1.0;
        return tmp;
    }else{
        return mat;
    }
}

Mat rodrigues(const Mat& p){
    Mat tmp;
    Rodrigues(p,tmp);
    return tmp;
}

void LieToRT(InputArray Lie, OutputArray _R, OutputArray _T){
                                                                                                                                            //std::cout << "\n\nLieToRT(InputArray Lie, OutputArray _R, OutputArray _T) chk_0 #############"<<std::flush;
    Mat p = Lie.getMat();
    _R.create(3,3,CV_32FC1);
    Mat R = _R.getMat();
    _T.create(3,1,CV_32FC1);
    Mat T = _T.getMat();
    if(p.cols==1){
        p = p.t();
    }
    rodrigues(p.colRange(Range(0,3))).copyTo(R);
    Mat(p.colRange(Range(3,6)).t()).copyTo(T);
}

void RTToLie(Matx33f R, Matx13f T, Matx16f &Lie ){
                                                                                                                                            //std::cout << "\n\nRTToLie(Matx33f R, Matx13f T, Matx16f Lie ) chk_0 #############"<<std::flush;
    Matx13f r(0,0,0);
    cv::Rodrigues(R, r);                         PRINT_MATX13F(r,);  PRINT_MATX13F(T,);                                                   // Makes so3 algebra from SO3 Matx33f
    Matx16f temp(r.operator()(0,0), r.operator()(0,1), r.operator()(0,2),   T.operator()(0,0), T.operator()(0,1), T.operator()(0,2) );
    Lie = temp.get_minor<1,6>(0,0);              PRINT_MATX16F(Lie, RTToLie(..));
}

Matx16f RTToLie(Matx33f _R, Matx13f _T){
                                                                                                                                            //std::cout << "\n\nRTToLie(Matx33f _R, Matx13f _T) chk_0 #############"<<std::flush;
    Matx16f P;
    RTToLie(_R,_T,P);
    return P;
}

void PToLie(Matx44f P, Matx16f &Lie){           PRINT_MATX44F(P,);
                                                                                                                                            std::cout << "\n\nPToLie(Matx44f P, Matx16f Lie) chk_0 #############"<<std::flush;
    Matx33f R( P.get_minor<3,3>(0,0) );         //PRINT_MATX33F(R,);
    Matx31f T( P.get_minor<3,1>(0,3) );         PRINT_MATX13F(T.t(),);
    RTToLie(R,T.t(),Lie);                       PRINT_MATX16F(Lie, PToLie(Matx44f P, Matx16f Lie));
}

Matx16f PToLie(Matx44f P){
                                                                                                                                            //std::cout << "\n\nPToLie(Matx44f P) chk_0 #############"<<std::flush;
    Matx16f Lie;
    PToLie(P, Lie);                             //PRINT_MATX16F(Lie, PToLie(..));
    return Lie;
}

void RTToP(InputArray _R, InputArray _T, OutputArray _P ){
                                                                                                                                            //std::cout << "\n\nRTToP (InputArray _R, InputArray _T, OutputArray _P ) chk_0 #############"<<std::flush;
    Mat R = _R.getMat();
    Mat T = _T.getMat();
    Mat P = _P.getMat();
    hconcat(R,T,P);
    make4x4(P).copyTo(_P);
}

Mat RTToP(InputArray _R, InputArray _T){
                                                                                                                                            //std::cout << "\n\nRTToP (InputArray _R, InputArray _T) chk_0 #############"<<std::flush;
    Mat R = _R.getMat();
    Mat T = _T.getMat();
    Mat P;
    hconcat(R,T,P);
    make4x4(P);
    return P;
}

Matx44f LieToP_Matx(Matx16f Lie){
                                                                                                                                            //std::cout << "\n\nLieToP_Matx chk_0 #############"<<std::flush;
                                                //PRINT_MATX16F(Lie,  LieToP_Matx(Matx16f Lie) );
    Matx13f r( Lie.get_minor<1,3>(0,0) );       //PRINT_MATX13F(r,    LieToP_Matx(Matx16f Lie) );
    Matx33f R;
    Rodrigues(r,R);                             //PRINT_MATX33F(R, LieToP_Matx(Matx16f Lie) );                                              // makes rotation Mat from SO3 Lie vector.

    Matx44f P = Matx44f::zeros();
    for (int row=0; row<3;row++)    for(int col=0; col<3; col++)    P.operator()(row,col) = R.operator()(row,col);
    for(int row=0; row<3; row++)  {                                 P.operator()(row,3) = Lie.operator()(0,row+3); }

    P.operator()(3,3)=1;                        //PRINT_MATX44F(P, LieToP_Matx(Matx16f Lie)_2 )
    return P;
}

Matx16f LieSub(Matx16f A, Matx16f B){
                                                                                                                                            //std::cout << "\n\nLieSub chk_0 #############"<<std::flush;
    Matx44f Pa = LieToP_Matx(A);
    Matx44f Pb = LieToP_Matx(B);
    Matx16f out;
    PToLie(Pa*Pb.inv(),out);
    return out;
}

Matx16f LieAdd(Matx16f A, Matx16f B){
                                                                                                                                            //std::cout << "\n\nLieSub chk_0 #############"<<std::flush;
    Matx44f Pa = LieToP_Matx(A);
    Matx44f Pb = LieToP_Matx(B);
    Matx16f out;
    PToLie(Pa*Pb,out);
    return out;
}
/*
template<class tp>
tp median_(const Mat& _M) {
    Mat M=_M.clone();
    int iSize=M.cols*M.rows;
    tp* dpSorted=(tp*)M.data;
                                                                                                                                            // Allocate an array of the same size and sort it.
    std::sort (dpSorted, dpSorted+iSize);
                                                                                                                                            // Middle or average of middle values in the sorted array.
    tp dMedian = 0.0;
    if ((iSize % 2) == 0) {
        dMedian = (dpSorted[iSize/2] + dpSorted[(iSize/2) - 1])/2.0;
    } else {
        dMedian = dpSorted[iSize/2];
    }
    return dMedian;
}

double median(const Mat& M) {                                                                                                               // NB only used for tp median_(const Mat& _M) above, which recasts to type tp.
    if(M.type()==CV_32FC1)
        return median_<float>(M);
    if(M.type()==CV_64FC1)
        return median_<double>(M);
    if(M.type()==CV_32SC1)
        return median_<int>(M);
    if(M.type()==CV_16UC1)
        return median_<unsigned int>(M);
    assert(!"Unsupported type");
}
*/

void Matx44f_To_float16arry(Matx44f matx, float arry[16]){
    for (int i=0; i<16; i++){ arry[i] = matx.operator()(i/4, i%4);}
                                                                                                                                            // PRINT_FLOAT_16( arry , Matx44f_To_float16arry );
}


void float16arry_To_Matx44f(float arry[16], Matx44f matx){
    for (int i=0; i<16; i++){ matx.operator()(i/4, i%4) = arry[i] ;}
}


cv::Matx44f getPose(Mat R, Mat T, int verbosity){																							// Mat R, Mat T, Matx44f& pose  // NB Matx::operator()() does not copy, but creates a submatrix. => would be updated when R & T are updated.
	int local_verbosity_threshold = V_DYNAMIC_SLAM_GETPOSE;
																																			if(verbosity>local_verbosity_threshold) { cout << "\n getPose chk_0"<<flush;}
	cv::Matx44f pose;
	for (int i=0; i<9; i++) pose.operator()(i/3,i%3) = 1 * R.at<float>(i/3,i%3);															if(verbosity>local_verbosity_threshold) { PRINT_MAT33F(R,);  PRINT_MATX44F(pose,); }
	for (int i=0; i<3; i++) pose.operator()(i,3)      = T.at<float>(i);
	for (int i=0; i<3; i++) pose.operator()(3,i)      = 0.0f;
	pose.operator()(3,3) = 1.0f;
	return pose;
}


cv::Matx44f getInvPose(cv::Matx44f pose, int verbosity) {	                                                                               // Matx44f pose, Matx44f& inv_pose
	int local_verbosity_threshold = V_DYNAMIC_SLAM_GETINVPOSE;                                                                             //verbosity_mp["Dynamic_slam::getInvPose"];

	cv::Matx44f local_inv_pose;
	cv::Matx33f local_rotation;
	cv::Matx31f local_translation;
	cv::Matx31f inv_local_translation;

	for (int i=0; i<3; i++) { for (int j=0; j<3; j++)	{    local_inv_pose.operator()(i,j) = pose.operator()(j,i); } }
	for (int i=0; i<3; i++) { for (int j=0; j<3; j++)	{    local_rotation.operator()(i,j) = pose.operator()(i,j); } }
	for (int i=0; i<3; i++) 							{ local_translation.operator()(i,0) = pose.operator()(i,3); }

	inv_local_translation = - local_rotation.t() * local_translation;
	for (int i=0; i<3; i++) local_inv_pose.operator()(i,3) = inv_local_translation.operator()(i,0);
	for (int i=0; i<4; i++) local_inv_pose.operator()(3,i) =                  pose.operator()(3,i);
																																			if(verbosity>local_verbosity_threshold){
																																				cout << "\n getInvPose(..) #############################################" << flush;
																																				PRINT_MATX44F(pose,);
																																				PRINT_MATX31F(local_translation,        "getInvPose(cv::Matx44f pose)" );
																																				PRINT_MATX31F(inv_local_translation,    "getInvPose(cv::Matx44f pose)" );
																																				PRINT_MATX44F(local_inv_pose,           "getInvPose(cv::Matx44f pose)" );
                                                                                                                                                cout << endl;
                                                                                                                                                PRINT_MATX44F(pose * local_inv_pose,    "getInvPose(cv::Matx44f pose)" );
                                                                                                                                                PRINT_MATX44F(local_inv_pose *pose,     "getInvPose(cv::Matx44f pose)" );
																																				cout << "\n getInvPose(..) Finished #####################################" << flush;
																																			}
	return local_inv_pose;
																																			/*
																																			*  Inverse of a transformation matrix:
																																			*  http://www.info.hiroshima-cu.ac.jp/~miyazaki/knowledge/teche0053
																																			*
																																			*   {     |   }-1       {       |        }
																																			*   {  R  | t }     =   {  R^T  |-R^T .t }
																																			*   {_____|___}         {_______|________}
																																			*   {0 0 0| 1 }         {0  0  0|    1   }
																																			*
																																			*/
}


cv::Matx44f generate_invK_(cv::Matx44f K_, int verbosity){
	int local_verbosity_threshold = V_DYNAMIC_SLAM_GENERATE_INVK_; //verbosity_mp["Dynamic_slam::generate_invK_"];// 0;
	cv::Matx44f inv_K_;

	float fx   =  K_.operator()(0,0);
	float fy   =  K_.operator()(1,1);
	float skew =  K_.operator()(0,1);
	float cx   =  K_.operator()(0,2);
	float cy   =  K_.operator()(1,2);
																																			if(verbosity>local_verbosity_threshold) {
																																				cout << "\ngenerate_invK_chk 1 ####################\n";
																																				cout<<"\nfx="<<fx <<"\nfy="<<fy <<"\nskew="<<skew <<"\ncx="<<cx <<"\ncy= "<<cy;
																																				cout << flush;
																																			}
	///////////////////////////////////////////////////////////////////// Inverse camera intrinsic matrix, see:
	// https://www.imatest.com/support/docs/pre-5-2/geometric-calibration-deprecated/projective-camera/#:~:text=Inverse,lines%20from%20the%20camera%20center.
	inv_K_ = inv_K_.zeros();
	inv_K_.operator()(0,0)  = 1.0/fx;  																										if(verbosity>local_verbosity_threshold) cout<<"\n1.0/fx="<<1.0/fx;
	inv_K_.operator()(1,1)  = 1.0/fy;  																										if(verbosity>local_verbosity_threshold) cout<<"\n1.0/fy="<<1.0/fy;
	inv_K_.operator()(2,2)  = 1.0;
	inv_K_.operator()(3,3)  = 1.0;                                                                                                          // NB This would be an orthographic projection,
                                                                                                                                            // but in the kernels we divide by Z to produce perspective projecton.
                                                                                                                                            // The pure perpective transform is not ivertable for distances at infinity.
                                                                                                                                            // See
                                                                                                                                            // https://learnwebgl.brown37.net/08_projections/projections_perspective.html
                                                                                                                                            // https://learnwebgl.brown37.net/08_projections/projections_ortho.html
	inv_K_.operator()(0,1)  = -skew/(fx*fy);
	inv_K_.operator()(0,2)  = (cy*skew - cx*fy)/(fx*fy);
	inv_K_.operator()(1,2)  = -cy/fy;
																																			if(verbosity>local_verbosity_threshold) {
																																				cv::Matx44f test_K_ = inv_K_ * K_;
																																				PRINT_MATX44F(test_K_,test_camera_intrinsic_matrix inversion);
																																				//PRINT_MATX44F(pose,);
																																				//PRINT_MATX44F(inv_old_pose,);
																																				PRINT_MATX44F(K_,);
																																				PRINT_MATX44F(inv_K_,);
																																				PRINT_MATX44F( K_ * inv_K_,);
                                                                                                                                                PRINT_MATX44F( inv_K_ * K_,);
																																				cout << "\nDynamic_slam::generate_invK_ Finished ####################"<<endl<<flush;
																																			}
	return inv_K_;
}

