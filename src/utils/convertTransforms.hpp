#ifndef DTAM_UTILS_HPP
#define DTAM_UTILS_HPP

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h> // req for types e.g. CV_BGR2GRAY
#include <opencv2/calib3d/calib3d.hpp>
#include "print_functions.hpp"
using namespace cv;

struct m33 {
	float data[9];
};

struct m34 {
	float data[12];
};

struct float3 {
	float data[3];
};
static Mat  makeGray(Mat image){
    if (image.channels()!=1) {
        cvtColor(image, image, CV_BGR2GRAY);
    }
    return image;
}


static Mat make4x4(const Mat& mat){
    if (mat.rows!=4||mat.cols!=4){
        Mat tmp=Mat::eye(4,4,mat.type());
        tmp(Range(0,mat.rows),Range(0,mat.cols))=mat*1.0;
        return tmp;
    }else{
        return mat;
    }
}

static Mat rodrigues(const Mat& p){
    Mat tmp;
    Rodrigues(p,tmp);
    return tmp;
}

static void LieToRT(InputArray Lie, OutputArray _R, OutputArray _T){
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

static void RTToLie(Matx33f R, Matx13f T, Matx16f &Lie ){
    //std::cout << "\n\nRTToLie(Matx33f R, Matx13f T, Matx16f Lie ) chk_0 #############"<<std::flush;
    Matx13f r(0,0,0);
    cv::Rodrigues(R, r);                // PRINT_MATX13F(r,);  PRINT_MATX13F(T,);       // Makes so3 algebra from SO3 Matx33f
    Matx16f temp(r.operator()(0,0), r.operator()(0,1), r.operator()(0,2),   T.operator()(0,0), T.operator()(0,1), T.operator()(0,2) );
    Lie = temp.get_minor<1,6>(0,0);     // PRINT_MATX16F(Lie, RTToLie(..));
}

static Matx16f RTToLie(Matx33f _R, Matx13f _T){
    //std::cout << "\n\nRTToLie(Matx33f _R, Matx13f _T) chk_0 #############"<<std::flush;
    Matx16f P;
    RTToLie(_R,_T,P);

    return P;
}

static void PToLie(Matx44f P, Matx16f &Lie){    //PRINT_MATX44F(P,);
    //std::cout << "\n\nPToLie(Matx44f P, Matx16f Lie) chk_0 #############"<<std::flush;
    Matx33f R( P.get_minor<3,3>(0,0) );         //PRINT_MATX33F(R,);
    Matx31f T( P.get_minor<3,1>(0,3) );         //PRINT_MATX13F(T.t(),);
    RTToLie(R,T.t(),Lie);                       //PRINT_MATX16F(Lie, PToLie(Matx44f P, Matx16f Lie));
}

static Matx16f PToLie(Matx44f P){
    //std::cout << "\n\nPToLie(Matx44f P) chk_0 #############"<<std::flush;
    Matx16f Lie;
    PToLie(P, Lie);                             //PRINT_MATX16F(Lie, PToLie(..));
    return Lie;
}

static void RTToP(InputArray _R, InputArray _T, OutputArray _P ){
    //std::cout << "\n\nRTToP (InputArray _R, InputArray _T, OutputArray _P ) chk_0 #############"<<std::flush;
    Mat R = _R.getMat();
    Mat T = _T.getMat();
    Mat P = _P.getMat();
    hconcat(R,T,P);
    make4x4(P).copyTo(_P);
}

static Mat RTToP(InputArray _R, InputArray _T){
    //std::cout << "\n\nRTToP (InputArray _R, InputArray _T) chk_0 #############"<<std::flush;
    Mat R = _R.getMat();
    Mat T = _T.getMat();
    Mat P;
    hconcat(R,T,P);
    make4x4(P);
    return P;
}

static Matx44f LieToP_Matx(Matx16f Lie){
    //std::cout << "\n\nLieToP_Matx chk_0 #############"<<std::flush;
                                                //PRINT_MATX16F(Lie,  LieToP_Matx(Matx16f Lie) );
    Matx13f r( Lie.get_minor<1,3>(0,0) );       //PRINT_MATX13F(r,    LieToP_Matx(Matx16f Lie) );
    Matx33f R;
    Rodrigues(r,R);                             //PRINT_MATX33F(R, LieToP_Matx(Matx16f Lie) );        // makes rotation Mat from SO3 Lie vector.

    Matx44f P = Matx44f::zeros();
    for (int row=0; row<3;row++)    for(int col=0; col<3; col++)    P.operator()(row,col) = R.operator()(row,col);
    for(int row=0; row<3; row++)  {                                 P.operator()(row,3) = Lie.operator()(0,row+3); }

    P.operator()(3,3)=1;                        //PRINT_MATX44F(P, LieToP_Matx(Matx16f Lie)_2 )
    return P;
}

static Matx16f LieSub(Matx16f A, Matx16f B){
    //std::cout << "\n\nLieSub chk_0 #############"<<std::flush;
    Matx44f Pa = LieToP_Matx(A);
    Matx44f Pb = LieToP_Matx(B);
    Matx16f out;
    PToLie(Pa*Pb.inv(),out);
    return out;
}

static Matx16f LieAdd(Matx16f A, Matx16f B){
    //std::cout << "\n\nLieSub chk_0 #############"<<std::flush;
    Matx44f Pa = LieToP_Matx(A);
    Matx44f Pb = LieToP_Matx(B);
    Matx16f out;
    PToLie(Pa*Pb,out);
    return out;
}

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

static double median(const Mat& M) { // NB only used for tp median_(const Mat& _M) above, which recasts to type tp.
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


static void Matx44f_To_float16arry(Matx44f matx, float arry[16]){
    for (int i=0; i<16; i++){ arry[i] = matx.operator()(i/4, i%4);}
}

static void float16arry_To_Matx44f(float arry[16], Matx44f matx){
    for (int i=0; i<16; i++){ matx.operator()(i/4, i%4) = arry[i] ;}
}





#endif 
