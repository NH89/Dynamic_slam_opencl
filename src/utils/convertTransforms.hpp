#ifndef DTAM_UTILS_HPP
#define DTAM_UTILS_HPP

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h> // req for types e.g. CV_BGR2GRAY
#include <opencv2/calib3d/calib3d.hpp>

#include "print_functions.hpp"
#include "verbosity.hpp"

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
Mat  makeGray(Mat image);


Mat make4x4(const Mat& mat);

Mat rodrigues(const Mat& p);

void LieToRT(InputArray Lie, OutputArray _R, OutputArray _T);

void RTToLie(Matx33f R, Matx13f T, Matx16f &Lie );

Matx16f RTToLie(Matx33f _R, Matx13f _T);

void PToLie(Matx44f P, Matx16f &Lie);

Matx16f PToLie(Matx44f P);

void RTToP(InputArray _R, InputArray _T, OutputArray _P );

Mat RTToP(InputArray _R, InputArray _T);

Matx44f LieToP_Matx(Matx16f Lie);

Matx16f LieSub(Matx16f A, Matx16f B);

Matx16f LieAdd(Matx16f A, Matx16f B);

template<class tp>  tp median_(const Mat& _M);

double median(const Mat& M);

void Matx44f_To_float16arry(Matx44f matx, float arry[16]);

void float16arry_To_Matx44f(float arry[16], Matx44f matx);

cv::Matx44f getPose(Mat R, Mat T, int verbosity);

cv::Matx44f getInvPose(cv::Matx44f pose, int verbosity);

cv::Matx44f generate_invK_(cv::Matx44f K_, int verbosity);

#endif 
