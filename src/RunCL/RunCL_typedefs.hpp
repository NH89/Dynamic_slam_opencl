#ifndef RUNCL_TYPEDEFS_H
#define RUNCL_TYPEDEFS_H

#include <opencv2/core.hpp>

typedef cv::Matx<double, 5, 5> Matx55d;													// used for Hessian of camera matrix params & of lens distortion params.
typedef cv::Matx<double, 1, 5> Matx15d;


#endif
