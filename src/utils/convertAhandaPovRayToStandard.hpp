#ifndef CONVERTAHANDAPOVRAYTOSTANDARD_H_INCLUDED
#define CONVERTAHANDAPOVRAYTOSTANDARD_H_INCLUDED

#include <opencv2/opencv.hpp>
#include "conf_params.hpp"

void convertAhandaPovRayToStandard_2(	Json::Value obj_ , const char *filepath,  cv::Mat& R,  cv::Mat& T, cv::Mat& cameraMatrix);

void convertAhandaPovRayToStandard(		Json::Value obj_ , const char * filepath, cv::Mat& R,  cv::Mat& T,  cv::Mat& cameraMatrix);

cv::Mat loadDepthAhanda(				Json::Value obj_ , std::string filename, int r,int c,cv::Mat cameraMatrix);

#endif // CONVERTAHANDAPOVRAYTOSTANDARD_H_INCLUDED
