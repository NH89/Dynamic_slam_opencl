#ifndef NH89_PRINT_FNS_HPP
#define NH89_PRINT_FNS_HPP

#include "../RunCL/RunCL_typedefs.hpp"

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>
#include <opencv2/core.hpp>
#include <jsoncpp/json/json.h>

#define PRINT_MATX33F(a,txt) std::cout << "\n\n" << #txt <<" "<< #a << " = " << std::flush ; print_matx33f(a);

#define PRINT_MATX44F(a,txt) std::cout << "\n\n" << #txt <<" "<< #a << " = " << std::flush ; print_matx44f(a);

#define PRINT_MATX55D(a,txt) std::cout << "\n\n" << #txt <<" "<< #a << " = " << std::flush ; print_matx55d(a);


#define PRINT_MATX66F(a,txt) std::cout << "\n\n" << #txt <<" "<< #a << " = " << std::flush ; print_matx66f(a);

#define PRINT_MATX61F(a,txt) std::cout << "\n\n" << #txt <<" "<< #a << " = " << std::flush ; print_matx61f(a);

#define PRINT_MATX16F(a,txt) std::cout << "\n\n" << #txt <<" "<< #a << " = " << std::flush ; print_matx16f(a);


#define PRINT_MATX15D(a,txt) std::cout << "\n\n" << #txt <<" "<< #a << " = " << std::flush ; print_matx15d(a);


#define PRINT_MATX13F(a,txt) std::cout << "\n\n" << #txt <<" "<< #a << " = " << std::flush ; print_matx13f(a);

#define PRINT_MATX31F(a,txt) std::cout << "\n\n" << #txt <<" "<< #a << " = " << std::flush ; print_matx31f(a);


#define PRINT_FLOAT_9(a,txt) std::cout << "\n\n" << #txt <<" "<< #a << " = " << std::flush ; print_float_9(a);

#define PRINT_FLOAT_16(a,txt) std::cout << "\n\n" << #txt <<" "<< #a << " = " << std::flush ; print_float_16(a);

#define PRINT_MAT44F(a,txt) std::cout << "\n\n" << #txt <<" "<< #a << " = " << std::flush ; print_matf(a, 4, 4);

#define PRINT_MAT33F(a,txt) std::cout << "\n\n" << #txt <<" "<< #a << " = " << std::flush ; print_matf(a, 3, 3);


void print_matx33f(cv::Matx33f matx);

void print_matx44f(cv::Matx44f matx);

void print_matx55d(Matx55d matx);		// NB own typedef


void print_matx66f(cv::Matx66f matx);

void print_matx61f(cv::Matx61f matx);

void print_matx16f(cv::Matx16f matx);


void print_matx13f(cv::Matx13f matx);

void print_matx31f(cv::Matx31f matx);


void print_matx15d(Matx15d matx);		// NB own typedef


void print_float_9(float float_9[9]);

void print_float_16(float float_16[16]);

void print_json_float_9(Json::Value obj, std::string name);

void print_matf(cv::Mat mat, int rows, int cols);


#endif /*PRINT_FNS_HPP*/
