#include "CV_chk.hpp"

std::string checkCVtype(int code){

    switch(code){
        //case CV_8UC :  return "CV_8UC ";
        case CV_8UC1 :  return "CV_8UC1";
        case CV_8UC2 :  return "CV_8UC2 ";
        case CV_8UC3 :  return "CV_8UC3 ";
        case CV_8UC4 :  return "CV_8UC4 ";

        //case CV_8SC :  return "CV_8SC";
        case CV_8SC1 :  return "CV_8SC1";
        case CV_8SC2 :  return "CV_8SC2";
        case CV_8SC3 :  return "CV_8SC3";
        case CV_8SC4 :  return "CV_8SC4";

        //case CV_16UC :  return "CV_16UC ";
        case CV_16UC1 :  return "CV_16UC1";
        case CV_16UC2 :  return "CV_16UC2 ";
        case CV_16UC3 :  return "CV_16UC3 ";
        case CV_16UC4 :  return "CV_16UC4 ";

        //case CV_16SC :  return "CV_16SC ";
        case CV_16SC1 :  return "CV_16SC1";
        case CV_16SC2 :  return "CV_16SC2 ";
        case CV_16SC3 :  return "CV_16SC3 ";
        case CV_16SC4 :  return "CV_16SC4 ";

        //case CV_32FC :  return "CV_32FC ";
        case CV_32FC1 :  return "CV_32FC1";
        case CV_32FC2 :  return "CV_32FC2 ";
        case CV_32FC3 :  return "CV_32FC3 ";
        case CV_32FC4 :  return "CV_32FC4 ";

        //case CV_32SC :  return "CV_32SC ";
        case CV_32SC1 :  return "CV_32SC1";
        case CV_32SC2 :  return "CV_32SC2 ";
        case CV_32SC3 :  return "CV_32SC3 ";
        case CV_32SC4 :  return "CV_32SC4 ";

        //case CV_64FC :  return "CV_64FC";
        case CV_64FC1 :  return "CV_64FC1";
        case CV_64FC2 :  return "CV_64FC2";
        case CV_64FC3 :  return "CV_64FC3";
        case CV_64FC4 :  return "CV_64FC4";
    }
    return "_code_not_found_";
};
