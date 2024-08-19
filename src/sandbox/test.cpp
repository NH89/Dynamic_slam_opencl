// build with :
// g++ -I/usr/local/include/opencv4  -L/usr/local/lib/  -o test test.cpp -lopencv_dnn -lopencv_ml -lopencv_calib3d -lopencv_imgcodecs -lopencv_video -lopencv_gapi -lopencv_stitching -lopencv_features2d -lopencv_objdetect -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_videoio -lopencv_flann -lopencv_photo

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;


int main(){
    Mat image = Mat::ones(640, 480, CV_8UC1);
    Mat dest;

    image.convertTo(dest, CV_16FC1);

    imshow("dest", dest);
    waitKey(-1);
}
