// This file converts the file format used at http://www.doc.ic.ac.uk/~ahanda/HighFrameRateTracking/downloads.html
// into the standard [R|T] world -> camera format used by OpenCV
// It is based on a file they provided there, but makes the world coordinate system right handed, with z up, x right, and y forward.
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <string.h>
#include <stdio.h>
#include "convertAhandaPovRayToStandard.hpp"

using namespace cv;
using namespace std;
Vec3f direction;
Vec3f upvector;
void convertAhandaPovRayToStandard(int_map verbosity_mp, const char *filepath,  Mat& R,  Mat& T, Mat& cameraMatrix){
    int verbosity = verbosity_mp["verbosity"];//0;//  link to gobal verbosity
    int local_verbosity_threshold = verbosity_mp["convertAhandaPovRayToStandard"];//2;
    																													if(verbosity>local_verbosity_threshold) cout << "\n convertAhandaPovRayToStandard_chk 0"<<flush;
	
    char     text_file_name[600];                                               // open .txt file
    sprintf(text_file_name,"%s",filepath);
    ifstream cam_pars_file(text_file_name);
    if( !cam_pars_file.is_open() ){  cerr<<"Failed to open param file, check location of sample trajectory!"<<endl<< flush;  cout<< flush; exit(1); }
    char     readlinedata[300];
    Point3f  direction;
    Point3f  upvector;
    Point3f  posvector;
    Point3f  rightvector;
    																													if(verbosity>local_verbosity_threshold) {
                                                                                                                            cout << "\n convertAhandaPovRayToStandard_chk 1"<<endl<<flush;
                                                                                                                        }
	
    while(1){
        cam_pars_file.getline(readlinedata,300);                                // read line
        if ( cam_pars_file.eof() ) break;
        istringstream iss;

        if ( strstr(readlinedata,"cam_dir")!= NULL){                            // "cam_dir" direction of optical axis
            string cam_dir_str(readlinedata);
            cam_dir_str = cam_dir_str.substr(cam_dir_str.find("= [")+3);
            cam_dir_str = cam_dir_str.substr(0,cam_dir_str.find("]"));
            iss.str(cam_dir_str);
            iss >> direction.x;    iss.ignore(1,',');
            iss >> direction.z;    iss.ignore(1,',');
            iss >> direction.y;    iss.ignore(1,',');
        }
        if ( strstr(readlinedata,"cam_up")!= NULL){                             // "cam_up" orientation of image wrt world
            string cam_up_str(readlinedata);
            cam_up_str = cam_up_str.substr(cam_up_str.find("= [")+3);
            cam_up_str = cam_up_str.substr(0,cam_up_str.find("]"));
            iss.str(cam_up_str);
            iss >> upvector.x;     iss.ignore(1,',');
            iss >> upvector.z;     iss.ignore(1,',');
            iss >> upvector.y;     iss.ignore(1,',');
        }
        if ( strstr(readlinedata,"cam_pos")!= NULL){                            // "cam_pos" camera position
            string cam_pos_str(readlinedata);
            cam_pos_str = cam_pos_str.substr(cam_pos_str.find("= [")+3);
            cam_pos_str = cam_pos_str.substr(0,cam_pos_str.find("]"));
            iss.str(cam_pos_str);
            iss >> posvector.x;    iss.ignore(1,',');
            iss >> posvector.z;    iss.ignore(1,',');
            iss >> posvector.y;    iss.ignore(1,',');
        }

        if ( strstr(readlinedata,"cam_right")!= NULL){                            // "cam_right" pixel aspect ratio
            string cam_right_str(readlinedata);
            cam_right_str = cam_right_str.substr(cam_right_str.find("= [")+3);
            cam_right_str = cam_right_str.substr(0,cam_right_str.find("]"));
            iss.str(cam_right_str);
            iss >> rightvector.x;    iss.ignore(1,',');
            iss >> rightvector.z;    iss.ignore(1,',');
            iss >> rightvector.y;    iss.ignore(1,',');
        }
    }
    																													if(verbosity>local_verbosity_threshold) cout << "\n convertAhandaPovRayToStandard_chk 2"<<flush;
	
    R        = Mat(3,3,CV_32F);                                                 // compute rotation & translation
    R.row(0) = Mat(direction.cross(upvector)).t();
    R.row(1) = Mat(-upvector).t();
    R.row(2) = Mat(direction).t();
    T        = -R*Mat(posvector);
    // debug
    Point3f test_point = {1,1,1}; 
    Mat test_T = -R*Mat(test_point);
    																													if(verbosity>local_verbosity_threshold) {
                                                                                                                            cout << "\n convertAhandaPovRayToStandard_chk 3"<<flush;
                                                                                                                            cout << "\n posvector="<< posvector.x << ", " << posvector.y << ", " << posvector.z << endl << flush;
                                                                                                                            cout << "\n test_T="<< test_T.at<float>(0) << ", " << test_T.at<float>(1) << ", " << test_T.at<float>(2) << endl << flush;
                                                                                                                        }
	
    float focal_length  = norm(direction);                                      // compute intrinsic cameraMatrix
    float aspect_ratio  = norm(rightvector)/norm(upvector);
    float angle         = norm(rightvector)/norm(direction);
    int   height        = 480;
    int   width         = 640;
    float Ox            = (width +1)*0.5;
    float Oy            = (height+1)*0.5;
    float fx            = width  * norm(direction) / norm(rightvector);         // pixel size
    float fy            = height * norm(direction) / norm(upvector);
    																													if(verbosity>local_verbosity_threshold) cout << "\n convertAhandaPovRayToStandard_chk 4"<<flush;
    float K[9]  = {fx,    0,   Ox, \
                    0,   fy,   Oy, \
                    0,    0,    1
    };
    cameraMatrix = cv::Mat(3,3, CV_32FC1);
    for (int i=0; i<9; i++){ cameraMatrix.at<float>(i/3, i%3) = K[i]; }

    cam_pars_file.close();
    																													if(verbosity>local_verbosity_threshold) cout << "\n convertAhandaPovRayToStandard_chk 5"<<flush;
	
}

Mat loadDepthAhanda(int_map verbosity_mp, string filename, int r,int c,Mat cameraMatrix){
    int verbosity = verbosity_mp["verbosity"];
    int local_verbosity_threshold = verbosity_mp["loadDepthAhanda"];

    ifstream in(filename.c_str());
    int sz=r*c;
    Mat_<float> out(r,c);
    float * p=(float *)out.data;
    for(int i=0;i<sz;i++){
        in>>p[i];
        assert(p[i]!=0);
    }
    Mat_<float> K = cameraMatrix;
    float fx=K(0,0);
    float fy=K(1,1);
    float cx=K(0,2);
    float cy=K(1,2);
    for (int i=0;i<r;i++){
        for (int j=0;j<c;j++,p++){
            float x=j;
            float y=i;
            x=(x-cx)/fx;
            y=(y-cy)/fy;
            *p=*p/sqrt(x*x+y*y+1);      // converts radial depth to z-depth
        }
    }
    return out;
}

/*
 * Example: "scene_00_0000.txt",   from "ahanda-icl/Trajectory_for_Variable_Frame-Rate/200fps/200fps_GT_archieve/"
cam_pos      = [149.376, 451.41, -285.9]';
cam_dir      = [0.421738, -0.409407, 0.809026]';
cam_up       = [-0.0482194, 0.880868, 0.470899]';
cam_lookat   = [0, 0, 1]';
cam_sky      = [0, 1, 0]';
cam_right    = [1.20423, 0.316017, -0.467833]';
cam_fpoint   = [0, 0, 10]';
cam_angle    = 90;
*/

/*
 * Octave file: "getcamK_octave.m" in "ahanda-icl/camera_codes/Octave/getcamK_octave.m"
function K = getcamK_octave(cam_file)
#file reads the camera file (e.g. "scene_00_0231.txt") and gives out the K matrix.
#https://www.doc.ic.ac.uk/~ahanda/VaFRIC/codes.html

source(cam_file);
focal  =         norm(cam_dir) ;
aspect =         norm(cam_right)   / norm(cam_up) ;
angle  = 2*atan( norm(cam_right)/2 / norm(cam_dir)  ) ;

M      = 480; %cam_height
N      = 640; %cam.width

width  = N;
height = M;

% pixel size
psx = 2*focal*tan(0.5*angle)/N ;
psy = 2*focal*tan(0.5*angle)/aspect/M ;

psx   = psx / focal;
psy   = psy / focal ;
%
 Sx = psx;
 Sy = psy;

Ox = (width+1)*0.5;
Oy = (height+1)*0.5;

f = focal;

K = [1/psx     0     Ox;
       0    1/psy    Oy;
       0      0     1];

K(2,2) = -K(2,2);

end
*/
