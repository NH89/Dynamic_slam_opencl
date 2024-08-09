#include "print_functions.hpp"
using namespace std;

void print_matx33f(cv::Matx33f matx){
     for(int row=0;row<3;row++){
         cout<<"  \n";
         for(int col=0; col<3;col++){
             cout << matx.operator()(row,col) << ", \t";
         }
    }
    cout<<flush;
}

void print_matx44_32f(cv::Matx44f matx){
     for(int row=0;row<3;row++){
         cout<<"  \n";
         for(int col=0; col<3;col++){
             cout << matx.operator()(row,col) << ", \t";
         }
    }
    cout<<flush;
}

void print_matx44f(cv::Matx44f matx){
     for(int row=0;row<4;row++){
         cout<<"  \n";
         for(int col=0; col<4;col++){
             cout << matx.operator()(row,col) << ", \t";
         }
    }
    cout<<endl<<flush;
}

void print_matx61f(cv::Matx61f matx){
     for(int col=0; col<6;col++){
         cout << matx.operator()(col) << ", \t";
         }
    cout<<flush;
}

void print_matx16f(cv::Matx16f matx){
     for(int col=0; col<6;col++){
         cout << matx.operator()(0,col) << ", \t";
         }
    cout<<flush;
}

void print_matx13f(cv::Matx13f matx){
     for(int col=0; col<3;col++){
         cout << matx.operator()(0,col) << ", \t";
         }
    cout<<flush;
}

void print_matx31f(cv::Matx31f matx){
     for(int row=0; row<3;row++){
         cout << matx.operator()(row,0) << ", \t";
         }
    cout<<flush;
}

//cout << "\nT = "<<T.operator()(0)<<", "<<T.operator()(1)<<", "<<T.operator()(2)<<endl<<flush;


void print_float_6(float float_6[6]){
     for(int col=0;col<6;col++){
        cout << float_6[col] << ", \t";
    }
    cout<<flush;
}

void print_float_9(float float_9[9]){
     for(int row=0;row<3;row++){
         cout<<"  \n";
         for(int col=0; col<3;col++){
             cout << float_9[row*3 + col] << ", \t";
         }
    }
    cout<<flush;
}

void print_float_16(float float_16[16]){
     for(int row=0;row<4;row++){
         cout<<"  \n";
         for(int col=0; col<4;col++){
             cout << float_16[row*4 + col] << ", \t";
         }
    }
    cout<<flush;
}

void print_json_float_9(Json::Value obj, std::string name){
    cout << "\n\nobj["<<name<<"] ="<<flush;
    for (int row=0; row<3; row++){
        cout << "\n";
        for (int col=0; col<3; col++) cout<<"\t"<< obj[name][row*3 + col].asFloat() <<","<<flush;
    }
}

void print_matf(cv::Mat mat, int rows, int cols){
    for(int row=0;row<rows;row++){
         cout<<"  \n";
         for(int col=0; col<cols;col++){
             cout << mat.at<float>(row,col) << ", \t";
         }
    }
    cout<<flush;
}




/*
void print_float_4_16(float float_4_16[4*16]){
    for(int chan=0;chan<4;chan++){
        for(int row=0;row<4;row++){
            cout<<"  \n";
            for(int col=0; col<4;col++){
                cout << float_4_16[chan*16 + row*4 + col] << ", \t";
            }
        }
        cout<<"\n chan = "<< chan <<flush;
    }
}
*/
