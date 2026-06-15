#include "print_functions.hpp"
using namespace std;

#define PRECISION 15

void print_matx33f(cv::Matx33f matx){
     for(int row=0;row<3;row++){
         cout<<"  \n";
         for(int col=0; col<3;col++){
             cout << setw(9) << matx.operator()(row,col) << ", \t\t";
         }
    }
    cout<<flush;
}

void print_matx44f(cv::Matx44f matx){
    cout << fixed << setprecision( PRECISION ) ;
    for(int row=0;row<4;row++){
         cout<<"  \n";
         for(int col=0; col<4;col++){
             cout << setw(9)<< matx.operator()(row,col) << ", \t\t";
         }
    }
    cout<<endl<<flush;
}

void print_matx55d(Matx55d matx){
    cout << fixed << setprecision( PRECISION ) ;
    for(int row=0;row<5;row++){
         cout<<"  \n";
         for(int col=0; col<5;col++){
             cout << setw(9)<< matx.operator()(row,col) << ", \t\t";
         }
    }
    cout<<endl<<flush;
}

void print_matx66f(cv::Matx66f matx){
    cout << fixed << setprecision( PRECISION ) ;
    for(int row=0;row<6;row++){
         cout<<"  \n";
         for(int col=0; col<6;col++){
             cout << setw(9)<< matx.operator()(row,col) << ", \t\t";
         }
    }
    cout<<endl<<flush;
}

void print_matx61f(cv::Matx61f matx){
    cout<<"  \n";
    for(int col=0; col<6;col++){
         cout << matx.operator()(col) << ", \t\t";
         }
    cout<<flush;
}

void print_matx16f(cv::Matx16f matx){
    cout<<"  \n";
    for(int col=0; col<6;col++){
         cout << matx.operator()(0,col) << ", \t\t";
         }
    cout<<flush;
}



void print_matx13f(cv::Matx13f matx){
    cout<<"  \n";
    for(int col=0; col<3;col++){
         cout << matx.operator()(0,col) << ", \t\t";
         }
    cout<<flush;
}

void print_matx31f(cv::Matx31f matx){
    cout<<"  \n";
    for(int row=0; row<3;row++){
         cout << matx.operator()(row,0) << ", \t\t";
         }
    cout<<flush;
}

void print_matx15d(Matx15d matx){
    cout<<"  \n";
    for(int col=0; col<5;col++){
         cout << matx.operator()(0,col) << ", \t\t";
         }
    cout<<flush;
}


//cout << "\nT = "<<T.operator()(0)<<", "<<T.operator()(1)<<", "<<T.operator()(2)<<endl<<flush;


void print_float_6(float float_6[6]){
    cout<<"  \n";
    for(int col=0;col<6;col++){
        cout << float_6[col] << ", \t\t";
    }
    cout<<flush;
}

void print_float_9(float float_9[9]){
    cout << fixed << setprecision( PRECISION ) ;
     for(int row=0;row<3;row++){
         cout<<"  \n";
         for(int col=0; col<3;col++){
             cout  << setw(9) << float_9[row*3 + col] << ", \t\t";
         }
    }
    cout<<flush;
}

void print_float_16(float float_16[16]){
    cout << fixed << setprecision( PRECISION ) ;
     for(int row=0;row<4;row++){
         cout<<"  \n";
         for(int col=0; col<4;col++){
             cout  << setw(9) << float_16[row*4 + col] << ", \t\t";
         }
    }
    cout<<flush;
}


void print_cl_float16(cl_float16 flt16){
	cout << fixed << setprecision( PRECISION ) ;
	cout << "\n"
	<< flt16.s0		<< ", "
	<< flt16.s1		<< ", "
	<< flt16.s2		<< ", "
	<< flt16.s3		<< ", "
	<< "\n"
	<< flt16.s4		<< ", "
	<< flt16.s5		<< ", "
	<< flt16.s6		<< ", "
	<< flt16.s7		<< ", "
	<< "\n"
	<< flt16.s8		<< ", "
	<< flt16.s9		<< ", "
	<< flt16.sa		<< ", "
	<< flt16.sb		<< ", "
	<< "\n"
	<< flt16.sc		<< ", "
	<< flt16.sd		<< ", "
	<< flt16.se		<< ", "
	<< flt16.sf		<< ", "
	<<flush;
}



void print_json_float_9(Json::Value obj, std::string name){
    cout << fixed << setprecision( PRECISION ) ;
    cout << "\n\nobj["<<name<<"] ="<<flush;
    for (int row=0; row<3; row++){
        cout << "\n";
        for (int col=0; col<3; col++) cout << setw(9) <<"\t\t"<< obj[name][row*3 + col].asFloat() <<","<<flush;
    }
}

void print_matf(cv::Mat mat, int rows, int cols){
    cout << fixed << setprecision( PRECISION ) ;
    for(int row=0;row<rows;row++){
         cout<<"  \n";
         for(int col=0; col<cols;col++){
             cout << setw(9) << mat.at<float>(row,col) << ", \t\t";
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
                cout << float_4_16[chan*16 + row*4 + col] << ", \t\t";
            }
        }
        cout<<"\n chan = "<< chan <<flush;
    }
}
*/
