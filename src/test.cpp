/*
#include <iostream>

struct Foo
{
    int n;
    Foo()
    {
        int i = 0;
        std::cout << "static constructor\n"  <<  2 << i;
    }
};

Foo f; // static object

int main()
{
    std::cout << "main function\n";
}
*/



#include <vector>
#include <string>
#include <fstream>
#include <filesystem>
#include <set>

namespace fs = std::filesystem;
using namespace std;


int main(){


    fs::path                root;
    vector<fs::path>        txt;
    vector<fs::path>        png;
    vector<fs::path>        depth;



}
