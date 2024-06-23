# ifndef CV_CHK
# define CV_CHK
//         C1	C2	C3	C4
// CV_8U	0	8	16	24
// CV_8SC	1	9	17	25
// CV_16U	2	10	18	26
// CV_16S	3	11	19	27
// CV_32S	4	12	20	28
// CV_32F	5	13	21	29
// CV_64FC	6	14	22	30
#include <string>
#include <opencv2/core.hpp>

std::string checkCVtype(int code);

#endif
