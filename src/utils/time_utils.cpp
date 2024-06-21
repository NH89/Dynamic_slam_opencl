#include "time_utils.hpp"

std::string date_time_string(     ){

    auto now        = std::chrono::system_clock::now();
	auto in_time_t  = std::chrono::system_clock::to_time_t(now);
    auto ms         = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;

	std::stringstream datetime;
	datetime << "_" << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d_%X");
    datetime << '.' << std::setfill('0') << std::setw(3) << ms.count();
    datetime << "_" << std::put_time(std::localtime(&in_time_t), "%a") << "_";

    std::string   datetime_str = datetime.str();
    return datetime_str;
}


// #include <chrono>
// #include <ctime>
// #include <iomanip>
// #include <sstream>
// #include <string>
//
// // This is a portable method using the C++11 chrono library:
//
// std::string time_in_HH_MM_SS_MMM()  // From https://stackoverflow.com/questions/24686846/get-current-time-in-milliseconds-or-hhmmssmmm-format
// {
//     using namespace std::chrono;
//
//     // get current time
//     auto now = system_clock::now();
//
//     // get number of milliseconds for the current second
//     // (remainder after division into seconds)
//     auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;
//
//     // convert to std::time_t in order to convert to std::tm (broken time)
//     auto timer = system_clock::to_time_t(now);
//
//     // convert to broken time
//     std::tm bt = *std::localtime(&timer);
//
//     std::ostringstream oss;
//
//     oss << std::put_time(&bt, "%H:%M:%S"); // HH:MM:SS
//     oss << '.' << std::setfill('0') << std::setw(3) << ms.count();
//
//     return oss.str();
// }
