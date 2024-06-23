#include "RunCL.hpp"


string  RunCL::checkerror(int input) {
		int errorCode = input;
		switch (errorCode) {
		case -9999:											return "Illegal read or write to a buffer";		// NVidia error code
		case CL_DEVICE_NOT_FOUND:							return "CL_DEVICE_NOT_FOUND";
		case CL_DEVICE_NOT_AVAILABLE:						return "CL_DEVICE_NOT_AVAILABLE";
		case CL_COMPILER_NOT_AVAILABLE:						return "CL_COMPILER_NOT_AVAILABLE";
		case CL_MEM_OBJECT_ALLOCATION_FAILURE:				return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
		case CL_OUT_OF_RESOURCES:							return "CL_OUT_OF_RESOURCES";
		case CL_OUT_OF_HOST_MEMORY:							return "CL_OUT_OF_HOST_MEMORY";
		case CL_PROFILING_INFO_NOT_AVAILABLE:				return "CL_PROFILING_INFO_NOT_AVAILABLE";
		case CL_MEM_COPY_OVERLAP:							return "CL_MEM_COPY_OVERLAP";
		case CL_IMAGE_FORMAT_MISMATCH:						return "CL_IMAGE_FORMAT_MISMATCH";
		case CL_IMAGE_FORMAT_NOT_SUPPORTED:					return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
		case CL_BUILD_PROGRAM_FAILURE:						return "CL_BUILD_PROGRAM_FAILURE";
		case CL_MAP_FAILURE:								return "CL_MAP_FAILURE";
		case CL_MISALIGNED_SUB_BUFFER_OFFSET:				return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
		case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:	return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
		case CL_INVALID_VALUE:								return "CL_INVALID_VALUE";
		case CL_INVALID_DEVICE_TYPE:						return "CL_INVALID_DEVICE_TYPE";
		case CL_INVALID_PLATFORM:							return "CL_INVALID_PLATFORM";
		case CL_INVALID_DEVICE:								return "CL_INVALID_DEVICE";
		case CL_INVALID_CONTEXT:							return "CL_INVALID_CONTEXT";
		case CL_INVALID_QUEUE_PROPERTIES:					return "CL_INVALID_QUEUE_PROPERTIES";
		case CL_INVALID_COMMAND_QUEUE:						return "CL_INVALID_COMMAND_QUEUE";
		case CL_INVALID_HOST_PTR:							return "CL_INVALID_HOST_PTR";
		case CL_INVALID_MEM_OBJECT:							return "CL_INVALID_MEM_OBJECT";
		case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:			return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
		case CL_INVALID_IMAGE_SIZE:							return "CL_INVALID_IMAGE_SIZE";
		case CL_INVALID_SAMPLER:							return "CL_INVALID_SAMPLER";
		case CL_INVALID_BINARY:								return "CL_INVALID_BINARY";
		case CL_INVALID_BUILD_OPTIONS:						return "CL_INVALID_BUILD_OPTIONS";
		case CL_INVALID_PROGRAM:							return "CL_INVALID_PROGRAM";
		case CL_INVALID_PROGRAM_EXECUTABLE:					return "CL_INVALID_PROGRAM_EXECUTABLE";
		case CL_INVALID_KERNEL_NAME:						return "CL_INVALID_KERNEL_NAME";
		case CL_INVALID_KERNEL_DEFINITION:					return "CL_INVALID_KERNEL_DEFINITION";
		case CL_INVALID_KERNEL:								return "CL_INVALID_KERNEL";
		case CL_INVALID_ARG_INDEX:							return "CL_INVALID_ARG_INDEX";
		case CL_INVALID_ARG_VALUE:							return "CL_INVALID_ARG_VALUE";
		case CL_INVALID_ARG_SIZE:							return "CL_INVALID_ARG_SIZE";
		case CL_INVALID_KERNEL_ARGS:						return "CL_INVALID_KERNEL_ARGS";
		case CL_INVALID_WORK_DIMENSION:						return "CL_INVALID_WORK_DIMENSION";
		case CL_INVALID_WORK_GROUP_SIZE:					return "CL_INVALID_WORK_GROUP_SIZE";
		case CL_INVALID_WORK_ITEM_SIZE:						return "CL_INVALID_WORK_ITEM_SIZE";
		case CL_INVALID_GLOBAL_OFFSET:						return "CL_INVALID_GLOBAL_OFFSET";
		case CL_INVALID_EVENT_WAIT_LIST:					return "CL_INVALID_EVENT_WAIT_LIST";
		case CL_INVALID_EVENT:								return "CL_INVALID_EVENT";
		case CL_INVALID_OPERATION:							return "CL_INVALID_OPERATION";
		case CL_INVALID_GL_OBJECT:							return "CL_INVALID_GL_OBJECT";
		case CL_INVALID_BUFFER_SIZE:						return "CL_INVALID_BUFFER_SIZE";
		case CL_INVALID_MIP_LEVEL:							return "CL_INVALID_MIP_LEVEL";
		case CL_INVALID_GLOBAL_WORK_SIZE:					return "CL_INVALID_GLOBAL_WORK_SIZE";
#if CL_HPP_MINIMUM_OPENCL_VERSION >= 200
		case CL_INVALID_DEVICE_QUEUE:						return "CL_INVALID_DEVICE_QUEUE";
		case CL_INVALID_PIPE_SIZE:							return "CL_INVALID_PIPE_SIZE";
#endif
		default:											return "unknown error code";
		}
}

/*
string  RunCL::checkCVtype(int input) {
		int errorCode = input;
		switch (errorCode) {
			//case	CV_8U:										return "CV_8U";
			case	CV_8UC1:										return "CV_8UC1";
			case	CV_8UC2:										return "CV_8UC2";
			case	CV_8UC3:										return "CV_8UC3";
			case	CV_8UC4:										return "CV_8UC4";

			//case	CV_8S:										return "CV_8S";
			case	CV_8SC1:										return "CV_8SC1";
			case	CV_8SC2:										return "CV_8SC2";
			case	CV_8SC3:										return "CV_8SC3";
			case	CV_8SC4:										return "CV_8SC4";

			//case	CV_16U:										return "CV_16U";
			case	CV_16UC1:										return "CV_16UC1";
			case	CV_16UC2:										return "CV_16UC2";
			case	CV_16UC3:										return "CV_16UC3";
			case	CV_16UC4:										return "CV_16UC4";

			//case	CV_16S:										return "CV_16S";
			case	CV_16SC1:										return "CV_16SC1";
			case	CV_16SC2:										return "CV_16SC2";
			case	CV_16SC3:										return "CV_16SC3";
			case	CV_16SC4:										return "CV_16SC4";

			//case	CV_32S:										return "CV_32S";
			case	CV_32SC1:										return "CV_32SC1";
			case	CV_32SC2:										return "CV_32SC2";
			case	CV_32SC3:										return "CV_32SC3";
			case	CV_32SC4:										return "CV_32SC4";

			//case	CV_16F
			case	CV_16FC1:										return "CV_16FC1";
			case	CV_16FC2:										return "CV_16FC2";
			case	CV_16FC3:										return "CV_16FC3";
			case	CV_16FC4:										return "CV_16FC4";

			//case	CV_32F:										return "CV_32F";
			case	CV_32FC1:										return "CV_32FC1";
			case	CV_32FC2:										return "CV_32FC2";
			case	CV_32FC3:										return "CV_32FC3";
			case	CV_32FC4:										return "CV_32FC4";

			//case	CV_64F:										return "CV_64F";
			case	CV_64FC1:										return "CV_64FC1";
			case	CV_64FC2:										return "CV_64FC2";
			case	CV_64FC3:										return "CV_64FC3";
			case	CV_64FC4:										return "CV_64FC4";

			default:										return "unknown CV_type code";
		}
}
*/
