#include "opencl.h"
#include "image.h"
#include <iostream>
#include <fstream>
#include <cassert>

namespace core {
	
CLComputeProvider::CLComputeProvider(std::string resourcePath) : resourcePath(resourcePath) {
	std::vector<cl::Platform> all_platforms;
	cl::Platform::get(&all_platforms);
	if (all_platforms.size() == 0){
		std::cerr << "No platforms found. Check OpenCL installation!\n";
		exit(1);
	}
	cl::Platform defaultPlatform = all_platforms[0];
	
	std::vector<cl::Device> all_devices;
	defaultPlatform.getDevices(CL_DEVICE_TYPE_GPU, &all_devices);
	if (all_devices.size()==0){
		std::cerr << "No devices found. Check OpenCL installation!\n";
		exit(1);
	}
	
	devices = { all_devices[0] };
	for (const auto &device: devices) {
		std::cout << "OpenCL information:\n";
		std::cout << "\tDevice Name: " << device.getInfo<CL_DEVICE_NAME>() << "\n";
		std::cout << "\tDevice Type: " << device.getInfo<CL_DEVICE_TYPE>();
		std::cout << " (GPU: " << CL_DEVICE_TYPE_GPU << ", CPU: " << CL_DEVICE_TYPE_CPU << ")" << "\n";
		std::cout << "\tDevice Vendor: " << device.getInfo<CL_DEVICE_VENDOR>() << "\n";
	}
	context = cl::Context(getDevices());
}

const cl::Context &CLComputeProvider::getContext() const {
	return context;
}

cl::CommandQueue &CLComputeProvider::getMainQueue() {
	if (mainQueue) {
		return *mainQueue;
	}
	mainQueue.reset(new cl::CommandQueue(context, devices[0]));
	return *mainQueue;
}

const std::vector<cl::Device> &CLComputeProvider::getDevices() const {
	return devices;
}

void CLComputeProvider::build(cl::Program &program) {
	if (program.build(getDevices()) != CL_SUCCESS){
		std::cerr << "Error building CL program: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(getDevices()[0]) << "\n";
		exit(1);
	}
}
	
std::unique_ptr<cl::Image2D> CLComputeProvider::createImage(const Image &image) {
	if (image.getComponentCount() != 4) {
		assert(false && "Invalid image format");
		return nullptr;
	}
	cl::ImageFormat format(CL_RGBA, CL_UNORM_INT8);
	return std::unique_ptr<cl::Image2D>(new cl::Image2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, format, image.getWidth(), image.getHeight(), 0, (void*)const_cast<uint8_t *>(image.data())));
}

std::string CLComputeProvider::loadSource(const char *name) {
	std::string path = resourcePath + "/" + name;
	std::ifstream t(path);
	std::string source((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
	return source;
}

}

