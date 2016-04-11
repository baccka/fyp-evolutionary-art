#pragma once

// TODO: move 2 core
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include "../cl2.hpp"
#include <vector>

namespace core {
	
class Image;
	
class CLComputeProvider {
	std::string resourcePath;
	cl::Context context;
	std::vector<cl::Device> devices;
	std::unique_ptr<cl::CommandQueue> mainQueue;
public:
	
	CLComputeProvider(std::string resourcePath);
	
	const cl::Context &getContext() const;
	
	cl::CommandQueue &getMainQueue();
	
	const std::vector<cl::Device> &getDevices() const;

	void build(cl::Program &program);
	
	std::unique_ptr<cl::Image2D> createImage(const Image &image);
	
	std::string loadSource(const char *name);
};

} // end namespace core
