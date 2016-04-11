#pragma once 

#include "../core/opencl.h"
#include "../core/options.h"

namespace image {

// An abstract OpenCL image filter.
class CLFilter {
public:
	virtual ~CLFilter() { }
	
	virtual const cl::Image2D &apply(const cl::Image2D &input, int width, int height) = 0;
};

// Creates an OpenCL based Sobel edge detection filter.
std::unique_ptr<CLFilter> createCLSobelFilter(core::CLComputeProvider &clProvider);

} // end namespace image
