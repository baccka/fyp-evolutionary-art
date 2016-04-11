#include "imageFilters.h"

using namespace core;

// Sobel edge detection kernel.
static const char *source = R"(
const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

void kernel sobel(write_only image2d_t output, read_only image2d_t input) {
	int2 coord = (int2) (get_global_id(0), get_global_id(1));
	int width = get_global_size(0);
	int height = get_global_size(1);
	
	// 3x3 gather.
	float4 m[9] = { 0 };
	/*for (int y = coord.y == 0 ? 0 : -1, lastY = coord.y == (height - 1) ? 0 : 1; y <= lastY; ++y) {
		for (int x = coord.x == 0 ? 0 : -1, lastX = coord.x == (width - 1) ? 0 : 1; x <= lastX; ++x) {
			m[y * 3 + x] = read_imagef(input, sampler, coord + (int2)(x, y));
		}
	}*/
	for (int y = -1; y <= 1; ++y) {
		for (int x = -1; x <= 1; ++x) {
			m[(y + 1) * 3 + (x + 1)] = read_imagef(input, sampler, coord + (int2)(x, y));
		}
	}

	// Horizontal 3x3 convolution.
	// -1 0 +1
	// -2 0 +2
	// -1 0 +1
	float4 gx = m[0] - m[2] + // m[0] * 1 + m[1] * 0 + m[2] * -1
		m[3] * 2.0f - 2.0f * m[5] + // m[3] * 2 + m[4] * 0 + m[5] * -2
		m[6] - m[8]; // m[6] * 1 + m[7] * 0 + m[8] * -1
	
	// Vertical 3x3 convolution.
	// 1 2 1
	// 0 0 0
	// -1 -2 -1
	float4 gy = -m[0] - 2.0f * m[1] - m[2] + // m[0] * -1 + m[1] * -2 + m[2] * -1
		// m[3] * 0 + m[4] * 0 + m[5] * 0
		m[6] + 2.0f * m[7] + m[8]; // m[7] * 1 + m[8] * 2 + m[9] * 1
	
	// Approximate magnitude.
	float4 g = fabs(fabs(gx) + fabs(gy));
	float grayscale = clamp( (g.x + g.y + g.z ) / 3.0f, 0.0f, 1.0f);
	
	write_imagef(output, coord, (float4) (grayscale, grayscale, grayscale, 1.0f));
}
)";

namespace image {

// Implements the Sobel edge detection kernel using OpenCL.
class CLSobelFilter : public CLFilter {
	CLComputeProvider &clProvider;
	std::unique_ptr<cl::Program> program;
	std::unique_ptr<cl::Image2D> output;
	int resultWidth = 0, resultHeight = 0;
public:
	CLSobelFilter(CLComputeProvider &clProvider) : clProvider(clProvider) {
		program.reset(new cl::Program(clProvider.getContext(), source));
		clProvider.build(*program);
	}

	const cl::Image2D &apply(const cl::Image2D &input, int width, int height) override {
		if (!output || width != resultWidth || height != resultHeight) {
			// Create a one channel image.
			cl::ImageFormat format(CL_R, CL_UNORM_INT8);
			output.reset(new cl::Image2D(clProvider.getContext(), CL_MEM_READ_WRITE, format, width, height));
			resultWidth = width;
			resultHeight = height;
		}
		cl::Kernel f(*program, "sobel");
		f.setArg(0, *output);
		f.setArg(1, input);
		clProvider.getMainQueue().enqueueNDRangeKernel(f, cl::NullRange, cl::NDRange(width, height));
		return *output;
	}
};

std::unique_ptr<CLFilter> createCLSobelFilter(CLComputeProvider &clProvider) {
	return std::unique_ptr<CLFilter>(new CLSobelFilter(clProvider));
}
	
}