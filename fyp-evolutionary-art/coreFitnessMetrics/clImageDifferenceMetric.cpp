#include "coreFitnessMetrics.h"
#include "../core/clFitnessMetric.h"
#include <cassert>
#include <cmath>

using namespace core;

// Image diff kernel.
static const char *source = R"(
const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
void kernel imageDiff(global float *error, read_only image2d_t i, read_only image2d_t target) {
	int2 coord = (int2) (get_global_id(0), get_global_id(1));
	float4 src = read_imagef(i, sampler, coord);
	float4 dst = read_imagef(target, sampler, coord);
	float4 diff = src - dst;
	error[get_global_id(1)*get_global_size(0) + get_global_id(0)] = dot(diff.xyz, diff.xyz);
}
)";

namespace fitnessMetrics {
	
class CLImageDifferenceMetric : public CLImageFitnessMetric {
	std::unique_ptr<cl::Program> program;
	std::unique_ptr<cl::Image2D> targetImage;
	std::unique_ptr<cl::Buffer> errors;
	std::vector<float> hostErrors;
public:
	CLImageDifferenceMetric(const TreeGenomeImageGenerationOptions &options, CLComputeProvider &clProvider, std::unique_ptr<cl::Image2D> targetImage) : CLImageFitnessMetric(options, clProvider), targetImage(std::move(targetImage)) {
		program.reset(new cl::Program(clProvider.getContext(), source));
		clProvider.build(*program);
		errors.reset(new cl::Buffer(clProvider.getContext(), CL_MEM_READ_WRITE, options.imageWidth * options.imageHeight * sizeof(float)));
		hostErrors.resize(options.imageWidth * options.imageHeight);
	}

	float measureFitnessForIndividual(const cl::Image2D &individual) override {
		assert(hostErrors.size() == options.imageWidth * options.imageHeight);
		cl::Kernel f(*program, "imageDiff");
		f.setArg(0, *errors);
		f.setArg(1, individual);
		f.setArg(2, *targetImage);
		auto &queue = clProvider.getMainQueue();
		queue.enqueueNDRangeKernel(f, cl::NullRange, cl::NDRange(options.imageWidth, options.imageHeight));
		queue.enqueueReadBuffer(*errors, true, 0, hostErrors.size() * sizeof(float), hostErrors.data());
		// Sum the errors here.. (FIXME: do it on device)
		float err = 0.0f;
		for (auto x : hostErrors) { err += x; }
		return 1.0f - sqrtf(err);
	}
};

std::unique_ptr<CLImageFitnessMetric> createCLImageDifferenceMetric(const TreeGenomeImageGenerationOptions &options, CLComputeProvider &clProvider, const Image &targetImage) {
	if (targetImage.getWidth() != options.imageWidth || targetImage.getHeight() != options.imageHeight) {
		assert(false && "Invalid target dimensions");
		return nullptr;
	}
	return std::unique_ptr<CLImageFitnessMetric>(new CLImageDifferenceMetric(options, clProvider, clProvider.createImage(targetImage)));
}

}
