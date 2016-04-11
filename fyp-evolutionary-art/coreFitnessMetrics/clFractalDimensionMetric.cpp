#include "coreFitnessMetrics.h"
#include "../imageFilters/imageFilters.h"
#include <cassert>
#include <cmath>

using namespace core;
using namespace image;

// Image diff kernel.
static const char *source = R"(
const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

void kernel imageBoxCount(global int *boxes, read_only image2d_t i, const int boxSize) {
	int2 coord = ((int2) (get_global_id(0), get_global_id(1))) * boxSize;
	int coverage = 0;
	for (int y = 0; y < boxSize; y++) {
		for (int x = 0; x < boxSize; x++) {
			float4 val = read_imagef(i, sampler, coord + (int2)(x, y));
			if ((val.x + val.y + val.z)/3.0f > 0.5f) {
				coverage++;
			}
		}
	}
	boxes[get_global_id(1)*get_global_size(0) + get_global_id(0)] = coverage;
}
)";

namespace fitnessMetrics {
	
class CLFractalDimensionMetric : public CLImageFitnessMetric {
	std::unique_ptr<cl::Program> program;
	std::unique_ptr<cl::Buffer> boxes;
	std::unique_ptr<CLFilter> edgeDetectionFilter;
public:
	CLFractalDimensionMetric(const TreeGenomeImageGenerationOptions &options, CLComputeProvider &clProvider) : CLImageFitnessMetric(options, clProvider) {
		program.reset(new cl::Program(clProvider.getContext(), source));
		clProvider.build(*program);
		boxes.reset(new cl::Buffer(clProvider.getContext(), CL_MEM_READ_WRITE, options.imageWidth * options.imageHeight * sizeof(int)));
		edgeDetectionFilter = createCLSobelFilter(clProvider);
	}

	float computeFractalDimension(int boxSize, const cl::Image2D &individual) {
		if ((options.imageWidth % boxSize) != 0 || (options.imageHeight % boxSize) != 0) {
			assert(false && "Invalid box / image size combination");
			return 0.0f;
		}
		auto boxRange = cl::NDRange(options.imageWidth/boxSize, options.imageHeight/boxSize);
		cl::Kernel f(*program, "imageBoxCount");
		f.setArg(0, *boxes);
		f.setArg(1, individual);//edgeDetectionFilter->apply(individual, options.imageWidth, options.imageHeight));
		f.setArg(2, boxSize);
		auto &queue = clProvider.getMainQueue();
		queue.enqueueNDRangeKernel(f, cl::NullRange, boxRange);
		std::vector<int> bs(boxRange[0]*boxRange[1]);
		queue.enqueueReadBuffer(*boxes, true, 0, bs.size()*sizeof(int), bs.data());
		//
		int N = 0;
		for (auto b : bs) {
			if (b) N++;
		}
		return std::logf(float(N)) / std::logf(1.0f / (float(boxSize) / float(options.imageWidth)));
	}

	float measureFitnessForIndividual(const cl::Image2D &individual) override {
		float x = computeFractalDimension(options.imageWidth / 4, individual);
		float y = computeFractalDimension(options.imageWidth / 8, individual);
		float z = computeFractalDimension(options.imageWidth / 16, individual);
		//float w = computeFractalDimension(4);
		float d = (x + y + z) / 3.0f;
		// M (I) = max(0, 1 − |1.35 − d(I)|)
		return std::max(0.0f, 1.0f - std::abs(1.35f - d));
	}
};

std::unique_ptr<CLImageFitnessMetric> createCLFractalDimensionMetric(const TreeGenomeImageGenerationOptions &options, CLComputeProvider &clProvider) {
	return std::unique_ptr<CLImageFitnessMetric>(new CLFractalDimensionMetric(options, clProvider));
}
	
}
