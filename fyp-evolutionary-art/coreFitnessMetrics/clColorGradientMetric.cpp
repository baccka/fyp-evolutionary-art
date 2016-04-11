#include "coreFitnessMetrics.h"
#include <cassert>
#include <cmath>

using namespace core;

// Image diff kernel.
static const char *source = R"(
const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

void kernel imageColorGradient(global float *output, read_only image2d_t i, const float dSquared, const float thresholdOfDetection) {
	int2 coord = ((int2) (get_global_id(0), get_global_id(1)));
	// Component's portion:
	// (r_i_j)^2 = ((r_i_j - r_i+1_j+1)^2 + (r_i+1_j - r_i_j+1)^2) / d^2
	float4 v00 = read_imagef(i, sampler, coord);
	float4 v11 = read_imagef(i, sampler, coord + (int2)(1, 1));
	float4 v10 = read_imagef(i, sampler, coord + (int2)(1, 0));
	float4 v01 = read_imagef(i, sampler, coord + (int2)(0, 1));
	float4 val = (pow(v00 - v11, 2.0f) + pow(v10 - v01, 2.0f)) / dSquared;
	// Overall gradient (or stimulus):
	float stimulus = sqrt(val.r + val.g + val.b);
	// Response:
	float response = log(stimulus/thresholdOfDetection);
	output[get_global_id(1)*get_global_size(0) + get_global_id(0)] = stimulus;
}
)";

namespace fitnessMetrics {

class CLColorGradientMetric : public CLImageFitnessMetric {
	std::unique_ptr<cl::Program> program;
	std::unique_ptr<cl::Buffer> response;
public:
	CLColorGradientMetric(const TreeGenomeImageGenerationOptions &options, CLComputeProvider &clProvider) : CLImageFitnessMetric(options, clProvider) {
		program.reset(new cl::Program(clProvider.getContext(), source));
		clProvider.build(*program);
		response.reset(new cl::Buffer(clProvider.getContext(), CL_MEM_READ_WRITE, (options.imageWidth - 1) * (options.imageHeight - 1) * sizeof(float)));
	}
	
	
	float computeDFN(const cl::Image2D &individual, float dSquared, float thresholdOfDetection, float expectedMean, float expectedStddev) {
		auto &queue = clProvider.getMainQueue();
		// Compute the response
		cl::Kernel f(*program, "imageColorGradient");
		f.setArg(0, *response);
		f.setArg(1, individual);
		f.setArg(2, dSquared);
		f.setArg(3, thresholdOfDetection);
		queue.enqueueNDRangeKernel(f, cl::NullRange, cl::NDRange(options.imageWidth - 1, options.imageHeight - 1));
		std::vector<float> r((options.imageWidth - 1)*(options.imageHeight - 1));
		queue.enqueueReadBuffer(*response, true, 0, r.size()*sizeof(float), r.data());
		// Compute the mean
		float sum = 0.0f;
		float mn = r[0], mx = r[0];
		float sumSquared = 0.0f;
		for (auto x: r) {
			sum += x;
			mn = std::min(x, mn);
			mx = std::max(x, mx);
			sumSquared += x * x;
		}
		float mean = sumSquared/sum;
		float y = 0.0f;
		for (auto x: r) {
			float d = x - mean;
			y += x * (d * d);
		}
		float stddev = y / sum;
		
		return powf(mean - expectedMean, 2.0f) + powf(stddev - expectedStddev, 2.0f);
		
		/*// Tabulate the response into a histogram
		float binWidth = stddev/100.0f;
		auto responseToBin = [=] (float x) -> int {
			return (int)((x - mn)/binWidth);
		};
		std::vector<float> histogram(responseToBin(mx) + 1, 0.0f);
		for (auto x: r) {
			histogram[responseToBin(x)] += x;
		}
		//
		float DFN = 0.0f;
		std::vector<float> normalHistogram(histogram.size(), 0.0f);
		for (size_t i = 0; i < histogram.size(); ++i) {
			if (normalHistogram[i] < 0.0001) { continue; }
			DFN += histogram[i]*logf(histogram[i]/normalHistogram[i]);
		}
		DFN *= 1000.0f;
		return 0.0f;*/
	}
	
	float measureFitnessForIndividual(const cl::Image2D &individual) override {
		float dfn = computeDFN(individual, powf(hypotf(options.imageWidth, options.imageHeight)*0.5*0.001, 2.0f), /*threshold*/2.0f, 3.75f, 0.75f);
		return 1.0f - dfn;
	}
};

std::unique_ptr<CLImageFitnessMetric> createCLColorGradientMetric(const TreeGenomeImageGenerationOptions &options, CLComputeProvider &clProvider) {
	return std::unique_ptr<CLImageFitnessMetric>(new CLColorGradientMetric(options, clProvider));
}
	
}
