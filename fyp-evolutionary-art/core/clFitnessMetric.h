#pragma once

#include "opencl.h"
#include "options.h"

namespace core {

// An abstract OpenCL based fitness metric.
class CLImageFitnessMetric {
protected:
	CLComputeProvider &clProvider;
	TreeGenomeImageGenerationOptions options;
public:
	CLImageFitnessMetric(const TreeGenomeImageGenerationOptions &options, CLComputeProvider &clProvider);
	virtual ~CLImageFitnessMetric() {}
	
	virtual float measureFitnessForIndividual(const cl::Image2D &individual) = 0;
};
	
} // end namespace core
