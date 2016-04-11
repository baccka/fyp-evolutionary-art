#include "clFitnessMetric.h"

namespace core {
	
CLImageFitnessMetric::CLImageFitnessMetric(const TreeGenomeImageGenerationOptions &options, CLComputeProvider &clProvider) : clProvider(clProvider), options(options) {		
}

}