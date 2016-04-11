#pragma once

#include "../core/image.h"
#include "../core/clFitnessMetric.h"

namespace fitnessMetrics {

std::unique_ptr<core::CLImageFitnessMetric> createCLImageDifferenceMetric(const core::TreeGenomeImageGenerationOptions &options, core::CLComputeProvider &clProvider, const core::Image &targetImage);
	
std::unique_ptr<core::CLImageFitnessMetric> createCLFractalDimensionMetric(const core::TreeGenomeImageGenerationOptions &options, core::CLComputeProvider &clProvider);

std::unique_ptr<core::CLImageFitnessMetric> createCLColorGradientMetric(const core::TreeGenomeImageGenerationOptions &options, core::CLComputeProvider &clProvider);
	
std::unique_ptr<core::CLImageFitnessMetric> createCLImageMirrorYAxisMetric(const core::TreeGenomeImageGenerationOptions &options, core::CLComputeProvider &clProvider);
	
} // end namespace fitnessMetrics
