#pragma once

#include "genome.h"
#include "grammar.h"
#include "../core/clGenerator.h"
#include "../core/ephemerals.h"
#include <memory>

namespace pixelFunction {
	
/// Creates an OpenCL based pixel function image generator.
std::unique_ptr<core::CLTreeGenomeImageGenerator> createCLTreeGenomeImageGenerator(const genetic::grammar::Grammar &grammar, const core::TreeGenomeImageGenerationOptions &options, core::CLComputeProvider &clProvider, const core::Ephemerals &ephemerals);

} // end namespace pixelFunction
