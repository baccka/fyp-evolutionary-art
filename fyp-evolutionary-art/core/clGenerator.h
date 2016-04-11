#pragma once

#include "opencl.h"
#include "options.h"
#include "genome.h"
#include <vector>

namespace core {

// An abstract OpenCL based image generator.
// Subclasses can implement custom genotype (tree genome) -> phenotype (image) mappings.
class CLTreeGenomeImageGenerator {
	std::unique_ptr<cl::Program> program;
	std::unique_ptr<cl::Image2D> image;
protected:
	CLComputeProvider &clProvider;
	cl::Program::Sources kernelSources;
	TreeGenomeImageGenerationOptions options;

	cl::Program &getProgram() const {
		return *program;
	}
public:
	CLTreeGenomeImageGenerator(const TreeGenomeImageGenerationOptions &options, CLComputeProvider &clProvider);
	virtual ~CLTreeGenomeImageGenerator() {}
	
	cl::Image2D &getImage();

	virtual void compileIndividual(size_t individualId, const genetic::TreeGenome &individual) = 0;
	
	virtual void generateIndividualImage(size_t individualId, const genetic::TreeGenome &individual) = 0;
	
	virtual void clear();

	virtual void build();
};

} // end namespace core
