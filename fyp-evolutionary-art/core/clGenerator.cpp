#include "clGenerator.h"
#include <iostream>

namespace core {

CLTreeGenomeImageGenerator::CLTreeGenomeImageGenerator(const TreeGenomeImageGenerationOptions &options, CLComputeProvider &clProvider) : clProvider(clProvider), options(options) {
}
	
cl::Image2D &CLTreeGenomeImageGenerator::getImage() {
	if (image) {
		return *image;
	}
	cl::ImageFormat format(CL_RGBA, CL_UNORM_INT8);
	image.reset(new cl::Image2D(clProvider.getContext(), CL_MEM_READ_WRITE, format, options.imageWidth, options.imageHeight));
	return *image;
}
	
void CLTreeGenomeImageGenerator::clear() {
	kernelSources.clear();
}

void CLTreeGenomeImageGenerator::build() {
	std::cout << "Building the program\n";
	program.reset(new cl::Program(clProvider.getContext(), kernelSources));
	clProvider.build(*program);
	std::cout << "Program done\n";
}
	
}
