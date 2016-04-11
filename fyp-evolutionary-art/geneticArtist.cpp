#include "geneticArtist.h"
#include "imageGenerator.h"
#include "rampedHalfAndHalfInitializer.h"
#include <cstdlib>
#include <cmath>

struct ImageGenerator {
	std::string resourcePath;
	EvolutionParameters params;
	std::unique_ptr<PixelFunctionImageGenerator> traits;
    std::unique_ptr<Population> population;
	std::vector<float> pixels;
	
	ImageGenerator(const char *resourcePath) : resourcePath(resourcePath) {
		params.rng = std::mt19937(242805);
        params.mutationRate = 0.1;
        params.crossoverRate = 0.9;
		
		traits.reset(new PixelFunctionImageGenerator(resourcePath, params));

        population.reset(new Population(200, params, *traits));
		InitializerDelegate delegate(traits->genomeGrammar().typeByName("individual"));//"individual"));
		{
			RampedHalfAndHalfInitializer<EvolutionParameters::RNG> init(traits->genomeGrammar(), params.rng, &delegate);
			population->initialize(5 /*5*/, init);
		}
        population->dump();
    }
};

ImageGenerator *createImageGenerator(const char *resourcePath) {
    return new ImageGenerator(resourcePath);
}

const float *getPixels(ImageGenerator *imageGenerator) {
	return imageGenerator->pixels.data();
}

const float *generatePixels(ImageGenerator *imageGenerator) {
    for (int i = 0; i < 1; ++i) {
		auto begin = std::chrono::high_resolution_clock::now();
        imageGenerator->population->nextGeneration(/*doDump*/ false);
		auto end = std::chrono::high_resolution_clock::now();
		std::cout << "One generation took: " << std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() << " microseconds\n";
    }
    auto bestIndividualId = imageGenerator->population->evaluateGeneration();
	imageGenerator->pixels = std::move(imageGenerator->traits->readImagePixels(bestIndividualId, (*imageGenerator->population)[bestIndividualId]));
    imageGenerator->population->dump(false);
    return imageGenerator->pixels.data();
}

void destroyImageGenerator(ImageGenerator *imageGenerator) {
    delete imageGenerator;
}
