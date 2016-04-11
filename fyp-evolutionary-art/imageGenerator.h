#pragma once

#include "geneticProgramming.h"
#include "treeCompiler.h"
#include "treeGenerator.h"
#include "pixelFunctionGenerator/pixelFunctionGenerator.h"
#include "coreFitnessMetrics/coreFitnessMetrics.h"
#include "core/image.h"
#include "rampedHalfAndHalfInitializer.h"
//#include "clGenerator.h"
#include <vector>
#include <iostream>

using namespace core;
using namespace genetic;
using namespace grammar;

const Type scalarRoot = type("float_0");
const Type scalar = type("float");
const Type vec = type("float3");
const Type individualType = type("individual");
static const Grammar imageGrammar = Grammar({ scalarRoot, scalar, vec, individualType }, {
	terminal("x", scalarRoot, 100),
	terminal("y", scalarRoot, 100),
	terminal("ephemeral", scalarRoot, 20),
	binaryFunction("+", scalarRoot, {scalarRoot, scalarRoot}, 50),
	binaryFunction("-", scalarRoot, {scalarRoot, scalarRoot}, 50),
	binaryFunction("*", scalarRoot, {scalarRoot, scalarRoot}, 50),
	binaryFunction("/", scalarRoot, {scalarRoot, scalarRoot}, 50),
	binaryFunction("min", scalarRoot, {scalarRoot, scalarRoot}, 50),
	binaryFunction("max", scalarRoot, {scalarRoot, scalarRoot}, 50),
	binaryFunction("pow", scalarRoot, {scalarRoot, scalarRoot}, 50),
	unaryFunction("-", scalarRoot, scalarRoot, 50),
	unaryFunction("sin", scalarRoot, scalarRoot, 50),
	unaryFunction("cos", scalarRoot, scalarRoot, 50),
	unaryFunction("sqrt", scalarRoot, scalarRoot, 50),
	unaryFunction("sign", scalarRoot, scalarRoot, 50),
	// Bitwise operations
	binaryFunction("^", scalarRoot, {scalarRoot, scalarRoot}, 50),
	binaryFunction("&", scalarRoot, {scalarRoot, scalarRoot}, 50),
	binaryFunction("|", scalarRoot, {scalarRoot, scalarRoot}, 50),
	// Noise functions
    //binaryFunction("hash", scalarRoot, {scalarRoot, scalarRoot}, 50),
	binaryFunction("fbm4", scalarRoot, {scalarRoot, scalarRoot}, 50),
	binaryFunction("fbm6", scalarRoot, {scalarRoot, scalarRoot}, 50),
	
	terminal("x", scalar, 100),
	terminal("y", scalar, 100),
	terminal("ephemeral", scalar, 20),
	binaryFunction("+", scalar, {scalar, scalar}, 50),
	binaryFunction("-", scalar, {scalar, scalar}, 50),
	binaryFunction("*", scalar, {scalar, scalar}, 50),
	binaryFunction("/", scalar, {scalar, scalar}, 50),
	binaryFunction("min", scalar, {scalar, scalar}, 50),
	binaryFunction("max", scalar, {scalar, scalar}, 50),
	binaryFunction("pow", scalar, {scalar, scalar}, 50),
	unaryFunction("-", scalar, scalar, 50),
	unaryFunction("sin", scalar, scalar, 50),
	unaryFunction("cos", scalar, scalar, 50),
	unaryFunction("sqrt", scalar, scalar, 50),
	unaryFunction("sign", scalar, scalar, 50),
	// Bitwise operations
	binaryFunction("^", scalar, {scalar, scalar}, 50),
	binaryFunction("&", scalar, {scalar, scalar}, 50),
	binaryFunction("|", scalar, {scalar, scalar}, 50),
	// Noise functions
    //binaryFunction("hash", scalar, {scalar, scalar}, 50),
	binaryFunction("fbm4", scalar, {scalar, scalar}, 50),
	binaryFunction("fbm6", scalar, {scalar, scalar}, 50),
	// Calls
	binaryFunction("call", scalar, {scalar, scalar}, 600),
	// RGB restriction.
	ternaryFunction("(float3)", vec, {scalar, scalar, scalar}, 10),
	
	ternaryFunction("function-set", individualType, { scalarRoot, scalar, vec }, 1)
});
/*const Type scalar = type("float");
const Type vec = type("float3");
static const Grammar imageGrammar = Grammar({ scalar, vec }, {
	terminal("x", scalar, 100),
	terminal("y", scalar, 100),
	terminal("ephemeral", scalar, 20),
	binaryFunction("+", scalar, {scalar, scalar}, 50),
	binaryFunction("-", scalar, {scalar, scalar}, 50),
	binaryFunction("*", scalar, {scalar, scalar}, 50),
	binaryFunction("/", scalar, {scalar, scalar}, 50),
	binaryFunction("min", scalar, {scalar, scalar}, 50),
	binaryFunction("max", scalar, {scalar, scalar}, 50),
	binaryFunction("pow", scalar, {scalar, scalar}, 20),
	unaryFunction("-", scalar, scalar, 50),
	unaryFunction("sin", scalar, scalar, 50),
	unaryFunction("cos", scalar, scalar, 50),
	unaryFunction("sqrt", scalar, scalar, 50),
	unaryFunction("sign", scalar, scalar, 50),
	// Bitwise operations
	binaryFunction("^", scalar, {scalar, scalar}, 50),
	binaryFunction("&", scalar, {scalar, scalar}, 50),
	binaryFunction("|", scalar, {scalar, scalar}, 50),
	// Noise functions.
	binaryFunction("fbm4", scalar, {scalar, scalar}, 50),
	binaryFunction("fbm6", scalar, {scalar, scalar}, 50),
	// RGB restriction.
	ternaryFunction("(float3)", vec, {scalar, scalar, scalar}, 10)
});*/

class InitializerDelegate: public RampedHalfAndHalfInitializerDelegate {
public:
	TreeGenomeType rootType;

	InitializerDelegate(TreeGenomeType rootType) : rootType(rootType) { }
	
	bool generateFull(TreeGenerator<EvolutionParameters::RNG> &generator, TreeGenome::Builder &builder, int maxDepth) override {
		generator.generateFull(builder, maxDepth + 1, rootType);
		return true;
	}

	bool generateGrow(TreeGenerator<EvolutionParameters::RNG> &generator, TreeGenome::Builder &builder, int maxDepth) override {
		generator.generateGrow(builder, maxDepth + 1, rootType);
		return true;
	}
};

class PixelFunctionImageGenerator: public EvolvingPopulationDelegate {
public:
    static const unsigned imageSize = 512;
	EvolutionParameters &params;
	
	TreeGenomeImageGenerationOptions options;
    std::unique_ptr<Image> targetImage;
	std::unique_ptr<Ephemerals> ephemerals;
	std::unique_ptr<CLComputeProvider> clProvider;
	std::unique_ptr<CLTreeGenomeImageGenerator> generator;
	std::unique_ptr<CLImageFitnessMetric> fitnessMetric;
	std::unique_ptr<CLImageFitnessMetric> fitnessMetric2;
    
	PixelFunctionImageGenerator(std::string resourcePath, EvolutionParameters &params) : params(params) {
        targetImage.reset(new Image("/Users/alex/Documents/target_1.png"));
		
		options.imageWidth = imageSize;
		options.imageHeight = imageSize;
		
		ephemerals.reset(new Ephemerals(params, /*count=*/ 20));
		clProvider.reset(new CLComputeProvider(resourcePath));
		generator = pixelFunction::createCLTreeGenomeImageGenerator(imageGrammar, options, *clProvider, *ephemerals);
		fitnessMetric =/* fitnessMetrics::createCLColorGradientMetric(options, *clProvider);*/fitnessMetrics::createCLFractalDimensionMetric(options, *clProvider);//fitnessMetrics::createCLImageDifferenceMetric(options, *clProvider, *targetImage);
		fitnessMetric2 = fitnessMetrics::createCLImageDifferenceMetric(options, *clProvider, *targetImage);
    }
    
    TreeGenome generateRandomTreeOfType(TreeGenomeType type) override {
		TreeGenome genome;
		TreeGenerator<EvolutionParameters::RNG> generator(genomeGrammar(), params.rng);
		TreeGenome::Builder builder(genome);
		generator.generateGrow(builder, /*maxDepth=*/ 3, type);
		return genome;
    }
	
	std::vector<float> readImagePixels(size_t individualId, const genetic::TreeGenome &individual) {
		std::vector<float> results(options.imageWidth * options.imageHeight * 4); // RGBA
		generator->clear();
		generator->compileIndividual(individualId, individual);
		generator->build();
		std::vector<uint8_t> ps(results.size());
		generator->generateIndividualImage(individualId, individual);
		clProvider->getMainQueue().enqueueReadImage(generator->getImage(), true, {0,0,0}, {512,512,1}, 0, 0, ps.data());
		for (size_t i = 0; i < results.size(); ++i) {
			results[i] = float(ps[i]) / 255.0f;
		}
		return results;
	}
	
	void computeFitness(const std::vector<TreeGenome> &individuals, std::vector<float> &fitnesses) override {
		const size_t batchSize = individuals.size();
		for (size_t batch = 0; batch < individuals.size(); batch += batchSize) {
			if (batch != 0) {
				sleep(1);
			}
			generator->clear();
			size_t end = std::min(individuals.size(), batch + batchSize);
			for (size_t i = batch; i < end; ++i) {
				generator->compileIndividual(i, individuals[i]);
			}
			generator->build();
			for (size_t i = batch; i < end; ++i) {
				fitnesses[i] = computeFitnessForIndividual(i, individuals[i]);
			}
		}
	}
	
    float computeFitnessForIndividual(size_t individualId, const TreeGenome &individual) {
		generator->generateIndividualImage(individualId, individual);
		//float p2 = individual.getNodeCount() > 150 ? 100.0f : 0.0f;
		//return fitnessMetric2->measureFitnessForIndividual(generator->getImage()) - p2;
		
		float err = fitnessMetric->measureFitnessForIndividual(generator->getImage());
		float err2 = 0.0f;// * fitnessMetric2->measureFitnessForIndividual(generator->getImage());
		float penalty = individual.getNodeCount() > 150 ? 100.0f : 0.0f;
		return (err + err2) - penalty;
		//return 1.0f - penalty - sqrtf(err);
		
		/*float dfn = clgen->computeDFN(512, 512, powf(hypotf(512, 512)*0.5*0.001, 2.0f), threshold/2.0f, 3.75f, 0.75f);
        float d = clgen->computeFractalDimension();
        return std::max(0.0f, 1.0f - std::fabsf(1.35f - d) - dfn*0.125f);*/
    }
	
	const Grammar &genomeGrammar() override {
		return imageGrammar;
	}
};
