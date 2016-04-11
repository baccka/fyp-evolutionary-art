#include "pixelFunctionGenerator.h"
#include "../core/clGenerator.h"
#include "../core/ephemerals.h"
#include "treeCompiler.h"
#include <sstream>

using namespace core;
using namespace genetic;

namespace pixelFunction {
	
unsigned ephemeralId(const Ephemerals &ephemerals, const grammar::Definition &definition, const TreeGenome::Node &node) {
	auto value = node.value - definition.getNodeValue();
	auto ephemeralCount = ephemerals.size();
	assert(value < definition.getWeight());
	assert(definition.getWeight() >= ephemeralCount);
	auto rangeOfValue = (definition.getWeight() / ephemeralCount);
	assert(rangeOfValue * ephemeralCount == definition.getWeight());
	auto result = value / rangeOfValue;
	assert(result < ephemerals.size());
	return result;
}

class OpenCLTreeGenomeCompiler : public TreeGenomeCompiler, public TreeGenomeCompilerDelegate {
	const Ephemerals &ephemerals;
	grammar::DefinitionSet ephemeralDefs;
public:

	OpenCLTreeGenomeCompiler(const grammar::Grammar &grammar, const Ephemerals &ephemerals) : TreeGenomeCompiler(grammar), ephemerals(ephemerals) {
		ephemeralDefs = grammar["ephemeral"];
		delegate = this;
	}

	bool printTerminal(const grammar::Definition &definition, const TreeGenome::Node &node, std::ostream &os) override {
		if (ephemeralDefs.contains(definition.getDefinitionId())) {
			os << "ephemerals[" << ephemeralId(ephemerals, definition, node) << "]";
			return true;
		}
		return false;
	}
	
	bool printFunction(const grammar::Definition &definition, const TreeGenome::Node &node, std::ostream &os) override {
		switch (definition.getName()[0]) {
			case '^': case '&': case '|':
				assert(definition.getNumArguments() == 2);
				os << "as_float(as_uint(";
				print(node[0], os);
				os << ") " << definition.getName() << " as_uint(";
				print(node[1], os);
				os << "))";
				return true;
			default:
				return false;
		}
	}
	
	bool treeGenomeCompilerShouldPrintFunctionAsOperator(const grammar::Definition &definition) override {
		switch (definition.getName()[0]) {
			case '+': case '-': case '*': case '/':
				return true;
			default:
				return false;
		}
	}
};

std::string compileToOpenCLKernel(size_t i, const grammar::Grammar &grammar, const TreeGenome &genome, const Ephemerals &ephemerals) {
	std::stringstream source;
	source << "void kernel k" << i << "(write_only image2d_t output, __constant float *ephemerals) {\n";
	source << "  int2 coord = (int2) (get_global_id(0), get_global_id(1));\n";
	source << "  float x = (float)get_global_id(0) / (float)get_global_size(0);\n";
	source << "  float y = (float)get_global_id(1) / (float)get_global_size(1);\n";
	source << "  float3 rgb = (float3)(clamp(";

	/*auto scalarType = grammar.typeByName("float");
	auto rootType = grammar[genome.first()].getType();
	if (rootType == scalarType) {
		source << ""
	} else {
		assert(rootType == grammar.typeByName("float3"));
		
	}*/
	OpenCLTreeGenomeCompiler compiler(grammar, ephemerals);
	compiler.print(genome, source);
	source << ", 0.0f, 1.0f));\n";
	source << "  write_imagef(output, coord, (float4) (rgb, 1.0f));\n";
	source << "}\n";
	return source.str();
}

class OpenCLTreeGenomeStackIntepreter {
protected:
	const grammar::Grammar &grammar;
	const Ephemerals &ephemerals;
	grammar::DefinitionSet ephemeralDef;
	// Custom instructions.
	bool supportsCalls;
	grammar::DefinitionSet callDefs;
	grammar::DefinitionSet functionSetDef;
	unsigned ret2InstructionID;
public:
	// Constants.
	static const int instructionOpMask = 0xFF;
	static const int instructionOpBits = 8;
	
	OpenCLTreeGenomeStackIntepreter(const grammar::Grammar &grammar, const Ephemerals &ephemerals) : grammar(grammar), ephemerals(ephemerals) {
		ephemeralDef = grammar["ephemeral"];
		callDefs = grammar["call"];
		functionSetDef = grammar["function-set"];
		supportsCalls = !callDefs.isEmpty();
		ret2InstructionID = (unsigned)grammar.size();
	}

	std::string sourceForInterpreter() {
		std::stringstream source;
		
		OpenCLTreeGenomeCompiler compiler(grammar, ephemerals);
		
		auto scalarType = grammar.typeByName("float");
        auto scalarType2 = grammar.typeByName("float_0");
		
		// Function.
		source << R"(
void kernel interpret(write_only image2d_t output, __constant float *ephemerals, __global uint *instructions, int begin, int end) {
int2 coord = (int2) (get_global_id(0), get_global_id(1));
float x = (float)get_global_id(0) / (float)get_global_size(0);
float y = (float)get_global_id(1) / (float)get_global_size(1);
)";
		
		// Stack.
		int stackSize = 64;
		int vecStackSize = 2;
		source << "float stack[" << stackSize << "]; int stackPointer = 0;\n";
		source << "float3 vecStack[" << vecStackSize << "]; int vecStackPointer = 0;\n";
		// Registers.
		source << "float r0, r1, r2;\n";

		// Instruction loop.
		source << "for (int i = begin; i < end; ++i) {\n";
		source << "uint instruction = instructions[i];\n";
		source << "switch (instruction & " << instructionOpMask << ") {\n";
		for (auto &&def : grammar.definitions()) {
			if (functionSetDef.contains(def.getDefinitionId())) {
				// Skip function sets.
				continue;
			}
			source << "case " << def.getDefinitionId() << ": \n";
			if (def.isTerminal()) {
				source << "  stack[stackPointer] = ";
				if (def.getName() == std::string("ephemeral")) {
					source << "ephemerals[instruction >> " << instructionOpBits << "]";
				} else {
					source << def.getName();
				}
				source << "; stackPointer++;\n";
				source << "  break;\n";
				continue;
			}
			// Function
			if (callDefs.contains(def.getDefinitionId())) {
				// CALL
				// Preserve the current X and Y on the stack and set them to be the two top stack entries.
				source << "  r0 = x; x = stack[stackPointer - 1]; stack[stackPointer - 1] = r0;\n";
				source << "  r0 = y; y = stack[stackPointer - 2]; stack[stackPointer - 2] = r0;\n";
				source << "  break;\n";
				continue;
			}
			if (def.getNumArguments() == 1) {
				//
				// TODO: type.
				source << "  stack[stackPointer - 1] = " << def.getName() << " (stack[stackPointer - 1]);\n";
				
			} else {
				// Prologue
				for (int i = 0; i < def.getNumArguments(); ++i) {
					source << "  r" << i  << " = stack[stackPointer - " << (i + 1) << "];\n";
				}
				source << "  stackPointer -= " << def.getNumArguments() << ";\n";
				bool isScalar = def.getType() == scalarType || def.getType() == scalarType2;
				if (isScalar) {
					source << "  stack[stackPointer] = ";
				} else {
					source << "  vecStack[vecStackPointer] = ";
				}
				const char firstChar = def.getName()[0];
				if (compiler.treeGenomeCompilerShouldPrintFunctionAsOperator(def)) {
					assert(def.getNumArguments() == 2);
					source << "r0 " << def.getName() << " r1";
				} else if(firstChar == '^' || firstChar == '&' || firstChar == '|') {
					assert(def.getNumArguments() == 2);
					source << "as_float(as_uint(r0) " << def.getName() << " as_uint(r1))";
				} else if (def.getNumArguments() == 2) {
					source << def.getName() << "(r0, r1)";
				} else if (def.getNumArguments() == 3) {
					source << def.getName() << "(r0, r1, r2)";
				} else {
					assert(false);
				}
				if (isScalar) {
					source << "; stackPointer++;\n";
				} else {
					source << "; vecStackPointer++;\n";
				}
			}
			source << "  break;\n";
		}
		if (supportsCalls) {
			// Restore the old X and Y and move the return value to the top of the stack.
			source << "case " << ret2InstructionID << ":\n";
			source << "  r0 = stack[stackPointer - 1];\n";
			source << "  x = stack[stackPointer - 2]; \n";
			source << "  y = stack[stackPointer - 3]; \n";
			source << "  stackPointer -= 2;\n";
			source << "  stack[stackPointer - 1] = r0;\n";
			source << "  break;\n";
		}
		source << "}\n";
		source << "}\n";
		source << "float3 rgb = (float3)(clamp(vecStack[vecStackPointer - 1], 0.0f, 1.0f));\n";
		source << "write_imagef(output, coord, (float4) (rgb, 1.0f));\n";
		source << "}\n";
		return source.str();
	}
};
	
class OpenCLTreeGenomeStackIntepreterCompiler: public OpenCLTreeGenomeStackIntepreter {
	const TreeGenome &tree;
	size_t currentFunction = 0;
public:
	
	OpenCLTreeGenomeStackIntepreterCompiler(const grammar::Grammar &grammar, const Ephemerals &ephemerals, const TreeGenome &tree) : OpenCLTreeGenomeStackIntepreter(grammar, ephemerals), tree(tree) {
	}
	
	// Compile to instructions.
	void compile(const TreeGenome::Node &node, std::vector<unsigned> &os) {
		// TODO: verify stack size.
		// Children, right to left.
		for (size_t i = node.size(); i != 0;) {
			--i;
			compile(node[i], os);
		}
		//
		const auto &def = grammar[node];
		unsigned instruction = def.getDefinitionId();
		assert(instruction <= instructionOpMask);
		if (ephemeralDef.contains(def.getDefinitionId())) {
			instruction |= ephemeralId(ephemerals, def, node) << instructionOpBits;
		}
		os.push_back(instruction);
		if (callDefs.contains(def.getDefinitionId())) {
			// Emit the inlined function.
			inlineCallToLowerFunction(os);
		}
	}
	
	void inlineCallToLowerFunction(std::vector<unsigned> &os) {
		const auto &root = tree.first();
		const auto &rootDef = grammar[root];
		assert(functionSetDef.contains(rootDef.getDefinitionId()));
		assert(currentFunction > 0);
		currentFunction -= 1;
		compile(root[currentFunction], os);
		// Emit a return.
		os.push_back(ret2InstructionID);
		currentFunction += 1;
	}

	void compile(std::vector<unsigned> &os) {
		currentFunction = 0;
		const auto &root = tree.first();
		const auto &rootDef = grammar[root];
		if (functionSetDef.contains(rootDef.getDefinitionId())) {
			currentFunction = root.size() - 1;
			compile(root[currentFunction], os);
			return;
		}
		compile(root, os);
	}
};
	

// OpenCL based pixel function generator.
class PixelFunctionCLTreeGenomeImageGenerator: public CLTreeGenomeImageGenerator {
	const grammar::Grammar &grammar;
	const Ephemerals &ephemerals;
	std::unique_ptr<cl::Buffer> ephemeralBuffer;
	std::string source;
	std::unique_ptr<cl::Buffer> instructionBuffer;
	std::vector<unsigned> instructions;
	std::unordered_map<size_t, std::pair<size_t, size_t>> instructionMappings;
public:
	PixelFunctionCLTreeGenomeImageGenerator(const grammar::Grammar &grammar, const TreeGenomeImageGenerationOptions &options, CLComputeProvider &clProvider, const Ephemerals &ephemerals) : CLTreeGenomeImageGenerator(options, clProvider), grammar(grammar), ephemerals(ephemerals) {
		ephemeralBuffer.reset(new cl::Buffer(clProvider.getContext(), CL_MEM_READ_WRITE, ephemerals.size() * sizeof(float)));
		cl::copy(clProvider.getMainQueue(), ephemerals.begin(), ephemerals.end(), *ephemeralBuffer);
		
		source = clProvider.loadSource("functions.cl");
		//
		OpenCLTreeGenomeStackIntepreter interpreter(grammar, ephemerals);
		kernelSources.push_back(source);
		kernelSources.push_back(interpreter.sourceForInterpreter());
        std::cout << interpreter.sourceForInterpreter();
		CLTreeGenomeImageGenerator::build();
	}
	
	void clear() override {
		instructions.clear();
		instructionMappings.clear();
		
	}

	void build() override {
		instructionBuffer.reset(new cl::Buffer(clProvider.getContext(), CL_MEM_READ_ONLY, instructions.size() * sizeof(unsigned)));
		cl::copy(clProvider.getMainQueue(), instructions.begin(), instructions.end(), *instructionBuffer);
	}

	void compileIndividual(size_t individualId, const genetic::TreeGenome &individual) override {
		OpenCLTreeGenomeStackIntepreterCompiler compiler(grammar, ephemerals, individual);
		size_t i = instructions.size();
		compiler.compile(instructions);
		instructionMappings.insert(std::make_pair(individualId, std::make_pair(i, instructions.size())));
	}

	void generateIndividualImage(size_t individualId, const genetic::TreeGenome &individual) override {
		cl::Kernel f(getProgram(),"interpret");
		auto i = instructionMappings.find(individualId);
		assert(i != instructionMappings.end());
		f.setArg(0, getImage());
		f.setArg(1, *ephemeralBuffer);
		f.setArg(2, *instructionBuffer);
		f.setArg(3, (int)i->second.first);
		f.setArg(4, (int)i->second.second);
		clProvider.getMainQueue().enqueueNDRangeKernel(f, cl::NullRange, cl::NDRange(options.imageWidth, options.imageHeight));
	}
};
	
// OpenCL based pixel function generator.
class PixelFunctionCLTreeGenomeImageGeneratorDirectCompiler: public CLTreeGenomeImageGenerator {
	const grammar::Grammar &grammar;
	const Ephemerals &ephemerals;
	std::unique_ptr<cl::Buffer> ephemeralBuffer;
	std::string source;
public:
	PixelFunctionCLTreeGenomeImageGeneratorDirectCompiler(const grammar::Grammar &grammar, const TreeGenomeImageGenerationOptions &options, CLComputeProvider &clProvider, const Ephemerals &ephemerals) : CLTreeGenomeImageGenerator(options, clProvider), grammar(grammar), ephemerals(ephemerals) {
		ephemeralBuffer.reset(new cl::Buffer(clProvider.getContext(), CL_MEM_READ_WRITE, ephemerals.size() * sizeof(float)));
		cl::copy(clProvider.getMainQueue(), ephemerals.begin(), ephemerals.end(), *ephemeralBuffer);
		
		source = clProvider.loadSource("functions.cl");
	}
	
	void clear() override {
		CLTreeGenomeImageGenerator::clear();
		kernelSources.push_back(source);
	}
	
	void build() override {
		CLTreeGenomeImageGenerator::build();
	}
	
	void compileIndividual(size_t individualId, const genetic::TreeGenome &individual) override {
		kernelSources.push_back(compileToOpenCLKernel(individualId, grammar, individual, ephemerals));
	}
	
	void generateIndividualImage(size_t individualId, const genetic::TreeGenome &individual) override {
		std::stringstream name;
		name << "k" << individualId;
		cl::Kernel f(getProgram(), name.str().c_str());
		f.setArg(0, getImage());
		f.setArg(1, *ephemeralBuffer);
		clProvider.getMainQueue().enqueueNDRangeKernel(f, cl::NullRange, cl::NDRange(options.imageWidth, options.imageHeight));
	}
};
	
std::unique_ptr<CLTreeGenomeImageGenerator> createCLTreeGenomeImageGenerator(const grammar::Grammar &grammar, const TreeGenomeImageGenerationOptions &options, CLComputeProvider &clProvider, const Ephemerals &ephemerals) {
	return std::unique_ptr<CLTreeGenomeImageGenerator>(new PixelFunctionCLTreeGenomeImageGenerator(grammar, options, clProvider, ephemerals));
}

}
