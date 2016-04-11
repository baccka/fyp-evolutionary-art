#include "ephemerals.h"
#include <random>

namespace core {

Ephemerals::Ephemerals(genetic::EvolutionParameters &params, unsigned count, float min, float max) {
	std::uniform_real_distribution<float> sampler(min, max);
	for (unsigned i = 0; i < count; ++i) {
		values.push_back(sampler(params.rng));
	}
}
	
}