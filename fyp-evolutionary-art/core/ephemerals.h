#pragma once

#include "geneticProgramming.h"
#include <vector>

namespace core {
	
// Contains ephemeral random values that remain constant throughout the run.
class Ephemerals {
	std::vector<float> values;
public:
	Ephemerals(genetic::EvolutionParameters &params, unsigned count = 10, float min = -1.0f, float max = 1.0f);

	unsigned size() const {
		return (unsigned)values.size();
	}

	float operator [](unsigned i) const {
		return values[i];
	}
	
	std::vector<float>::const_iterator begin() const {
		return values.begin();
	}
	
	std::vector<float>::const_iterator end() const {
		return values.end();
	}
};
	
} // end namespace core