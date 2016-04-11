#pragma once

#include <stdlib.h>

namespace core {

class Image {
	int width;
	int height;
	int componentCount;
	uint8_t *pixels;
	
public:
	Image(const char *filename);
	~Image();
	
	int getWidth() const { return width; }
	int getHeight() const { return height; }
	int getComponentCount() const { return componentCount; }
	
	const uint8_t *data() const { return pixels; }
};

} // end namespace core
