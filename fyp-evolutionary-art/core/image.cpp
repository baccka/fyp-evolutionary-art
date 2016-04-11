#include "image.h"
// TODO: move 2 core
#define STB_IMAGE_IMPLEMENTATION
#include "../stb-image.h"

namespace core {

Image::Image(const char *filename) {
	pixels = stbi_load(filename, &width, &height, &componentCount, 0);
	assert(pixels != nullptr);
}

Image::~Image() {
	stbi_image_free(pixels);
}

}
