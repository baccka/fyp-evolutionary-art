#pragma once

struct ImageGenerator;

#ifdef __cplusplus
extern "C" {
#endif

struct ImageGenerator *createImageGenerator(const char *resourcePath);
const float *getPixels(struct ImageGenerator *imageGenerator);
const float *generatePixels(struct ImageGenerator *imageGenerator);
void destroyImageGenerator(struct ImageGenerator *imageGenerator);

#ifdef __cplusplus
}
#endif