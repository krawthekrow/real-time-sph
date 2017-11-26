#pragma once

#include <GL/glew.h>
#include <glm/glm.hpp>

using namespace glm;

class TextureUtils {
public:
    static const int MAX_SCREEN_WIDTH = 1920;
    static const int MAX_SCREEN_HEIGHT = 1080;
    static GLuint GenComputeTexture2D(const ivec2 texSize);
    static GLuint GenComputeTexture3D(const ivec3 texSize);
};
