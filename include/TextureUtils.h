#pragma once

#include <GL/glew.h>
#include <glm/glm.hpp>

using namespace glm;

class TextureUtils {
public:
    static GLuint GenComputeTexture2D(const ivec2 texSize);
    static GLuint GenComputeTexture3D(const ivec3 texSize);
};
