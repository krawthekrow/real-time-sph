#include <GL/glew.h>
#include <glm/glm.hpp>

#include "TextureUtils.h"

using namespace std;
using namespace glm;

GLuint TextureUtils::GenComputeTexture2D(const ivec2 texSize) {
    GLuint texID;
    glGenTextures(1, &texID);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texID);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_R32F,
        texSize.x,
        texSize.y,
        0,
        GL_RED,
        GL_FLOAT,
        NULL);

    glBindImageTexture(0, texID, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F);

    return texID;
}

GLuint TextureUtils::GenComputeTexture3D(const ivec3 texSize) {
    GLuint texID;
    glGenTextures(1, &texID);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, texID);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage3D(
        GL_TEXTURE_3D,
        0,
        GL_R32F,
        texSize.x,
        texSize.y,
        texSize.z,
        0,
        GL_RED,
        GL_FLOAT,
        NULL);

    glBindImageTexture(0, texID, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R32F);

    return texID;
}
