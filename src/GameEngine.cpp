#include <cstdio>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "GlfwUtils.h"
#include "ShaderManager.h"
#include "Shaders.h"
#include "TextureUtils.h"

#include "GameEngine.h"

using namespace std;
using namespace glm;

static const double PI = acos(-1);

GameEngine::GameEngine()
    : camera(vec3(0, 0, -100), 0, PI),
      mouseTogglePressed(false) {}

void GameEngine::Init(GLFWwindow *_window) {
    window = _window;
    cameraController = CameraController(_window, &camera);

    disableCursor();

    glClearColor(0.0f, 0.0f, 0.4f, 0.0f);

    // procGenEngine.Init();
    sphEngine.Init();

    GLint viewportParams[4];
    glGetIntegerv(GL_VIEWPORT, viewportParams);
    SetViewportDimensions(viewportParams[2], viewportParams[3]);

    prevTime = 0.0;
}

void GameEngine::Update() {
    mat4 viewMatrix = camera.GetViewMatrix();
    mat4 modelMatrix = mat4(1.0f);
    mat4 mvMatrix = viewMatrix * modelMatrix;
    mat4 pMatrix = projectionMatrix;
    mat4 mvpMatrix = projectionMatrix * viewMatrix * modelMatrix;

    // procGenEngine.Update(mvpMatrix);
    sphEngine.Update(mvMatrix, pMatrix);

    double currentTime = glfwGetTime();
    float timeStep = (float)(currentTime - prevTime);
    prevTime = currentTime;

    if (fpsCounter.Update(timeStep)) {
        printf("%.3f\n", fpsCounter.millisecondsPerFrame);
    }

    cameraController.Update(timeStep);

    if (GlfwUtils::IsKeyPressed(window, GLFW_KEY_E)) {
        if (!mouseTogglePressed) {
            if (glfwGetInputMode(window, GLFW_CURSOR) ==
                GLFW_CURSOR_NORMAL) {
                disableCursor();
            } else {
                enableCursor();
            }
            mouseTogglePressed = true;
        }
    } else {
        mouseTogglePressed = false;
    }
}

bool GameEngine::CheckWindowCanClose() {
    return !GlfwUtils::IsKeyPressed(window, GLFW_KEY_ESCAPE);
}

void GameEngine::SetViewportDimensions(
    const int width, const int height) {
    viewportDimensions = ivec2(width, height);

    projectionMatrix = perspective(
        radians(45.0f),
        (float)viewportDimensions[0] / (float)viewportDimensions[1],
        0.1f,
        10000.0f);
}

void GameEngine::disableCursor() {
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glfwSetCursorPos(window, 0.0, 0.0);
}

void GameEngine::enableCursor() {
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    glfwSetCursorPos(
        window, viewportDimensions.x / 2, viewportDimensions.y / 2);
}
