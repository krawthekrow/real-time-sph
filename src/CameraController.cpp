#include <cstdio>

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include "Camera.h"
#include "GlfwUtils.h"

#include "CameraController.h"

using namespace glm;

CameraController::CameraController() {}

CameraController::CameraController(GLFWwindow *_window, Camera *_camera)
    : mouseSensitivity(3.0f), movementSpeed(50.0f) {
    window = _window;
    camera = _camera;
}

void CameraController::Update(const float &timeStep) {
    if (glfwGetInputMode(window, GLFW_CURSOR) == GLFW_CURSOR_DISABLED) {
        double cursorPos[2];
        glfwGetCursorPos(window, &cursorPos[0], &cursorPos[1]);
        glfwSetCursorPos(window, 0, 0);
        float mouseAngleScale = 0.001f;
        camera->AdjustYaw(
            mouseSensitivity * cursorPos[0] * mouseAngleScale);
        camera->AdjustPitch(
            -mouseSensitivity * cursorPos[1] * mouseAngleScale);
    }

    const bool goForward =
        GlfwUtils::IsKeyPressed(window, GLFW_KEY_UP) ||
        GlfwUtils::IsKeyPressed(window, GLFW_KEY_W);
    const bool goBackward =
        GlfwUtils::IsKeyPressed(window, GLFW_KEY_DOWN) ||
        GlfwUtils::IsKeyPressed(window, GLFW_KEY_S);
    const bool goRight =
        GlfwUtils::IsKeyPressed(window, GLFW_KEY_RIGHT) ||
        GlfwUtils::IsKeyPressed(window, GLFW_KEY_D);
    const bool goLeft =
        GlfwUtils::IsKeyPressed(window, GLFW_KEY_LEFT) ||
        GlfwUtils::IsKeyPressed(window, GLFW_KEY_A);
    const bool goUp =
        GlfwUtils::IsKeyPressed(window, GLFW_KEY_SPACE);
    const bool goDown =
        GlfwUtils::IsKeyPressed(window, GLFW_KEY_LEFT_SHIFT);

    camera->AdjustPosition(
        vec3(
            (goRight ? 1 : 0) - (goLeft ? 1 : 0),
            (goUp ? 1 : 0) - (goDown ? 1 : 0),
            (goForward ? 1 : 0) - (goBackward ? 1 : 0)) *
        timeStep * movementSpeed);

    camera->Update();
}
