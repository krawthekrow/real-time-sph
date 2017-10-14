#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include "Camera.h"
#include "CameraController.h"
#include "FpsCounter.h"
#include "ProcGenEngine.h"

using namespace std;
using namespace glm;

class GameEngine {
public:
    GameEngine();
    void Init(GLFWwindow *_window);
    void Update();
    bool CheckWindowCanClose();
    void SetViewportDimensions(const int width, const int height);

private:
    GLFWwindow *window;
    ProcGenEngine procGenEngine;
    FpsCounter fpsCounter;

    double prevTime;
    bool mouseTogglePressed;
    ivec2 viewportDimensions;

    mat4 projectionMatrix;
    Camera camera;
    CameraController cameraController;

    bool isKeyPressed(int key);
    void disableCursor();
    void enableCursor();
};
