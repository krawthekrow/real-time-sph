#pragma once

#include <GLFW/glfw3.h>

class GlfwUtils {
public:
    static bool IsKeyPressed(GLFWwindow *window, const int key);
};
