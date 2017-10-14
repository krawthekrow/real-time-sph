#include <GLFW/glfw3.h>

#include "GlfwUtils.h"

bool GlfwUtils::IsKeyPressed(GLFWwindow *window, const int key) {
    return glfwGetKey(window, key) == GLFW_PRESS;
}
