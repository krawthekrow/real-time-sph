#include <cstdio>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "GameEngine.h"

using namespace std;
using namespace glm;

GameEngine gameEngine;

void glfwErrorCallback(int error, const char *description) {
    fprintf(stderr, "GLFW error %d: %s\n", error, description);
}

void onWindowResize(
    GLFWwindow *window, const int width, const int height) {
    glViewport(0, 0, width, height);
    gameEngine.SetViewportDimensions(width, height);
}

int main() {

    // INIT GLFW

    glfwSetErrorCallback(glfwErrorCallback);
    if (!glfwInit()) {
        fprintf(stderr, "Failed to initialise GLFW.\n");
        return -1;
    }

    // glfwWindowHint(GLFW_SAMPLES, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    GLFWwindow *window =
        glfwCreateWindow(1024, 782, "Real Time SPH", NULL, NULL);
    if (window == NULL) {
        fprintf(stderr, "Failed to open GLFW window.\n");
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glewExperimental = true;
    if (glewInit() != GLEW_OK) {
        fprintf(stderr, "Failed to initialise GLEW.\n");
        return -1;
    }

    glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
    glfwSwapInterval(1);

    glfwSetWindowSizeCallback(window, onWindowResize);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

    // INIT GAME ENGINE

    srand(588472); // Randomly chosen random seed for testing determinism
    gameEngine.Init(window);

    // GAME LOOP

    do {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        gameEngine.Update();
        glfwSwapBuffers(window);
        glfwPollEvents();
    } while (gameEngine.CheckWindowCanClose() &&
             glfwWindowShouldClose(window) == 0);

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
