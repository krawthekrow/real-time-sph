#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include "Camera.h"

using namespace glm;

class CameraController {
public:
    CameraController();
    CameraController(GLFWwindow *_window, Camera *_camera);
    void Update(const float timeStep);

private:
    GLFWwindow *window;
    Camera *camera;

    float mouseSensitivity;
    float movementSpeed;
};
