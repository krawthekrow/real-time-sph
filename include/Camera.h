#pragma once

#include <glm/glm.hpp>

using namespace glm;

class Camera {
public:
    Camera(const vec3 _position, const double _pitch, const double _yaw);

    void AdjustPosition(const vec3 delta);
    void AdjustYaw(const double delta);
    void AdjustPitch(const double delta);

    mat4 GetViewMatrix();

    void Update();

private:
    vec3 position;
    double pitch, yaw;

    vec3 cameraForward, cameraUp;
};
