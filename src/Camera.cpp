#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "Camera.h"

using namespace std;

static const double PI = acos(-1);

Camera::Camera(
    const vec3 _position, const double _pitch, const double _yaw)
    : position(_position), pitch(_pitch), yaw(_yaw) {
    Update();
}

void Camera::AdjustPosition(const vec3 delta) {
    const vec3 playerForward(sin(yaw), 0, -cos(yaw));
    const vec3 playerRight = cross(cameraForward, cameraUp);
    const vec3 playerUp = cross(playerRight, playerForward);
    const vec3 transformedDelta =
        mat3(playerRight, playerUp, playerForward) * delta;
    position += transformedDelta;
}

void Camera::AdjustYaw(const double delta) {
    yaw += delta;
    yaw = mod(yaw, 2 * PI);
}

void Camera::AdjustPitch(const double delta) {
    pitch += delta;
    pitch = clamp(pitch, -0.9 * PI / 2, 0.9 * PI / 2);
}

mat4 Camera::GetViewMatrix() {
    return lookAt(position, position + cameraForward, cameraUp);
}

void Camera::Update() {
    cameraForward =
        vec3(cos(pitch) * sin(yaw), sin(pitch), -cos(pitch) * cos(yaw));
    const float cameraUpPitch = pitch + PI / 2.0f;
    cameraUp = vec3(
        cos(cameraUpPitch) * sin(yaw),
        sin(cameraUpPitch),
        -cos(cameraUpPitch) * cos(yaw));
}
