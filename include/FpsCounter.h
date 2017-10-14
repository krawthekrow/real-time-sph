#pragma once

class FpsCounter {
private:
    float cumTime;
    int numFrames;

public:
    FpsCounter();
    float millisecondsPerFrame;
    void Reset();
    bool Update(const float timeStep);
};
