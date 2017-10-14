#include "FpsCounter.h"

FpsCounter::FpsCounter()
    : cumTime(0.0f), numFrames(0), millisecondsPerFrame(0.0f) {}

void FpsCounter::Reset() {
    cumTime = 0.0f;
    numFrames = 0;
    millisecondsPerFrame = 0.0f;
}

bool FpsCounter::Update(const float timeStep) {
    cumTime += timeStep;
    numFrames++;
    if (cumTime > 0.5) {
        millisecondsPerFrame = cumTime / numFrames * 1000.0f;
        cumTime = 0.0f;
        numFrames = 0;
        return true;
    } else {
        return false;
    }
}
