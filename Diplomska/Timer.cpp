#include "Timer.h"

Timer* Timer::instance = nullptr;

Timer::Timer() {
	lastFrame = 0;
	delta = 0;
}

Timer* Timer::getInstance() {
	if (instance != nullptr) {
		return instance;
	}
	instance = new Timer();
	return instance;
}

uint64_t Timer::getTime() {
	return duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
}

void Timer::update() {
	uint64_t currentTime = getTime();
	delta = currentTime - lastFrame;
	lastFrame = currentTime;
}

uint64_t Timer::getDelta() {
	return delta;
}

float Timer::getFPS() {
	return 1.0f / (delta / 1.0e6);
}
