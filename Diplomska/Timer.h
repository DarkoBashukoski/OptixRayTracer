#ifndef TIMER_H
#define TIMER_H

#include <chrono>

using namespace std::chrono;

class Timer {
private:
	static Timer* instance;
	Timer();
	uint64_t lastFrame;
	uint64_t delta;
public:
	Timer(const Timer& obj) = delete;
	static Timer* getInstance();
	uint64_t getTime();
	void update();
	uint64_t getDelta();
	float getFPS();
};

#endif // !TIMER_H
