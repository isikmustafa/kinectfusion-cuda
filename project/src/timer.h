#pragma once

#include <chrono>

class Timer
{
public:
	Timer();

	void start();
	double getTime();

private:
	std::chrono::time_point<std::chrono::system_clock> m_start_time;
};