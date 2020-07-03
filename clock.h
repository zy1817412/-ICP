#ifndef CLOCK_H
#define CLOCK_H

#include <windows.h>

class sapClock
{
public:
	sapClock()
	{

	}

	~sapClock()
	{
	
	}
	
	inline void begin()
	{
		QueryPerformanceCounter(&counterBegin);
		QueryPerformanceFrequency(&frequency);
	}
	
	inline void end()
	{
		QueryPerformanceCounter(&counterEnd);
		interval = 1000 * ((double)counterEnd.QuadPart - (double)counterBegin.QuadPart) / frequency.QuadPart;
	}
	
	inline double getInterval()
	{
		return interval;
	}

	inline void printInterval(const char *name) const
	{
		printf("%s: %lfms\n", name, interval);
	}

private:
	LARGE_INTEGER frequency;
	LARGE_INTEGER counterBegin;
	LARGE_INTEGER counterEnd;
	double interval;
};

#endif 
