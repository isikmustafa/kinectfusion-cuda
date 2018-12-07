#pragma once

#include <cuda_runtime.h>

class CudaEvent
{
public:
	CudaEvent();
	~CudaEvent();

	void record();
	void synchronize();

	static float calculateElapsedTime(const CudaEvent& start, const CudaEvent& end);

private:
	cudaEvent_t m_event;
};