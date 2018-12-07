#include "cuda_event.h"
#include "cuda_utils.h"

CudaEvent::CudaEvent()
{
	HANDLE_ERROR(cudaEventCreate(&m_event));
}

CudaEvent::~CudaEvent()
{
	HANDLE_ERROR(cudaEventDestroy(m_event));
}

void CudaEvent::record()
{
	HANDLE_ERROR(cudaEventRecord(m_event, 0));
}

void CudaEvent::synchronize()
{
	HANDLE_ERROR(cudaEventSynchronize(m_event));
}

float CudaEvent::calculateElapsedTime(const CudaEvent& start, const CudaEvent& end)
{
	float elapsed_time = 0.0f;
	HANDLE_ERROR(cudaEventElapsedTime(&elapsed_time, start.m_event, end.m_event));

	return elapsed_time;
}