#pragma once

#include "depth_frame.h"

#include <SDL_opengl.h>
#include <SDL.h>
#undef main
#include <string>
#include <cuda_runtime.h>

struct INuiSensor;

class Window
{
public:
	Window();
	~Window();

	void getKinectData(DepthFrame& depth_frame) const;
	void draw();
	void setWindowTitle(const std::string& str) const;
	cudaSurfaceObject_t get_content() const { return m_content; }

private:
	SDL_Window* m_window{ nullptr };
	GLuint m_texture{ 0 };
	cudaSurfaceObject_t m_content{ 0 };
	cudaArray* m_content_array{ nullptr };
	cudaGraphicsResource* m_resource{ nullptr };
	HANDLE m_depth_stream{ nullptr };
	INuiSensor* m_sensor{ nullptr };
};