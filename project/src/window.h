#pragma once

#include <string>

#include <SDL_opengl.h>
#define SDL_MAIN_HANDLED
#include <SDL.h>
#include <cuda_runtime.h>
#include "depth_map.h"

struct INuiSensor;

class Window
{
public:
	Window(const bool use_kinect);
	~Window();

	void getKinectData(DepthMap& depth_frame) const;
	void draw();
	void setWindowTitle(const std::string& str) const;
	cudaSurfaceObject_t get_content() const { return m_content; }
	void handleInput(); // If not called, state of buttons or mouse is not be updated.
	const std::array<bool, 4>& getWasdState(); // Returns if W-A-S-D pressed or not.
	const std::array<glm::ivec2, 2>& getMouseState(); // Returns previous and current mouse position
	bool isMousePressed();

private:
	SDL_Window* m_window{ nullptr };
	GLuint m_texture{ 0 };
	cudaSurfaceObject_t m_content{ 0 };
	cudaArray* m_content_array{ nullptr };
	cudaGraphicsResource* m_resource{ nullptr };
	HANDLE m_depth_stream{ nullptr };
	INuiSensor* m_sensor{ nullptr };
	std::array<bool, 4> m_wasd_state{ false, false, false, false };
	std::array<glm::ivec2, 2> m_mouse_state;
	bool m_mouse_pressed{ false };
};