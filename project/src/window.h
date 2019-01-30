#pragma once

#include <string>

#include <SDL_opengl.h>
#define SDL_MAIN_HANDLED
#include <SDL.h>
#include <cuda_runtime.h>
#include "depth_map.h"

struct INuiSensor;

struct KeyboardState
{
	bool w{ false };
	bool a{ false };
	bool s{ false };
	bool d{ false };
	bool enter{ false };
	bool space{ false };
};

struct MouseState
{
	glm::ivec2 previous_position;
	glm::ivec2 current_position;
	bool pressed{ false };
};

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
	const KeyboardState& getKeyboardState() const;
	const MouseState& getMouseState() const;

private:
	SDL_Window* m_window{ nullptr };
	GLuint m_texture{ 0 };
	cudaSurfaceObject_t m_content{ 0 };
	cudaArray* m_content_array{ nullptr };
	cudaGraphicsResource* m_resource{ nullptr };
	HANDLE m_depth_stream{ nullptr };
	INuiSensor* m_sensor{ nullptr };
	KeyboardState m_keyboard_state;
	MouseState m_mouse_state;
};