#include "window.h"
#include "cuda_utils.h"

#include <cuda_gl_interop.h>
#include <stdexcept>
#include <cuda_runtime.h>

constexpr int gWidth = 640;
constexpr int gHeight = 480;
constexpr bool gVSync = true;

Window::Window()
{
	if (SDL_Init(SDL_INIT_VIDEO) < 0)
	{
		throw std::runtime_error("SDL could not initialize!");
	}

	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);

	m_window = SDL_CreateWindow("KinectFusion", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, gWidth, gHeight, SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN);

	if (!m_window)
	{
		throw std::runtime_error("Window could not be created!");
	}

	if (!SDL_GL_CreateContext(m_window))
	{
		throw std::runtime_error("OpenGL context could not be created!");
	}

	if (SDL_GL_SetSwapInterval(gVSync) < 0)
	{
		throw std::runtime_error("VSync cannot be set!");
	}

	//Initialize textures and set up the CUDA interop.
	glGenTextures(1, &m_texture);
	glBindTexture(GL_TEXTURE_2D, m_texture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, gWidth, gHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
	HANDLE_ERROR(cudaGraphicsGLRegisterImage(&m_resource, m_texture, GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard));
	glBindTexture(GL_TEXTURE_2D, 0);

	//OpenGL setup
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glDisable(GL_DEPTH_TEST);
	glEnable(GL_TEXTURE_2D);

	//Camera setup
	glViewport(0, 0, gWidth, gHeight);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	//Allocate CUDA array and create surface object.
	cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
	HANDLE_ERROR(cudaMallocArray(&m_content_array, &channel_desc, gWidth, gHeight, cudaArraySurfaceLoadStore));

	cudaResourceDesc res_desc;
	memset(&res_desc, 0, sizeof(res_desc));
	res_desc.resType = cudaResourceTypeArray;

	res_desc.res.array.array = m_content_array;
	HANDLE_ERROR(cudaCreateSurfaceObject(&m_content, &res_desc));
}

Window::~Window()
{
	HANDLE_ERROR(cudaDestroySurfaceObject(m_content));
	HANDLE_ERROR(cudaFreeArray(m_content_array));
	SDL_DestroyWindow(m_window);
	SDL_Quit();
}

void Window::draw()
{
	//CUDA interop.
	cudaArray* texture_ptr = nullptr;
	HANDLE_ERROR(cudaGraphicsMapResources(1, &m_resource, 0));
	HANDLE_ERROR(cudaGraphicsSubResourceGetMappedArray(&texture_ptr, m_resource, 0, 0));
	HANDLE_ERROR(cudaMemcpyArrayToArray(texture_ptr, 0, 0, m_content_array, 0, 0, gWidth * gHeight * 4, cudaMemcpyDeviceToDevice));
	HANDLE_ERROR(cudaGraphicsUnmapResources(1, &m_resource, 0));

	//Draw the quad.
	glClear(GL_COLOR_BUFFER_BIT);
	glBindTexture(GL_TEXTURE_2D, m_texture);
	glBegin(GL_QUADS);
	{
		glTexCoord2f(0.0f, 0.0f);
		glVertex2f(-1.0f, 1.0f);

		glTexCoord2f(1.0f, 0.0f);
		glVertex2f(1.0f, 1.0f);

		glTexCoord2f(1.0f, 1.0f);
		glVertex2f(1.0f, -1.0f);

		glTexCoord2f(0.0f, 1.0f);
		glVertex2f(-1.0f, -1.0f);
	}
	glEnd();

	SDL_GL_SwapWindow(m_window);
}

void Window::setWindowTitle(const std::string& str) const
{
	SDL_SetWindowTitle(m_window, ("KinectFusion --- " + str).c_str());
}