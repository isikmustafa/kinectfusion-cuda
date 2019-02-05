# An implementation of [KinectFusion](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ismar2011.pdf) on CUDA platform.

How to build?
 - Clone the repository recursively (to also clone the submodules)
 - Go here https://www.libsdl.org/download-2.0.php to download precompiled dynamic and static libraries. Put them under 3rd\SDL2-2.0.9\libs. You can build the SDL if the precompiled libraries are not suitable for you.
 - Open the visual studio project (works fine with VS 2017). Build and run it.
 - We were able to run it 30 frames per second on our Nvidia GTX 960M graphics card.
