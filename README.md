Check out the [demo video](https://youtu.be/OTSuqxK--3Y)!

Implementation of real-time smoothed-particle hydrodynamics (SPH) with screen-space rendering for my 6.837 (computer graphics) final project (from 2017). Uses a pipeline similar to [Zhang](https://drive.google.com/file/d/0B2macuhZeO18TzQ0UmNsRUt6bUE/view), but for the GPU. SPH physics is based on [Pirovano](https://www.politesi.polimi.it/bitstream/10589/33101/1/2011_12_Pirovano.pdf), uniform grid collision detection on [Le Grand](https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch32.html) and screen-space rendering on [Green](http://developer.download.nvidia.com/presentations/2010/gdc/Direct3D_Effects.pdf).

Requires a CUDA-enabled GPU with compute capability 2.0 or higher and OpenGL 4.2 or higher. Currently only builds on Linux with cmake. Requires glm 0.9.9-a2. Only tested with CUDA 9.0, though it might work with older versions. Simulates 3000 particles at about 300 fps on a NVIDIA GeForce GT 750M.

More information provided in the [writeup](doc/writeup.pdf).

Usage:

- Escape quits the program.
- Mouse moves the camera direction.
- WASD/Shift/Space moves the camera position.
- E frees the mouse from the window.
- Period ('.') pauses/unpauses the simulation.
- Comma (',') pauses/unpauses the box rotation.
- V makes the box rotate quickly.
- B makes the box rotate slowly.
- G resets the box rotation velocity.
- Z changes the smoothing mode.
- X shows the depth map.
- C disables rendering.
- R and T changes the Z threshold for particle rendering.
