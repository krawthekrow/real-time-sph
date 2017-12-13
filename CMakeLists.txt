cmake_minimum_required (VERSION 3.1)
project (Experiments)
# set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CXX_FLAGS "--std=c++11")
set (CMAKE_BINARY_DIR ${PROJECT_SOURCE_DIR}/bin)
set (EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set (LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set (CMAKE_SOURCE_DIR ${PROJECT_SOURCE_DIR}/src)
include_directories ("${PROJECT_SOURCE_DIR}/include")
include_directories ("${PROJECT_SOURCE_DIR}/shaders")

find_package (CUDA QUIET REQUIRED)

cuda_add_executable (main
    src/main.cpp
    src/GlfwUtils.cpp
    src/CudaUtils.cpp
    src/ShaderManager.cpp
    src/Camera.cpp
    src/CameraController.cpp
    src/FpsCounter.cpp
    src/TexturedQuadRenderer.cpp
    src/Shaders.cpp
    src/GlobalDebugSwitches.cpp
    src/GameEngine.cpp
    src/SphEngine.cpp
    src/SphCuda.cu
    src/FluidRenderer.cpp
)

find_package (PkgConfig REQUIRED)
pkg_search_module (GLFW REQUIRED glfw3)
include_directories (${GLFW_INCLUDE_DIRS})
target_link_libraries(main ${GLFW_STATIC_LIBRARIES})

find_package (GLEW REQUIRED)
if (GLEW_FOUND)
    include_directories(${GLEW_INCLUDE_DIRS})
    target_link_libraries(main ${GLEW_LIBRARIES})
endif()

find_library (NVTX_LIBRARY nvToolsExt PATHS ENV LD_LIBRARY_PATH)
target_link_libraries(main ${NVTX_LIBRARY})
