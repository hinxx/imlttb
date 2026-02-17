cmake_minimum_required(VERSION 3.21)
project(imlttb)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Dependencies ---
include(FetchContent)

# 1. GLFW
find_package(glfw3 REQUIRED)

# 2. Dear ImGui (Docking Branch)
FetchContent_Declare(
  imgui
  GIT_REPOSITORY https://github.com/ocornut/imgui.git
  GIT_TAG        docking
)
FetchContent_MakeAvailable(imgui)

# 3. ImPlot
FetchContent_Declare(
  implot
  GIT_REPOSITORY https://github.com/epezent/implot.git
  GIT_TAG        v0.17
)
FetchContent_GetProperties(implot)
if(NOT implot_POPULATED)
  FetchContent_Populate(implot)
endif()

# 4. GLEW & OpenGL (System)
find_package(OpenGL REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)

# --- ImGui Library Wrapper ---
# Create a static library for ImGui to handle sources and backends cleanly
add_library(imgui_lib STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/imgui_demo.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)
target_include_directories(imgui_lib PUBLIC 
    ${imgui_SOURCE_DIR} 
    ${imgui_SOURCE_DIR}/backends
)
target_link_libraries(imgui_lib PUBLIC glfw OpenGL::GL)

# --- ImPlot Library Wrapper ---
add_library(implot_lib STATIC
    ${implot_SOURCE_DIR}/implot.cpp
    ${implot_SOURCE_DIR}/implot_items.cpp
    ${implot_SOURCE_DIR}/implot_demo.cpp
)
target_include_directories(implot_lib PUBLIC ${implot_SOURCE_DIR})
target_link_libraries(implot_lib PUBLIC imgui_lib)

# --- Main Executable ---
add_executable(imlttb main.cpp)

target_include_directories(imlttb PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
# Enable AVX2 for SIMD optimizations
target_compile_options(imlttb PRIVATE -mavx2)
target_link_libraries(imlttb PRIVATE OpenMP::OpenMP_CXX)

target_link_libraries(imlttb PRIVATE
    imgui_lib
    implot_lib
)