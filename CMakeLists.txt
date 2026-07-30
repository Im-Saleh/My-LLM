# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.16)
project(slm VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(SLM_WITH_GUI      "Build the Dear ImGui dashboard (needs OpenGL + GLFW)" ON)
option(SLM_WITH_LIBTORCH "Use libtorch (PyTorch C++) as the tensor backend"     OFF)
option(SLM_NATIVE_ARCH   "Optimise for the build machine (-march=native)"       OFF)
option(SLM_BUILD_TESTS   "Build the gradient checking test binary"              ON)
option(SLM_GLFW_WAYLAND  "Build the bundled GLFW with Wayland support"          OFF)

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -fno-math-errno -DNDEBUG")

# ----------------------------------------------------------------- core library
set(SLM_CORE_SOURCES
  src/core/config.cpp
  src/core/dataset.cpp
  src/core/optim.cpp
  src/core/params.cpp
  src/core/serialize.cpp
  src/core/tensor_common.cpp
  src/model.cpp
  src/tokenizer.cpp
)
if(SLM_WITH_LIBTORCH)
  list(APPEND SLM_CORE_SOURCES src/core/tensor_torch.cpp)
else()
  list(APPEND SLM_CORE_SOURCES src/core/tensor_native.cpp src/core/gemm.cpp)
endif()

add_library(slm_core STATIC ${SLM_CORE_SOURCES})
target_include_directories(slm_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_options(slm_core PRIVATE -Wall -Wextra -Wno-unused-parameter)

if(SLM_NATIVE_ARCH)
  target_compile_options(slm_core PUBLIC -march=native)
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(slm_core PUBLIC OpenMP::OpenMP_CXX)
endif()
find_package(Threads REQUIRED)
target_link_libraries(slm_core PUBLIC Threads::Threads)

if(SLM_WITH_LIBTORCH)
  find_package(Torch REQUIRED)
  target_link_libraries(slm_core PUBLIC "${TORCH_LIBRARIES}")
  target_compile_definitions(slm_core PUBLIC SLM_BACKEND_TORCH=1)
  target_compile_options(slm_core PUBLIC ${TORCH_CXX_FLAGS})
else()
  target_compile_definitions(slm_core PUBLIC SLM_BACKEND_NATIVE=1)
endif()

# ---------------------------------------------------- self-training executable
add_library(slm_live STATIC
  src/app.cpp
  src/chat.cpp
  src/coordinator.cpp
  src/gui.cpp
  src/gui_terminal.cpp
  src/telemetry.cpp
  src/train_continual.cpp
  src/train_feedback.cpp
  src/train_selfgen.cpp
  src/trainer.cpp
)
target_link_libraries(slm_live PUBLIC slm_core)

# ------------------------------------------------------------------- ImGui GUI
if(SLM_WITH_GUI)
  find_package(OpenGL)
  if(NOT OpenGL_FOUND)
    message(WARNING "OpenGL not found - building without the ImGui dashboard")
    set(SLM_WITH_GUI OFF)
  endif()
endif()

if(SLM_WITH_GUI)
  include(FetchContent)
  set(FETCHCONTENT_QUIET OFF)

  find_package(glfw3 3.3 QUIET)
  if(NOT glfw3_FOUND)
    message(STATUS "system GLFW not found - fetching and building it")
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
    # Wayland needs wayland-scanner/protocol headers at build time; X11 (plus
    # XWayland) works everywhere, so it is the default.
    set(GLFW_BUILD_WAYLAND ${SLM_GLFW_WAYLAND} CACHE BOOL "" FORCE)
    set(GLFW_BUILD_X11 ON CACHE BOOL "" FORCE)
    FetchContent_Declare(glfw
      GIT_REPOSITORY https://github.com/glfw/glfw.git
      GIT_TAG 3.4
      GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(glfw)
  endif()

  FetchContent_Declare(imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG v1.91.5
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(imgui)

  add_library(imgui STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp)
  target_include_directories(imgui PUBLIC ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends)
  target_link_libraries(imgui PUBLIC glfw OpenGL::GL)

  target_link_libraries(slm_live PUBLIC imgui)
  target_compile_definitions(slm_live PUBLIC SLM_WITH_GUI=1)
endif()

add_executable(slm src/main.cpp)
target_link_libraries(slm PRIVATE slm_live)

if(SLM_BUILD_TESTS)
  add_executable(slm_gradcheck src/tests/gradcheck.cpp)
  target_link_libraries(slm_gradcheck PRIVATE slm_core)
  enable_testing()
  add_test(NAME gradcheck COMMAND slm_gradcheck)
endif()

# --------------------------------------------------------------------- install
include(GNUInstallDirs)
install(TARGETS slm RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY configs/ DESTINATION ${CMAKE_INSTALL_DATADIR}/slm/configs)
install(FILES scripts/make_sample_data.py scripts/fetch_data.sh
        DESTINATION ${CMAKE_INSTALL_DATADIR}/slm/scripts
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE
                    WORLD_READ WORLD_EXECUTE)
install(FILES README.md docs/ARCHITECTURE.fa.md docs/COORDINATOR.fa.md
        DESTINATION ${CMAKE_INSTALL_DOCDIR})

# ------------------------------------------------------------------ CPack (deb)
set(CPACK_PACKAGE_NAME "slm")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_PACKAGE_CONTACT "slm maintainers <slm@example.org>")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY
    "Small language model with a hybrid self-training pipeline and live monitor")
set(CPACK_DEBIAN_PACKAGE_SECTION "science")
set(CPACK_DEBIAN_PACKAGE_DEPENDS "libc6, libstdc++6, libgomp1")
set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "amd64")
set(CPACK_GENERATOR "DEB;TGZ")
include(CPack)
