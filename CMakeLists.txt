# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.16)
project(slm VERSION 0.3.0 LANGUAGES CXX)

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
  src/core/quant.cpp
  src/core/serialize.cpp
  src/core/tensor_common.cpp
  src/core/text.cpp
  src/memory.cpp
  src/model.cpp
  src/qmodel.cpp
  src/spt_assets.cpp
  src/telemetry.cpp
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

# ------------------------------------- model backends, agent tools, debate
# Sits between the tensor/model core and the application: it may use the model
# and the telemetry hub, and it must NOT know about the trainers or the GUI.
add_library(slm_agent STATIC
  src/agent/codebase.cpp
  src/agent/http.cpp
  src/agent/runtime.cpp
  src/agent/tool_codebase.cpp
  src/agent/tool_shell.cpp
  src/agent/tool_web.cpp
  src/agent/tools.cpp
  src/backend/backend_gguf.cpp
  src/backend/backend_spt.cpp
  src/backend/model_backend.cpp
  src/debate/debate.cpp
)
target_link_libraries(slm_agent PUBLIC slm_core)
target_compile_options(slm_agent PRIVATE -Wall -Wextra -Wno-unused-parameter)

if(SLM_WITH_CURL)
  find_package(CURL QUIET)
  if(CURL_FOUND)
    target_link_libraries(slm_agent PUBLIC CURL::libcurl)
    target_compile_definitions(slm_agent PUBLIC SLM_WITH_CURL=1)
    message(STATUS "web tools: libcurl ${CURL_VERSION_STRING}")
  else()
    message(STATUS "web tools: libcurl not found - falling back to the curl binary")
  endif()
endif()

# ---------------------------------------------------------------- llama.cpp
if(SLM_WITH_LLAMA)
  set(LLAMA_BUILD_TESTS   OFF CACHE BOOL "" FORCE)
  set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(LLAMA_BUILD_SERVER  OFF CACHE BOOL "" FORCE)
  set(LLAMA_BUILD_TOOLS   OFF CACHE BOOL "" FORCE)
  set(LLAMA_CURL          OFF CACHE BOOL "" FORCE)
  set(GGML_NATIVE         ON  CACHE BOOL "" FORCE)
  if(SLM_LLAMA_DIR AND EXISTS "${SLM_LLAMA_DIR}/include/llama.h")
    message(STATUS "llama.cpp: using ${SLM_LLAMA_DIR}")
    add_subdirectory("${SLM_LLAMA_DIR}" "${CMAKE_BINARY_DIR}/llama-build" EXCLUDE_FROM_ALL)
  else()
    include(FetchContent)
    FetchContent_Declare(llama_cpp
      GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
      GIT_TAG b6833
      GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(llama_cpp)
  endif()
  target_link_libraries(slm_agent PUBLIC llama)
  target_compile_definitions(slm_agent PUBLIC SLM_WITH_LLAMA=1)
endif()

# ---------------------------------------------------- self-training executable
add_library(slm_live STATIC
  src/app.cpp
  src/gui_text.cpp
  src/chat.cpp
  src/coordinator.cpp
  src/gui.cpp
  src/gui_agent.cpp
  src/gui_terminal.cpp
  src/train_continual.cpp
  src/train_feedback.cpp
  src/train_selfgen.cpp
  src/trainer.cpp
)
target_link_libraries(slm_live PUBLIC slm_core slm_agent)

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

  # Correct Persian/Arabic rendering needs real text shaping.  Without these two
  # libraries the dashboard still builds and runs, it just cannot draw connected
  # Persian script (the terminal dashboard is the fallback for that).
  find_package(PkgConfig QUIET)
  if(PkgConfig_FOUND)
    pkg_check_modules(FREETYPE2 QUIET freetype2)
    pkg_check_modules(HARFBUZZ QUIET harfbuzz)
  endif()
  if(FREETYPE2_FOUND AND HARFBUZZ_FOUND)
    message(STATUS "text shaping: freetype ${FREETYPE2_VERSION} + harfbuzz ${HARFBUZZ_VERSION}")
    target_include_directories(slm_live PUBLIC ${FREETYPE2_INCLUDE_DIRS} ${HARFBUZZ_INCLUDE_DIRS})
    target_link_libraries(slm_live PUBLIC ${FREETYPE2_LIBRARIES} ${HARFBUZZ_LIBRARIES})
    target_link_directories(slm_live PUBLIC ${FREETYPE2_LIBRARY_DIRS} ${HARFBUZZ_LIBRARY_DIRS})
    target_compile_definitions(slm_live PUBLIC SLM_WITH_TEXT_SHAPING=1)
  else()
    message(WARNING "freetype2/harfbuzz not found - the GUI will not shape Persian text\n"
                    "   install: libfreetype-dev libharfbuzz-dev (Debian) / freetype-devel harfbuzz-devel (Fedora)")
  endif()
endif()

add_executable(slm src/main.cpp)
target_link_libraries(slm PRIVATE slm_live)

if(SLM_BUILD_TESTS)
  add_executable(slm_gradcheck src/tests/gradcheck.cpp)
  target_link_libraries(slm_gradcheck PRIVATE slm_core)
  add_executable(slm_texttest src/tests/text_test.cpp)
  target_link_libraries(slm_texttest PRIVATE slm_core)
  add_executable(slm_qtest src/tests/quant_test.cpp)
  target_link_libraries(slm_qtest PRIVATE slm_core)
  add_executable(slm_agenttest src/tests/agent_test.cpp)
  target_link_libraries(slm_agenttest PRIVATE slm_agent)
  add_executable(slm_codebasetest src/tests/codebase_test.cpp)
  target_link_libraries(slm_codebasetest PRIVATE slm_agent)
  add_executable(slm_debatetest src/tests/debate_test.cpp)
  target_link_libraries(slm_debatetest PRIVATE slm_agent)
  enable_testing()
  add_test(NAME gradcheck COMMAND slm_gradcheck)
  add_test(NAME text COMMAND slm_texttest)
  add_test(NAME quant COMMAND slm_qtest)
  add_test(NAME agent COMMAND slm_agenttest)
  add_test(NAME codebase COMMAND slm_codebasetest)
  add_test(NAME debate COMMAND slm_debatetest)
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
        docs/AGENT.fa.md docs/QUANTIZATION.fa.md
        DESTINATION ${CMAKE_INSTALL_DOCDIR})

# The model ships with the package: without it `slm up` has to bootstrap a
# starter model, which works but is not what someone installing a package wants.
install(FILES models/spt.slmtok models/spt.slm models/spt-q4.slmq
        DESTINATION ${CMAKE_INSTALL_DATADIR}/slm/models OPTIONAL)

# A launcher, so the dashboard is also one double click.
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/packaging/slm.desktop.in
               ${CMAKE_CURRENT_BINARY_DIR}/slm.desktop @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/slm.desktop
        DESTINATION ${CMAKE_INSTALL_DATADIR}/applications)

# ------------------------------------------------------------------ CPack (deb)
set(CPACK_PACKAGE_NAME "slm")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_PACKAGE_CONTACT "slm maintainers <slm@example.org>")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY
    "Small language model with a hybrid self-training pipeline and live monitor")
set(CPACK_DEBIAN_PACKAGE_SECTION "science")
set(CPACK_DEBIAN_PACKAGE_DEPENDS "libc6, libstdc++6, libgomp1")
set(CPACK_DEBIAN_PACKAGE_RECOMMENDS "libcurl4, libglfw3, libharfbuzz0b, libfreetype6")
set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "amd64")
set(CPACK_GENERATOR "DEB;TGZ")
include(CPack)
