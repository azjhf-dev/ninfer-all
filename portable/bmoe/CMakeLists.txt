# BigMoeOnEdge engine build for the portable companion. This is NOT the native
# NInfer engine and not the plain llama.cpp companion: it is a separate
# MoE-streaming engine (Helldez/BigMoeOnEdge, Apache-2.0) built on a pinned
# llama.cpp revision, for MoE GGUF models larger than RAM/VRAM.
cmake_minimum_required(VERSION 3.21)
project(ninfer_portable_bmoe LANGUAGES CXX)

option(NINFER_PORTABLE_BMOE_VULKAN "Enable the Vulkan backend for the 680M trial" OFF)

include(FetchContent)

# Conservative CPU targeting: no build-host ISA inheritance, no NVIDIA, no
# unqualified ROCm. BigMoeOnEdge builds one CPU backend module per ISA tier
# (BMOE_BACKEND_MODULES ON by default) and loads what the machine supports.
set(GGML_NATIVE OFF CACHE BOOL "Do not inherit the build machine's ISA" FORCE)
set(GGML_CUDA OFF CACHE BOOL "No NVIDIA dependency" FORCE)
set(GGML_HIP OFF CACHE BOOL "No unqualified ROCm path" FORCE)
set(GGML_VULKAN ${NINFER_PORTABLE_BMOE_VULKAN} CACHE BOOL "Opt-in Vulkan" FORCE)
set(BMOE_BUILD_CLI ON CACHE BOOL "Build bmoe-cli" FORCE)
set(BMOE_BUILD_SERVER ON CACHE BOOL "Build bmoe-server" FORCE)
set(BMOE_BUILD_DESKTOP OFF CACHE BOOL "No webview/desktop app in this companion" FORCE)
set(BMOE_BUILD_TESTS OFF CACHE BOOL "Upstream test suite not built by the launcher" FORCE)

# Pinned engine source: BigMoeOnEdge 0.43.1 commit.
FetchContent_Declare(bmoe
  URL https://codeload.github.com/Helldez/BigMoeOnEdge/tar.gz/600dc2966467b035c4d5de09c839985b2dc5c286
  URL_HASH SHA256=b2f1ac1f336ba5722291b8e5caa80e32d6aa08523b3efb2b1c02a9ba89a6b01f)
# Pinned llama.cpp: the Helldez fork commit the BigMoeOnEdge submodule points at
# (upstream plus one commit, the expert-readiness hook used by --overlap).
FetchContent_Declare(bmoe_llama
  URL https://codeload.github.com/Helldez/llama.cpp/tar.gz/ac3098770cde1d495befaa7d1066076652190aa2
  URL_HASH SHA256=d6ebac8fc5cebfe5d7066fb642385500882b0233a889a2c373402abb2414cb0d)

FetchContent_GetProperties(bmoe)
if(NOT bmoe_POPULATED)
  FetchContent_Populate(bmoe)
  FetchContent_GetProperties(bmoe_llama)
  if(NOT bmoe_llama_POPULATED)
    FetchContent_Populate(bmoe_llama)
  endif()
  # Place the pinned llama.cpp where the BigMoeOnEdge submodule would sit.
  file(COPY ${bmoe_llama_SOURCE_DIR}/ DESTINATION ${bmoe_SOURCE_DIR}/third_party/llama.cpp)

  # Upstream 0.43.1 assumes it is the top-level project: server/CMakeLists.txt
  # references third_party through ${CMAKE_SOURCE_DIR}. As a FetchContent
  # subproject that path points at this wrapper, so rewrite it to the real
  # location. Pinned commit, so fail loudly if the pattern ever disappears.
  file(READ ${bmoe_SOURCE_DIR}/server/CMakeLists.txt _server_cmake)
  set(_wrong [=[${CMAKE_SOURCE_DIR}/third_party/]=])
  set(_right [=[${CMAKE_CURRENT_SOURCE_DIR}/../third_party/]=])
  string(FIND "${_server_cmake}" "${_wrong}" _found)
  if(_found EQUAL -1)
    message(FATAL_ERROR "BigMoeOnEdge server CMake no longer uses the expected third_party path; review before building as a subproject.")
  endif()
  string(REPLACE "${_wrong}" "${_right}" _server_cmake "${_server_cmake}")
  file(WRITE ${bmoe_SOURCE_DIR}/server/CMakeLists.txt "${_server_cmake}")
endif()

add_subdirectory(${bmoe_SOURCE_DIR} ${bmoe_BINARY_DIR})
