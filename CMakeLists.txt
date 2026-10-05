cmake_minimum_required(VERSION 3.20)

project(quantum_pricing
  VERSION 0.1.0
  DESCRIPTION "Imaginary-time Schrödinger solver for option pricing under a projected stochastic-volatility model"
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(DEFINED SKBUILD)
  set(_qp_default_standalone OFF)
  set(_qp_default_python ON)
else()
  set(_qp_default_standalone ON)
  set(_qp_default_python OFF)
endif()

option(QP_BUILD_TESTS "Build the C++ unit tests" ${_qp_default_standalone})
option(QP_BUILD_APPS "Build the command-line tools" ${_qp_default_standalone})
option(QP_BUILD_PYTHON "Build the pybind11 extension module" ${_qp_default_python})
option(QP_NATIVE "Optimise for the host CPU (-march=native)" OFF)

# ---------------------------------------------------------------------------
# Core library
# ---------------------------------------------------------------------------
add_library(qpcore STATIC
  cpp/src/black76.cpp
  cpp/src/tridiagonal.cpp
  cpp/src/effective_vol.cpp
  cpp/src/schrodinger.cpp
  cpp/src/pricer.cpp
  cpp/src/spectral.cpp)
add_library(qp::core ALIAS qpcore)
target_include_directories(qpcore PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include)

if(MSVC)
  target_compile_options(qpcore PRIVATE /W4)
else()
  target_compile_options(qpcore PRIVATE -Wall -Wextra -Wpedantic -Wshadow -Wconversion
                                        -Wno-sign-conversion)
  if(QP_NATIVE)
    target_compile_options(qpcore PUBLIC -march=native)
  endif()
endif()

# ---------------------------------------------------------------------------
# Command-line tools
# ---------------------------------------------------------------------------
if(QP_BUILD_APPS)
  add_executable(qp_price cpp/apps/qp_price.cpp)
  target_link_libraries(qp_price PRIVATE qp::core)
  add_executable(qp_bench cpp/apps/qp_bench.cpp)
  target_link_libraries(qp_bench PRIVATE qp::core)
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(QP_BUILD_TESTS)
  include(FetchContent)
  FetchContent_Declare(doctest
    GIT_REPOSITORY https://github.com/doctest/doctest.git
    GIT_TAG v2.4.12
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(doctest)

  enable_testing()
  file(GLOB QP_TEST_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cpp/tests/test_*.cpp)
  add_executable(qp_tests cpp/tests/main.cpp ${QP_TEST_SOURCES})
  target_link_libraries(qp_tests PRIVATE qp::core doctest::doctest)
  add_test(NAME qp_tests COMMAND qp_tests)
endif()

# ---------------------------------------------------------------------------
# Python extension
# ---------------------------------------------------------------------------
if(QP_BUILD_PYTHON)
  find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(_qpcore cpp/bindings/module.cpp)
  target_link_libraries(_qpcore PRIVATE qp::core)
  install(TARGETS _qpcore DESTINATION qpricing)
endif()
