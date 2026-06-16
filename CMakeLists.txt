cmake_minimum_required(VERSION 3.16)
project(quicktok VERSION 0.4.0 LANGUAGES CXX)

option(QUICKTOK_BUILD_TESTS "Build quicktok tests" OFF)
option(QUICKTOK_BUILD_EXAMPLE "Build the hello example" OFF)
option(QUICKTOK_NATIVE "Tune for the build host (-march=native)" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

# --- Python extension build (driven by scikit-build-core / pip) ---
# When building a wheel, build ONLY the pybind11 module and bundle the data,
# then stop — the C++ library targets/install below are for native consumers.
if(DEFINED SKBUILD)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(_quicktok
    python/src/_quicktok.cpp
    src/quicktok.cpp
    src/trie2_mb.cpp)
  target_include_directories(_quicktok PRIVATE include src)
  install(TARGETS _quicktok LIBRARY DESTINATION quicktok)
  install(DIRECTORY data/ DESTINATION quicktok/data
    FILES_MATCHING PATTERN "*.vocab" PATTERN "*.bin" PATTERN "*.special" PATTERN "*.meta")
  return()
endif()

add_library(quicktok
  src/quicktok.cpp
  src/trie2_mb.cpp
  src/cabi.cpp)
add_library(quicktok::quicktok ALIAS quicktok)

target_include_directories(quicktok PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_include_directories(quicktok PRIVATE src)

# consumers inherit the C++20 requirement automatically
target_compile_features(quicktok PUBLIC cxx_std_20)

# Portable by default. The kernel autovectorizes from baseline SSE2 / NEON, so a
# portable binary keeps most of the speed; QUICKTOK_NATIVE squeezes the rest on a
# machine you build and run on yourself.
if(QUICKTOK_NATIVE)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-march=native" QT_HAS_MARCH_NATIVE)
  if(QT_HAS_MARCH_NATIVE)
    target_compile_options(quicktok PRIVATE -march=native)
  endif()
endif()

set_target_properties(quicktok PROPERTIES
  VERSION ${PROJECT_VERSION} SOVERSION 0 POSITION_INDEPENDENT_CODE ON)

find_package(Threads REQUIRED)
target_link_libraries(quicktok PUBLIC Threads::Threads)

if(QUICKTOK_BUILD_TESTS)
  enable_testing()
  add_executable(test_quicktok test/test_quicktok.cpp)
  target_link_libraries(test_quicktok PRIVATE quicktok)
  add_test(NAME quicktok_test
           COMMAND test_quicktok ${CMAKE_CURRENT_SOURCE_DIR}/data
           WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
endif()

if(QUICKTOK_BUILD_EXAMPLE)
  add_executable(hello examples/hello.cpp)
  target_link_libraries(hello PRIVATE quicktok)
endif()

# --- install + package config (find_package(quicktok) for consumers) ---
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS quicktok EXPORT quicktokTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(FILES include/quicktok.hpp include/quicktok.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY data/ DESTINATION ${CMAKE_INSTALL_DATADIR}/quicktok
  FILES_MATCHING PATTERN "*.vocab" PATTERN "*.bin" PATTERN "*.special" PATTERN "*.meta")

install(EXPORT quicktokTargets
  FILE quicktokTargets.cmake NAMESPACE quicktok::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/quicktok)
configure_package_config_file(
  cmake/quicktokConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/quicktokConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/quicktok)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/quicktokConfigVersion.cmake
  VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/quicktokConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/quicktokConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/quicktok)
