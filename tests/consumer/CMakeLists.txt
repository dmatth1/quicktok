# Minimal downstream consumer: proves `find_package(quicktok)` + the installed
# CMake config/targets work for an external project. Built in CI against an
# install tree (see the cmake-test job).
cmake_minimum_required(VERSION 3.16)
project(quicktok_consumer CXX)
find_package(quicktok REQUIRED)
add_executable(consumer app.cpp)
target_link_libraries(consumer PRIVATE quicktok::quicktok)
