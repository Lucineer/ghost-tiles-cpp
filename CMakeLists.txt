cmake_minimum_required(VERSION 3.16)
project(ghost_tiles LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(ghost_tiles INTERFACE)
target_include_directories(ghost_tiles INTERFACE include)

enable_testing()
add_executable(test_ghost_tiles tests/test_ghost_tiles.cpp)
target_link_libraries(test_ghost_tiles PRIVATE ghost_tiles)
add_test(NAME ghost_tiles COMMAND test_ghost_tiles)
