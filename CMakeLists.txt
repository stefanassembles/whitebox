cmake_minimum_required(VERSION 3.14)
project(simple_tokenizer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Fetch nlohmann/json
include(FetchContent)
FetchContent_Declare(
  json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.2
)
FetchContent_MakeAvailable(json)

# Find system-installed pybind11
find_package(pybind11 REQUIRED CONFIG)

# Build the module
pybind11_add_module(simple_tokenizer src/tokenizer.cpp)

target_link_libraries(simple_tokenizer PRIVATE nlohmann_json::nlohmann_json)

# Optional: Output the .so file into the py/ directory
set_target_properties(simple_tokenizer PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/py
)

