cmake_minimum_required(VERSION 3.23)
project(nanogpt-cpp LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(spdlog REQUIRED)
find_package(Boost REQUIRED COMPONENTS regex)
find_package(absl REQUIRED)
find_package(OpenMP)
find_package(GTest REQUIRED)

add_library(tokenizer_lib STATIC src/tokenizer.cc src/bpe.cc)
target_include_directories(tokenizer_lib PUBLIC src)
target_link_libraries(tokenizer_lib PUBLIC spdlog::spdlog Boost::regex absl::flat_hash_map)

add_executable(nanogpt-cpp src/main.cc)
target_link_libraries(nanogpt-cpp tokenizer_lib OpenMP::OpenMP_CXX)

enable_testing()
add_executable(tokenizer_test tests/tokenizer_test.cc)
target_link_libraries(tokenizer_test tokenizer_lib GTest::gtest_main)
target_compile_definitions(tokenizer_test PRIVATE TEST_SRCDIR="${CMAKE_SOURCE_DIR}")
add_test(NAME tokenizer_test COMMAND tokenizer_test)
