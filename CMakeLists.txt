cmake_minimum_required(VERSION 3.23)
project(gptcpp LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(spdlog REQUIRED)
find_package(Boost REQUIRED COMPONENTS regex)
find_package(absl REQUIRED)

add_library(tokenizer_lib STATIC src/tokenizer.cc src/bpe.cc)
target_include_directories(tokenizer_lib PUBLIC src)
target_link_libraries(tokenizer_lib PUBLIC spdlog::spdlog Boost::regex absl::flat_hash_map)

add_executable(gptcpp src/main.cc)
target_link_libraries(gptcpp tokenizer_lib)
