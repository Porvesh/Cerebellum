cmake_minimum_required(VERSION 3.16)
project(cerebellum CXX)

# C++20 everywhere, no compiler-specific extensions, warnings on.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
add_compile_options(-Wall -Wextra)

# The hard real-time core stays header-only. The optional Python bridge is a
# compiled POSIX adapter and never runs on the control thread.
add_library(cerebellum_core INTERFACE)
target_include_directories(cerebellum_core INTERFACE ${CMAKE_SOURCE_DIR}/include)

find_package(Threads REQUIRED)

add_library(cerebellum_python_bridge STATIC src/python_chunk_generator.cpp)
target_link_libraries(cerebellum_python_bridge PUBLIC cerebellum_core Threads::Threads)

# --- tests ---
# One dependency-free test per component, each returning its failure count.
enable_testing()

add_executable(test_config tests/test_config.cpp)
target_link_libraries(test_config PRIVATE cerebellum_core)
add_test(NAME config COMMAND test_config)

# Timing links Threads::Threads for sleep_until — the harness is single-threaded,
# but std::this_thread pulls in pthread on libstdc++.
add_executable(test_timing tests/test_timing.cpp)
target_link_libraries(test_timing PRIVATE cerebellum_core Threads::Threads)
add_test(NAME timing COMMAND test_timing)

add_executable(test_chunk_queue tests/test_chunk_queue.cpp)
target_link_libraries(test_chunk_queue PRIVATE cerebellum_core)
add_test(NAME chunk_queue COMMAND test_chunk_queue)

add_executable(test_chunk_queue_mt tests/test_chunk_queue_mt.cpp)
target_link_libraries(test_chunk_queue_mt PRIVATE cerebellum_core Threads::Threads)
add_test(NAME chunk_queue_mt COMMAND test_chunk_queue_mt)

add_executable(test_loop tests/test_loop.cpp)
target_link_libraries(test_loop PRIVATE cerebellum_core Threads::Threads)
add_test(NAME loop COMMAND test_loop)

find_package(Python3 COMPONENTS Interpreter REQUIRED)
add_executable(test_python_chunk_generator tests/test_python_chunk_generator.cpp)
target_link_libraries(test_python_chunk_generator PRIVATE cerebellum_python_bridge)
target_compile_definitions(test_python_chunk_generator PRIVATE
    CEREBELLUM_TEST_PYTHON="${Python3_EXECUTABLE}"
    CEREBELLUM_TEST_PYTHONPATH="${CMAKE_SOURCE_DIR}/python")
add_test(NAME python_chunk_generator COMMAND test_python_chunk_generator)

# ThreadSanitizer for the concurrency tests — the chunk queue has one producer
# (inference worker) and one consumer (control thread), so it earns the same
# scrutiny Retina's buffers got. Off by default; flip on with -DCEREBELLUM_TSAN=ON
# in a separate build dir.
option(CEREBELLUM_TSAN "Build the concurrency tests with ThreadSanitizer" OFF)

if(CEREBELLUM_TSAN)
    message(STATUS "cerebellum: ThreadSanitizer enabled")
    target_compile_options(test_chunk_queue_mt PRIVATE -fsanitize=thread -g -O1)
    target_link_options(test_chunk_queue_mt PRIVATE -fsanitize=thread)
    target_compile_options(test_loop PRIVATE -fsanitize=thread -g -O1)
    target_link_options(test_loop PRIVATE -fsanitize=thread)
endif()
