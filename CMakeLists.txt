cmake_minimum_required(VERSION 3.16)
project(cerebellum CXX)

# C++20 everywhere, no compiler-specific extensions, warnings on.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
add_compile_options(-Wall -Wextra)

# The runtime is header-only: config, chunk queue, timing harness, control loop.
# An INTERFACE library carries no compiled code — it hands its include path to
# anything that links it.
add_library(cerebellum_core INTERFACE)
target_include_directories(cerebellum_core INTERFACE ${CMAKE_SOURCE_DIR}/include)

find_package(Threads REQUIRED)

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

# As components land, register them here:
#
#   add_executable(test_chunk_queue tests/test_chunk_queue.cpp)
#   target_link_libraries(test_chunk_queue PRIVATE cerebellum_core)
#   add_test(NAME chunk_queue COMMAND test_chunk_queue)
#
# Anything touching two threads also links Threads::Threads.

# ThreadSanitizer for the concurrency tests — the chunk queue has one producer
# (inference worker) and one consumer (control thread), so it earns the same
# scrutiny Retina's buffers got. Off by default; flip on with -DCEREBELLUM_TSAN=ON
# in a separate build dir.
option(CEREBELLUM_TSAN "Build the concurrency tests with ThreadSanitizer" OFF)

if(CEREBELLUM_TSAN)
    message(STATUS "cerebellum: ThreadSanitizer enabled")
    # Add -fsanitize=thread to the MT test targets as they appear, e.g.
    #   target_compile_options(test_chunk_queue_mt PRIVATE -fsanitize=thread -g -O1)
    #   target_link_options(test_chunk_queue_mt    PRIVATE -fsanitize=thread)
endif()
