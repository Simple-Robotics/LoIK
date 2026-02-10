#
# Copyright (C) 2024 INRIA
#

add_project_private_dependency(example-robot-data REQUIRED)

function(add_loik_benchmark name)
  set(bench_name "${PROJECT_NAME}-bench-cpp-${name}")
  set(bench_file ${name}.cpp)

  message(STATUS "Adding benchmark ${bench_name}")
  add_executable(${bench_name} ${bench_file})
  set_standard_output_directory(${bench_name})
  set_target_properties(${bench_name} PROPERTIES LINKER_LANGUAGE CXX)
  target_link_libraries(${bench_name} PRIVATE ${PROJECT_NAME} benchmark::benchmark example-robot-data::example-robot-data)

  if(${BUILD_WITH_EXTENDED_TESTING})
    target_include_directories(${bench_name} PRIVATE ${PROJECT_SOURCE_DIR})
    target_link_libraries(${bench_name} PRIVATE problem-utils)
  endif()

endfunction()


add_loik_benchmark(timings)