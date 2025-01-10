#
# Copyright (C) 2024 INRIA
#

add_project_dependency(Boost COMPONENTS unit_test_framework REQUIRED)

add_project_private_dependency(example-robot-data REQUIRED)

function(add_loik_test name)
  set(test_name "${PROJECT_NAME}-test-cpp-${name}")
  set(test_file ${name}.cpp)

  add_unit_test(${test_name} ${test_file})
  set_standard_output_directory(${test_name})
  set_target_properties(${test_name} PROPERTIES LINKER_LANGUAGE CXX)

  # define macros required by boost_test see:
  # https://www.boost.org/doc/libs/1_78_0/libs/test/doc/html/boost_test/usage_variants.html define
  # module name, replace '-' by '_'
  set(MODULE_NAME "${name}Test")
  string(REGEX REPLACE "-" "_" MODULE_NAME ${MODULE_NAME})

  target_compile_definitions(${test_name} PRIVATE BOOST_TEST_DYN_LINK
                                                  BOOST_TEST_MODULE=${MODULE_NAME})

  target_link_libraries(${test_name} PRIVATE ${PROJECT_NAME} Boost::unit_test_framework
                                             example-robot-data::example-robot-data)

  if(${BUILD_WITH_EXTENDED_TESTING})
    target_include_directories(${test_name} PRIVATE ${PROJECT_SOURCE_DIR})
    target_link_libraries(${test_name} PRIVATE problem-utils)
  endif()
endfunction()

add_loik_test(loik-loid)

if(${BUILD_WITH_EXTENDED_TESTING})
  add_loik_test(loik-extended)
endif()


string(TOUPPER "${CMAKE_BUILD_TYPE}" uppercase_CMAKE_BUILD_TYPE)
if(uppercase_CMAKE_BUILD_TYPE STREQUAL DEBUG)
  set_tests_properties(${PROJECT_NAME}-test-cpp-loik-loid PROPERTIES TIMEOUT 3000)
endif()

add_loik_test(loik-loid-data)
