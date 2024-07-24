#
# Copyright (C) 2024 INRIA
#

cmake_minimum_required(VERSION 3.16)

set(PROJECT_NAME loik)
set(PROJECT_DESCRIPTION "Low-Order Inverse Kinematics")
set(PROJECT_URL "https://github.com/Simple-Robotics/LoIK")
set(PROJECT_CUSTOM_HEADER_EXTENSION "hpp")
set(PROJECT_CUSTOM_HEADER_DIR "loik")
set(PROJECT_USE_KEYWORD_LINK_LIBRARIES True)
# To enable jrl-cmakemodules compatibility with workspace we must define the two following lines
set(PROJECT_AUTO_RUN_FINALIZE FALSE)
set(PROJECT_SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR})

set(CXX_DISABLE_WERROR True)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(PROJECT_USE_CMAKE_EXPORT True)

option(INSTALL_DOCUMENTATION "Generate and install the documentation" ON)

# Check if the submodule cmake have been initialized
set(JRL_CMAKE_MODULES "${CMAKE_CURRENT_LIST_DIR}/cmake")
if(EXISTS "${JRL_CMAKE_MODULES}/base.cmake")
  message(STATUS "JRL cmakemodules found in 'cmake/' git submodule")
else()
  find_package(jrl-cmakemodules QUIET CONFIG)
  if(jrl-cmakemodules_FOUND)
    get_property(
      JRL_CMAKE_MODULES
      TARGET jrl-cmakemodules::jrl-cmakemodules
      PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
    message(STATUS "JRL cmakemodules found on system at ${JRL_CMAKE_MODULES}")
  elseif(${CMAKE_VERSION} VERSION_LESS "3.14.0")
    message(
      FATAL_ERROR
        "\nCan't find jrl-cmakemodules. Please either:\n"
        "  - use git submodule: 'git submodule update --init'\n"
        "  - or install https://github.com/jrl-umi3218/jrl-cmakemodules\n"
        "  - or upgrade your CMake version to >= 3.14 to allow automatic fetching\n")
  else()
    message(STATUS "JRL cmakemodules not found. Let's fetch it.")
    include(FetchContent)
    FetchContent_Declare("jrl-cmakemodules"
                         GIT_REPOSITORY "https://github.com/jrl-umi3218/jrl-cmakemodules.git")
    FetchContent_MakeAvailable("jrl-cmakemodules")
    FetchContent_GetProperties("jrl-cmakemodules" SOURCE_DIR JRL_CMAKE_MODULES)
  endif()
endif()

set(DOXYGEN_USE_MATHJAX YES)
set(DOXYGEN_USE_TEMPLATE_CSS YES)

# Use BoostConfig module distributed by boost library instead of using FindBoost module distributed
# by CMake
if(POLICY CMP0167)
  cmake_policy(SET CMP0167 NEW)
endif()
include(${JRL_CMAKE_MODULES}/base.cmake)
compute_project_args(PROJECT_ARGS LANGUAGES CXX)
project(${PROJECT_NAME} ${PROJECT_ARGS})
set(CMAKE_VERBOSE_MAKEFILE TRUE)

include(${JRL_CMAKE_MODULES}/boost.cmake)
include(${JRL_CMAKE_MODULES}/ide.cmake)
include(${JRL_CMAKE_MODULES}/apple.cmake)
include(CMakeDependentOption)

apply_default_apple_configuration()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE
      Release
      CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel"
                                               "RelWithDebInfo")
endif()

# --- OPTIONS ----------------------------------------
option(BUILD_WITH_VERSION_SUFFIX "Build libraries with version appended to suffix" OFF)
option(ENABLE_TEMPLATE_INSTANTIATION "Template instantiation of the main library" ON)

# --- OPTIONAL DEPENDENCIES -------------------------
option(INITIALIZE_WITH_NAN "Initialize Eigen entries with NaN" OFF)
option(CHECK_RUNTIME_MALLOC "Check if some memory allocations are performed at runtime" OFF)

# Variable containing all the cflags definition relative to optional dependencies and options
set(CFLAGS_DEPENDENCIES)

if(INITIALIZE_WITH_NAN)
  message(STATUS "Initialize with NaN all the Eigen entries.")
  add_compile_definitions(EIGEN_INITIALIZE_MATRICES_BY_NAN)
endif()

if(CHECK_RUNTIME_MALLOC)
  message(STATUS "Check if some memory allocations are performed at runtime.")
  add_compile_definitions(LOIK_EIGEN_CHECK_MALLOC)
  add_compile_definitions(EIGEN_RUNTIME_NO_MALLOC)
endif()

if(ENABLE_TEMPLATE_INSTANTIATION)
  add_compile_definitions(LOIK_ENABLE_TEMPLATE_INSTANTIATION)
  list(APPEND CFLAGS_DEPENDENCIES "-DLOIK_ENABLE_TEMPLATE_INSTANTIATION")
endif()

macro(TAG_LIBRARY_VERSION target)
  set_target_properties(${target} PROPERTIES SOVERSION ${PROJECT_VERSION})
endmacro()

# ----------------------------------------------------
# --- DEPENDENCIES -----------------------------------
# ----------------------------------------------------
add_project_dependency(Eigen3 3.4.0 REQUIRED PKG_CONFIG_REQUIRES "eigen3 >= 3.4.0")
add_project_dependency(pinocchio REQUIRED PKG_CONFIG_REQUIRES "pinocchio >= 3.0.0")

set(BOOST_REQUIRED_COMPONENTS serialization)
set_boost_default_options()
export_boost_default_options()
add_project_dependency(Boost REQUIRED COMPONENTS ${BOOST_REQUIRED_COMPONENTS})

# --- MAIN LIBRARY ----------------------------------------
set(LIB_HEADER_DIR ${PROJECT_SOURCE_DIR}/include/${PROJECT_NAME})
set(LIB_HEADERS
    ${LIB_HEADER_DIR}/fwd.hpp
    ${LIB_HEADER_DIR}/ik-id-description.hpp
    ${LIB_HEADER_DIR}/ik-id-description-optimized.hpp
    ${LIB_HEADER_DIR}/loik-loid-data.hpp
    ${LIB_HEADER_DIR}/loik-loid-data.hxx
    ${LIB_HEADER_DIR}/loik-loid-data-optimized.hpp
    ${LIB_HEADER_DIR}/loik-loid-data-optimized.hxx
    ${LIB_HEADER_DIR}/loik-loid.hpp
    ${LIB_HEADER_DIR}/loik-loid.hxx
    ${LIB_HEADER_DIR}/loik-loid-optimized.hpp
    ${LIB_HEADER_DIR}/loik-loid-optimized.hxx
    ${LIB_HEADER_DIR}/task-solver-base.hpp)
set(LIB_TEMPLATE_INSTANTIATION_HEADERS
    ${LIB_HEADER_DIR}/loik-loid.txx ${LIB_HEADER_DIR}/loik-loid-optimized.txx
    ${LIB_HEADER_DIR}/loik-loid-data.txx ${LIB_HEADER_DIR}/loik-loid-data-optimized.txx)

set(LIB_SOURCE_DIR ${PROJECT_SOURCE_DIR}/src)
set(LIB_SOURCES)
set(LIB_TEMPLATE_INSTANTIATION_SOURCES
    ${LIB_SOURCE_DIR}/loik-loid.cpp ${LIB_SOURCE_DIR}/loik-loid-optimized.cpp
    ${LIB_SOURCE_DIR}/loik-loid-data.cpp ${LIB_SOURCE_DIR}/loik-loid-data-optimized.cpp)

if(ENABLE_TEMPLATE_INSTANTIATION)
  list(APPEND LIB_HEADERS ${LIB_TEMPLATE_INSTANTIATION_HEADERS})
  list(APPEND LIB_SOURCES ${LIB_TEMPLATE_INSTANTIATION_SOURCES})
endif()

function(set_standard_output_directory target)
  set_target_properties(
    ${target}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin
               LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib
               ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
endfunction()

function(create_library)
  # get the compile defs
  get_directory_property(CURRENT_COMPILE_DEFINITIONS COMPILE_DEFINITIONS)

  set(library_type INTERFACE)
  if(LIB_SOURCES)
    set(library_type SHARED)
  endif()

  add_library(${PROJECT_NAME} ${library_type} ${LIB_HEADERS} ${LIB_SOURCES})
  add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})
  set_standard_output_directory(${PROJECT_NAME})
  set_target_properties(
    ${PROJECT_NAME}
    PROPERTIES LINKER_LANGUAGE CXX
               INSTALL_RPATH "\$ORIGIN"
               VERSION ${PROJECT_VERSION})

  cxx_flags_by_compiler_frontend(MSVC "NOMINMAX" OUTPUT PUBLIC_DEFINITIONS)
  target_compile_definitions(${PROJECT_NAME} PUBLIC ${CURRENT_COMPILE_DEFINITIONS}
                                                    ${PUBLIC_DEFINITIONS})

  if(BUILD_LIBRARY_WITH_VERSION_SUFFIX)
    tag_library_version(${PROJECT_NAME})
  endif()

  target_link_libraries(${PROJECT_NAME} PUBLIC pinocchio::pinocchio)
  target_link_libraries(${PROJECT_NAME} PUBLIC Eigen3::Eigen)
  target_link_libraries(${PROJECT_NAME} PUBLIC Boost::boost)

  target_include_directories(
    ${PROJECT_NAME}
    PUBLIC $<INSTALL_INTERFACE:include> $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
           $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)

  cxx_flags_by_compiler_frontend(
    MSVC "/bigobj"
    OUTPUT PUBLIC_OPTIONS
    FILTER)
  target_compile_options(${PROJECT_NAME} PUBLIC ${PUBLIC_OPTIONS})
endfunction()

create_library()

add_header_group(LIB_HEADERS)
add_source_group(LIB_SOURCES)
# Allow jrl-cmakemodules to install headers in the right directory
set(${PROJECT_NAME}_HEADERS ${LIB_HEADERS})

install(
  TARGETS ${PROJECT_NAME}
  EXPORT ${TARGETS_EXPORT_NAME}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()

# --- PACKAGING ----------------------------------------------------------------
macro(EXPORT_VARIABLE var_name var_value)
  get_directory_property(has_parent PARENT_DIRECTORY)
  if(has_parent)
    set(${var_name}
        ${var_value}
        PARENT_SCOPE)
  else()
    set(${var_name} ${var_value})
  endif()
  string(APPEND PACKAGE_EXTRA_MACROS "\nset(${var_name} ${var_value})")
endmacro()

if(ENABLE_TEMPLATE_INSTANTIATION)
  export_variable(LOIK_ENABLE_TEMPLATE_INSTANTIATION ON)
endif()

pkg_config_append_libs(${PROJECT_NAME})
pkg_config_append_boost_libs(${BOOST_REQUIRED_COMPONENTS})
pkg_config_append_cflags("${CFLAGS_DEPENDENCIES}")

setup_project_finalize()
