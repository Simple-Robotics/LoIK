#
# Copyright (C) 2024 INRIA
#

set(PROBLEM-UTILS-LIB-DIR ${CMAKE_CURRENT_LIST_DIR})
set(PROBLEM-UTILS-LIB-HEADERS 
    ${PROBLEM-UTILS-LIB-DIR}/problem-utils.hpp)
set(PROBLEM-UTILS-LIB-SOURCES 
    ${PROBLEM-UTILS-LIB-DIR}/problem-utils.cpp)


add_library(problem-utils STATIC ${PROBLEM-UTILS-LIB-HEADERS} ${PROBLEM-UTILS-LIB-SOURCES})
target_include_directories(problem-utils PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(problem-utils PRIVATE ${PROJECT_NAME} Boost::boost)