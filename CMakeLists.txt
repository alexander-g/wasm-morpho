cmake_minimum_required(VERSION 3.12)
project(traininglib_cpp)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_VERBOSE_MAKEFILE on)


set(LIB_SRC_FILES
    morphology.cpp
)


#suppress a warning regarding FetchContent
cmake_policy(SET CMP0135 NEW)
include(FetchContent)


if(DEFINED CMAKE_TOOLCHAIN_FILE AND CMAKE_TOOLCHAIN_FILE MATCHES "emscripten" OR DEFINED ENV{EMSCRIPTEN})
  set(RUNNING_EMSCRIPTEN TRUE)
else()
  set(RUNNING_EMSCRIPTEN FALSE)
endif()


set(EIGEN_URL https://gitlab.com/libeigen/eigen/-/archive/5.0.0/eigen-5.0.0.zip)
message("Maybe downloading ${EIGEN_URL}")
FetchContent_Declare(eigen 
    URL ${EIGEN_URL}
)
FetchContent_MakeAvailable(eigen)


if(NOT RUNNING_EMSCRIPTEN)

set(PYBIND_URL https://github.com/pybind/pybind11/archive/refs/tags/v3.0.1.zip)
message("Maybe downloading ${PYBIND_URL}")
FetchContent_Declare(pybind 
    URL ${PYBIND_URL}
)
FetchContent_MakeAvailable(pybind)


pybind11_add_module(traininglib_cpp_ext main_pybind.cpp ${LIB_SRC_FILES})

target_include_directories(traininglib_cpp_ext PRIVATE ${eigen_SOURCE_DIR})
target_compile_options(traininglib_cpp_ext PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang,Intel>: -O3 >
  $<$<CXX_COMPILER_ID:MSVC>: /O2 /GL>
)

endif()




if(RUNNING_EMSCRIPTEN)

add_library(traininglib_cpp STATIC 
    main_wasm.cpp
    ${LIB_SRC_FILES}
)
target_include_directories(traininglib_cpp PRIVATE ${eigen_SOURCE_DIR})
# SIMD!
target_compile_options(traininglib_cpp PRIVATE
  $<$<CXX_COMPILER_ID:Clang>: -O3 -msimd128 >
)


add_custom_target(wasm-js ALL
    COMMAND ${CMAKE_COMMAND} -E echo "Running emcc"
    BYPRODUCTS ${CMAKE_BINARY_DIR}/traininglib_cpp.js
    COMMAND emcc 
      libtraininglib_cpp.a
        -sEXPORT_ES6 
        -sMODULARIZE=1 
        -sSINGLE_FILE=1 
        -sEXPORT_NAME="traininglib_cpp" 
        -sEXPORTED_RUNTIME_METHODS=addFunction,Asyncify
        -sEXPORTED_FUNCTIONS=_skeletonize_wasm,_malloc,_free
        -sENVIRONMENT=web 
        -sALLOW_TABLE_GROWTH 
        -sALLOW_MEMORY_GROWTH=1
        -sWASM_BIGINT 
        -sASYNCIFY 
        -O3
        -o traininglib_cpp.js
    DEPENDS traininglib_cpp
)
endif()
