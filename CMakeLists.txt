cmake_minimum_required(VERSION 3.12)
project(wasm_morpho)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_VERBOSE_MAKEFILE on)


set(LIB_SRC_FILES
    src/morphology.cpp
)


#suppress a warning regarding FetchContent
cmake_policy(SET CMP0135 NEW)
include(FetchContent)


if(NOT DEFINED eigen_SOURCE_DIR)
  set(
    EIGEN_URL 
    https://gitlab.com/libeigen/eigen/-/archive/5.0.0/eigen-5.0.0.zip
  )
  message("Maybe downloading ${EIGEN_URL}")
  FetchContent_Declare(eigen 
      URL ${EIGEN_URL}
  )
  FetchContent_MakeAvailable(eigen)
  # note to self: making eigen_SOURCE_DIR available to upstream
  set(eigen_SOURCE_DIR "${eigen_SOURCE_DIR}" CACHE INTERNAL "")
endif()


add_library(morpho_core STATIC 
    ${LIB_SRC_FILES}
)
target_include_directories(morpho_core PUBLIC ${eigen_SOURCE_DIR})






if(DEFINED CMAKE_TOOLCHAIN_FILE AND CMAKE_TOOLCHAIN_FILE MATCHES "emscripten" OR DEFINED ENV{EMSCRIPTEN})
  set(RUNNING_EMSCRIPTEN TRUE)
else()
  set(RUNNING_EMSCRIPTEN FALSE)
endif()





if(NOT RUNNING_EMSCRIPTEN)


set(PYBIND_URL https://github.com/pybind/pybind11/archive/refs/tags/v3.0.1.zip)
message("Maybe downloading ${PYBIND_URL}")
FetchContent_Declare(pybind 
    URL ${PYBIND_URL}
)
FetchContent_MakeAvailable(pybind)


pybind11_add_module(morpho_py main_pybind.cpp ${LIB_SRC_FILES})
#target_include_directories(morpho_py PRIVATE ${eigen_SOURCE_DIR})
target_link_libraries(morpho_py PRIVATE morpho_core)
target_compile_options(morpho_py PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang,Intel>: -O3 >
  $<$<CXX_COMPILER_ID:MSVC>: /O2 /GL>
)

endif()




if(RUNNING_EMSCRIPTEN)

add_library(wasm_morpho STATIC 
    main_wasm.cpp
    ${LIB_SRC_FILES}
)
#target_include_directories(wasm_morpho PRIVATE ${eigen_SOURCE_DIR})
target_link_libraries(wasm_morpho PRIVATE morpho_core)
# SIMD!
target_compile_options(wasm_morpho PRIVATE
  $<$<CXX_COMPILER_ID:Clang>: -O3 -msimd128 >
)


add_custom_target(wasm-js ALL
    COMMAND ${CMAKE_COMMAND} -E echo "Running emcc"
    BYPRODUCTS ${CMAKE_BINARY_DIR}/wasm_morpho.js
    COMMAND emcc 
      libwasm_morpho.a
        -sEXPORT_ES6 
        -sMODULARIZE=1 
        -sSINGLE_FILE=1 
        -sEXPORT_NAME="wasm_morpho" 
        -sEXPORTED_RUNTIME_METHODS=addFunction,Asyncify
        -sEXPORTED_FUNCTIONS=_skeletonize_wasm,_malloc,_free
        -sENVIRONMENT=web 
        -sALLOW_TABLE_GROWTH 
        -sALLOW_MEMORY_GROWTH=1
        -sWASM_BIGINT 
        -sASYNCIFY 
        -O3
        -o wasm_morpho.js
    DEPENDS wasm_morpho
)
endif()
