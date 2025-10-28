cmake_minimum_required(VERSION 3.12)
project(traininglib_cpp)


#suppress a warning regarding FetchContent
cmake_policy(SET CMP0135 NEW)


include(FetchContent)
set(PYBIND_URL https://github.com/pybind/pybind11/archive/refs/tags/v3.0.1.zip)
message("Maybe downloading ${PYBIND_URL}")
FetchContent_Declare(pybind 
    URL ${PYBIND_URL}
)
FetchContent_MakeAvailable(pybind)
#add_subdirectory(bui/pybind11)


set(EIGEN_URL https://gitlab.com/libeigen/eigen/-/archive/5.0.0/eigen-5.0.0.zip)
message("Maybe downloading ${EIGEN_URL}")
FetchContent_Declare(eigen 
    URL ${EIGEN_URL}
)
FetchContent_MakeAvailable(eigen)

pybind11_add_module(
    traininglib_cpp_ext 
        main.cpp 
        morphology.cpp
)
target_include_directories(traininglib_cpp_ext PRIVATE ${eigen_SOURCE_DIR})



