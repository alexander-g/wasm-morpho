#pragma once

#include <Eigen/Dense>
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "./morphology.hpp"





namespace py = pybind11;

typedef py::array_t<bool, py::array::c_style | py::array::forcecast> py_bool_array;


/** numpy 2-D bool array to eigen tensor  */
EigenMapToBinaryMap boolarray_to_eigen_tensor(const py_bool_array& x);



