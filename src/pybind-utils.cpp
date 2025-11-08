#include "./pybind-utils.hpp"





EigenMapToBinaryMap boolarray_to_eigen_tensor(const py_bool_array& x) {
    if (x.ndim() != 2)
        throw std::runtime_error("input must be a 2D boolean array");
    const ssize_t d0 = x.shape(0);
    const ssize_t d1 = x.shape(1);

    py::buffer_info xbufinfo = x.request();
    const EigenMapToBinaryMap t((bool*)xbufinfo.ptr, d0, d1);
    return t;
}







