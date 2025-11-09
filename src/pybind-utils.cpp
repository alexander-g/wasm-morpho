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



py::array_t<int64_t> indices2d_to_array(const Indices2D &x) {
    const size_t n = (size_t)x.size();
    py::array_t<int64_t> y({n, (size_t)2});

    if (n == 0)
        return y;

    // future-proofing
    static_assert(
        sizeof(Index2D) == 2 * sizeof(int64_t), 
        "Expecting Index2D to be tightly packed"
    );
    static_assert(
        std::is_trivial<Index2D>::value, 
        "Expecting Index2D to be trivially copyable"
    );

    // copy raw bytes: vector data is contiguous
    std::memcpy(y.mutable_data(), x.data(), n * sizeof(Index2D));
    return y;
}




