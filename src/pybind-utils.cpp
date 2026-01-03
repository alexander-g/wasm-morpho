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




/** List of np arrays [N,3] (row, start, len) to native RLE */
ListOfRLEComponents py_list_to_rle_components(const py::list& components){
    ListOfRLEComponents output;
    output.reserve(components.size());

    for(const py::handle &component_handle: components) {
        const py::array array = py::array::ensure(component_handle);
        if (!array) 
            throw std::runtime_error("not an array");
        if (array.ndim() != 2 || array.shape(1) != 3)
            throw std::runtime_error("array must have shape [N,3]");
        
        const py::array_t<uint32_t, py::array::c_style|py::array::forcecast> a32 = 
            array;
        const size_t nrows = a32.shape(0);
        
        RLEComponent component;
        component.reserve(nrows);

        const auto a32data = a32.unchecked<2>();
        for(int i = 0; i < nrows; i++) {
            const uint32_t row   = a32data(i, 0);
            const uint32_t start = a32data(i, 1);
            const uint32_t len   = a32data(i, 2);
            
            component.push_back( RLERun{
                .row   = row,
                .start = start,
                .len   = len
            } );
        }

        output.push_back(component);
    }
    return output;
}


/**  Native RLE to list of np arrays [N,3] (row, start, len) */
py::list rle_components_to_py_list(const ListOfRLEComponents& components) {
    py::list output;

    for(const RLEComponent& component: components) {
        const size_t nrows = component.size();
        // (N,3) uint32 array
        py::array_t<uint32_t, py::array::c_style> array({nrows, (size_t)3});

        auto array_unchecked = array.mutable_unchecked<2>();
        for(int i = 0; i < nrows; i++) {
            const RLERun &run = component[i];
            array_unchecked(i, 0) = run.row;
            array_unchecked(i, 1) = run.start;
            array_unchecked(i, 2) = run.len;
        }
        output.append(array);
    }
    return output;
}

