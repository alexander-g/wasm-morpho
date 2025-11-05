#include <utility>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

#include "./src/morphology.hpp"



namespace py = pybind11;

typedef py::array_t<bool, py::array::c_style | py::array::forcecast> py_bool_array;


static EigenMapToBinaryMap array_to_eigen_tensor(const py_bool_array& x) {
    if (x.ndim() != 2)
        throw std::runtime_error("input must be a 2D boolean array");
    const ssize_t d0 = x.shape(0);
    const ssize_t d1 = x.shape(1);

    py::buffer_info xbufinfo = x.request();
    const EigenMapToBinaryMap t((bool*)xbufinfo.ptr, d0, d1);
    return t;
}

static py::array_t<int> int_vector_to_array(const std::vector<int>& x) {
    py::array_t<int> y(x.size());
    std::memcpy(y.mutable_data(), x.data(), x.size() * sizeof(int));
    return y;
}

static py::array_t<int64_t> indices2d_to_array(const Indices2D &x) {
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



py::array_t<bool> skeletonize_pybind(py_bool_array& x) {
    py::buffer_info xbufinfo = x.request();
    const EigenMapToBinaryMap x_t(array_to_eigen_tensor(x));
    
    EigenBinaryMap y_t = skeletonize(x_t);
    
    const Eigen::Index d0 = static_cast<Eigen::Index>(x.shape(0));
    const Eigen::Index d1 = static_cast<Eigen::Index>(x.shape(1));

    // copy output into new numpy array
    py::array_t<bool> y({(size_t)d0, (size_t)d1});
    py::buffer_info ybufinfo = y.request();
    EigenMapToBinaryMap((bool*)ybufinfo.ptr, d0, d1) = y_t;
    return y;
}

static py::dict dfs_py(
    const py_bool_array&     mask_py,
    const std::pair<int,int> start_py
) {
    const EigenBinaryMap mask = array_to_eigen_tensor(mask_py);
    const Index2D start = {start_py.first, start_py.second};

    const DFS_Result res = dfs(mask, start);

    py::array_t<int64_t> visited_py = indices2d_to_array(res.visited);
    py::array_t<int> preds_py  = int_vector_to_array(res.predecessors);
    py::array_t<int> leaves_py = int_vector_to_array(res.leaves);

    py::dict out;
    out["visited"]      = visited_py;
    out["predecessors"] = preds_py;
    out["leaves"]       = leaves_py;
    return out;
}

static py::array_t<int> concom_py(const py_bool_array& mask_py) {
    const EigenBinaryMap mask = array_to_eigen_tensor(mask_py);
    const CCResult res = connected_components(mask);

    const Eigen::Index d0 = res.labelmap.dimension(0);
    const Eigen::Index d1 = res.labelmap.dimension(1);

    py::array_t<int> y({ d0, d1 });
    py::buffer_info ybufinfo = y.request();
    Eigen::TensorMap<Eigen::Tensor<int, 2, Eigen::RowMajor>>((int*)ybufinfo.ptr, d0, d1) = res.labelmap;
    return y;
}


PYBIND11_MODULE(morpho_pyext, m) {
    m.doc() = "some morphology functions";

    m.def("skeletonize", skeletonize_pybind, py::arg().noconvert());
    m.def(
        "dfs", 
        &dfs_py, 
        py::arg("mask").noconvert(), 
        py::arg("start"),
        "Depth-first search on a 2D binary image"
    );
    m.def("connected_components", &concom_py, py::arg("mask").noconvert());
}

