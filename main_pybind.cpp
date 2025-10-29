#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

#include "./morphology.hpp"



namespace py = pybind11;



py::array_t<bool> skeletonize_pybind(py::array_t<bool> x) {
    const auto r = x.unchecked<2>();

    const Eigen::Index d0 = static_cast<Eigen::Index>(x.shape(0));
    const Eigen::Index d1 = static_cast<Eigen::Index>(x.shape(1));


    py::buffer_info xbufinfo = x.request();

    const EigenMapToBinaryMap x_t((bool*)xbufinfo.ptr, d0, d1);
        
    EigenBinaryMap y_t = skeletonize(x_t);

    // copy
    py::array_t<bool> y({(size_t)d0, (size_t)d1});
    py::buffer_info ybufinfo = y.request();
    EigenMapToBinaryMap((bool*)ybufinfo.ptr, d0, d1) = y_t;
    return y;
}




PYBIND11_MODULE(traininglib_cpp_ext, m) {
    m.doc() = "traininglib c++ extensions";

    m.def("skeletonize", skeletonize_pybind, py::arg().noconvert());
}

