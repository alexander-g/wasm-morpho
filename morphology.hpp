#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>


typedef Eigen::Tensor<bool, 2, Eigen::RowMajor> EigenBinaryMap;
typedef Eigen::TensorMap<EigenBinaryMap> EigenMapToBinaryMap;

/** Skeletonization as in https://doi.org/10.1145/357994.358023 */
EigenBinaryMap skeletonize(const EigenMapToBinaryMap input);

