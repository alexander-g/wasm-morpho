#pragma once

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>


typedef Eigen::Tensor<bool, 2, Eigen::RowMajor> EigenBinaryMap;
typedef Eigen::TensorMap<EigenBinaryMap> EigenMapToBinaryMap;

/** Skeletonization as in https://doi.org/10.1145/357994.358023 */
EigenBinaryMap skeletonize(const EigenBinaryMap& input);



typedef struct Index2D {
    Eigen::Index i;
    Eigen::Index j;

    bool operator==(const Index2D& other) const { 
        return i == other.i && j == other.j; 
    }
} Index2D;
typedef std::vector<Index2D> Indices2D;



typedef struct DFS_Result {
    /** Indices/pixels in order of visit  */
    Indices2D visited;

    /** Predecessor pixels along a path. Values indexing into `visited`. */
    std::vector<int> predecessors;

    /** Terminal pixels, first/last in a path. Values indexing into `visited` */
    std::vector<int> leaves;
} DFS_Result;


/** Depth-first search*/
DFS_Result dfs(const EigenBinaryMap& input, const Index2D& start);



struct CCResult {
    Eigen::Tensor<int, 2, Eigen::RowMajor> labelmap;
    int n_labels;

    std::vector<DFS_Result> dfs_results;
};

CCResult connected_components(const EigenBinaryMap& input);

