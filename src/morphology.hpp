#pragma once


#include <expected>
#include <string>
#include <unordered_set>
#include <utility>

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>


typedef Eigen::Tensor<bool, 2, Eigen::RowMajor> EigenBinaryMap;
typedef Eigen::TensorMap<EigenBinaryMap>        EigenMapToBinaryMap;
typedef Eigen::Tensor<bool, 1, Eigen::RowMajor> EigenBinaryRow;
typedef Eigen::Tensor<int32_t, 1, Eigen::RowMajor> EigenIntRow;

/** Skeletonization as in https://doi.org/10.1145/357994.358023 */
EigenBinaryMap skeletonize(const EigenBinaryMap& input);



typedef struct Index2D {
    Eigen::Index i;
    Eigen::Index j;

    bool operator==(const Index2D& other) const { 
        return i == other.i && j == other.j; 
    }
} Index2D;
typedef std::vector<Index2D>   Indices2D;
typedef std::vector<Indices2D> ListOfIndices2D;



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




struct int32_pair_hash {
    size_t operator()(const std::pair<int32_t,int32_t>& p) const noexcept {
        return (static_cast<uint64_t>(uint32_t(p.first)) << 32) ^ uint32_t(p.second);
    }
};

typedef 
std::unordered_set<std::pair<int32_t, int32_t>, int32_pair_hash> Int32PairSet;


struct CCResultStreaming {
    ListOfIndices2D components;
};

/** Compute connected components by feeding one row at a time. */
struct StreamingConnectedComponents {

    /** Feed an additional row of the binary image */
    [[nodiscard]] std::expected<std::monostate, std::string> 
    push_image_row(const EigenBinaryRow&);
    
    /** Compute the final result */
    CCResultStreaming finalize();


    private:
    int row_number =  0;
    int row_width  = -1;
    EigenIntRow  previous_row;
    Int32PairSet equivalent_labels;
    ListOfIndices2D all_components = {{}};

};


// TODO: move to a util file

/** Extract a row from a binary image tensor. Shape [W] */
std::expected<EigenBinaryRow, std::string> 
row_slice(const EigenBinaryMap& t, Eigen::Index i);

