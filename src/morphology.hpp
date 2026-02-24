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

// run-length encoding (RLE) to save memory
struct RLERun {
    uint32_t row;   // row index
    uint32_t start; // start column (inclusive)
    uint32_t len;   // length of run
};

// a single connected component in RLE format
typedef std::vector<RLERun> RLEComponent;
typedef std::vector<RLEComponent> ListOfRLEComponents;

struct CCResultStreaming {
    ListOfRLEComponents components;
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
    ListOfRLEComponents all_components = {{}};
};


/** Total number of pixels in a component */
uint64_t rle_component_size(const RLEComponent& component);

/** Convert a component encoded in RLE format into dense pixel coordinates */
Indices2D rle_component_to_dense(const RLEComponent& comp);

/** Convert a component encoded in RLE format into a dense hollow contour. */
Indices2D rle_component_to_contour(const RLEComponent& component);

/** Convert multiple components encoded in RLE format into dense contours. */
ListOfIndices2D rle_components_to_contour(const ListOfRLEComponents& components);

/** Convert dense pixel coordinates into a component encoded in RLE format */
RLEComponent dense_to_rle_component(
    const Indices2D& dense,
    bool already_sorted = false
);

ListOfRLEComponents dense_to_rle_components(
    const ListOfIndices2D& dense, 
    bool already_sorted = false
);

/** Make sure there is only one run per row. */
void coalesce_rle_runs(RLEComponent& runs);


// TODO: move to a util file

/** Extract a row from a binary image tensor. Shape [W] */
std::expected<EigenBinaryRow, std::string> 
row_slice(const EigenBinaryMap& t, Eigen::Index i);

