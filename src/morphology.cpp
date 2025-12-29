#include <iostream>
#include <ranges>
#include <unordered_set>

#include "./morphology.hpp"


typedef Eigen::Array<bool,8,1> Array8b;



static Array8b get_neighbors_of_pixel(
    const EigenBinaryMap& x, 
    Eigen::Index i, 
    Eigen::Index j
) {
    Array8b P;
    P <<
        x(i-1,j  ),
        x(i-1,j+1),
        x(i  ,j+1),
        x(i+1,j+1),
        x(i+1,j  ),
        x(i+1,j-1),
        x(i  ,j-1),
        x(i-1,j-1);
    return P;
}

/** Compute if the number of neighbors of a pixel is >=2 and <=6 */
static bool _skeletonize_condition_a(const Array8b&  P) {
    const int sum = (P.cast<int>()).sum();
    return sum >= 2 && sum <= 6;
}

/** Compute the number of 0-1 transitions in P2...P9 */
static bool _skeletonize_condition_b(const Array8b&  P) {
    const int T_23 = !P(0) && P(1);
    const int T_34 = !P(1) && P(2);
    const int T_45 = !P(2) && P(3);
    const int T_56 = !P(3) && P(4);
    const int T_67 = !P(4) && P(5);
    const int T_78 = !P(5) && P(6);
    const int T_89 = !P(6) && P(7);
    const int T_92 = !P(7) && P(0);  // not in paper?
    const int T_sum = T_23 + T_34 + T_45 + T_56 + T_67 + T_78 + T_89 + T_92;
    return T_sum == 1;
}

/** Condition c): P2 * P4 * P6 = 0 */
const bool _skeletonize_condition_c0(const Array8b& P) {
    return (P(0) && P(2) && P(4)) == 0;
}

/** Condition d): P4 * P6 * P8 == 0 */
const bool _skeletonize_condition_d0(const Array8b& P) {
    return (P(2) && P(4) && P(6)) == 0;
}

/** Condition c'): P2 * P4 * P8 = 0 */
const bool _skeletonize_condition_c1(const Array8b& P) {
    return (P(0) && P(2) && P(6)) == 0;
}

/** Condition d'): P2 * P6 * P8 == 0 */
const bool _skeletonize_condition_d1(const Array8b& P) {
    return (P(0) && P(4) && P(6)) == 0;
}



static EigenBinaryMap _skeletonize_subiteration(
    const EigenBinaryMap& x,
    const Indices2D& nonzeroindices,
    bool  subiter_2
) {
    EigenBinaryMap mask(x.dimensions());
    mask.setZero();

    for( const Index2D& index: nonzeroindices )
        if( x(index.i, index.j) ){
            const Array8b P = get_neighbors_of_pixel(x, index.i, index.j);
            const bool a = _skeletonize_condition_a(P);
            const bool b = _skeletonize_condition_b(P);
            const bool c = 
                !subiter_2
                    ? _skeletonize_condition_c0(P) 
                    : _skeletonize_condition_c1(P);
            const bool d = 
                !subiter_2
                    ? _skeletonize_condition_d0(P) 
                    : _skeletonize_condition_d1(P);

            mask(index.i, index.j) = (a && b && c && d);
        }

    return mask;
}



static auto remove_padding(EigenBinaryMap& x) {
    Eigen::array<Eigen::Index,2> offsets{1, 1};
    Eigen::array<Eigen::Index,2> new_dims{x.dimension(0)-2, x.dimension(1)-2};
    return x.slice(offsets, new_dims);
}

static EigenBinaryMap copy_and_pad(const EigenBinaryMap& x) {
    Eigen::array<std::pair<Eigen::Index,Eigen::Index>, 2> paddings{
        std::make_pair(1,1), 
        std::make_pair(1,1)
    };
    return x.pad(paddings).eval();
}


/** I cannot get `bool ok = mask.any()` to work ...help. */
bool _any(const EigenBinaryMap& mask) {
    for (Eigen::Index i = 0; i < mask.dimension(0); i++)
        for (Eigen::Index j = 0; j < mask.dimension(1); j++)
            if( mask(i,j) )
                return true;
    return false;
}

Indices2D argwhere(const EigenBinaryMap& x) {
    Indices2D result;
    for (Eigen::Index i = 1; i < x.dimension(0)-1; i++)
        for (Eigen::Index j = 1; j < x.dimension(1)-1; j++)
            if( x(i,j) )
                result.push_back({i,j});
    return result;
}


EigenBinaryMap skeletonize(const EigenBinaryMap& input) {
    const auto dim0 = input.dimension(0);
    const auto dim1 = input.dimension(1);

    EigenBinaryMap padded = copy_and_pad(input);
    const Indices2D nonzeroindices = argwhere(padded);

    bool finished = false;
    while(!finished) {
        const EigenBinaryMap mask1 = 
            _skeletonize_subiteration(padded, nonzeroindices, false);
        padded = padded * !mask1;

        const EigenBinaryMap mask2 = 
            _skeletonize_subiteration(padded, nonzeroindices, true);
        padded = padded * !mask2;

        finished = !_any(mask1) && !_any(mask2);
    }

    const auto output = remove_padding(padded);
    return output;
}



Indices2D valid_neighbor_indices(
    const Index2D& p, 
    const EigenBinaryMap& image,
    bool  _8way
) {
    Indices2D result;
    result.reserve(8);

    const auto dim0 = image.dimension(0);
    const auto dim1 = image.dimension(1);

    if(p.i > 0)
        result.push_back( {p.i-1, p.j} );
    if(p.j > 0)
        result.push_back( {p.i,   p.j-1} );
    if(p.i < dim0 -1)
        result.push_back( {p.i+1, p.j} );
    if(p.j < dim1 -1)
        result.push_back( {p.i,   p.j+1} );
    

    if(_8way) {
        if(p.i > 0       && p.j > 0)
            result.push_back( {p.i-1, p.j-1} );
        if(p.i > 0       && p.j < dim1 -1)
            result.push_back( {p.i-1, p.j+1} );
        if(p.i < dim0 -1 && p.j > 0)
            result.push_back( {p.i+1, p.j-1} );
        if(p.i < dim0 -1 && p.j < dim1 -1)
            result.push_back( {p.i+1, p.j+1} );
    }
    return result;
}

bool is_in(const Index2D& p, const Indices2D& indices) {
    return std::find(indices.begin(), indices.end(), p) != indices.end();
}

uint64_t ravel_index2d(const Index2D& p, uint64_t imagewidth) {
    return p.i * imagewidth + p.j;
}





/** Depth-first search*/
DFS_Result dfs(
    const EigenBinaryMap& input,
    const Index2D& start
) {
    const uint64_t width = input.dimension(1);

    // stack: next index/pixel to visit and its predecessor
    std::vector<std::pair<Index2D, int>> stack;
    stack.push_back( {start, -1} );

    DFS_Result result;
    result.leaves.push_back(0);

    // set of visited indices for faster lookup
    std::unordered_set<uint64_t> visited_set;
    // or better a binary map of the same size as input?

    while(!stack.empty()) {
        const auto next = std::move(stack.back());
        stack.pop_back();

        const Index2D& p = next.first;
        const int predecessor = next.second;

        if( visited_set.contains( ravel_index2d(p, width) ) )
            continue;
        
        const int p_i = result.visited.size();
        result.visited.push_back(p);
        result.predecessors.push_back(predecessor);
        visited_set.insert( ravel_index2d(p, width) );


        bool no_unvisited_neighbors = true;
        const Indices2D neighbors = valid_neighbor_indices(p, input, true);

        // reverse because horizontal and vertical neighbors have priority
        for(const Index2D& neighbor: std::views::reverse(neighbors)) {
            if( visited_set.contains( ravel_index2d(neighbor, width) ) )
                continue;

            if( input(neighbor.i, neighbor.j) ){
                stack.push_back( {neighbor, p_i} );
                no_unvisited_neighbors = false;
            }
        }

        if(no_unvisited_neighbors)
            result.leaves.push_back(p_i);
    }
    return result;
}



template<typename T>
void scatter(
    Eigen::Tensor<T, 2, Eigen::RowMajor>& x, 
    const Indices2D& indices, 
    T value
) {
    for(const Index2D& p: indices)
        x(p.i, p.j) = value;
}


CCResult connected_components(const EigenBinaryMap& input) {
    const int H = input.dimension(0), W = input.dimension(1);
    Eigen::Tensor<int, 2, Eigen::RowMajor> labelmap(H, W);
    labelmap.setZero();
    int nextlabel = 1;
    std::vector<DFS_Result> dfs_results;

    for (Eigen::Index i = 0; i < H; i++)
        for (Eigen::Index j = 0; j < W; j++) {
            if(input(i,j) == 0 || labelmap(i,j) != 0)
                continue;
            
            const DFS_Result dfs_result = dfs(input, {i,j});
            scatter(labelmap, dfs_result.visited, nextlabel);
            dfs_results.push_back(dfs_result);
            nextlabel++;
        }

    return {labelmap, nextlabel-1, dfs_results};
}



std::expected<std::monostate, std::string> 
StreamingConnectedComponents::push_image_row(const EigenBinaryRow& row) {
    const int row_width = (int)row.dimension(0);
    if(this->row_width == -1)
        this->row_width = row_width;
    else if(this->row_width != row_width)
        return std::unexpected(
            "Received a row with a different number of pixels than before." 
        );
    
    Eigen::Tensor<int32_t, 1, Eigen::RowMajor>  current_row(row_width);
    current_row.setZero();
    const bool is_first_row = (this->row_number == 0);

    for(int i = 0; i < row_width; i++) {
        if(row(i)) {
            const bool is_first_col = (i == 0);
            const bool is_last_col  = (i == row_width-1);

            // labels left, top, topright, topleft
            const int32_t label_l  = (is_first_col)? 0 : current_row(i-1);
            const int32_t label_t  = (is_first_row)? 0 : this->previous_row(i);
            const int32_t label_tr = 
                (is_first_row || is_last_col)? 0 : this->previous_row(i+1);
            const int32_t label_tl = 
                (is_first_row || is_first_col)? 0 : this->previous_row(i-1);
            
            const int32_t this_label = 
                label_tr? label_tr :
                label_t?  label_t  :
                label_tl? label_tl :
                label_l?  label_l  :
                          this->all_components.size();
            
            if(label_l && label_l != this_label)
                this->equivalent_labels.insert({label_l, this_label});

            if(this_label >= this->all_components.size())
                this->all_components.push_back({});

            const Index2D coordinate = {this->row_number, i};
            this->all_components[ this_label ].push_back(coordinate);
            current_row(i) = this_label;
        }
    }
    this->row_number++;
    this->previous_row = current_row;

    return std::monostate{};
}



/** Concatenate two ranges */
template<std::ranges::input_range R1, std::ranges::input_range R2>
auto concat_copy(const R1& a, const R2& b) {
    using T = std::ranges::range_value_t<R1>;
    std::vector<T> out;
    out.reserve(std::ranges::size(a) + std::ranges::size(b));
    out.insert(out.end(), std::ranges::begin(a), std::ranges::end(a));
    out.insert(out.end(), std::ranges::begin(b), std::ranges::end(b));
    return out;
}


/** Build adjacency and return connected components */
std::vector<std::vector<int32_t>> connected_clusters(const Int32PairSet& edges){
    // adjacency list
    std::unordered_map<int32_t, std::vector<int32_t>> adj;
    adj.reserve(edges.size()*2);

    for(const auto& e: edges) {
        const int32_t u = e.first;
        const int32_t v = e.second;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    std::vector<std::vector<int32_t>> components;
    components.reserve(adj.size());

    std::unordered_set<int32_t> visited;
    visited.reserve(adj.size());

    std::vector<int32_t> stack;
    for (auto const& kv : adj) {
        const int32_t start = kv.first;
        if (visited.contains(start)) 
            continue;

        stack.clear();
        stack.push_back(start);
        visited.insert(start);

        std::vector<int32_t> component;
        while(!stack.empty()) {
            const int32_t node = stack.back(); 
            stack.pop_back();

            component.push_back(node);
            for(const int32_t neighbor: adj[node]) {
                if(!visited.contains(neighbor)) {
                    visited.insert(neighbor);
                    stack.push_back(neighbor);
                }
            }
        }
        components.push_back(std::move(component));
    }
    return components;
}


CCResultStreaming StreamingConnectedComponents::finalize() {
    if(this->all_components.size() == 0)
        return {.components = {}};
    
    std::vector<std::vector<Index2D>> output;


    const auto clusters = connected_clusters(this->equivalent_labels);
    for(const auto& cluster: clusters) {
        std::vector<Index2D> merged_component;
        for(const int32_t index: cluster){
            merged_component = 
                concat_copy(merged_component, this->all_components[index]);
            this->all_components[index].resize(0);
        }
        output.push_back(merged_component);
    }

    // remaining components not in equivalent_labels
    for(int i = 0; i < this->all_components.size(); i++ ) {
        if(this->all_components[i].empty())
            continue;

        output.push_back(this->all_components[i]);
        this->all_components[i].resize(0);
    }

    return {.components = output};;
}





/** Extract a row from a binary image tensor. Shape [W] */
std::expected<EigenBinaryRow, std::string> 
row_slice(const EigenBinaryMap& t, Eigen::Index i) {
    const Eigen::Index H = t.dimension(0);
    const Eigen::Index W = t.dimension(1);

    if(i < 0 || i >= H)
        return std::unexpected("Row slice index out of range");

    const Eigen::array<Eigen::Index, 2> offsets = { i, 0 };
    const Eigen::array<Eigen::Index, 2> extents = { 1, W };
    return t.slice(offsets, extents)
            .reshape(Eigen::array<Eigen::Index, 1>{ W });
}


