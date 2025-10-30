#include <iostream>

#include "./morphology.hpp"


typedef Eigen::Array<bool,8,1> Array8b;

typedef struct Index2D {
    Eigen::Index i;
    Eigen::Index j;
} Index2D;
typedef std::vector<Index2D> Indices2D;



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


EigenBinaryMap skeletonize(const EigenMapToBinaryMap input) {
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
