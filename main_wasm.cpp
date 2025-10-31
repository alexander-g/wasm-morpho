#include <Eigen/Dense>
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

#include "./src/morphology.hpp"



enum ERRORCODES {
    OK = 0,
    OUTPUTBUFFER_TOO_SMALL = -1,
};



extern "C" {

// TODO: exception handling
int skeletonize_wasm(
    // input
    bool*   binarymap,
    size_t  width,
    size_t  height,
    // output
    bool*   output,
    size_t  output_size
){
    if(output_size < width * height)
        return OUTPUTBUFFER_TOO_SMALL;

    const EigenMapToBinaryMap x_t(binarymap, height, width);
    EigenBinaryMap y_t = skeletonize(x_t);

    // copy result to output
    EigenMapToBinaryMap(output, height, width) = y_t;
    
    return OK;
}

}
