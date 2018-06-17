/*
MIT License

Copyright (c) 2015-2018 Ardavan Kanani

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
#ifndef IMAGE_CUH
#define IMAGE_CUH

/*Based on PBRT image resize chapter. */
__forceinline__ __device__
int wrapCoord( int c, int res, cudaTextureAddressMode addressmode ) {
    int result;
    if ( addressmode == cudaAddressModeClamp ) {
        result = clamp( c, 0, res - 1 );
    } else {
        result = NOOR::mod( c, res );
    }
    assert( 0 <= result && result < res );
    return result;
}

// Texture Function Definitions
__forceinline__ __device__
float Lanczos( float x, float tau = 2.0f ) {
    x = fabsf( x );
    if ( x < 1e-5f ) return 1.f;
    if ( x > 1.f ) return 0.f;
    x *= NOOR_PI;
    const float s = sinf( x / tau ) / ( x / tau );
    const float lanczos = sinf( x ) / x;
    return s * lanczos;
}

struct ResampleWeight {
    NOOR::noor_float4 weight{ make_float4( 0 ) };
    int firstTexel{ 0 };
};


// Cuda resize image 
template<class T>
__global__
void resize_kernel(
    cudaSurfaceObject_t src_surfaceobj
    , cudaSurfaceObject_t dst_surfaceobj
    , const cudaExtent old_extent
    , const cudaExtent new_extent
    , int num_channels
    , cudaTextureAddressMode addressmode
) {
    extern __shared__ ResampleWeight wt[];
    const int new_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int new_y = blockIdx.y * blockDim.y + threadIdx.y;

    const float filterwidth = 2.f;
    const float xscale = (float) old_extent.width / (float) new_extent.width;
    const float yscale = (float) old_extent.height / (float) new_extent.height;

    const int center_x = ( new_x + .5f ) * xscale;
    const int center_y = ( new_y + .5f ) * yscale;

    ResampleWeight temp;
    if ( threadIdx.y == 0 ) {
        // Compute image resampling weights for i'th texel
        float sumWts = 0.f;
        temp.firstTexel = floorf( center_x - filterwidth + .5f );
        for ( int j = 0; j < 4; ++j ) {
            const float pos = temp.firstTexel + j + .5f;
            temp.weight[j] = Lanczos( ( pos - center_x ) / filterwidth );
            sumWts += temp.weight[j];
        }
        // Normalize filter weights for texel resampling
        temp.weight /= sumWts;
        wt[threadIdx.x] = temp;
    }
    __syncthreads();
    temp = wt[threadIdx.x];
    T color{ 0 };
    for ( int j = 0; j < 4; ++j ) {
        int old_x = temp.firstTexel + j;
        old_x = wrapCoord( old_x, old_extent.width, addressmode );
        if ( old_x >= 0 && old_x < old_extent.width )
            color += temp.weight[j] * ( surf2Dread<T>( src_surfaceobj, old_x * sizeof( T ), center_y ) );
    }
    if ( threadIdx.x == 0 ) {
        // Compute image resampling weights for i'th texel
        float sumWts = 0.f;
        temp.firstTexel = floorf( center_y - filterwidth + 0.5f );
        for ( int j = 0; j < 4; ++j ) {
            const float pos = temp.firstTexel + j + .5f;
            temp.weight[j] = Lanczos( ( pos - center_y ) / filterwidth );
            sumWts += temp.weight[j];
        }
        // Normalize filter weights for texel resampling
        temp.weight /= sumWts;
        wt[threadIdx.y] = temp;
    }
    __syncthreads();
    temp = wt[threadIdx.y];
    for ( int j = 0; j < 4; ++j ) {
        int old_y = temp.firstTexel + j;
        old_y = wrapCoord( old_y, old_extent.height, addressmode );
        if ( old_y >= 0 && old_y < old_extent.height )
            color += temp.weight[j] * ( surf2Dread<T>( src_surfaceobj, center_x * sizeof( T ), old_y ) );
    }
    surf2Dwrite<T>( color, dst_surfaceobj, new_x * sizeof( T ), new_y );
}

#endif /* IMAGE_CUH */
