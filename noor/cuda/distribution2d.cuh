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

/*
pbrt source code is Copyright(c) 1998-2016
Matt Pharr, Greg Humphreys, and Wenzel Jakob.

This file is part of pbrt.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

- Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

- Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/
#ifndef DISTRIBUTION2D_CUH
#define DISTRIBUTION2D_CUH
/* based on PBRT 2D distribution */

#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) \
    ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))
__global__
void update_func( cudaTextureObject_t texobj,
                  float* _func,
                  int width,
                  int height
) {
    const int row = blockIdx.y;
    const int col = threadIdx.x;
    float func = 0.0f;
    const float delta = 1.0f;
    const float n = powf( 2.0f*delta + 1.0f, 2.0f );
    float u, v;
    for ( int y = -delta; y <= delta; ++y ) {
        for ( int x = -delta; x <= delta; ++x ) {
            u = ( x + col ) / (float) width;
            v = ( y + row ) / (float) height;
            func += NOOR::rgb2Y( tex2D<float4>( texobj, u, v ) );
        }
    }
    _func[row * ( width + 1 ) + col] = func / n;
}

__global__
void prescan( float* func, float* cdf, int n ) {
    extern __shared__ float temp[];
    int tid = threadIdx.x;
    int bid = blockIdx.y;
    int offset = 1;

    int ai = tid;
    int bi = tid + ( n / 2 );

    int bankOffsetA = CONFLICT_FREE_OFFSET( ai );
    int bankOffsetB = CONFLICT_FREE_OFFSET( bi );
    temp[ai + bankOffsetA] = func[bid * ( n + 1 ) + ai] / n;
    temp[bi + bankOffsetB] = func[bid * ( n + 1 ) + bi] / n;
    // build sum in place up the tree
    for ( int d = n >> 1; d > 0; d >>= 1 ) {
        __syncthreads();
        if ( tid < d ) {
            int ai = offset*( 2 * tid + 1 ) - 1;
            int bi = offset*( 2 * tid + 2 ) - 1;
            ai += CONFLICT_FREE_OFFSET( ai );
            bi += CONFLICT_FREE_OFFSET( bi );
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }

    if ( tid == 0 ) { temp[n - 1 + CONFLICT_FREE_OFFSET( n - 1 )] = 0; }

    for ( int d = 1; d < n; d *= 2 ) // traverse down tree & build scan
    {
        offset >>= 1;
        __syncthreads();
        if ( tid < d ) {

            int ai = offset*( 2 * tid + 1 ) - 1;
            int bi = offset*( 2 * tid + 2 ) - 1;
            ai += CONFLICT_FREE_OFFSET( ai );
            bi += CONFLICT_FREE_OFFSET( bi );

            float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();
    cdf[bid * ( n + 1 ) + ai] = temp[ai + bankOffsetA];
    cdf[bid * ( n + 1 ) + bi] = temp[bi + bankOffsetB];
    __syncthreads();
    if ( tid == 0 ) {
        const float funcInt = cdf[bid * ( n + 1 ) + n - 1] +
            func[bid * ( n + 1 ) + n - 1] / n;
        cdf[bid * ( n + 1 ) + n] = funcInt;
        func[bid * ( n + 1 ) + n] = funcInt;
    }
    __syncthreads();
}

__global__
void process_cdf( float* _func, float* _cdf, int width, int height ) {
    const int row = blockIdx.y;
    const int col = threadIdx.x;
    const int tid = threadIdx.x;

    __shared__ float funcInt;
    if ( tid == 0 )
        funcInt = _cdf[row * ( width + 1 ) + width];
    __syncthreads();
    if ( funcInt == 0 ) {
        _cdf[row*( width + 1 ) + col + 1] = float( col + 1 ) / float( width );
    } else {
        _cdf[row*( width + 1 ) + col + 1] /= funcInt;
    }
    __syncthreads();
    if ( tid == 0 && height > 1 ) {
        _func[height * ( width + 1 ) + row] = funcInt;
    }
    __syncthreads();
}

template<class T>
class CudaDistribution2D {
    float* _func;
    float* _cdf;
    int _width, _height;
    cudaTextureObject_t _func_texobj;
    cudaTextureObject_t _cdf_texobj;
    size_t _size_bytes;
public:
    // Distribution2D Public Methods
    CudaDistribution2D() = default;
    __host__
        CudaDistribution2D( const T& tex ) :
        _width( clamp( NOOR::nearestPow2( tex.width() ), 32, 1024 ) ),
        _height( clamp( NOOR::nearestPow2( tex.height() ), 32, 1024 ) ),
        _size_bytes( ( _height * ( _width + 2 ) + 1 ) * sizeof( float ) ) {
        NOOR::malloc( (void**) &_func, _size_bytes );
        NOOR::memset( (void*) _func, 0, _size_bytes );

        NOOR::malloc( (void**) &_cdf, _size_bytes );
        NOOR::memset( (void*) _cdf, 0, _size_bytes );
        update( tex );
        NOOR::create_1d_texobj( &_func_texobj, _func, _size_bytes, NOOR::_float_channelDesc );
        NOOR::create_1d_texobj( &_cdf_texobj, _cdf, _size_bytes, NOOR::_float_channelDesc );
    }

    __host__
        void free() {
        cudaDestroyTextureObject( _func_texobj );
        cudaDestroyTextureObject( _cdf_texobj );
        cudaFree( _func );
        cudaFree( _cdf );
    }

    __host__
        void update( const T& tex ) {
        dim3 block( _width, 1, 1 );
        dim3 grid( 1, _height, 1 );
        update_func << <grid, block >> > ( tex.getReadTexObj(), _func, _width, _height );
        checkNoorErrors( cudaPeekAtLastError() );
        checkNoorErrors( cudaDeviceSynchronize() );
        block = dim3( _width / 2, 1, 1 );
        grid = dim3( 1, _height, 1 );
        size_t shmsize = ( _width + 1 ) * sizeof( float );
        prescan << <grid, block, shmsize >> > ( _func, _cdf, _width );
        checkNoorErrors( cudaPeekAtLastError() );
        checkNoorErrors( cudaDeviceSynchronize() );
        block = dim3( _width, 1, 1 );
        grid = dim3( 1, _height, 1 );
        process_cdf << <grid, block >> > ( _func, _cdf, _width, _height );
        checkNoorErrors( cudaPeekAtLastError() );
        checkNoorErrors( cudaDeviceSynchronize() );
        block = dim3( _height / 2, 1, 1 );
        grid = dim3( 1, 1, 1 );
        shmsize = ( _height + 1 ) * sizeof( float );
        prescan << <grid, block, shmsize >> > ( &_func[_height*( _width + 1 )], &_cdf[_height*( _width + 1 )], _height );
        checkNoorErrors( cudaPeekAtLastError() );
        checkNoorErrors( cudaDeviceSynchronize() );
        block = dim3( _height, 1, 1 );
        grid = dim3( 1, 1, 1 );
        process_cdf << <grid, block >> > ( &_func[_height*( _width + 1 )], &_cdf[_height*( _width + 1 )], _height, 1 );
        checkNoorErrors( cudaPeekAtLastError() );
        checkNoorErrors( cudaDeviceSynchronize() );
    }

private:
    __device__
        float getCdf( int row, int col ) const {
        const int index = row*( _width + 1 ) + col;
        return  tex1Dfetch<float>( _cdf_texobj, index );
    }
    __device__
        float getFunc( int row, int col ) const {
        const int index = row*( _width + 1 ) + col;
        return  tex1Dfetch<float>( _func_texobj, index );
    }
    __device__
        float getFuncInt( int row ) const {
        const int index = row*( _width + 1 ) + ( row == _height ? _height : _width );
        return  tex1Dfetch<float>( _func_texobj, index );
    }
    __device__
        float getMarginalFuncInt() const {
        return getFuncInt( _height );
    }
    __device__
        int findInterval( int row, int n, float x ) const {
        int m;
        int l = 0;
        int r = n;
        int interval = -1;
        while ( l <= r ) {
            m = l + ( r - l ) / 2;
            if ( getCdf( row, m ) <= x ) {
                l = m + 1;
                interval = m;
            } else {
                r = m - 1;
            }
        }
        return clamp( interval, 0, n-1 );
    }
    __device__
        float sampleContinuous1D( int row, int n, float u, float& pdf, int *off = nullptr ) const {
        // Find surrounding CDF segments and offset
        int offset = findInterval( row, n, u );

        if ( off ) *off = offset;
        // Compute offset along CDF segment
        const float func = getFunc( row, offset );
        const float cdf = getCdf( row, offset );
        const float cdf_n = getCdf( row, offset + 1 );
        const float funcInt = getFuncInt( row );

        float du = u - cdf;
        if ( ( cdf_n - cdf ) > 0 ) {
            du /= ( cdf_n - cdf );
        }
        // Compute PDF for sampled offset
        pdf = ( funcInt > 0 ) ? func / funcInt : 0;
        return ( offset + du ) / n;
    }
public:
    __device__
        int sampleDiscrete1D( float u, float& pdf, float *uRemapped = nullptr ) const {
        // Find surrounding CDF segments and offset

        const int offset = findInterval( 0, _width, u );

        const float func = getFunc( 0, offset );
        const float cdf = getCdf( 0, offset );
        const float cdf_n = getCdf( 0, offset + 1 );
        const float funcInt = getFuncInt( 0 );

        pdf = ( funcInt > 0 ) ? func / ( funcInt * _width ) : 0;
        if ( uRemapped ) *uRemapped = ( u - cdf ) / ( cdf_n - cdf );
        return offset;
    }

    __device__
        float sampleContinuous1D( float u, float& pdf, int *off = nullptr ) const {
        return sampleContinuous1D( 0, _width, u, pdf, off );
    }
    __device__
        float Pdf( int index ) const {
        return getFunc( 0, index ) / ( getFuncInt( 0 ) * _width );
    }
    __device__
        float Pdf( const float2& p ) const {
        const int iu = clamp( int( p.x * _width ), 0, _width - 1 );
        const int iv = clamp( int( p.y * _height ), 0, _height - 1 );
        return getFunc( iv, iu ) / getFuncInt( _height );
    }
    __device__
        float2 sampleContinuous2D( const float2 &u, float& pdf ) const {
        float2 pdfs;
        int v;
        // marginal distribution
        const float y = sampleContinuous1D( _height, _height + 1, u.y, pdfs.y, &v );
        // conditional distribution
        const float x = sampleContinuous1D( v, _width + 1, u.x, pdfs.x );
        if (x>=1 || y>=1)
            printf( "x %f y %f\n", x, y );
        pdf = pdfs.x * pdfs.y;
        return make_float2( x, y );
    }
};

template<class T>
class CudaDistribution2DManager {
public:
    CudaDistribution2D<T> _distribution2d;
    CudaDistribution2DManager() = default;
    __host__
        CudaDistribution2DManager( const T& tex ) :
        _distribution2d( tex ) {}

    __host__
        void update( const T& tex ) {
        _distribution2d.update( tex );
    }
    __host__
        void free() {
        _distribution2d.free();
    }
};

#endif /* DISTRIBUTION2D_CUH */