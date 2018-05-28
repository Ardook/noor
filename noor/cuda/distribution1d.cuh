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
#ifndef CUDADISTRIBUTION1D_CUH
#define CUDADISTRIBUTION1D_CUH
/* based on PBRT 1D distribution */

class CudaDistribution1D {
    float* _func;
    float* _cdf;
    float _funcInt;
    int _n;
    cudaTextureObject_t _func_texobj;
    cudaTextureObject_t _cdf_texobj;

    __host__
        void process( const float* func ) {
        std::vector<float> cdf( _n + 1 );
        cdf[0] = 0;
        // Compute integral of step_function at $x_i$
        for ( int i = 1; i <= _n; ++i ) {
            cdf[i] = cdf[i - 1] + func[i - 1] / _n;
        }

        // Transform step_function integral into CDF
        _funcInt = cdf[_n];
        if ( _funcInt == 0 ) {
            for ( int i = 1; i <= _n; ++i ) cdf[i] = float( i ) / float( _n );
        } else {
            for ( int i = 1; i <= _n; ++i ) cdf[i] /= _funcInt;
        }
        NOOR::memcopy( _func, (void*) func, _n * sizeof( float ) );
        NOOR::memcopy( _cdf, (void*) &cdf[0], ( _n + 1 ) * sizeof( float ) );
        NOOR::create_1d_texobj( &_func_texobj, _func, _n * sizeof( float ), NOOR::_float_channelDesc );
        NOOR::create_1d_texobj( &_cdf_texobj, _cdf, ( _n + 1 ) * sizeof( float ), NOOR::_float_channelDesc );
    }
public:
    // Distribution2D Public Methods
    CudaDistribution1D() = default;
    __host__
        CudaDistribution1D( const float* func, int n ) :
        _n( n ) {
        NOOR::malloc( (void**) &_func, _n * sizeof( float ) );
        NOOR::malloc( (void**) &_cdf, ( _n + 1 ) * sizeof( float ) );
        process( func );
    }
    __host__
        void free() {
        cudaDestroyTextureObject( _func_texobj );
        cudaDestroyTextureObject( _cdf_texobj );
        cudaFree( _func );
        cudaFree( _cdf );
    }
    __device__
        float getCdf( int i ) const {
        return  tex1Dfetch<float>( _cdf_texobj, i );
    }
    __device__
        float getFunc( int i ) const {
        return  tex1Dfetch<float>( _func_texobj, i );
    }
    __device__
        float getCount() const { return _n; }
    __device__
        float getFuncInt( int row ) const { return _funcInt; }
    __device__
        int findInterval( float x ) const {
        int m;
        int l = 0;
        int r = _n + 1;
        int interval = -1;
        float cdf;
        while ( l <= r ) {
            m = l + ( r - l ) / 2;
            cdf = getCdf( m );

            if ( cdf == x )
                return m;
            if ( cdf < x ) {
                l = m + 1;
                interval = m;
            } else
                r = m - 1;
        }
        return clamp( interval, 0, _n - 1 );
    }

    __device__
        int sampleDiscrete1D( float u, float *pdf = nullptr, float *uRemapped = nullptr ) const {
        // Find surrounding CDF segments and _offset_
        const int offset = findInterval( u );

        const float func = getFunc( offset );
        const float cdf = getCdf( offset );
        const float cdf_n = getCdf( offset + 1 );

        if ( pdf ) *pdf = ( _funcInt > 0 ) ? func / ( _funcInt * _n ) : 0;
        if ( uRemapped )
            *uRemapped = ( u - cdf ) / ( cdf_n - cdf );
        return offset;
    }

    __device__
        float sampleContinuous1D( float u, float *pdf, int *off = nullptr ) const {
        // Find surrounding CDF segments and _offset_
        int offset = findInterval( u );

        if ( off ) *off = offset;
        // Compute offset along CDF segment
        const float func = getFunc( offset );
        const float cdf = getCdf( offset );
        const float cdf_n = getCdf( offset + 1 );

        float du = u - cdf;
        if ( ( cdf_n - cdf ) > 0 ) {
            du /= ( cdf_n - cdf );
        }
        // Compute PDF for sampled offset
        if ( pdf ) *pdf = ( _funcInt > 0 ) ? func / _funcInt : 0;

        return ( offset + du ) / _n;
    }
};

struct CudaDistribution1DManager {
    CudaDistribution1D _distribution1d;
    CudaDistribution1DManager() = default;
    __host__
        CudaDistribution1DManager( const float* func, int n ) {
        _distribution1d = CudaDistribution1D( func, n );
    }
    __host__
        void free() {
        _distribution1d.free();
    }
};


#endif /* CUDADISTRIBUTION1D_CUH */