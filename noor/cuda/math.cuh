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
**/
#ifndef CUDAMATH_CUH
#define CUDAMATH_CUH
namespace NOOR {
    struct noor_int3 : public int3 {
        __device__ 
            int& operator[]( int i ) {
            assert( 0 <= i && i < 3 );
            int* start = &x;
            return *( start + i );
        }

        __device__ 
            int operator[]( int i ) const {
            assert( 0 <= i && i < 3 );
            if ( i == 0 ) return x;
            if ( i == 1 ) return y;
            else return z;
        }
    };
    struct noor_uint4 : public uint4 {
        noor_uint4() = default;
        __device__ 
            explicit noor_uint4( const uint4& u ) : uint4( u ) {}
        __device__ 
            uint operator[]( int i ) const {
            assert( 0 <= i && i < 4 );
            if ( i == 0 ) return x;
            if ( i == 1 ) return y;
            if ( i == 2 ) return z;
            else return w;
        }
    };
    struct noor_float2 : public float2 {
        noor_float2() = default;
        __device__
            explicit noor_float2( const float2& f ) : float2( f ) {}
        __device__ 
            const float& operator[]( int i ) const {
            assert( 0 <= i && i < 2 );
            const float* start = &x;
            return *( start + i );
        }
        __device__ 
            float& operator[]( int i ) {
            assert( 0 <= i && i < 2 );
            float* start = &x;
            return *( start + i );
        }
    };

    struct noor_float3 : public float3 {
        noor_float3() = default;
        __device__ 
            explicit noor_float3( const float3& f ) : float3( f ) {}
        __device__ 
            const float& operator[]( int i )const {
            assert( 0 <= i && i < 3 );
            const float* start = &x;
            return *( start + i );
        }
        __device__ 
            float& operator[]( int i ) {
            assert( 0 <= i && i < 3 );
            float* start = &x;
            return *( start + i );
        }
    };
    struct noor_float4 : public float4 {
        noor_float4() = default;
        __device__ 
            explicit noor_float4( const float4& f ) : float4( f ) {}
        __device__ 
            float& operator[]( int i ) {
            assert( 0 <= i && i < 4 );
            float* start = &x;
            return *( start + i );
        }
        __device__ 
            const float& operator[]( int i ) const {
            assert( 0 <= i && i < 4 );
            const float* start = &x;
            return *( start + i );
        }
    };
    class Matrix2x2 {
        noor_float2 _row0;
        noor_float2 _row1;
    public:
        Matrix2x2() = default;

        __device__ 
            Matrix2x2( const ::float2& row0, const ::float2& row1 ) :
            _row0( row0 )
            , _row1( row1 ) {}

        __device__ 
            const noor_float2& operator[]( int i ) const {
            if ( i == 0 ) return _row0;
            return _row1;
        }

        __device__ 
            float determinant() const {
            return  _row0.x * _row1.y - _row1.x*_row0.y;
        }

        __device__ 
            void transpose2x2() {
            const Matrix2x2& m = *this;
            const noor_float2 row0{ make_float2( m[0][0], m[1][0] ) };
            const noor_float2 row1{ make_float2( m[0][1], m[1][1] ) };
            _row0 = row0;
            _row1 = row1;
        }

        __device__ 
            void inverse() {
            const float invdet = 1.0f / determinant();
            const Matrix2x2& m = *this;
            const noor_float2 row0{ make_float2( m[1][1] * invdet, -m[0][1] * invdet ) };
            const noor_float2 row1{ make_float2( -m[1][0] * invdet,  m[0][0] * invdet ) };
            _row0 = row0;
            _row1 = row1;
        }
    };

    class Matrix3x3 {
        noor_float3 _row0;
        noor_float3 _row1;
        noor_float3 _row2;
    public:
        Matrix3x3() = default;

        __device__ 
            Matrix3x3(
            const ::float3& row0
            , const ::float3& row1
            , const ::float3& row2
            ) :
            _row0( row0 )
            , _row1( row1 )
            , _row2( row2 ) {}

        __device__ 
            const noor_float3& operator[]( int i ) const {
            if ( i == 0 ) return _row0;
            if ( i == 1 ) return _row1;
            return _row2;
        }

        __device__ __host__
            float3 operator*( const float3& v ) const {
            return  make_float3( dot( _row0, v ), dot( _row1, v ), dot( _row2, v ) );
        }

        __device__ 
            void transpose3x3() {
            const Matrix3x3& m = *this;
            const noor_float3 row0{ make_float3( m[0][0], m[1][0], m[2][0] ) };
            const noor_float3 row1{ make_float3( m[0][1], m[1][1], m[2][1] ) };
            const noor_float3 row2{ make_float3( m[0][2], m[1][2], m[2][2] ) };

            _row0 = row0;
            _row1 = row1;
            _row2 = row2;
        }
    };

    __forceinline__	__device__ 
        void solve2x2( const Matrix2x2& A, const noor_float2& B, float& x0, float& x1 ) {
        const float det = A.determinant();
        if ( fabsf( det ) < 1e-10f ) {
            x0 = x1 = 0;
            return;
        }
        const float inv_det = 1.0f / det;
        x0 = ( A[1][1] * B[0] - A[0][1] * B[1] ) * inv_det;
        x1 = ( A[0][0] * B[1] - A[1][0] * B[0] ) * inv_det;
        if ( isnan( x0 ) || isnan( x1 ) ) {
            x0 = x1 = 0;
            return;
        }
        return;
    }
} /* end NOOR namespace */
#endif /* CUDAMATH_CUH */