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
#ifndef CUDATRANSFORM_CUH
#define CUDATRANSFORM_CUH
class CudaTransform {
    NOOR::Matrix3x3 _scale_rotate;
    float3 _translate;
public:
    CudaTransform() = default;

    __host__ __device__
        CudaTransform(
        const float3& row0
        , const float3& row1
        , const float3& row2
        ) :
        _scale_rotate( row0, row1, row2 )
        , _translate( make_float3( 0.0f ) ) {}

    __host__ __device__
        CudaTransform(
        const float4& row0
        , const float4& row1
        , const float4& row2
        ) :
        _scale_rotate( make_float3( row0 ), make_float3( row1 ), make_float3( row2 ) )
        , _translate( make_float3( row0.w, row1.w, row2.w ) ) {}

    __device__ __host__
        const float3& getTranslation() const {
        return _translate;
    }

    __device__ 
        float3 transformPoint( const float3& p ) const {
        return _translate + _scale_rotate * p;
    }

    __device__ __host__
        float3 transformVector( const float3& v ) const {
        return _scale_rotate * v;
    }

    __device__ 
        float3 transformNormal( const float3& n ) const {
        return normalize( transformVector( n ) );
    }
};
#endif