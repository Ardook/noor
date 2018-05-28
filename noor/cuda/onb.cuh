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
#ifndef CUDAONB_CUH
#define CUDAONB_CUH

/**
* Orthonormal basis
*/
class CudaONB {
public:
    __device__
        CudaONB(
        const float3& n
        , const float3& t
        , const float3& b
        ) :
        _u( t )
        , _v( b )
        , _w( n ) {}

    // transform from the coordinate system represented by ONB
    __device__
        float3 toWorld( const float3& v ) const {
        return ( v.x*_u + v.y*_v + v.z*_w );
    }

    // transform to the coordinate system represented by ONB
    __device__
        float3 toLocal( const float3& v ) const {
        const float x = dot( v, _u );
        const float y = dot( v, _v );
        const float z = dot( v, _w );
        return make_float3( x, y, z );
    }
    const float3& _u;
    const float3& _v;
    const float3& _w;
};
#endif /* CUDAONB_CUH */
