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
#ifndef CUDASHAPE_CUH
#define CUDASHAPE_CUH

class CudaRay;
class CudaRNG;
class CudaIntersection;
class CudaShape {
public:
    float3 _center;
    float3 _u;
    float3 _v;
    float3 _n;
    float _area;
    float _radius;
    float _radius2;
    int _light_idx;
    AreaMeshLightType _type;
    CudaShape() = default;

    __host__ __device__
        CudaShape(
        const float3& center,
        const float3& u,
        const float3& v,
        const float3& n,
        AreaMeshLightType type,
        int light_idx = 0
        ) :
        _center( center ),
        _u( u ),
        _v( v ),
        _n( n ),
        _light_idx( light_idx ),
        _type( type ) {
        _radius = 0.5f*fminf( length( _u ), length( _v ) );
        _radius2 = _radius*_radius;
        if ( _type == QUAD ) {
            _area = length( cross( _u, _v ) );
        } else if ( _type == SPHERE ) {
            _area = NOOR_4PI*_radius2;
            _center = _center + .5f*( _u + _v );
        } else if ( _type == DISK ) {
            _area = NOOR_PI*_radius2;
            _center = _center + .5f*( _u + _v );
        }
    }

    __host__ __device__
        void getBounds( float3& min, float3& max )const {
        if ( _type == QUAD ) {
            min = fminf( fminf( fminf( _center, _center + _u ), _center + _v ), _center + _u + _v );
            max = fmaxf( fmaxf( fmaxf( _center, _center + _u ), _center + _v ), _center + _u + _v );
        } else if ( _type == SPHERE ) {
            const float3 bounds = make_float3( _radius );
            min = _center - bounds;
            max = _center + bounds;
        } else if ( _type == DISK ) {
            const float3 u = normalize( _u );
            const float3 v = normalize( _v );
            float3 r = _center + _radius * u;
            float3 t = _center + _radius * v;
            float3 l = _center - _radius * u;
            float3 b = _center - _radius * v;
            min = fminf( fminf( fminf( l, r ), t ), b );
            max = fmaxf( fmaxf( fmaxf( l, r ), t ), b );
        }
    }
#ifdef __CUDACC__
    __device__
        float pdf() const { return 1.f / _area; }
    __device__
        float pdf(
        const CudaIntersection& I,
        const float3& i
        ) const;
    __device__
        bool intersect( const CudaRay& ray, float3& p, float3& n ) const;
    __device__
        bool intersect( const CudaRay& ray ) const;
    __device__
        void sample(
        const CudaIntersection& I,
        float3& p,
        float& pdf
        ) const;
#endif
};
#endif /* CUDASHAPE_CUH */