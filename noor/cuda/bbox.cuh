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
#ifndef CUDABBOX_CUH
#define CUDABBOX_CUH

struct CudaMinMax {
    float3 _min;
    float3 _max;
    __device__
        const float3& operator[]( int i ) const {
        return i == 0 ? _min : _max;
    }
};

class CudaBBox {
public:
    __device__
        CudaBBox( const float3& a, const float3& b ) {
        _minmax._min = a; _minmax._max = b;
    }
    __device__
        const float3& min() const {
        return _minmax[0];
    }

    __device__
        const float3& max() const {
        return _minmax[1];
    }

    __device__
        const float3& centroid() const {
        return ( _minmax[0] + _minmax[1] )*0.5f;
    }

    __device__
        float radius() const {
        return length( ( _minmax[0] - _minmax[1] )*0.5f - _minmax[1] );
    }

    __device__
        bool intersect( const CudaRay& ray ) const {
        float tMin = ( _minmax[ray.getPosneg()[0]].x - ray.getOrigin().x ) * ray.getInvDir().x;
        float tMax = ( _minmax[1 - ray.getPosneg()[0]].x - ray.getOrigin().x ) * ray.getInvDir().x;
        float tyMin = ( _minmax[ray.getPosneg()[1]].y - ray.getOrigin().y ) * ray.getInvDir().y;
        float tyMax = ( _minmax[1 - ray.getPosneg()[1]].y - ray.getOrigin().y ) * ray.getInvDir().y;

        if ( tMin > tyMax || tyMin > tMax ) return false;
        if ( tyMin > tMin ) tMin = tyMin;
        if ( tyMax < tMax ) tMax = tyMax;

        float tzMin = ( _minmax[ray.getPosneg()[2]].z - ray.getOrigin().z ) * ray.getInvDir().z;
        float tzMax = ( _minmax[1 - ray.getPosneg()[2]].z - ray.getOrigin().z ) * ray.getInvDir().z;

        if ( tMin > tzMax || tzMin > tMax ) return false;
        if ( tzMin > tMin ) tMin = tzMin;
        if ( tzMax < tMax ) tMax = tzMax;
        return ( tMin < ray.getTmax() ) && ( tMax > 0 );
    }
    CudaMinMax _minmax;
};

#endif /* CUDABBOX_CUH */
