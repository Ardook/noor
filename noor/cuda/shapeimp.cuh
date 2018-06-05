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
#ifndef CUDASHAPEIMP_CUH
#define CUDASHAPEIMP_CUH
#include "geometry.cuh"

__forceinline__ __device__
float CudaShape::pdf(
    const CudaIntersection& I,
    const float3& wi
) const {
    float pdf = 0.f;
    if ( _type == QUAD || _type == DISK ) {
        float3 p, n;
        const CudaRay ray( I._p, wi );
        if ( !intersect( ray, p, n ) ) {
            return 0.f;
        }
        // Convert light sample weight to solid angle measure
        pdf = NOOR::length2( I._p - p ) / ( NOOR::absDot( n, -1.0f*ray.getDir() ) * _area );
    } else if ( _type == SPHERE ) {
        // Compute general sphere PDF
        float sinThetaMax2 = _radius2 / NOOR::length2( I._p - _center );
        float cosThetaMax = sqrtf( fmaxf( 0.f, 1.f - sinThetaMax2 ) );
        pdf = NOOR::uniformConePdf( cosThetaMax );
    }
    if ( isinf( pdf ) ) {
        pdf = 0.f;
    }
    return pdf;
}

__forceinline__ __device__
void CudaShape::sample( const CudaIntersection& I,
                        float3& p,
                        float& pdf
) const {
    if ( _type == QUAD ) {
        sampleQuad( *this, I, p, pdf );
    } else if ( _type == SPHERE ) {
        sampleSphere( *this, I, p, pdf );
    } else if ( _type == DISK ) {
        sampleDisk( *this, I, p, pdf );
    }
}

__forceinline__ __device__
bool CudaShape::intersect( const CudaRay& ray,
                           float3& p,
                           float3& n
) const {
    if ( _type == QUAD )
        return intersectQuad( *this, ray, p, n );
    else if ( _type == SPHERE )
        return intersectSphere( *this, ray, p, n );
    else if ( _type == DISK )
        return intersectDisk( *this, ray, p, n );
    else
        return false;
}

__forceinline__ __device__
bool CudaShape::intersect( const CudaRay& ray ) const {
    if ( _type == QUAD )
        return intersectQuad( *this, ray );
    else if ( _type == SPHERE )
        return intersectSphere( *this, ray );
    else if ( _type == DISK )
        return intersectDisk( *this, ray );
    else
        return false;
}
#endif
