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
#ifndef CUDAGEOMETRY_CUH
#define CUDAGEOMETRY_CUH

__forceinline__ __device__
void sampleQuad(
    const CudaShape& quad,
    const CudaIntersection& I,
    const CudaRNG& rng,
    float3& p,
    float& pdf
) {
    p = quad._center + quad._u*rng() + quad._v*rng() + _constant_spec._reflection_bias*quad._n;
    const float3 wi = I._p - p;
    const float dist2 = NOOR::length2( wi );
    pdf = dist2 / ( NOOR::absDot( quad._n, -1.0f*wi ) * quad._area );
    if ( isinf( pdf ) ) pdf = 0.f;
}

__forceinline__ __device__
void sampleSphere(
    const CudaShape& sphere,
    const CudaIntersection& I,
    const CudaRNG& rng,
    float3& p,
    float& pdf
) {
    const float2 r = make_float2( rng(), rng() );
    float3 w = normalize( sphere._center - I._p );
    float3 u, v;
    NOOR::coordinateSystem( w, u, v );
    // Compute theta and phi values for sample in cone
    const float sinThetaMax2 = sphere._radius2 / NOOR::length2( I._p - sphere._center );
    const float cosThetaMax = sqrtf( fmaxf( 0.f, 1.f - sinThetaMax2 ) );
    const float cosTheta = ( 1.f - r.x ) + r.x * cosThetaMax;
    const float sinTheta = sqrtf( fmaxf( 0.f, 1.f - cosTheta * cosTheta ) );
    const float phi = r.y * NOOR_2PI;

    // Compute angle alpha from center of sphere to sampled point on surface
    const float dc = length( I._p - sphere._center );
    const float ds = dc * cosTheta - sqrtf( fmaxf( 0.f, sphere._radius2 - dc * dc * sinTheta * sinTheta ) );
    const float cosAlpha = ( dc * dc + sphere._radius2 - ds * ds ) / ( 2.f * dc * sphere._radius );
    const float sinAlpha = sqrtf( fmaxf( 0.f, 1.f - cosAlpha * cosAlpha ) );
    // Compute surface normal and sampled point on sphere
    float3 n = NOOR::sphericalDirection( sinAlpha, cosAlpha, phi, -u, -v, -w );
    p = sphere._center + sphere._radius * n;
    p *= sphere._radius / length( p - sphere._center );
    p += _constant_spec._reflection_bias * n;
    // uniform cone PDF
    pdf = NOOR::uniformConePdf( cosThetaMax );
}


__forceinline__ __device__
void sampleDisk(
    const CudaShape& disk,
    const CudaIntersection& I,
    const CudaRNG& rng,
    float3& p,
    float& pdf
) {
    float2 pOnDisk = NOOR::concentricSampleDisk( make_float2( rng(), rng() ) );
    p = make_float3( pOnDisk.x * disk._radius, pOnDisk.y * disk._radius, 0.f );
    const float3& w = disk._n;
    float3 u, v;
    NOOR::coordinateSystem( w, u, v );
    p = p.x*u + p.y*v + p.z*w + disk._center + _constant_spec._reflection_bias*disk._n;
    const float3 wi = I._p - p;
    const float dist2 = NOOR::length2( wi );
    pdf = dist2 / ( NOOR::absDot( disk._n, -1.0f*wi ) * disk._area );
    if ( isinf( pdf ) ) pdf = 0.f;
}

__forceinline__ __device__
bool intersectQuad(
    const CudaShape& quad,
    const CudaRay& ray
) {
    float t_hit;
    float denom = dot( quad._n, ray.getDir() );
    if ( denom == 0 ) return false;
    t_hit = -( dot( ray.getOrigin(), quad._n ) - dot( quad._center, quad._n ) ) / denom;
    if ( t_hit > ray.getTmax() ) return false;
    const float3 p_hit = ray.pointAtParameter( t_hit );
    const float3 inplane = p_hit - quad._center;
    const float t0 = dot( inplane, quad._u ) / dot( quad._u, quad._u );
    if ( t0 < 0 || t0 > 1 ) return false;
    const float t1 = dot( inplane, quad._v ) / dot( quad._v, quad._v );
    if ( t1 < 0 || t1 > 1 ) return false;
    ray.setTmax( t_hit );
    return true;
}

__forceinline__ __device__
bool intersectSphere( const CudaShape& sphere,
                   const CudaRay& ray ) {
    const float3 o = ray.getOrigin() - sphere._center;
    const float3& d = ray.getDir();

    float A = NOOR::length2( d );
    float B = 2 * dot( o, d );
    float C = NOOR::length2( o ) - sphere._radius2;

    float nearT, farT;
    if ( !NOOR::solveQuadratic( A, B, C, nearT, farT ) )
        return false;

    const float mint = 0.f;
    const float maxt = ray.getTmax();
    if ( !( nearT <= maxt && farT >= mint ) ) /* NaN-aware conditionals */
        return false;

    if ( nearT < mint ) {
        if ( farT > maxt )
            return false;
        ray.setTmax( farT );
    } else {
        ray.setTmax( nearT );
    }
    return true;
}

__forceinline__ __device__
bool intersectDisk(
    const CudaShape& disk,
    const CudaRay& ray
) {
    float denom = dot( disk._n, ray.getDir() );
    if ( denom == 0 ) return false;
    float t_hit = -( dot( ray.getOrigin(), disk._n ) - dot( disk._center, disk._n ) ) / denom;
    if ( t_hit > ray.getTmax() ) return false;
    const float3 p = ray.pointAtParameter( t_hit ) - disk._center;
    if ( NOOR::length2( p ) > disk._radius2 ) {
        return false;
    }
    ray.setTmax( t_hit );
    return true;
}

__forceinline__ __device__
bool intersectQuad(
    const CudaShape& quad,
    const CudaRay& ray,
    float3& p,
    float3& n
) {
    if ( intersectQuad( quad, ray ) ) {
        n = quad._n;
        p = ray.pointAtParameter( ray.getTmax() );
        return true;
    }
    return false;
}

__forceinline__ __device__
bool intersectSphere(
    const CudaShape& sphere,
    const CudaRay& ray,
    float3& p,
    float3& n
) {
    if ( intersectSphere( sphere, ray ) ) {
        p = ray.pointAtParameter( ray.getTmax() );
        n = normalize( p - sphere._center );
        return true;
    }
    return false;
}

__forceinline__ __device__
bool intersectDisk(
    const CudaShape& disk,
    const CudaRay& ray,
    float3& p,
    float3& n
) {
    if ( intersectDisk( disk, ray ) ) {
        n = disk._n;
        p = ray.pointAtParameter( ray.getTmax() );
        return true;
    }
    return false;
}
#endif
