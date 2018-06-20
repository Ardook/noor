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
#ifndef CUDAMOLLERTRIANGLE_CUH
#define CUDAMOLLERTRIANGLE_CUH

__forceinline__ __device__
bool intersect(
    const CudaRay& ray,
    CudaIntersection& I,
    uint tri_idx )
{
    const float EPSILON = NOOR_EPSILON;
    const uint4 attr_idx = _mesh_manager.getAttrIndex( tri_idx );
    const float3 vertex0 = make_float3( _mesh_manager.getVertex( attr_idx.x ) );
    const float3 vertex1 = make_float3( _mesh_manager.getVertex( attr_idx.y ) );
    const float3 vertex2 = make_float3( _mesh_manager.getVertex( attr_idx.z ) );
    const float3 edge1 = vertex1 - vertex0;
    const float3 edge2 = vertex2 - vertex0;
    const float3 h = cross( ray.getDir(), edge2 );
    const float a = dot( edge1, h );
    if ( a > -EPSILON && a < EPSILON )
        return false;
    const float f = 1.f / a;
    const float3 s = ray.getOrigin() - vertex0;

    const float u = f * ( dot( s, h ) );
    if ( u < 0.0 || u > 1.0 )
        return false;
    const float3 q = cross( s, edge1 );
    const float v = f * dot( ray.getDir(), q );
    if ( v < 0.0 || u + v > 1.0 )
        return false;
    const float t = ( f * dot( edge2, q ) );
    if ( t < 0.0f || t > ray.getTmax() ) {
        return false;
    }
    I._u = u;
    I._v = v;
    I._tri_idx = tri_idx;
    I.setMaterialType( DIFFUSE );
    ray.setTmax( t );
    return true;
}

__forceinline__ __device__
bool intersect(
    const CudaRay& ray,
    uint tri_idx )
{
    const float EPSILON = NOOR_EPSILON;
    const uint4 attr_idx = _mesh_manager.getAttrIndex( tri_idx );
    const float3 vertex0 = make_float3( _mesh_manager.getVertex( attr_idx.x ) );
    const float3 vertex1 = make_float3( _mesh_manager.getVertex( attr_idx.y ) );
    const float3 vertex2 = make_float3( _mesh_manager.getVertex( attr_idx.z ) );
    const float3 edge1 = vertex1 - vertex0;
    const float3 edge2 = vertex2 - vertex0;
    const float3 h = cross( ray.getDir(), edge2 );
    const float a = dot( edge1, h );
    if ( a > -EPSILON && a < EPSILON )
        return false;
    const float f = 1.f / a;
    const float3 s = ray.getOrigin() - vertex0;

    const float u = f * ( dot( s, h ) );
    if ( u < 0.0 || u > 1.0 )
        return false;
    const float3 q = cross( s, edge1 );
    const float v = f * dot( ray.getDir(), q );
    if ( v < 0.0 || u + v > 1.0 )
        return false;
    const float t = ( f * dot( edge2, q ) );
    if ( t < 0.0f || t > ray.getTmax() ) {
        return false;
    }
    return true;
}

#endif /* CUDAMOLLERTRIANGLE_CUH */