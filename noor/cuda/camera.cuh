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
#ifndef CUDACAMERA_CUH
#define CUDACAMERA_CUH

/* Based on PBRT Cameras */
__forceinline__ __host__
void cudatransform( CudaTransform& t, const glm::mat4& m ) {
    t = CudaTransform( V2F4( glm::row( m, 0 ) ), V2F4( glm::row( m, 1 ) ), V2F4( glm::row( m, 2 ) ) );
}

class CudaCamera {
public:
    CudaCamera() = default;
    __host__
        void update(
        const glm::mat4& cameraToWorld,
        const glm::mat4& rasterToCamera,
        int w,
        int h,
        float lens_radius,
        float focal_length,
        CameraType camera_type
        ) {
        cudatransform( _cameraToWorld, cameraToWorld );
        cudatransform( _rasterToCamera, rasterToCamera );
        _w = w;
        _h = h;
        _lens_radius = lens_radius;
        _focal_length = focal_length;
        _type = camera_type;
        _center = _w * ( _h >> 1 ) + ( _w >> 1 );
        _dxCamera = _rasterToCamera.transformVector( make_float3( 1, 0, 0 ) );
        _dyCamera = _rasterToCamera.transformVector( make_float3( 0, 1, 0 ) );
    }
    CudaTransform _cameraToWorld;
    CudaTransform _rasterToCamera;
    float3 _dxCamera, _dyCamera;
    float _lens_radius, _focal_length;
    int _w, _h;
    int _center;
    CameraType _type;
};

class CudaEnvCamera : public CudaCamera {
public:
    CudaEnvCamera() = default;

    __device__
        float3 genDir( float x, float y ) const {
        // Compute environment camera ray direction
        const float theta = NOOR_PI  * y / _h + NOOR_PI;
        const float phi = NOOR_2PI * x / _w - NOOR_PI_over_2;
        return normalize( NOOR::sphericalDirection( theta, phi, RIGHT_HANDED ) );
    }

    __device__
        CudaRay generateRay( int x, int y, const CudaRNG& rng ) const {
        const float raster_x = (float) x + rng() - 0.5f;
        const float raster_y = (float) y + rng() - 0.5f;
        float3 origin, origin_dx, origin_dy;
        origin = origin_dx = origin_dy = make_float3( 0.0f );

        const float3 dir = genDir( raster_x, raster_y );
        const float3 dir_dx = genDir( raster_x + 1, raster_y );
        const float3 dir_dy = genDir( raster_x, raster_y + 1 );
        CudaRay ray( origin,
                     dir,
                     origin_dx,
                     origin_dy,
                     dir_dx,
                     dir_dy );

        ray.transform( _cameraToWorld );
        return ray;
    }
};

class CudaPerspCamera : public CudaCamera {
public:
    CudaPerspCamera() = default;

    __device__
        CudaRay generateRay( int x, int y, const CudaRNG& rng ) const {
        const float raster_x = (float) x + rng() - 0.5f;
        const float raster_y = (float) y + rng() - 0.5f;

        const float3 pCamera = _rasterToCamera.transformPoint( make_float3( raster_x, raster_y, 0.0f ) );
        float3 origin = make_float3( 0.f );
        float3 dir = normalize( pCamera );
        float3 origin_dx, origin_dy;
        float3 dir_dx, dir_dy;
        if ( _lens_radius > 0.f ) {
            const float2 u = make_float2( rng(), rng() );
            // Sample point on lens
            float2 pLens = _lens_radius * NOOR::concentricSampleDisk( u );
            // Compute point on plane of focus
            float ft = -_focal_length / dir.z;
            //float3 pFocus = ray.pointAtParameter( ft );
            float3 pFocus = origin + ft*dir;
            // Update ray for effect of lens
            origin = make_float3( pLens.x, pLens.y, 0 );
            dir = normalize( pFocus - origin );

            const float3 dx = normalize( pCamera + _dxCamera );
            ft = -_focal_length / dx.z;
            pFocus = make_float3( 0, 0, 0 ) + ( ft * dx );
            origin_dx = make_float3( pLens.x, pLens.y, 0 );
            dir_dx = normalize( pFocus - origin_dx );

            const float3 dy = normalize( pCamera + _dyCamera );
            ft = -_focal_length / dy.z;
            pFocus = make_float3( 0, 0, 0 ) + ( ft * dy );
            origin_dy = make_float3( pLens.x, pLens.y, 0 );
            dir_dy = normalize( pFocus - origin_dy );
        } else {
            origin_dx = origin_dy = origin;
            dir_dx = normalize( pCamera + _dxCamera );
            dir_dy = normalize( pCamera + _dyCamera );
        }
        CudaRay ray( origin,
                     dir,
                     origin_dx,
                     origin_dy,
                     dir_dx,
                     dir_dy );

        ray.transform( _cameraToWorld );
        ray.scaleDifferentials( .25f );
        return ray;
    }
};

class CudaOrthoCamera : public CudaCamera {
public:
    CudaOrthoCamera() = default;

    __device__
        CudaRay generateRay( int x, int y, const CudaRNG& rng ) const {
        const float raster_x = (float) x + rng() - 0.5f;
        const float raster_y = (float) y + rng() - 0.5f;

        const float3 origin = _rasterToCamera.transformPoint( make_float3( raster_x, raster_y, 0.f ) );
        const float3 dir = make_float3( 0.f, 0.f, -1.f );
        const float3 origin_dx = origin + _dxCamera;
        const float3 origin_dy = origin + _dyCamera;
        const float3 dir_dx = dir;
        const float3 dir_dy = dir;
        CudaRay ray( origin,
                     dir,
                     origin_dx,
                     origin_dy,
                     dir_dx,
                     dir_dy );

        ray.transform( _cameraToWorld );
        return ray;
    }
};
__constant__
CudaCamera _constant_camera;


__forceinline__ __device__
CudaRay generateRay( int x, int y, const CudaRNG& rng ) {
    if ( _constant_camera._type == ORTHO )
        return ( (const CudaOrthoCamera&) _constant_camera ).generateRay( x, y, rng );
    else if ( _constant_camera._type == ENV )
        return ( (const CudaEnvCamera&) _constant_camera ).generateRay( x, y, rng );
    else
        return ( (const CudaPerspCamera&) _constant_camera ).generateRay( x, y, rng );
}
#endif /* CUDACAMERA_CUH */
