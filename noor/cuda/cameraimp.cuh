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
#ifndef CUDACAMERAIMP_CUH
#define CUDACAMERAIMP_CUH

/* Based on PBRT Cameras */
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
        CudaRay generateRay( uint x, uint y, const CudaRNG& rng ) const {
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
        float3 We( const CudaRay& ray, float2* raster2 )const {
        const float cosTheta = dot( ray.getDir(), 
                              _cameraToWorld.transformVector( make_float3( 0, 0, -1 ) ) );
        if ( cosTheta <= 0 ) return make_float3(0);

        // Map ray $(\p{}, \w{})$ onto the raster grid
        const float t = ( _lens_radius > 0 ? _focal_length : 1 ) / cosTheta;
        const float3 focus = ray.pointAtParameter( t );
        const float3 raster3 = _worldToRaster.transformPoint(focus);

        // Return raster position if requested
        if (raster2) *raster2 = make_float2( raster3.x, raster3.y );

        // Return zero importance for out of bounds points
        if ( raster3.x < 0 || raster3.x >= _w ||
             raster3.y < 0 || raster3.y >= _h )
            return make_float3(0);

        // Compute lens area of perspective camera
        const float lensArea = _lens_radius != 0 ? 
            ( NOOR_PI * _lens_radius * _lens_radius ) : 1;

        // Return importance for point on image plane
        const float cos2Theta = cosTheta * cosTheta;
        return make_float3( 1 / ( _image_area * lensArea * cos2Theta * cos2Theta ) );
    }

    __device__
    void Pdf_We( const CudaRay &ray, float *pdfPos, float *pdfDir ) const {
        // Interpolate camera matrix and fail if $\w{}$ is not forward-facing
        const float cosTheta = dot( ray.getDir(), 
                              _cameraToWorld.transformVector( make_float3( 0, 0, -1 ) ) );
        if ( cosTheta <= 0 ) {
            *pdfPos = *pdfDir = 0;
            return;
        }
        // Map ray $(\p{}, \w{})$ onto the raster grid
        const float t = ( _lens_radius > 0 ? _focal_length : 1 ) / cosTheta;
        const float3 focus = ray.pointAtParameter( t );
        const float3 raster3 = _worldToRaster.transformPoint( focus );

        // Return zero importance for out of bounds points
        if ( raster3.x < 0 || raster3.x >= _w ||
             raster3.y < 0 || raster3.y >= _h ) {
            *pdfPos = *pdfDir = 0;
            return;
        }

        // Compute lens area of perspective camera
        const float lensArea = _lens_radius != 0 ?
            ( NOOR_PI * _lens_radius * _lens_radius ) : 1;

        // Return importance for point on image plane
        *pdfPos = 1 / lensArea;
        *pdfDir = 1 / ( _image_area * cosTheta * cosTheta * cosTheta );
    }

    __device__
    float3 Sample_Wi( const CudaIntersection &I, 
                      const float2 &u,
                      float3 &wi, 
                      float &pdf,
                      float2 &raster2,
                      CudaVisibility &vis ) const {
        // Uniformly sample a lens interaction _lensIntr_
        const float2 lens = _lens_radius * NOOR::concentricSampleDisk( u );
        const float3 lensWorld = 
            _cameraToWorld.transformPoint( make_float3( lens.x, lens.y, 0 ) );
        const float3 n = 
            _cameraToWorld.transformVector( make_float3( 0, 0, -1 ) ) ;

        // Populate arguments and compute the importance value
        vis = CudaVisibility( I.getP(), lensWorld );
        wi = lensWorld - I.getP();
        const float dist = length( wi );
        wi /= dist;
        // Compute PDF for importance arriving at _ref_
        // Compute lens area of perspective camera
        const float lensArea = _lens_radius != 0 ?
            ( NOOR_PI * _lens_radius * _lens_radius ) : 1;
        pdf = ( dist * dist ) / ( NOOR::absDot( n, wi ) * lensArea );
        const CudaRay ray( lensWorld, wi );
        return We( ray, &raster2 );
    }

    __device__
        CudaRay generateRay( uint x, uint y, const CudaRNG& rng ) const {
        const float raster_x = (float) x + rng() - 0.5f;
        const float raster_y = (float) y + rng() - 0.5f;
        const float3 pCamera = _rasterToCamera.transformPoint( 
            make_float3( raster_x, raster_y, 0.0f ) );
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
        return ray;
    }
};

class CudaOrthoCamera : public CudaCamera {
public:
    CudaOrthoCamera() = default;

    __device__
        CudaRay generateRay( uint x, uint y, const CudaRNG& rng ) const {
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
CudaRay generateRay( uint x, uint y, const CudaRNG& rng ) {
    switch ( _constant_camera._type ) {
        case ORTHO:
            return ( (const CudaOrthoCamera&) _constant_camera ).generateRay( x, y, rng );
        case ENV:
            return ( (const CudaEnvCamera&) _constant_camera ).generateRay( x, y, rng );
        default:
            return ( (const CudaPerspCamera&) _constant_camera ).generateRay( x, y, rng );
    }
}
#endif /* CUDACAMERAIMP_CUH */
