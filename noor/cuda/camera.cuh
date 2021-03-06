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
        uint w,
        uint h,
        float lens_radius,
        float focal_length,
        CameraType camera_type
        ) {
        cudatransform( _cameraToWorld, cameraToWorld );
        cudatransform( _rasterToCamera, rasterToCamera );
        cudatransform( _worldToRaster,
                       glm::inverse( rasterToCamera ) * 
                       glm::inverse( cameraToWorld ) );
        _w = w;
        _h = h;
        _lens_radius = lens_radius;
        _focal_length = focal_length;
        _type = camera_type;
        _center = _w * ( _h >> 1 ) + ( _w >> 1 );
        _dxCamera = _rasterToCamera.transformVector( make_float3( 1, 0, 0 ) );
        _dyCamera = _rasterToCamera.transformVector( make_float3( 0, 1, 0 ) );

        const float3 ipMin = _rasterToCamera.transformPoint( make_float3( 0 ) );
        const float3 ipMax = _rasterToCamera.transformPoint( 
            make_float3( (float)_w, (float)_h, 0 ) );
        _image_area = fabsf( ( ipMax.x - ipMin.x ) * ( ipMax.y - ipMin.y ) );
    }
    CudaTransform _cameraToWorld;
    CudaTransform _rasterToCamera;
    CudaTransform _worldToRaster;
    float3 _dxCamera, _dyCamera;
    float _lens_radius, _focal_length, _image_area;
    uint _w, _h;
    uint _center;
    CameraType _type;
};
#endif /* CUDACAMERA_CUH */
