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
#ifndef PATH_TRACER_DATA_CUH
#define PATH_TRACER_DATA_CUH
#include "path_tracer.cuh"
#include "onb.cuh"
#include "rng.cuh"
#include "transformimp.cuh"
#include "ray.cuh"
#include "mesh.cuh"
#include "intersection.cuh"
#include "texture.cuh"
#include "distribution1d.cuh"
#include "distribution2d.cuh"
#include "skydome.cuh"
#include "materialimp.cuh"
#include "triangle.cuh"
#include "stack.cuh"
#include "bbox.cuh"
#include "shapeimp.cuh"
#include "lightimp.cuh"
#include "bvh.cuh"
#include "lookat.cuh"
#include "framebuffer.cuh"
#include "camera.cuh"
#include "render.cuh"
#include "bsdf.cuh"
#include "direct.cuh"
#include "scatter.cuh"
#include "accumulate.cuh"

std::unique_ptr<CudaRenderManager> _cuda_renderer;
//void init_framebuffer( GLuint* textureID, uint w, uint h ) {
//    _cuda_renderer->init_framebuffer( textureID, w, h );
//}

void update_cuda_spec( std::unique_ptr<CudaSpec>& spec ) {
    if ( spec->_outofsync ) {
        _host_spec = *spec;
        NOOR::memcopy_symbol_async( &_constant_spec, &_host_spec );
        spec->_outofsync = false;
    }
}

void load_cuda_data( const std::unique_ptr<CudaPayload>& payload,
                     CudaHosekSky& hosek_sky, int gpuID,
                     SkydomeType skydome_type,
                     GLuint* cudaTextureID,
                     int w, 
                     int h

) {
    _cuda_renderer = std::make_unique<CudaRenderManager>( payload, hosek_sky, gpuID, skydome_type, cudaTextureID, w, h );
}

void update_cuda_camera( const glm::mat4& cameraToWorld,
                         const glm::mat4& rasterToCamera, int w, int h,
                         float lens_radius,
                         float focal_length,
                         CameraType camera_type ) {
    _cuda_renderer->update_camera( cameraToWorld, rasterToCamera, w, h,
                                   lens_radius, focal_length, camera_type );
}

void update_cuda_sky() {
    _cuda_renderer->update_hoseksky();
}

void device_free_memory() {
    if ( _cuda_renderer ) _cuda_renderer.reset();
}

#endif /* PATH_TRACER_DATA_CUH */
