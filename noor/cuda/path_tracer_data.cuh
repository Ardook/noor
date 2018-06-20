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
#include "stack.cuh"
#include "bbox.cuh"
#include "shapeimp.cuh"
#include "lightimp.cuh"
#include "bvh.cuh"
#include "framebuffer.cuh"
#include "cameraimp.cuh"
#include "bsdf.cuh"
#include "render.cuh"
#include "direct.cuh"
#include "scatter.cuh"
#include "accumulate.cuh"

std::unique_ptr<CudaRenderManager> _render_manager;

void load_cuda_data( const std::unique_ptr<CudaPayload>& payload, 
                     const CudaHosekSky& hosek,
                     const CudaCamera& camera,
                     const CudaSpec& spec,
                     GLuint* textureID
) {
    _render_manager = std::make_unique<CudaRenderManager>( payload, hosek,
                                                          camera, spec, 
                                                          textureID );
}

void get_lookAt(float4& lookAt) {
    _render_manager->get_lookAt( lookAt );
}

void update_cuda_spec() {
    _render_manager->update_spec();
}

void update_cuda_camera() {
    _render_manager->update_camera();
}

void update_cuda_hosek() {
    _render_manager->update_hoseksky();
}

void device_free_memory() {
    if ( _render_manager ) _render_manager.reset();
}

#endif /* PATH_TRACER_DATA_CUH */
