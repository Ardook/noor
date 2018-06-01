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
#include "path_tracer_data.cuh"

__forceinline__ __device__
bool rr_end( const CudaRNG& rng, int bounce, float3& beta ) {
    // Russian Roulette 
    if ( bounce >= _constant_spec._rr ) {
        const float q = fmaxf( 0.05f, 1.0f - NOOR::maxcomp( beta ) );
        if ( rng() < q ) return true;
        beta /= 1.0f - q;
    }
    return false;
}

__forceinline__ __device__
float4 pathtracer(
    CudaRay& ray,
    const CudaRNG& rng,
    float4& lookAt,
    int tid
) {
    float3 L = _constant_spec._black;
    float3 beta = _constant_spec._white;
    float3 wi;
    bool specular_bounce = false;
    for ( unsigned char bounce = 0; bounce < _constant_spec._bounces; ++bounce ) {
        CudaIntersection I;
        I._tid = tid;
        if ( intersect( ray, I ) ) {
            if ( bounce == 0 ) {
                lookAt = make_float4( I._p, 1.f );
            }
            if ( I.isEmitter() && ( bounce == 0 || specular_bounce ) ) {
                L += beta * ( I._material_type & MESHLIGHT ?
                              _light_manager.Le( ray.getDir(), I._ins_idx ) :
                              _material_manager.getEmmitance( I )
                              );
                break;
            }
            if ( ray.isDifferential() && I.isBumped() ) {
                bump( I );
            }
            accumulate( I, rng, wi, beta, L, _constant_spec.is_mis_enabled() );
            if ( !I.isTransparentBounce() && ( dot( wi, I._n ) < 0 ) ) break;
            // Russian Roulette 
            if ( rr_end( rng, bounce, beta ) ) break;
            specular_bounce = I.isSpecularBounce();
            ray = I.spawnRay( ray, wi );
        } else { // no intersection 
            if ( _constant_spec.is_sky_light_enabled() && 
                ( bounce == 0 || specular_bounce ) ) {
                L += beta*_skydome_manager.evaluate( normalize( ray.getDir() ), false );
            }
            break;
        }
    }
    L = clamp( L, 0, 60.f );
    return make_float4( L, 1.f );
}

__global__
void path_tracer_kernel( uint frame_number ) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    int id = y * _constant_camera._w + x;
    CudaRNG rng( id, frame_number + clock64() );
    CudaRay ray = generateRay( x, y, rng );
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    float4 lookAt = make_float4( 0.f );
    const float4 new_color = pathtracer( ray, rng, lookAt, tid );
    _framebuffer_manager.set( new_color, id, frame_number );
    if ( id == _constant_camera._center ) {
        _device_lookAt = lookAt;
    }
}

__global__
void debug_skydome_kernel(
    uint frame_number
    , int width
    , int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int id = y * _constant_camera._w + x;
    float2 u = make_float2( x / (float) ( _constant_camera._w ), y / (float) ( _constant_camera._h ) );
    _framebuffer_manager.set(_skydome_manager.evaluate( u ), id);
    if ( _constant_spec.is_mis_enabled() ) {
        CudaRNG rng( id, frame_number + clock64() );
        u = _skydome_manager.importance_sample_uv( make_float2( rng(), rng() ) );
        u.x *= _constant_camera._w;
        u.y *= _constant_camera._h;
        id = int( u.y ) * ( _constant_camera._w ) + int( u.x );
        _framebuffer_manager.set(make_float4( 1.f, 0, 0, 1 ),id);
    }
}

void debug_skydome( uint frame_number ) {
    static const int width = _cuda_renderer->_host_camera._w;
    static const int height = _cuda_renderer->_host_camera._h;
    static const dim3 block( THREAD_W, THREAD_H, 1 );
    static const dim3 grid( width / block.x, height / block.y, 1 );
    debug_skydome_kernel << < grid, block >> > ( frame_number, width, height );
}

void cuda_path_tracer( unsigned int& frame_number ) {
    static const dim3 block( THREAD_W, THREAD_H, 1 );
    static const dim3 grid( _cuda_renderer->_host_camera._w / block.x, _cuda_renderer->_host_camera._h / block.y, 1 );
    cudaFuncSetCacheConfig( path_tracer_kernel, cudaFuncCachePreferL1 );
    const size_t shmsize = THREAD_N * _cuda_renderer->_host_spec._bvh_height * sizeof( uint );
    if ( _cuda_renderer->_host_spec._debug_sky )
        debug_skydome( frame_number );
    else
        path_tracer_kernel << <grid, block, shmsize >> > ( frame_number );
    checkNoorErrors( cudaPeekAtLastError() );
    checkNoorErrors( cudaDeviceSynchronize() );
    _cuda_renderer->_host_framebuffer_manager->update();
    ++frame_number;
}



