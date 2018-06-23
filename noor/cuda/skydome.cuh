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
#ifndef CUDASKYDOME_CUH
#define CUDASKYDOME_CUH

__global__
void update_skydome( cudaSurfaceObject_t surfwrite, int width, int height ) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const float2 u = make_float2( x / float( width ), y / float( height ) );
    float sinTheta;
    const float3 dir = NOOR::sphericalDirection( u, sinTheta );
    // PBRT page 850 3rd paragraph, sinTheta factor to uniform sample directions on sphere
    const float4 color = sinTheta * make_float4( _constant_hosek_sky.querySkyModel( dir ), 1.0f );
    surf2Dwrite( color, surfwrite, x * sizeof( float4 ), y );
}

template<class T>
class CudaSkyDomeManagerTemplate {
    T _tex;
    // Cuda array containing 2d texture images
    CudaDistribution2DManager<T> _distribution2d_manager;
    SkydomeType _type;

public:
    CudaSkyDomeManagerTemplate() = default;

    __host__
        CudaSkyDomeManagerTemplate( const T& tex, SkydomeType type ) :
        _tex( tex ),
        _type( type ) {
        if ( _type == PHYSICAL ) {
            dim3 blockSize( 16, 16, 1 );
            dim3 gridSize( ( (uint) _tex.width() + blockSize.x - 1 ) / blockSize.x,
                ( (uint) _tex.height() + blockSize.y - 1 ) / blockSize.y, 1 );
            update_skydome << <gridSize, blockSize >> > ( _tex.getWriteSurfaceObj(), _tex.width(), _tex.height() );
            _tex.update();
            checkNoorErrors( cudaPeekAtLastError() );
            checkNoorErrors( cudaDeviceSynchronize() );
        }
        _distribution2d_manager = CudaDistribution2DManager<T>( _tex );
    }

    __host__
        void update() {
        if ( _type == PHYSICAL ) {
            dim3 blockSize( 32, 32, 1 );
            dim3 gridSize( ( (uint) _tex.width() + blockSize.x - 1 ) / blockSize.x,
                ( (uint) _tex.height() + blockSize.y - 1 ) / blockSize.y, 1 );
            update_skydome << <gridSize, blockSize >> > ( _tex.getWriteSurfaceObj(), _tex.width(), _tex.height() );
            _tex.update();
            checkNoorErrors( cudaPeekAtLastError() );
            checkNoorErrors( cudaDeviceSynchronize() );
        }
        _distribution2d_manager.update( _tex );
    }

    __host__
        void free() {
        _distribution2d_manager.free();
    }

    __device__
        float Pdf( const float2& u ) const {
        return _distribution2d_manager._distribution2d.Pdf( u );
    }

    __device__
        float4 evaluate( const float2& u ) const {
        return _tex.evaluate( u );
    }

    __device__
        float3 evaluate( const float3& dir, bool constant = false ) const {
        if ( constant ) return SKYDOME_COLOR;
        const float phi = NOOR::sphericalPhi( dir ) * NOOR_inv2PI;
        const float theta = NOOR::sphericalTheta( dir ) * NOOR_invPI;
        return ( _type == PHYSICAL ) ?
            _constant_hosek_sky.querySkyModel( dir ) :
            make_float3( _tex.evaluate( phi, theta ) );
    }

    __device__
        float2 importance_sample_uv( const float2& u ) const {
        float pdf;
        return _distribution2d_manager._distribution2d.sampleContinuous2D( u, pdf );
    }

    __device__
        float3 cosine_sample_dir( const CudaIntersection& I,
                                  float& pdf ) const {
        const float2 u = make_float2( I._rng(), I._rng() );
        CudaONB onb( I._shading._n, I._shading._dpdu, I._shading._dpdv );
        const float3 dir = onb.toWorld( NOOR::cosineSampleHemisphere( u, RIGHT_HANDED ) );
        pdf = dot( dir, onb._w ) * NOOR_invPI;
        return dir;
    }

    __device__
        float3 uniform_sample_dir( const CudaIntersection& I,
                                   const CudaRNG& rng,
                                   float& pdf ) const {
        const float2 u = make_float2( rng(), rng() );
        float map_pdf = NOOR::uniformHemiSpherePdf();
        pdf = map_pdf / ( 2.0f * NOOR_PI * NOOR_PI );
        CudaONB onb( I._shading._n, I._shading._dpdu, I._shading._dpdv );
        return onb.toWorld( NOOR::uniformSampleHemisphere( u ) );
    }

    __device__
        float3 importance_sample_dir( const CudaRNG& rng, float& pdf ) const {
        float2 u = make_float2( rng(), rng() );
        float map_pdf = 1.0f;
        u = _distribution2d_manager._distribution2d.sampleContinuous2D( u, map_pdf );
        float sinTheta = 0.f;
        const float3 dir = NOOR::sphericalDirection( u, sinTheta );
        // PBRT page 850 3rd paragraph, sinTheta factor to uniform sample directions on sphere
        pdf = ( sinTheta == 0 ) ? 0.0f : map_pdf / ( 2.0f * NOOR_PI * NOOR_PI * sinTheta );
        return dir;
    }
};

using CudaSkyDomeManager = CudaSkyDomeManagerTemplate<Cuda2DTexture>;
//using CudaSkyDomeManager = CudaSkyDomeManagerTemplate<CudaMipMap>;
__constant__
CudaSkyDomeManager _skydome_manager;
#endif /* CUDASKYDOME_CUH */