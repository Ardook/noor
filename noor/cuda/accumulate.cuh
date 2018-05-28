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
#ifndef ACCUMULATE_CUH
#define ACCUMULATE_CUH

__forceinline__ __device__
void accumulateMIS(
    CudaIntersection& I
    , const CudaRNG& rng
    , float3& wi
    , float3& beta
    , float3& L
) {
    if ( I._material_type & DIFFUSE ) {
        const CudaDiffuseBSDF bsdf = factoryDiffuseBSDF( I );
        L += beta * directMIS( bsdf, I, rng );
        beta *= scatter( bsdf, I, rng, wi );
    } else if ( I._material_type & ORENNAYAR ) {
        const CudaOrenNayarBSDF bsdf = factoryOrenNayarBSDF( I );
        L += beta * directMIS( bsdf, I, rng );
        beta *= scatter( bsdf, I, rng, wi );
    } else if ( I._material_type & TRANSLUCENT ) {
        const CudaTranslucentBSDF bsdf = factoryTranslucentBSDF( I );
        L += beta * directMIS( bsdf, I, rng );
        beta *= scatter( bsdf, I, rng, wi );
    } else if ( I._material_type & GLASS ) {
        const CudaGlassBSDF bsdf = factoryGlassBSDF( I );
        L += beta * directMIS( bsdf, I, rng );
        beta *= scatter( bsdf, I, rng, wi );
    } else if ( I._material_type & GLOSSY ) {
        const CudaGlossyBSDF bsdf = factoryGlossyBSDF( I );
        L += beta * directMIS( bsdf, I, rng );
        beta *= scatter( bsdf, I, rng, wi );
    } else if ( I._material_type & METAL ) {
        const CudaMetalBSDF bsdf = factoryMetalBSDF( I );
        L += beta * directMIS( bsdf, I, rng );
        beta *= scatter( bsdf, I, rng, wi );
    } else if ( I._material_type & ROUGHGLASS ) {
        const CudaRoughGlassBSDF bsdf = factoryRoughGlassBSDF( I );
        L += beta * directMIS( bsdf, I, rng );
        beta *= scatter( bsdf, I, rng, wi );
    } else if ( I._material_type & MIRROR ) {
        const CudaSpecularReflectionBSDF bsdf = factoryMirrorBSDF( I );
        L += beta * directMIS( bsdf, I, rng );
        beta *= scatter( bsdf, I, rng, wi );
    } else if ( I._material_type & SUBSTRATE ) {
        const CudaFresnelBlendBSDF bsdf = factoryFresnelBlendBSDF( I );
        L += beta * directMIS( bsdf, I, rng );
        beta *= scatter( bsdf, I, rng, wi );
    } else if ( I._material_type & SMOOTHCOATING ) {
        const CudaCoatingBSDF bsdf = factorySmoothCoatingBSDF( I );
        L += beta * directMIS( bsdf, I, rng );
        beta *= scatter( bsdf, I, rng, wi );
    } else if ( I._material_type == SHADOW ) {
        const CudaShadowCatcherBSDF bsdf = factoryShadowCatcherBSDF( I );
        L += beta * directMIS( bsdf, I, rng );
        beta *= scatter( bsdf, I, rng, wi );
    }
}

__forceinline__ __device__
void accumulate(
    CudaIntersection& I
    , const CudaRNG& rng
    , float3& wi
    , float3& beta
    , float3& L
) {
    if ( I._material_type & DIFFUSE ) {
        const CudaDiffuseBSDF bsdf = factoryDiffuseBSDF( I );
        L += beta * direct( bsdf, I, rng );
        beta *= scatter( bsdf, I, rng, wi );
    } else if ( I._material_type & ORENNAYAR ) {
        const CudaOrenNayarBSDF bsdf = factoryOrenNayarBSDF( I );
        L += beta * direct( bsdf, I, rng );
        beta *= scatter( bsdf, I, rng, wi );
    } else if ( I._material_type & TRANSLUCENT ) {
        const CudaTranslucentBSDF bsdf = factoryTranslucentBSDF( I );
        L += beta * direct( bsdf, I, rng );
        beta *= scatter( bsdf, I, rng, wi );
    } else if ( I._material_type & GLASS ) {
        const CudaGlassBSDF bsdf = factoryGlassBSDF( I );
        L += beta * direct( bsdf, I, rng );
        beta *= scatter( bsdf, I, rng, wi );
    } else if ( I._material_type & GLOSSY ) {
        const CudaGlossyBSDF bsdf = factoryGlossyBSDF( I );
        L += beta * direct( bsdf, I, rng );
        beta *= scatter( bsdf, I, rng, wi );
    } else if ( I._material_type & METAL ) {
        const CudaMetalBSDF bsdf = factoryMetalBSDF( I );
        L += beta * direct( bsdf, I, rng );
        beta *= scatter( bsdf, I, rng, wi );
    } else if ( I._material_type & ROUGHGLASS ) {
        const CudaRoughGlassBSDF bsdf = factoryRoughGlassBSDF( I );
        L += beta * direct( bsdf, I, rng );
        beta *= scatter( bsdf, I, rng, wi );
    } else if ( I._material_type & MIRROR ) {
        const CudaSpecularReflectionBSDF bsdf = factoryMirrorBSDF( I );
        L += beta * direct( bsdf, I, rng );
        beta *= scatter( bsdf, I, rng, wi );
    } else if ( I._material_type & SUBSTRATE ) {
        const CudaFresnelBlendBSDF bsdf = factoryFresnelBlendBSDF( I );
        L += beta * direct( bsdf, I, rng );
        beta *= scatter( bsdf, I, rng, wi );
    } else if ( I._material_type & SMOOTHCOATING ) {
        const CudaCoatingBSDF bsdf = factorySmoothCoatingBSDF( I );
        L += beta * direct( bsdf, I, rng );
        beta *= scatter( bsdf, I, rng, wi );
    } else if ( I._material_type == SHADOW ) {
        const CudaShadowCatcherBSDF bsdf = factoryShadowCatcherBSDF( I );
        L += beta * direct( bsdf, I, rng );
        beta *= scatter( bsdf, I, rng, wi );
    }
}

__forceinline__ __device__
void bump( CudaIntersection& I ) {
    float du = .5f * ( fabsf( I._differential._dudx ) + fabsf( I._differential._dudy ) );
    float dv = .5f * ( fabsf( I._differential._dvdx ) + fabsf( I._differential._dvdy ) );

    du = du == 0 ? 0.001f : du;
    dv = dv == 0 ? 0.001f : dv;

    const float displace = _material_manager.getBump( I );
    const float scale = _material_manager.getBumpFactor( I );
    const float uDisplace = scale * ( _material_manager.getBump( I, du, 0 ) - displace ) / du;
    const float vDisplace = scale * ( _material_manager.getBump( I, 0, dv ) - displace ) / dv;
    // Compute bump-mapped differential geometry
    I._shading._dpdu += uDisplace* I._shading._n;
    I._shading._dpdv += vDisplace* I._shading._n;
    I._shading._n = NOOR::normalize( cross( I._shading._dpdu, I._shading._dpdv ) );

    I._shading._dpdu = NOOR::normalize( I._shading._dpdu - I._shading._n*dot( I._shading._n, I._shading._dpdu ) );
   // const float sign = dot( cross( n, I._shading._dpdu ), I._shading._dpdv ) < 0.0f ? -1.0f : 1.0f;
    I._shading._dpdv = NOOR::normalize( cross( I._shading._n, I._shading._dpdu ) );
    I._shading._n = NOOR::faceforward( I._shading._n, I._n );
}

__forceinline__ __device__
void accumulate(
    CudaIntersection& I,
    const CudaRNG& rng,
    float3& wi,
    float3& beta,
    float3& L,
    bool MIS
) {
  
    if ( MIS )
        accumulateMIS( I, rng, wi, beta, L );
    else
        accumulate( I, rng, wi, beta, L );
}

#endif /* ACCUMULATE_CUH */