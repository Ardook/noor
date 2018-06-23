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
bool rr_terminate( const CudaIntersection& I, int bounce, float3& beta ) {
    // Russian Roulette 
    if ( bounce >= _constant_spec._rr ) {
        const float q = fmaxf( 0.05f, 1.0f - NOOR::maxcomp( beta ) );
        if ( I._rng() < q ) {
            return true;
        }
        beta /= 1.0f - q;
    }
    return false;
}

__forceinline__ __device__
bool correct_sidedness( CudaIntersection& I ) {
    return ( !I.isTransparent() && ( dot( I.getWi(), I.getGn() ) < 0 ) ) ?
        false :
        true;
}

/* Based on PBRT bump mapping */
__forceinline__ __device__
void bump( const CudaIntersection& I, ShadingFrame& frame ) {
    float du = .5f * ( fabsf( I._differential._dudx ) + fabsf( I._differential._dudy ) );
    float dv = .5f * ( fabsf( I._differential._dvdx ) + fabsf( I._differential._dvdy ) );

    du = du == 0 ? 0.001f : du;
    dv = dv == 0 ? 0.001f : dv;

    const float displace = _material_manager.getBump( I );
    const float scale = _material_manager.getBumpFactor( I );
    const float uDisplace = scale * ( _material_manager.getBump( I, du, 0 ) - displace ) / du;
    const float vDisplace = scale * ( _material_manager.getBump( I, 0, dv ) - displace ) / dv;
    // Compute bump-mapped differential geometry
    frame._dpdu += uDisplace* I.getSn();
    frame._dpdv += vDisplace* I.getSn();
    frame._n = NOOR::normalize( cross( frame._dpdu, frame._dpdv ) );

    frame._dpdu = NOOR::normalize( frame._dpdu - frame._n*dot( frame._n, frame._dpdu ) );
    // const float sign = dot( cross( n, I.getSdpdu() ), I.getSdpdv() ) < 0.0f ? -1.0f : 1.0f;
    frame._dpdv = NOOR::normalize( cross( frame._n, frame._dpdu ) );
    frame._n = NOOR::faceforward( frame._n, I.getGn() );
}

/* Based on PBRT bump mapping */
__forceinline__ __device__
void bump( CudaIntersection& I ) {
    bump( I, I.getShadingFrame() );
}

__forceinline__ __device__
void accumulate(
    CudaIntersection& I,
    const CudaRay& ray,
    float3& beta,
    float3& L
) {
    if ( ray.isDifferential() && I.isBumped() ) {
        bump( I );
    }
    CudaBSDF bsdf;
    factoryBSDF( I, bsdf );
    L += ( _constant_spec.is_mis_enabled() ) ?
        beta * directMIS( bsdf, I ) :
        beta * direct( bsdf, I );
    beta *= scatter( bsdf, I );
}



#endif /* ACCUMULATE_CUH */