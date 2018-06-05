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
#ifndef DIRECT_CUH
#define DIRECT_CUH

__forceinline__ __device__
//bool occluded( const CudaIntersection& I, const CudaVisibility& v, int* light_idx = nullptr ) {
bool occluded( const CudaIntersection& I, const CudaVisibility& v ) {
    return ( intersectP( I.spawnShadowRay( v ), I ) );
}

__forceinline__ __device__
float3 direct(
    const CudaBSDF& bsdf
    , const CudaIntersection& I
) {
    if ( I.isSpecularBounce() )  return _constant_spec._black;
    BxDFType bsdf_flags = I.isSpecularBounce() ? BSDF_ALL : BxDFType( BSDF_ALL & ~BSDF_SPECULAR );
    float light_pdf;
    CudaLightRecord Lr;
    const float3 Li = _light_manager.sample_Li( I, Lr, light_pdf );
    if ( light_pdf == 0 || NOOR::isBlack( Li ) ) {
        return _constant_spec._black;
    }

    float3 f = bsdf.f( I, I._wo, Lr._vis._wi, bsdf_flags );
    if ( NOOR::isBlack( f ) ) {
        return _constant_spec._black;
    }
    f *= NOOR::absDot( Lr._vis._wi, I._shading._n );
    if ( !occluded( I, Lr._vis ) ) {
        if ( !I.isShadowCatcher() ) {
            return Li * f / light_pdf;
        } else {
            return f;
        }
    }
    return _constant_spec._black;
}

__forceinline__ __device__
float3 sampleLight( const CudaBSDF& bsdf,
                    const CudaIntersection& I,
                    int& light_idx
) {
    BxDFType bsdf_flags = I.isSpecularBounce() ? BSDF_ALL : BxDFType( BSDF_ALL & ~BSDF_SPECULAR );
    CudaLightRecord Lr;
    float3 Ld = _constant_spec._black;
    float light_pdf = 0.f;
    const float3 Li = _light_manager.sample_Li( I, Lr, light_pdf );
    if ( light_pdf == 0 || NOOR::isBlack( Li ) ) {
        return _constant_spec._black;
    }
    light_idx = Lr._light_idx;
    float3 f = bsdf.f( I, I._wo, Lr._vis._wi, bsdf_flags );
    if ( NOOR::isBlack( f ) ) return _constant_spec._black;
    f *= NOOR::absDot( Lr._vis._wi, I._shading._n );
    if ( !occluded( I, Lr._vis ) ) {
        if ( !I.isShadowCatcher() ) {
            if ( _light_manager.isDeltaLight( light_idx ) ) {
                Ld += Li * f / light_pdf;
            } else {
                const float scatter_pdf = bsdf.Pdf( I, I._wo, Lr._vis._wi, bsdf_flags );
                const float light_weight = NOOR::powerHeuristic( 1.f, light_pdf, 1.f, scatter_pdf );
                Ld += Li * f * light_weight / light_pdf;
            }
        } else {
            Ld += f;
        }
    }
    return Ld;
}

__forceinline__ __device__
float3 sampleBSDF( const CudaBSDF& bsdf,
                   const CudaIntersection& I,
                   int light_idx
) {
    BxDFType bsdf_flags = I.isSpecularBounce() ? BSDF_ALL : BxDFType( BSDF_ALL & ~BSDF_SPECULAR );
    float3 Ld = _constant_spec._black;
    if ( _light_manager.isDeltaLight( light_idx ) ) return Ld;
    BxDFType sampled_type;
    const float2 u = make_float2( I._rng(), I._rng() );
    float scatter_pdf, light_pdf;
    float3 wi;
    float3 f = bsdf.Sample_f( I, I._wo, wi, u, scatter_pdf, bsdf_flags, sampled_type );
    if ( NOOR::isBlack( f ) || scatter_pdf == 0 ) return Ld;
    f *= NOOR::absDot( wi, I._shading._n );
    float sampledSpecular = ( sampled_type & BSDF_SPECULAR ) != 0;
    float scatter_weight = 1.f;
    if ( !sampledSpecular ) {
        light_pdf = _light_manager.pdf_Li( I, wi, light_idx );
        if ( light_pdf == 0 ) {
            return Ld;
        }
        scatter_weight = NOOR::powerHeuristic( 1.f, scatter_pdf, 1.f, light_pdf );
    }
    const CudaRay shadow_ray = I.spawnShadowRay( wi, 2.f*_constant_spec._world_radius );
    if ( _light_manager.intersect( shadow_ray, light_idx ) ) {
        const CudaVisibility vis( I._p, shadow_ray.pointAtParameter( shadow_ray.getTmax() ) );
        if ( !occluded( I, vis ) ) {
            Ld += _light_manager.Le( wi, light_idx ) * f * scatter_weight / scatter_pdf;
        } 
    } else if ( _constant_spec.is_sky_light_enabled() ) {
        Ld += _skydome_manager.evaluate( wi ) * f * scatter_weight / scatter_pdf;
    }
    return Ld;
}

__forceinline__ __device__
float3 directMIS(
    const CudaBSDF& bsdf
    , const CudaIntersection& I
) {
    int light_idx;
    float3 Ld = sampleLight( bsdf, I, light_idx );
    if ( I._material_type != SHADOW )
        Ld += sampleBSDF( bsdf, I, light_idx );
    return Ld;
}

#endif /* DIRECT_CUH */