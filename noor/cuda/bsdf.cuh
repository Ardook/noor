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
#ifndef CUDABSDF_CUH
#define CUDABSDF_CUH
#include "bxdf.cuh"

__forceinline__ __device__
float Pdf(
    CudaBxDF* bxdf,
    const CudaIntersection& I,
    const float3 &wo,
    const float3 &wi
) {
    switch ( bxdf->_index ) {
        // reflections
        case ShadowCatcher:
            return ( (CudaShadowCatcher*) bxdf )->Pdf( I, wo, wi );
        case LambertReflection:
            return ( (CudaLambertianReflection*) bxdf )->Pdf( I, wo, wi );
        case SpecularReflectionNoOp:
        case SpecularReflectionDielectric:
            return ( (CudaSpecularReflection*) bxdf )->Pdf( I, wo, wi );
        case MicrofacetReflectionDielectric:
        case MicrofacetReflectionConductor:
            return ( (CudaMicrofacetReflection*) bxdf )->Pdf( I, wo, wi );

        // transmissions
        case LambertTransmission:
            return ( (CudaLambertianTransmission*) bxdf )->Pdf( I, wo, wi );
        case SpecularTransmission:
            return ( (CudaSpecularTransmission*) bxdf )->Pdf( I, wo, wi );
        case MicrofacetTransmission:
            return ( (CudaMicrofacetTransmission*) bxdf )->Pdf( I, wo, wi );

        // multi-lobes
        case FresnelBlend:
            return ( (CudaFresnelBlend*) bxdf )->Pdf( I, wo, wi );
        case FresnelSpecular:
            return ( (CudaFresnelSpecular*) bxdf )->Pdf( I, wo, wi );
        case ClearCoat:
            return ( (CudaClearCoat*) bxdf )->Pdf( I, wo, wi );
        default:
            return 0.f;
    }
}

__forceinline__ __device__
float3 f(
    CudaBxDF* bxdf,
    const CudaIntersection& I,
    const float3 &wo,
    const float3 &wi
) {
    switch ( bxdf->_index ) {
        // reflections
        case ShadowCatcher:
            return ( (CudaShadowCatcher*) bxdf )->f( I, wo, wi );
        case LambertReflection:
            return ( (CudaLambertianReflection*) bxdf )->f( I, wo, wi );
        case SpecularReflectionNoOp:
        case SpecularReflectionDielectric:
            return ( (CudaSpecularReflection*) bxdf )->f( I, wo, wi );
        case MicrofacetReflectionDielectric:
        case MicrofacetReflectionConductor:
            return ( (CudaMicrofacetReflection*) bxdf )->f( I, wo, wi );

        // transmissions
        case LambertTransmission:
            return ( (CudaLambertianTransmission*) bxdf )->f( I, wo, wi );
        case SpecularTransmission:
            return ( (CudaSpecularTransmission*) bxdf )->f( I, wo, wi );
        case MicrofacetTransmission:
            return ( (CudaMicrofacetTransmission*) bxdf )->f( I, wo, wi );

        // multi-lobes
        case FresnelBlend:
            return ( (CudaFresnelBlend*) bxdf )->f( I, wo, wi );
        case FresnelSpecular:
            return ( (CudaFresnelSpecular*) bxdf )->f( I, wo, wi );
        case ClearCoat:
            return ( (CudaClearCoat*) bxdf )->f( I, wo, wi );
        default:
            return _constant_spec._black;
    }
}

__forceinline__ __device__
float3 Sample_f(
    CudaBxDF* bxdf,
    const CudaIntersection& I,
    const float3 &wo,
    float3 &wi,
    const float2 &u,
    float &pdf,
    BxDFType &sampledType
) {
    switch ( bxdf->_index ) {
        // reflections
        case ShadowCatcher:
            return ( (CudaShadowCatcher*) bxdf )->Sample_f( I, wo, wi, u, pdf, sampledType );
        case LambertReflection:
            return ( (CudaLambertianReflection*) bxdf )->Sample_f( I, wo, wi, u, pdf, sampledType );
        case SpecularReflectionNoOp:
        case SpecularReflectionDielectric:
            return ( (CudaSpecularReflection*) bxdf )->Sample_f( I, wo, wi, u, pdf, sampledType );
        case MicrofacetReflectionDielectric:
        case MicrofacetReflectionConductor:
            return ( (CudaMicrofacetReflection*) bxdf )->Sample_f( I, wo, wi, u, pdf, sampledType );

        // transmissions
        case LambertTransmission:
            return ( (CudaLambertianTransmission*) bxdf )->Sample_f( I, wo, wi, u, pdf, sampledType );
        case SpecularTransmission:
            return ( (CudaSpecularTransmission*) bxdf )->Sample_f( I, wo, wi, u, pdf, sampledType );
        case MicrofacetTransmission:
            return ( (CudaMicrofacetTransmission*) bxdf )->Sample_f( I, wo, wi, u, pdf, sampledType );

        // multi-lobes
        case FresnelBlend:
            return ( (CudaFresnelBlend*) bxdf )->Sample_f( I, wo, wi, u, pdf, sampledType );
        case FresnelSpecular:
            return ( (CudaFresnelSpecular*) bxdf )->Sample_f( I, wo, wi, u, pdf, sampledType );
        case ClearCoat:
            return ( (CudaClearCoat*) bxdf )->Sample_f( I, wo, wi, u, pdf, sampledType );
        default:
            return _constant_spec._black;
    }
}

class CudaBSDF {
public:
    CudaBxDF* _bxdfs[8];
    int _nbxdfs;

    __device__
        CudaBSDF() :_nbxdfs( 0 ) {}

    __device__
        void Add( CudaBxDF *b ) {
        _bxdfs[_nbxdfs++] = b;
    }
    __device__
        int NumComponents( BxDFType flags ) const {
        int num = 0;
        for ( int i = 0; i < _nbxdfs; ++i )
            if ( _bxdfs[i]->MatchesFlags( flags ) ) ++num;
        return num;
    }

    __device__
        float Pdf(
        const CudaIntersection& I,
        const float3 &woWorld,
        const float3 &wiWorld,
        BxDFType flags
        ) const {
        if ( _nbxdfs == 0.f ) return 0.f;
        const CudaONB onb( I._shading._n, I._shading._dpdu, I._shading._dpdv );
        const float3 wo = onb.toLocal( woWorld );
        const float3 wi = onb.toLocal( wiWorld );
        if ( wo.z == 0.0f ) { return 0.0f; }

        float pdf = 0.f;
        int matchingComps = 0;
        for ( int i = 0; i < _nbxdfs; ++i )
            if ( _bxdfs[i]->MatchesFlags( flags ) ) {
                ++matchingComps;
                pdf += ::Pdf( _bxdfs[i], I, wo, wi );
            }
        return matchingComps > 0 ? pdf / matchingComps : 0.f;
    }

    __device__
        float3 f(
        const CudaIntersection& I,
        const float3 &woWorld,
        const float3 &wiWorld,
        BxDFType flags
        ) const {
        const CudaONB onb( I._shading._n, I._shading._dpdu, I._shading._dpdv );
        const float3 wo = onb.toLocal( woWorld );
        const float3 wi = onb.toLocal( wiWorld );
        const bool reflect = dot( wiWorld, I._n ) * dot( woWorld, I._n ) > 0;
        float3 f = _constant_spec._black;
        for ( int i = 0; i < _nbxdfs; ++i )
            if ( _bxdfs[i]->MatchesFlags( flags ) &&
                ( ( reflect && ( _bxdfs[i]->_type & BSDF_REFLECTION ) ) ||
                 ( !reflect && ( _bxdfs[i]->_type & BSDF_TRANSMISSION ) ) ) ) {
                f += ::f( _bxdfs[i], I, wo, wi );
            }
        return f;
    }

    __device__
        float3 Sample_f(
        const CudaIntersection& I,
        const float3 &woWorld,
        float3 &wiWorld,
        const float2 &u,
        float &pdf,
        BxDFType type,
        BxDFType &sampledType
        ) const {
        const int matchingComps = NumComponents( type );
        if ( matchingComps == 0 ) {
            pdf = 0;
            sampledType = BxDFType( 0 );
            return _constant_spec._black;
        }
        const int comp = min( (int) floorf( u.x * matchingComps ), matchingComps - 1 );
        const float2 uRemapped = make_float2( fminf( u.x * matchingComps - comp, NOOR_ONE_MINUS_EPSILON ), u.y );
        CudaBxDF *bxdf = nullptr;
        int count = comp;
        for ( int i = 0; i < _nbxdfs; ++i ) {
            if ( _bxdfs[i]->MatchesFlags( type ) && count-- == 0 ) {
                bxdf = _bxdfs[i];
                break;
            }
        }

        const CudaONB onb( I._shading._n, I._shading._dpdu, I._shading._dpdv );
        const float3 wo = onb.toLocal( woWorld );
        if ( wo.z == 0.0f ) {
            return _constant_spec._black;
        }
        pdf = 0;
        sampledType = bxdf->_type;
        float3 wi;
        float3 f = ::Sample_f( bxdf, I, wo, wi, uRemapped, pdf, sampledType );
        if ( pdf == 0 ) {
            sampledType = BxDFType( 0 );
            return _constant_spec._black;
        }
        wiWorld = onb.toWorld( wi );

        // Compute overall PDF with all matching _BxDF_s
        if ( !( bxdf->_type & BSDF_SPECULAR ) && matchingComps > 1 )
            for ( int i = 0; i < _nbxdfs; ++i )
                if ( _bxdfs[i] != bxdf && _bxdfs[i]->MatchesFlags( type ) )
                    pdf += ::Pdf( _bxdfs[i], I, wo, wi );
        if ( matchingComps > 1 ) pdf /= matchingComps;

        // Compute value of BSDF for sampled direction
        if ( !( bxdf->_type & BSDF_SPECULAR ) ) {
            bool reflect = dot( wiWorld, I._n ) * dot( woWorld, I._n ) > 0;
            //f = _constant_spec._black;
            for ( int i = 0; i < _nbxdfs; ++i )
                if ( _bxdfs[i] != bxdf && _bxdfs[i]->MatchesFlags( type ) &&
                     ( ( reflect && ( _bxdfs[i]->_type & BSDF_REFLECTION ) ) ||
                     ( !reflect && ( _bxdfs[i]->_type & BSDF_TRANSMISSION ) ) ) )
                    f += ::f( _bxdfs[i], I, wo, wi );
        }
        return f;
    }
};

__forceinline__ __device__
void factoryClearCoatBSDF( const CudaIntersection& I, CudaBSDF& bsdf ) {
    bsdf.Add( _bxdf_manager._bxdfs[ClearCoat] );
}

__forceinline__ __device__
void factoryShadowCatcherBSDF( const CudaIntersection& I, CudaBSDF& bsdf ) {
    bsdf.Add( _bxdf_manager._bxdfs[ShadowCatcher] );
}
__forceinline__ __device__
void factoryDiffuseBSDF( const CudaIntersection& I, CudaBSDF& bsdf ) {
    bsdf.Add( _bxdf_manager._bxdfs[LambertReflection] );
}
__forceinline__ __device__
void factoryMirrorBSDF( const CudaIntersection& I, CudaBSDF& bsdf ) {
    bsdf.Add( _bxdf_manager._bxdfs[SpecularReflectionNoOp] );
}
__forceinline__ __device__
void factoryMetalBSDF( const CudaIntersection& I, CudaBSDF& bsdf ) {
    bsdf.Add( _bxdf_manager._bxdfs[MicrofacetReflectionConductor] );
}
__forceinline__ __device__
void factoryRoughGlassBSDF( const CudaIntersection& I, CudaBSDF& bsdf ) {
   bsdf.Add( _bxdf_manager._bxdfs[MicrofacetReflectionDielectric] );
   bsdf.Add( _bxdf_manager._bxdfs[MicrofacetTransmission] );
}
__forceinline__ __device__
void factoryGlossyBSDF( const CudaIntersection& I, CudaBSDF& bsdf ) {
    bsdf.Add( _bxdf_manager._bxdfs[LambertReflection] );
    bsdf.Add( _bxdf_manager._bxdfs[MicrofacetReflectionDielectric] );
}
__forceinline__ __device__
void factorySubstrateBSDF( const CudaIntersection& I, CudaBSDF& bsdf ) {
    bsdf.Add( _bxdf_manager._bxdfs[FresnelBlend] );
}
__forceinline__ __device__
void factoryGlassBSDF( const CudaIntersection& I, CudaBSDF& bsdf ) {
    bsdf.Add( _bxdf_manager._bxdfs[FresnelSpecular] );
}

__forceinline__ __device__
void factoryTranslucentBSDF( const CudaIntersection& I, CudaBSDF& bsdf ) {
    bsdf.Add( _bxdf_manager._bxdfs[LambertReflection] );
    bsdf.Add( _bxdf_manager._bxdfs[LambertTransmission] );
}


__forceinline__ __device__
void factoryBSDF( const CudaIntersection& I, CudaBSDF& bsdf ) {
    switch ( I.getMaterialType() ) {
        case SHADOWCATCHER:
            factoryShadowCatcherBSDF( I, bsdf );
            break;
        case DIFFUSE:
            factoryDiffuseBSDF( I, bsdf );
            break;
        case TRANSLUCENT:
            factoryTranslucentBSDF( I, bsdf );
            break;
        case GLASS:
            factoryGlassBSDF( I, bsdf );
            break;
        case GLOSSY:
            factoryGlossyBSDF( I, bsdf );
            break;
        case METAL:
            factoryMetalBSDF( I, bsdf );
            break;
        case ROUGHGLASS:
            factoryRoughGlassBSDF( I, bsdf );
            break;
        case MIRROR:
            factoryMirrorBSDF( I, bsdf );
            break;
        case SUBSTRATE:
            factorySubstrateBSDF( I, bsdf );
            break;
        case CLEARCOAT:
            factoryClearCoatBSDF( I, bsdf );
            break;
        default:
            break;
    }
}
#endif /* CUDABSDF_CUH */
