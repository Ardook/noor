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
void bump( const CudaIntersection& I, CudaIntersection::ShadingFrame& frame ) {
    float du = .5f * ( fabsf( I._differential._dudx ) + fabsf( I._differential._dudy ) );
    float dv = .5f * ( fabsf( I._differential._dvdx ) + fabsf( I._differential._dvdy ) );

    du = du == 0 ? 0.001f : du;
    dv = dv == 0 ? 0.001f : dv;

    const float displace = _material_manager.getBump( I );
    const float scale = _material_manager.getBumpFactor( I );
    const float uDisplace = scale * ( _material_manager.getBump( I, du, 0 ) - displace ) / du;
    const float vDisplace = scale * ( _material_manager.getBump( I, 0, dv ) - displace ) / dv;
    // Compute bump-mapped differential geometry
    frame._dpdu += uDisplace* I._shading._n;
    frame._dpdv += vDisplace* I._shading._n;
    frame._n = NOOR::normalize( cross( frame._dpdu, frame._dpdv ) );

    frame._dpdu = NOOR::normalize( frame._dpdu - frame._n*dot( frame._n, frame._dpdu ) );
    //const float sign = dot( cross( frame._n, frame._dpdu ), frame._dpdv ) < 0.0f ? -1.0f : 1.0f;
    frame._dpdv = NOOR::normalize( cross( frame._n, frame._dpdu ) );
    frame._n = NOOR::faceforward( frame._n, I._n );
}

class CudaShadowCatcherBSDF {
    int _num_components{ 1 };
public:
    __device__
        float Pdf(
        const CudaIntersection& I,
        const float3 &wo,
        const float3 &wi,
        BxDFType flags
        ) const {
        return 1.f;
    }
    __device__
        float3 f(
        const CudaIntersection& I,
        const float3 &wo,
        const float3 &wi,
        BxDFType flags
        ) const {
        return _skydome_manager.evaluate( -1.f*wo, false ) / NOOR::absDot( wi, I._shading._n );
    }
    __device__
        float3 Sample_f(
        const CudaIntersection& I,
        const float3 &wo,
        float3 &wi,
        const float2 &u,
        float &pdf,
        BxDFType type,
        BxDFType &sampledType
        ) const {
        pdf = 1.f;
        const CudaONB onb( I._shading._n, I._shading._dpdu, I._shading._dpdv );
        wi = NOOR::uniformSampleHemisphere( u );
        wi = onb.toWorld( wi );
        return f( I, wo, wi, type );
    }
};

template <typename...> class CudaBSDF;
template<typename BXDF>
class CudaBSDF<BXDF> {
    BXDF _bxdf;
    int _num_components{ 1 };
public:
    __device__
        CudaBSDF( const BXDF& bxdf ) :
        _bxdf( bxdf ) {}
    __device__
        int NumComponents( BxDFType flags ) const {
        return ( _bxdf.MatchesFlags( flags ) ) ? 1 : 0;
    }
    __device__
        float Pdf(
        const CudaIntersection& I,
        const float3 &woWorld,
        const float3 &wiWorld,
        BxDFType flags
        ) const {
        if ( !_bxdf.MatchesFlags( flags ) ) return 0.0f;
        const CudaONB onb( I._shading._n, I._shading._dpdu, I._shading._dpdv );
        const float3 wo = onb.toLocal( woWorld );
        const float3 wi = onb.toLocal( wiWorld );
        if ( wo.z == 0.0f ) return 0.0f;
        return _bxdf.Pdf( wo, wi );
    }
    __device__
        float3 f(
        const CudaIntersection& I,
        const float3 &woWorld,
        const float3 &wiWorld,
        BxDFType flags
        ) const {
        if ( !_bxdf.MatchesFlags( flags ) ) return _constant_spec._black;
        const CudaONB onb( I._shading._n, I._shading._dpdu, I._shading._dpdv );
        const float3 wo = onb.toLocal( woWorld );
        const float3 wi = onb.toLocal( wiWorld );
        if ( wo.z == 0.0f ) {
            return _constant_spec._black;
        }
        const bool reflect = dot( wiWorld, I._n ) * dot( woWorld, I._n ) > 0.f;
        const bool mask = ( reflect && ( _bxdf.getType() & BSDF_REFLECTION ) ) ||
            ( !reflect && ( _bxdf.getType() & BSDF_TRANSMISSION ) );
        return mask ? _bxdf.f( wo, wi ) : _constant_spec._black;
    }
    __device__
        float3 Sample_f(
        const CudaIntersection& I,
        const float3 &woWorld,
        float3 &wi,
        const float2 &u,
        float &pdf,
        BxDFType type,
        BxDFType &sampledType
        ) const {
        if ( !_bxdf.MatchesFlags( type ) ) {
            pdf = 0;
            sampledType = BxDFType( 0 );
            return _constant_spec._black;
        }
        const CudaONB onb( I._shading._n, I._shading._dpdu, I._shading._dpdv );
        const float3 wo = onb.toLocal( woWorld );
        if ( wo.z == 0.0f ) {
            pdf = 0.0f;
            return _constant_spec._black;
        }
        const float3 f = _bxdf.Sample_f( wo, wi, u, pdf, sampledType );
        if ( pdf == 0.0f ) {
            sampledType = BxDFType( 0 );
            return _constant_spec._black;
        }
        wi = onb.toWorld( wi );
        return f;
    }
};

template<typename BXDF0, typename BXDF1>
class CudaBSDF<BXDF0, BXDF1> {
    BXDF0 _bxdf0;
    BXDF1 _bxdf1;
    int _num_components{ 2 };
public:
    __device__
        CudaBSDF( const BXDF0& bxdf0, const BXDF1& bxdf1 ) :
        _bxdf0( bxdf0 ),
        _bxdf1( bxdf1 ) {}
    __device__
        int NumComponents( BxDFType flags ) const {
        return ( _bxdf0.MatchesFlags( flags ) + _bxdf1.MatchesFlags( flags ) );
    }
    __device__
        float Pdf(
        const CudaIntersection& I,
        const float3 &woWorld,
        const float3 &wiWorld,
        BxDFType flags
        ) const {
        const CudaONB onb( I._shading._n, I._shading._dpdu, I._shading._dpdv );
        const float3 wo = onb.toLocal( woWorld );
        const float3 wi = onb.toLocal( wiWorld );
        if ( wo.z == 0.0f ) { return 0.0f; }
        int matchingComps = 0;
        float pdf = 0.0f;
        if ( _bxdf0.MatchesFlags( flags ) ) {
            ++matchingComps;
            pdf += _bxdf0.Pdf( wo, wi );
        }
        if ( _bxdf1.MatchesFlags( flags ) ) {
            ++matchingComps;
            pdf += _bxdf1.Pdf( wo, wi );
        }
        return matchingComps > 0 ? pdf / matchingComps : 0.0f;
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

        const bool mask0 = _bxdf0.MatchesFlags( flags )
            && ( ( reflect && ( _bxdf0.getType() & BSDF_REFLECTION ) )
                 || ( !reflect && ( _bxdf0.getType() & BSDF_TRANSMISSION ) ) );
        f += mask0 ? _bxdf0.f( wo, wi ) : _constant_spec._black;

        const bool mask1 = _bxdf1.MatchesFlags( flags )
            && ( ( reflect && ( _bxdf1.getType() & BSDF_REFLECTION ) )
                 || ( !reflect && ( _bxdf1.getType() & BSDF_TRANSMISSION ) ) );
        f += mask1 ? _bxdf1.f( wo, wi ) : _constant_spec._black;

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
        const int comp = min( (int) floorf( u.x * 2.f ), 1 );
        const float2 uRemapped = make_float2( fminf( u.x * 2.f - comp, NOOR_ONE_MINUS_EPSILON ), u.y );
        const CudaONB onb( I._shading._n, I._shading._dpdu, I._shading._dpdv );
        const float3 wo = onb.toLocal( woWorld );
        if ( wo.z == 0.0f ) {
            return _constant_spec._black;
        }
        pdf = 0.0f;
        float3 f;
        float3 wi;

        if ( comp == 0 && _bxdf0.MatchesFlags( type ) ) {
            f = _bxdf0.Sample_f( wo, wi, uRemapped, pdf, sampledType );
            const bool f0 = _bxdf0.notSpecular() && _bxdf1.MatchesFlags( type );
            pdf += f0 ? _bxdf1.Pdf( wo, wi ) : 0.0f;
        } else if ( comp == 1 && _bxdf1.MatchesFlags( type ) ) {
            f = _bxdf1.Sample_f( wo, wi, uRemapped, pdf, sampledType );
            const bool f1 = _bxdf1.notSpecular() && _bxdf0.MatchesFlags( type );
            pdf += f1 ? _bxdf0.Pdf( wo, wi ) : 0.0f;
        }
        if ( pdf == 0.0f ) {
            sampledType = BxDFType( 0 );
            return _constant_spec._black;
        }
        wiWorld = onb.toWorld( wi );
        pdf /= matchingComps;

        const bool reflect = dot( wiWorld, I._n ) * dot( woWorld, I._n ) >= 0;
        const bool f0 = ( comp == 1 && _bxdf1.notSpecular() && _bxdf0.MatchesFlags( type ) )
            && ( ( reflect && ( _bxdf0.getType() & BSDF_REFLECTION ) )
                 || ( !reflect && ( _bxdf0.getType() & BSDF_TRANSMISSION ) ) );
        f += f0 ? _bxdf0.f( wo, wi ) : _constant_spec._black;

        const bool f1 = ( comp == 0 && _bxdf0.notSpecular() && _bxdf1.MatchesFlags( type ) )
            && ( ( reflect && ( _bxdf1.getType() & BSDF_REFLECTION ) )
                 || ( !reflect && ( _bxdf1.getType() & BSDF_TRANSMISSION ) ) );
        f += f1 ? _bxdf1.f( wo, wi ) : _constant_spec._black;
        return f;
    }
};

/* Based on Mitsuba Renderer SmoothCoating */
template<typename SubstrateBXDF>
class CudaSmoothCoatingBSDF {
    SubstrateBXDF _substrate;
    CudaFresnelDielectric _fresnel;
    float3 _S;
    float3 _coatingSigma;
    float _coatingThickness;
    float _coatingWeight;
    float _eta;
    float _invEta;
public:
    __device__
        CudaSmoothCoatingBSDF(
        const SubstrateBXDF& substrate,
        const CudaFresnelDielectric& fresnel,
        const float3& S,
        const float3& coatingSigma,
        float coatingThickness,
        float coatingWeight
        ) :
        _substrate( substrate ),
        _fresnel( fresnel ),
        _S( S ),
        _coatingSigma( coatingSigma ),
        _coatingThickness( coatingThickness ),
        _coatingWeight( coatingWeight ),
        _eta( _fresnel._etaT / _fresnel._etaI ),
        _invEta( 1.f / _eta ) {}
    __device__
        float fresnelDielectricExt( float cosThetaI_, float &cosThetaT_, float eta ) const {
            /* Using Snell's law, calculate the squared sine of the
            angle between the normal and the transmitted ray */
        float scale = ( cosThetaI_ > 0 ) ? 1 / eta : eta,
            cosThetaTSqr = 1 - ( 1 - cosThetaI_*cosThetaI_ ) * ( scale*scale );

        /* Check for total internal reflection */
        if ( cosThetaTSqr <= 0.0f ) {
            cosThetaT_ = 0.0f;
            return 1.0f;
        }

        /* Find the absolute cosines of the incident/transmitted rays */
        float cosThetaI = fabsf( cosThetaI_ );
        float cosThetaT = sqrtf( cosThetaTSqr );

        float Rs = ( cosThetaI - eta * cosThetaT )
            / ( cosThetaI + eta * cosThetaT );
        float Rp = ( eta * cosThetaI - cosThetaT )
            / ( eta * cosThetaI + cosThetaT );

        cosThetaT_ = ( cosThetaI_ > 0 ) ? -cosThetaT : cosThetaT;

        /* No polarization -- return the unpolarized reflectance */
        return 0.5f * ( Rs * Rs + Rp * Rp );
    }
    /// Refract into the material, preserve sign of direction
    __device__
        float3 refractIn( const float3 &wi, float &R ) const {
        float cosThetaT;
        R = fresnelDielectricExt( fabsf( CosTheta( wi ) ), cosThetaT, _eta );
        return make_float3( _invEta*wi.x, _invEta*wi.y, -NOOR::sign( CosTheta( wi ) ) * cosThetaT );
    }

    /// Refract out of the material, preserve sign of direction
    __device__
        float3 refractOut( const float3 &wi, float &R ) const {
        float cosThetaT;
        R = fresnelDielectricExt( fabsf( CosTheta( wi ) ), cosThetaT, _invEta );
        return make_float3( _eta*wi.x, _eta*wi.y, -NOOR::sign( CosTheta( wi ) ) * cosThetaT );
    }
    __device__
        float3 f(
        const CudaIntersection& I,
        const float3 &woWorld,
        const float3 &wiWorld,
        BxDFType flags
        ) const {
        bool sampleSpecular = ( flags & BSDF_SPECULAR );
        bool sampleSubstrate = ( flags & _substrate.getType() );

        const CudaONB onb( I._shading._n, I._shading._dpdu, I._shading._dpdv );
        const float3 wo = onb.toLocal( woWorld );
        const float3 wi = onb.toLocal( wiWorld );
        if ( !SameHemisphere( wo, wi ) ) return _constant_spec._black;

        float3 result = _constant_spec._black;
        float F1;
        float3 wo_c = refractIn( wo, F1 );
        if ( sampleSpecular ) {
            result = _constant_spec._black;
        }
        if ( sampleSubstrate ) {
            float F0;
            float3 wi_c = refractIn( wi, F0 );
            if ( F0 == 1.f || F1 == 1.f ) return result;
            result += _substrate.f( wo_c, wi_c )*( 1.f - F0 )*( 1.f - F1 );
            float3 sigmaA = _coatingSigma * _coatingThickness;
            if ( !NOOR::isBlack( sigmaA ) ) {
                result *= NOOR::exp3f( -sigmaA * ( 1.f / AbsCosTheta( wo_c ) + 1.f / AbsCosTheta( wi_c ) ) );
            }
            result /= AbsCosTheta( wi_c );
        }
        return result;
    }

    __device__
        float Pdf(
        const CudaIntersection& I,
        const float3 &woWorld,
        const float3 &wiWorld,
        BxDFType flags
        ) const {
        bool sampleSpecular = ( flags & BSDF_SPECULAR );
        bool sampleSubstrate = ( flags & _substrate.getType() & BSDF_ALL );
        const CudaONB onb( I._shading._n, I._shading._dpdu, I._shading._dpdv );
        const float3 wo = onb.toLocal( woWorld );
        const float3 wi = onb.toLocal( wiWorld );
        if ( !SameHemisphere( wo, wi ) ) return 0;

        float F0;
        float3 wo_c = refractIn( wo, F0 );
        float probSpecular = F0*_coatingWeight /
            ( F0*_coatingWeight + ( 1 - F0 ) * ( 1 - _coatingWeight ) );
        float probSubstrate = 1.f - probSpecular;

        float pdf = 0.f;
        if ( sampleSpecular ) {
            pdf = sampleSubstrate ? probSpecular : 1.0f;
        }
        if ( sampleSubstrate ) {
            float F1;
            float3 wi_c = refractIn( wi, F1 );
            if ( F0 == 1.f || F1 == 1.f ) return 0.f;
            pdf *= _substrate.Pdf( wo_c, wi_c );
            pdf *= sampleSpecular ? pdf * probSubstrate : pdf;
        }
        return pdf;
    }
    __device__
        float3 Sample_f(
        const CudaIntersection& I,
        const float3 &woWorld,
        float3 &wiWorld,
        const float2 &u,
        float &pdf,
        BxDFType flags,
        BxDFType &sampledType
        ) const {
        bool sampleSpecular = ( flags & BSDF_SPECULAR );
        bool sampleSubstrate = ( flags & _substrate.getType() & BSDF_ALL );

        const CudaONB onb( I._shading._n, I._shading._dpdu, I._shading._dpdv );
        const float3 wo = onb.toLocal( woWorld );
        float3 wi = onb.toLocal( wiWorld );
        if ( wo.z == 0.0f ) return _constant_spec._black;

        float F0;
        float3 wo_c = refractIn( wo, F0 );
        float probSpecular = F0*_coatingWeight /
            ( F0*_coatingWeight + ( 1 - F0 ) * ( 1 - _coatingWeight ) );
        float probSubstrate = 1.f - probSpecular;
        bool choseSpecular = false;
        float2 sample( u );
        if ( sample.y < probSpecular ) {
            sample.y /= probSpecular;
            choseSpecular = true;
        } else {
            sample.y = ( sample.y - probSpecular ) / ( 1.f - probSpecular );
            choseSpecular = false;
        }
        float3 result;
        if ( choseSpecular ) {
            sampledType = BxDFType( BSDF_REFLECTION | BSDF_SPECULAR );
            wi = make_float3( -wo.x, -wo.y, wo.z );
            pdf = sampleSubstrate ? probSpecular : 1.0f;
            result = _S * F0 / AbsCosTheta( wi );
        } else {
            float3 wi_c;
            result = _substrate.Sample_f( wo_c, wi_c, sample, pdf, sampledType );
            float3 sigmaA = _coatingSigma * _coatingThickness;
            if ( !NOOR::isBlack( sigmaA ) ) {
                result *= NOOR::exp3f( -sigmaA * ( 1.f / AbsCosTheta( wo_c ) + 1.f / AbsCosTheta( wi_c ) ) );
            }
            float F1;
            wi = refractOut( wi_c, F1 );
            if ( F1 == 1.f ) {
                pdf = 0.f;
                return _constant_spec._black;
            }
            result *= ( 1.f - F0 )*( 1.f - F1 );
            if ( sampleSpecular ) {
                pdf *= probSubstrate;
            }
        }
        wiWorld = onb.toWorld( wi );
        return result;
    }
};

template class CudaBSDF<CudaOrenNayar>;
template class CudaBSDF<LambertionReflectionBxDF>;
template class CudaBSDF<CudaFresnelSpecular>;
template class CudaBSDF<ConductorReflectionBxDF>;
template class CudaBSDF<MirrorReflectionBxDF>;
template class CudaBSDF<FresnelBlend>;
template class CudaBSDF<DielectricReflectionBxDF, DielectricTransmissionBxDF>;
template class CudaBSDF<LambertionReflectionBxDF, DielectricReflectionBxDF>;
template class CudaBSDF<LambertionReflectionBxDF, LambertionTransmissionBxDF>;
template class CudaSmoothCoatingBSDF<ConductorReflectionBxDF>;

using CudaOrenNayarBSDF = CudaBSDF<CudaOrenNayar>;
//using CudaShadowCatcherBSDF = CudaBSDF<CudaShadowCatcher>;
using CudaDiffuseBSDF = CudaBSDF<LambertionReflectionBxDF>;
using CudaGlassBSDF = CudaBSDF<CudaFresnelSpecular>;
using CudaSpecularReflectionBSDF = CudaBSDF<MirrorReflectionBxDF>;
using CudaFresnelBlendBSDF = CudaBSDF<FresnelBlend>;
using CudaRoughGlassBSDF = CudaBSDF<DielectricReflectionBxDF, DielectricTransmissionBxDF>;
using CudaGlossyBSDF = CudaBSDF<LambertionReflectionBxDF, DielectricReflectionBxDF >;
using CudaTranslucentBSDF = CudaBSDF<LambertionReflectionBxDF, LambertionTransmissionBxDF>;
using CudaCoatingBSDF = CudaSmoothCoatingBSDF<ConductorReflectionBxDF>;
//using CudaMetalBSDF = CudaBSDF<ConductorReflectionBxDF>;
using CudaMetalBSDF = CudaBSDF<CudaScaledBxDF<ConductorReflectionBxDF>,CudaScaledBxDF<LambertionReflectionBxDF>>;

__forceinline__ __device__
CudaShadowCatcherBSDF factoryShadowCatcherBSDF( const CudaIntersection& I ) {
    //return CudaShadowCatcherBSDF(CudaShadowCatcher());
    return CudaShadowCatcherBSDF();
}

__forceinline__ __device__
CudaDiffuseBSDF factoryDiffuseBSDF( const CudaIntersection& I ) {
    const float3 R = _material_manager.getDiffuse( I );
    return CudaDiffuseBSDF( LambertionReflectionBxDF( R ) );
}

__forceinline__ __device__
CudaSpecularReflectionBSDF factoryMirrorBSDF( const CudaIntersection& I ) {
    const float3 S = _material_manager.getSpecular( I );
    const CudaFresnelNoOp fresnel;
    return CudaSpecularReflectionBSDF( MirrorReflectionBxDF( S, fresnel ) );
}

__forceinline__ __device__
CudaGlassBSDF factoryGlassBSDF( const CudaIntersection& I ) {
    const float3 S = _material_manager.getSpecular( I );
    const float3 T = _material_manager.getTransmission( I );
    const float3 etaI = make_float3( 1.0f );
    const float3 etaT = _material_manager.getIor( I );
    const CudaFresnelDielectric fresnel( etaI, etaT );
    return CudaGlassBSDF( CudaFresnelSpecular( S, T, fresnel ) );
}

__forceinline__ __device__
CudaMetalBSDF factoryMetalBSDF( const CudaIntersection& I ) {
    /*	R            G            B
        Silver      0.971519    0.959915    0.915324
        Aluminum    0.913183    0.921494    0.924524
        Gold        1           0.765557    0.336057
        Copper      0.955008    0.637427    0.538163
        Chromium    0.549585    0.556114    0.554256
        Nickel      0.659777    0.608679    0.525649
        Titanium    0.541931    0.496791    0.449419
        Cobalt      0.662124    0.654864    0.633732
        Platinum    0.672411    0.637331    0.585456
        */
    const float3 S = _material_manager.getSpecular( I );
    const float3 D = _material_manager.getDiffuse( I );
    const float3 etaI = make_float3( 1.0f );
    const float3 etaT = _material_manager.getIor( I );
    const float3 K = _material_manager.getK( I );
    const float2 Roughness = _material_manager.getRoughness( I );
    const CudaFresnelConductor fresnel( etaI, etaT, K );
    const CudaTrowbridgeReitzDistribution distribution( Roughness.x, Roughness.y );
    float mscale = _material_manager.getMetalness( I );
    float dscale = fabsf( 1.f - mscale );
    const CudaScaledBxDF<ConductorReflectionBxDF> mbxdf( ConductorReflectionBxDF( S, distribution, fresnel ), make_float3(mscale));
    const CudaScaledBxDF<LambertionReflectionBxDF> dbxdf( LambertionReflectionBxDF( D ), make_float3(dscale) );
    return CudaMetalBSDF( mbxdf, dbxdf );
}

__forceinline__ __device__
CudaRoughGlassBSDF factoryRoughGlassBSDF( const CudaIntersection& I ) {
    const float3 S = _material_manager.getSpecular( I );
    const float3 T = _material_manager.getTransmission( I );
    const float3 etaI = make_float3( 1.0f );
    const float3 etaT = _material_manager.getIor( I );
    const float2 Roughness = _material_manager.getRoughness( I );
    const CudaFresnelDielectric fresnel( etaI, etaT );
    const CudaTrowbridgeReitzDistribution distribution( Roughness.x, Roughness.y );
    return CudaRoughGlassBSDF( DielectricReflectionBxDF( S, distribution, fresnel )
                               , DielectricTransmissionBxDF( T, distribution, fresnel ) );
}

__forceinline__ __device__
CudaCoatingBSDF factorySmoothCoatingBSDF( const CudaIntersection& I ) {
    // substrate
    const float3 S = _material_manager.getSpecular( I );
    const float3 R = _material_manager.getDiffuse( I );

    const float3 SubstrateEtaI = make_float3( 1.0f );
    const float3 SubstrateEtaT = _material_manager.getIor( I );
    const float3 SubstrateK = _material_manager.getK( I );
    const float2 SubstrateRoughness = _material_manager.getRoughness( I );

    const CudaFresnelConductor SF( SubstrateEtaI, SubstrateEtaT, SubstrateK );
    const CudaTrowbridgeReitzDistribution SD( SubstrateRoughness.x, SubstrateRoughness.y );
    const ConductorReflectionBxDF substrate( R, SD, SF );

    // coating
    const float3 coatingSigma = make_float3( _material_manager.getCoatSigma( I ) );
    const float coatingWeight = _material_manager.getCoatWeight( I );
    const float coatingThickness = _material_manager.getCoatThickness( I );
    const float coatingIor = _material_manager.getCoatIOR( I );
    const float3 CoatingEtaI = make_float3( 1.0f );
    const float3 CoatingEtaT = make_float3( coatingIor );
    const CudaFresnelDielectric CF( CoatingEtaI, CoatingEtaT );
    return CudaCoatingBSDF(
        substrate,
        CF,
        S,
        coatingSigma,
        coatingThickness,
        coatingWeight
    );
}

__forceinline__ __device__
CudaGlossyBSDF factoryGlossyBSDF( const CudaIntersection& I ) {
    const float3 R = _material_manager.getDiffuse( I );
    const float3 S = _material_manager.getSpecular( I );
    const float3 etaI = make_float3( 1.0f );
    const float3 etaT = _material_manager.getIor( I );
    const float2 Roughness = _material_manager.getRoughness( I );
    const CudaFresnelDielectric fresnel( etaI, etaT );
    const CudaTrowbridgeReitzDistribution distribution( Roughness.x, Roughness.y );
    return CudaGlossyBSDF( LambertionReflectionBxDF( R ), DielectricReflectionBxDF( S, distribution, fresnel ) );
}

__forceinline__ __device__
CudaOrenNayarBSDF factoryOrenNayarBSDF( const CudaIntersection& I ) {
    const float3 R = _material_manager.getDiffuse( I );
    const float2 Roughness = _material_manager.getRoughness( I );
    return CudaOrenNayarBSDF( CudaOrenNayar( R, Roughness.x ) );
}

__forceinline__ __device__
CudaFresnelBlendBSDF factoryFresnelBlendBSDF( const CudaIntersection& I ) {
    const float3 R = _material_manager.getDiffuse( I );
    const float3 S = _material_manager.getSpecular( I );
    const float2 Roughness = _material_manager.getRoughness( I );
    const CudaTrowbridgeReitzDistribution distribution( Roughness.x, Roughness.y );
    return CudaFresnelBlendBSDF( FresnelBlend( R, S, distribution ) );
}

__forceinline__ __device__
CudaTranslucentBSDF factoryTranslucentBSDF( const CudaIntersection& I ) {
    const float3 R = _material_manager.getDiffuse( I );
    const float3 T = _material_manager.getDiffuse( I );
    return CudaTranslucentBSDF( LambertionReflectionBxDF( R ), LambertionTransmissionBxDF( T ) );
}

#endif /* CUDABSDF_CUH */
