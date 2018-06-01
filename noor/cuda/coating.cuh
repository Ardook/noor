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
#ifndef COATING_CUH
#define COATING_CUH
#include "bxdf.cuh"
class CudaFresnelConductor {
public:
    float3 _etaI;
    float3 _etaT;
    float3 _k;
    __device__
        CudaFresnelConductor( const float3 &etaI, const float3 &etaT, const float3 &k )
        : _etaI( etaI ), _etaT( etaT ), _k( k ) {}
    __device__
        float3 evaluate( float cosThetaI ) const {
        return fresnelConductor( cosThetaI, _etaI, _etaT, _k );
    }

};

class CudaFresnelDielectric {
public:
    float _etaI;
    float _etaT;
    __device__
        CudaFresnelDielectric( const float3& etaI, const float3& etaT ) : _etaI( etaI.x ), _etaT( etaT.x ) {}
    __device__
        float3 evaluate( float cosThetaI ) const {
        return make_float3( fresnelDielectric( cosThetaI, _etaI, _etaT ) );
    }
};

/* Based on Mitsuba Renderer SmoothCoating */
class CudaSmoothCoatingBSDF {
    CudaMicrofacetReflection _substrate;
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
        const CudaMicrofacetReflection& substrate,
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
        const CudaIntersection& I
        , const float3 &woWorld
        , const float3 &wiWorld
        , BxDFType flags
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
        const CudaIntersection& I
        , const float3 &woWorld
        , const float3 &wiWorld
        , BxDFType flags
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
        const CudaIntersection& I
        , const float3 &woWorld
        , float3 &wiWorld
        , const float2 &u
        , float &pdf
        , BxDFType flags
        , BxDFType &sampledType
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
#endif /* COATING_CUH */
