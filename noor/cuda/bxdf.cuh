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
#ifndef CUDABXDF_CUH
#define CUDABXDF_CUH
/* based on PBRT BSDF and BXDF's */

#include "fresnel.cuh"
__forceinline__ __device__
static float CosTheta( const float3 &w ) { return w.z; }
__forceinline__ __device__
static float Cos2Theta( const float3 &w ) { return w.z * w.z; }
__forceinline__ __device__
static float AbsCosTheta( const float3 &w ) { return fabsf( w.z ); }
__forceinline__ __device__
static float Sin2Theta( const float3 &w ) { return fmaxf( 0.0f, 1.0f - Cos2Theta( w ) ); }
__forceinline__ __device__
static float SinTheta( const float3 &w ) { return sqrtf( Sin2Theta( w ) ); }
__forceinline__ __device__
static float TanTheta( const float3 &w ) { return SinTheta( w ) / CosTheta( w ); }
__forceinline__ __device__
static float Tan2Theta( const float3 &w ) { return Sin2Theta( w ) / Cos2Theta( w ); }
__forceinline__ __device__
static float CosPhi( const float3 &w ) {
    const float sinTheta = SinTheta( w );
    return ( sinTheta == 0.0f ) ? 1.0f : clamp( w.x / sinTheta, -1.0f, 1.0f );
}
__forceinline__ __device__
static float SinPhi( const float3 &w ) {
    const float sinTheta = SinTheta( w );
    return ( sinTheta == 0.0f ) ? 0.0f : clamp( w.y / sinTheta, -1.0f, 1.0f );
}
__forceinline__ __device__
static float Cos2Phi( const float3 &w ) { return CosPhi( w ) * CosPhi( w ); }
__forceinline__ __device__
static float Sin2Phi( const float3 &w ) { return SinPhi( w ) * SinPhi( w ); }
__forceinline__ __device__
static float CosDPhi( const float3 &wa, const float3 &wb ) {
    return clamp(
        ( wa.x * wb.x + wa.y * wb.y ) / sqrtf( ( wa.x * wa.x + wa.y * wa.y ) *
        ( wb.x * wb.x + wb.y * wb.y ) ),
        -1.0f, 1.0f );
}
__forceinline__ __device__
static bool SameHemisphere( const float3 &w, const float3 &wp ) {
    return w.z * wp.z > 0;
}

__forceinline__ __device__
static float3 Reflect( const float3 &wo, const float3 &n ) {
    return -1.0f*wo + 2.0f * dot( wo, n ) * n;
}

__forceinline__ __device__
static bool Refract( const float3 &wi, const float3 &n, float eta, float3 &wt ) {
    const float cosThetaI = dot( n, wi );
    const float sin2ThetaI = fmaxf( 0.0f, 1.0f - cosThetaI * cosThetaI );
    const float sin2ThetaT = eta * eta * sin2ThetaI;

    // Handle total internal reflection for transmission
    if ( sin2ThetaT >= 1.0f ) {
        return false;
    }
    const float cosThetaT = sqrtf( 1.0f - sin2ThetaT );
    wt = eta * ( -1.0f*wi ) + ( eta * cosThetaI - cosThetaT ) * n;
    return true;
}

#include "distribution.cuh"

// BSDF Declarations
enum BxDFType {
    BSDF_REFLECTION = 1 << 0,
    BSDF_TRANSMISSION = 1 << 1,
    BSDF_DIFFUSE = 1 << 2,
    BSDF_GLOSSY = 1 << 3,
    BSDF_SPECULAR = 1 << 4,
    BSDF_ALL = BSDF_DIFFUSE | BSDF_GLOSSY |
    BSDF_SPECULAR | BSDF_REFLECTION | BSDF_TRANSMISSION
};
enum BxDFIndex {
    LambertReflection = 0,
    LambertTransmission,
    SpecularReflection,
    SpecularTransmission,
    MicrofacetReflection,
    MicrofacetTransmission,
    FresnelBlend,
    FresnelSpecular
};
class CudaBxDF {
public:
    BxDFType _type;
    BxDFIndex _index;
public:
    __device__
        CudaBxDF( BxDFType type, BxDFIndex index ) :_type( type ), _index( index ) {}
    __device__
        bool MatchesFlags( BxDFType t ) const { return ( _type & t ) == _type; }
    __device__
        BxDFType getType() const { return _type; }
    __device__
        bool isSpecular() const {
        return ( _type & BSDF_SPECULAR );
    }
    __device__
        bool notSpecular() const {
        return !isSpecular();
    }
    __device__
        float3 R( const CudaIntersection& I )const {
        return _material_manager.getDiffuse( I );
    }
    __device__
        float3 S( const CudaIntersection& I )const {
        return _material_manager.getSpecular( I );
    }
    __device__
        float3 T( const CudaIntersection& I )const {
        return _material_manager.getTransmission( I );
    }
    __device__
        float3 ior( const CudaIntersection& I )const {
        return _material_manager.getIor( I );
    }
    __device__
        float3 k( const CudaIntersection& I )const {
        return _material_manager.getK( I );
    }
    __device__
        float2 roughness( const CudaIntersection& I )const {
        return _material_manager.getRoughness( I );
    }

    __device__
        CudaFresnel factoryFresnel( const CudaIntersection& I ) const {
        const float3 etaI = make_float3( 1.f );
        const float3 etaT = ior( I );
        if ( I._material_type == METAL )
            return CudaFresnel( etaI, etaT, k( I ) );
        else if ( I._material_type == MIRROR ) {
            return CudaFresnel();
        } else {
            return CudaFresnel( etaI, etaT );
        }
    }
    __device__
        CudaTrowbridgeReitzDistribution factoryDistribution( const CudaIntersection& I ) const {
        const float2 r = _material_manager.getRoughness( I );
        return CudaTrowbridgeReitzDistribution( r );
    }
};

class CudaLambertianReflection : public CudaBxDF {
public:
    // LambertianReflection Public Methods
    __device__
        CudaLambertianReflection() :
        CudaBxDF( BxDFType( BSDF_REFLECTION | BSDF_DIFFUSE ), LambertReflection ) {}
    __device__
        float3 f( const CudaIntersection& I,
                  const float3 &wo,
                  const float3 &wi ) const {
        return R( I )*NOOR_invPI;
    }
    __device__
        float3 Sample_f(
        const CudaIntersection& I,
        const float3 &wo,
        float3 &wi,
        const float2 &u,
        float &pdf,
        BxDFType &sampledType
        ) const {
        sampledType = _type;
        // Cosine-sample the hemisphere, flipping the direction if necessary
        // default left handed since we are in bxdf coordinate system
        wi = NOOR::cosineSampleHemisphere( u );
        if ( wo.z < 0 ) wi.z *= -1.0f;
        pdf = Pdf( I, wo, wi );
        return f( I, wo, wi );
    }
    __device__
        float Pdf( const CudaIntersection& I,
                   const float3 &wo, const float3 &wi ) const {
        return SameHemisphere( wo, wi ) ? AbsCosTheta( wi ) * NOOR_invPI : 0.0f;
    }
};

class CudaLambertianTransmission : public CudaBxDF {
public:
    // LambertianTransmission Public Methods
    __device__
        CudaLambertianTransmission() :
        CudaBxDF( BxDFType( BSDF_TRANSMISSION | BSDF_DIFFUSE ), LambertTransmission ) {}
    __device__
        float3 f( const CudaIntersection& I,
                  const float3 &wo,
                  const float3 &wi ) const {
        return T( I )*NOOR_invPI;
    }
    __device__
        float3 Sample_f(
        const CudaIntersection& I,
        const float3 &wo,
        float3 &wi,
        const float2 &u,
        float &pdf,
        BxDFType &sampledType
        ) const {
        sampledType = _type;
        // Cosine-sample the hemisphere, flipping the direction if necessary
        // default left handed since we are in bxdf coordinate system
        wi = NOOR::cosineSampleHemisphere( u );
        if ( wo.z > 0.0f ) wi.z *= -1.0f;
        pdf = Pdf( I, wo, wi );
        return f( I, wo, wi );
    }
    __device__
        float Pdf( const CudaIntersection& I,
                   const float3 &wo, const float3 &wi ) const {
        return !SameHemisphere( wo, wi ) ? AbsCosTheta( wi ) * NOOR_invPI : 0.0f;
    }
};


class CudaFresnelSpecular : public CudaBxDF {
public:
    __device__
        CudaFresnelSpecular() :
        CudaBxDF( BxDFType( BSDF_REFLECTION | BSDF_TRANSMISSION | BSDF_SPECULAR ), FresnelSpecular ) {}

    __device__
        float3 f( const CudaIntersection& I,
                  const float3 &wo,
                  const float3 &wi ) const {
        return make_float3( 0.f );
    }

    __device__
        float3 Sample_f(
        const CudaIntersection& I,
        const float3 &wo,
        float3 &wi,
        const float2 &u,
        float &pdf,
        BxDFType &sampledType
        ) const {
        const CudaFresnel fresnel = factoryFresnel( I );
        float F = fresnel.evaluate( CosTheta( wo ) ).x;
        if ( u.x < F ) {
            // Compute specular reflection for _FresnelSpecular_
            // Compute perfect specular reflection direction
            wi = make_float3( -wo.x, -wo.y, wo.z );
            pdf = F;
            sampledType = BxDFType( BSDF_REFLECTION | BSDF_SPECULAR );
            return F * S( I ) / AbsCosTheta( wi );
        } else {
            // Compute specular transmission for _FresnelSpecular_
            // Figure out which eta is incident and which is transmitted
            const bool entering = CosTheta( wo ) > 0.0f;
            const float eta = entering ? ( fresnel._etaI / fresnel._etaT ).x :
                ( fresnel._etaT / fresnel._etaI ).x;

            // Compute ray direction for specular transmission
            if ( !Refract( wo, NOOR::faceforward( make_float3( 0, 0, 1 ), wo ), eta, wi ) ) {
                return _constant_spec._black;
            }
            // Account for non-symmetry with transmission to different medium
            pdf = 1.0f - F;
            sampledType = BxDFType( BSDF_TRANSMISSION | BSDF_SPECULAR );
            return T( I ) * eta * eta * ( 1.0f - F ) / AbsCosTheta( wi );
        }
    }
    __device__
        float Pdf( const CudaIntersection& I,
                   const float3 &wo, const float3 &wi ) const {
        return 0.0f;
    }
};


class CudaSpecularReflection : public CudaBxDF {
public:
    __device__
        CudaSpecularReflection() :
        CudaBxDF( BxDFType( BSDF_REFLECTION | BSDF_SPECULAR ), SpecularReflection ) {}
    __device__
        float3 f( const CudaIntersection& I,
                  const float3 &wo,
                  const float3 &wi ) const {
        return make_float3( 0.f );
    }
    __device__
        float3 Sample_f(
        const CudaIntersection& I,
        const float3 &wo,
        float3 &wi,
        const float2 &u,
        float &pdf,
        BxDFType &sampledType
        ) const {
        sampledType = _type;
        // Compute perfect specular reflection direction
        wi = make_float3( -wo.x, -wo.y, wo.z );
        pdf = 1.0f;
        const CudaFresnel fresnel = factoryFresnel( I );
        return fresnel.evaluate( CosTheta( wi ) ) * S( I ) / AbsCosTheta( wi );
    }

    __device__
        float Pdf( const CudaIntersection& I,
                   const float3 &wo, const float3 &wi ) const {
        return 0;
    }

};

class CudaSpecularTransmission : public CudaBxDF {
public:
    // SpecularTransmission Public Methods
    __device__
        CudaSpecularTransmission() :
        CudaBxDF( BxDFType( BSDF_TRANSMISSION | BSDF_SPECULAR ), SpecularTransmission ) {}
    __device__
        float3 f( const CudaIntersection& I,
                  const float3 &wo,
                  const float3 &wi ) const {
        return make_float3( 0.f );
    }
    __device__
        float3 Sample_f(
        const CudaIntersection& I,
        const float3 &wo,
        float3 &wi,
        const float2 &u,
        float &pdf,
        BxDFType &sampledType
        ) const {
        sampledType = _type;
        const CudaFresnel fresnel = factoryFresnel( I );
        const bool entering = CosTheta( wo ) > 0.0f;
        const float etaI = entering ? fresnel._etaI.x : fresnel._etaT.x;
        const float etaT = entering ? fresnel._etaT.x : fresnel._etaI.x;

        // Compute ray direction for specular transmission
        if ( !Refract( wo, NOOR::faceforward( make_float3( 0, 0, 1 ), wo ), etaI / etaT, wi ) ) {
            return _constant_spec._black;
        }
        pdf = 1.0f;
        float3 ft = T( I ) * ( make_float3( 1.f ) - fresnelDielectric( CosTheta( wi ), etaI, etaT ) );
        // Account for non-symmetry with transmission to different medium
        ft *= ( etaI * etaI ) / ( etaT * etaT );
        return ft / AbsCosTheta( wi );
    }
    __device__
        float Pdf( const CudaIntersection& I,
                   const float3 &wo, const float3 &wi ) const {
        return 0;
    }
};


class CudaFresnelBlend : public CudaBxDF {
public:
    // FresnelBlend Public Methods
    __device__
        CudaFresnelBlend() :
        CudaBxDF( BxDFType( BSDF_REFLECTION | BSDF_GLOSSY ), FresnelBlend ) {}
    __device__
        float3 SchlickFresnel( const CudaIntersection& I, float cosTheta ) const {
        return S( I ) + NOOR::pow5f( 1.f - cosTheta ) * ( make_float3( 1.f ) - S( I ) );
    }
    __device__
        float3 f( const CudaIntersection& I,
                  const float3 &wo,
                  const float3 &wi ) const {
        const float3 diffuse = ( 28.f / ( 23.f * NOOR_PI ) ) * R( I ) *
            ( make_float3( 1.f ) - S( I ) ) *
            ( 1.f - NOOR::pow5f( 1.f - .5f * AbsCosTheta( wi ) ) ) *
            ( 1.f - NOOR::pow5f( 1.f - .5f * AbsCosTheta( wo ) ) );
        float3 wh = wi + wo;
        if ( wh.x == 0 && wh.y == 0 && wh.z == 0 ) return make_float3( 0 );
        wh = NOOR::normalize( wh );
        const CudaTrowbridgeReitzDistribution  distribution = factoryDistribution( I );
        const float3 specular =
            distribution.D( wh ) /
            ( 4.f * NOOR::absDot( wi, wh ) * fmaxf( AbsCosTheta( wi ), AbsCosTheta( wo ) ) ) *
            SchlickFresnel( I, dot( wi, wh ) );
        return diffuse + specular;
    }

    __device__
        float3 Sample_f(
        const CudaIntersection& I,
        const float3 &wo,
        float3 &wi,
        const float2 &u,
        float &pdf,
        BxDFType &sampledType
        ) const {
        sampledType = _type;
        float2 lu = u;
        if ( lu.x < .5f ) {
            lu.x = fminf( 2.f * lu.x, NOOR_ONE_MINUS_EPSILON );
            // Cosine-sample the hemisphere, flipping the direction if necessary
            wi = NOOR::cosineSampleHemisphere( lu );
            if ( wo.z < 0 ) wi.z *= -1.f;
        } else {
            lu.x = fminf( 2.f * ( lu.x - .5f ), NOOR_ONE_MINUS_EPSILON );
            // Sample microfacet orientation $\wh$ and reflected direction $\wi$
            const CudaTrowbridgeReitzDistribution  distribution = factoryDistribution( I );
            const float3 wh = distribution.Sample_wh( wo, lu );
            wi = Reflect( wo, wh );
            if ( !SameHemisphere( wo, wi ) ) return make_float3( 0.f );
        }
        pdf = Pdf( I, wo, wi );
        return f( I, wo, wi );
    }
    __device__
        float Pdf( const CudaIntersection& I,
                   const float3 &wo, const float3 &wi ) const {
        if ( !SameHemisphere( wo, wi ) ) return 0.f;
        float3 wh = NOOR::normalize( wo + wi );
        const CudaTrowbridgeReitzDistribution  distribution = factoryDistribution( I );
        float pdf_wh = distribution.Pdf( wo, wh );
        return .5f * ( AbsCosTheta( wi ) * NOOR_invPI + pdf_wh / ( 4.f * dot( wo, wh ) ) );
    }
};


class CudaMicrofacetReflection : public CudaBxDF {
public:
    // MicrofacetReflection Public Methods
    __device__
        CudaMicrofacetReflection()
        : CudaBxDF( BxDFType( BSDF_REFLECTION | BSDF_GLOSSY ), MicrofacetReflection ) {}

    __device__
        float3 f( const CudaIntersection& I,
                  const float3 &wo,
                  const float3 &wi ) const {
        float cosThetaO = AbsCosTheta( wo ), cosThetaI = AbsCosTheta( wi );
        // Handle degenerate cases for microfacet reflection
        if ( cosThetaI == 0.0f || cosThetaO == 0.0f ) return _constant_spec._black;
        float3 wh = wi + wo;
        if ( wh.x == 0.0f && wh.y == 0 && wh.z == 0.0f ) return _constant_spec._black;
        wh = NOOR::normalize( wh );
        const CudaFresnel fresnel = factoryFresnel( I );
        const float3 F = fresnel.evaluate( dot( wo, wh ) );
        const CudaTrowbridgeReitzDistribution  distribution = factoryDistribution( I );
        return NOOR::sheen( wo, wi ) + R( I ) * distribution.D( wh ) *
            distribution.G( wo, wi ) * F /
            ( 4.0f * cosThetaI * cosThetaO );
    }

    __device__
        float3 Sample_f(
        const CudaIntersection& I,
        const float3 &wo,
        float3 &wi,
        const float2 &u,
        float &pdf,
        BxDFType &sampledType
        ) const {
        sampledType = _type;
        // Sample microfacet orientation $\wh$ and reflected direction $\wi$
        if ( wo.z == 0.0f ) return _constant_spec._black;
        const CudaTrowbridgeReitzDistribution  distribution = factoryDistribution( I );
        const float3 wh = distribution.Sample_wh( wo, u );
        wi = Reflect( wo, wh );
        if ( !SameHemisphere( wo, wi ) ) {
            pdf = 0.0f;
            return _constant_spec._black;
        }

        // Compute PDF of _wi_ for microfacet reflection
        pdf = distribution.Pdf( wo, wh ) / ( 4.0f * dot( wo, wh ) );
        return f( I, wo, wi );
    }
    __device__
        float Pdf( const CudaIntersection& I,
                   const float3 &wo, const float3 &wi ) const {
        if ( !SameHemisphere( wo, wi ) ) {
            return 0.0f;
        }
        const float3 wh = NOOR::normalize( wo + wi );
        const CudaTrowbridgeReitzDistribution  distribution = factoryDistribution( I );
        return distribution.Pdf( wo, wh ) / ( 4.0f * dot( wo, wh ) );
    }
};

class CudaMicrofacetTransmission : public CudaBxDF {
public:
    // MicrofacetReflection Public Methods
    __device__
        CudaMicrofacetTransmission() :
        CudaBxDF( BxDFType( BSDF_TRANSMISSION | BSDF_GLOSSY ), MicrofacetTransmission ) {}

    __device__
        float3 f( const CudaIntersection& I,
                  const float3 &wo,
                  const float3 &wi ) const {
        if ( SameHemisphere( wo, wi ) ) return _constant_spec._black;  // transmission only

        const float cosThetaO = CosTheta( wo );
        const float cosThetaI = CosTheta( wi );
        if ( cosThetaI == 0.0f || cosThetaO == 0.0f ) return _constant_spec._black;

        // Compute $\wh$ from $\wo$ and $\wi$ for microfacet transmission
        const CudaFresnel fresnel = factoryFresnel( I );
        float eta = CosTheta( wo ) > 0.0f ? ( fresnel._etaT / fresnel._etaI ).x :
            ( fresnel._etaI / fresnel._etaT ).x;
        float3 wh = NOOR::normalize( wo + wi * eta );
        wh *= NOOR::sign( wh.z );

        const float3 F = fresnel.evaluate( dot( wo, wh ) );

        const float sqrtDenom = dot( wo, wh ) + eta * dot( wi, wh );
        const float factor = 1.f / eta;
        const CudaTrowbridgeReitzDistribution  distribution = factoryDistribution( I );
        const float3 result = NOOR::sheen( wo, wi ) + ( make_float3( 1.f ) - F ) * T( I ) *
            fabsf( distribution.D( wh ) * distribution.G( wo, wi ) * eta * eta *
                   NOOR::absDot( wi, wh ) * NOOR::absDot( wo, wh ) * factor * factor /
                   ( cosThetaI * cosThetaO * sqrtDenom * sqrtDenom ) );
        return result;
    }
    __device__
        float3 Sample_f(
        const CudaIntersection& I,
        const float3 &wo,
        float3 &wi,
        const float2 &u,
        float &pdf,
        BxDFType &sampledType
        ) const {
        sampledType = _type;
        if ( wo.z == 0 ) return _constant_spec._black;
        const CudaTrowbridgeReitzDistribution  distribution = factoryDistribution( I );
        float3 wh = distribution.Sample_wh( wo, u );
        const CudaFresnel fresnel = factoryFresnel( I );
        const float eta = CosTheta( wo ) > 0 ? ( fresnel._etaI / fresnel._etaT ).x :
            ( fresnel._etaT / fresnel._etaI ).x;
        if ( !Refract( wo, wh, eta, wi ) ) return _constant_spec._black;
        pdf = Pdf( I, wo, wi );
        return f( I, wo, wi );
    }
    __device__
        float Pdf( const CudaIntersection& I,
                   const float3 &wo, const float3 &wi ) const {
        if ( SameHemisphere( wo, wi ) ) return 0;
        const float3 etaI = make_float3( 1.f );
        const float3 etaT = ior( I );
        const CudaFresnel fresnel = factoryFresnel( I );
        const float eta = CosTheta( wo ) > 0 ? ( fresnel._etaT / fresnel._etaI ).x :
            ( fresnel._etaI / fresnel._etaT ).x;
        const float3 wh = NOOR::normalize( wo + wi * eta );

        const float sqrtDenom = dot( wo, wh ) + eta * dot( wi, wh );
        const float dwh_dwi =
            fabsf( ( eta * eta * dot( wi, wh ) ) / ( sqrtDenom * sqrtDenom ) );
        const float2 r = roughness( I );
        const CudaTrowbridgeReitzDistribution  distribution = factoryDistribution( I );
        return distribution.Pdf( wo, wh ) * dwh_dwi;
    }
};

#define NUM_BXDFS 8

__global__
void setup_bxdfs( CudaBxDF** bxdfs,
                  CudaLambertianReflection* bxdf0,
                  CudaLambertianTransmission* bxdf1,
                  CudaSpecularReflection* bxdf2,
                  CudaSpecularTransmission* bxdf3,
                  CudaMicrofacetReflection* bxdf4,
                  CudaMicrofacetTransmission* bxdf5,
                  CudaFresnelBlend* bxdf6,
                  CudaFresnelSpecular* bxdf7
) {
    *bxdf0 = CudaLambertianReflection();
    *bxdf1 = CudaLambertianTransmission();
    *bxdf2 = CudaSpecularReflection();
    *bxdf3 = CudaSpecularTransmission();
    *bxdf4 = CudaMicrofacetReflection();
    *bxdf5 = CudaMicrofacetTransmission();
    *bxdf6 = CudaFresnelBlend();
    *bxdf7 = CudaFresnelSpecular();
    bxdfs[LambertReflection] = bxdf0;
    bxdfs[LambertTransmission] = bxdf1;
    bxdfs[SpecularReflection] = bxdf2;
    bxdfs[SpecularTransmission] = bxdf3;
    bxdfs[MicrofacetReflection] = bxdf4;
    bxdfs[MicrofacetTransmission] = bxdf5;
    bxdfs[FresnelBlend] = bxdf6;
    bxdfs[FresnelSpecular] = bxdf7;
}
__global__
void free_bxdfs( CudaBxDF** bxdfs ) {
    for ( int i = 0; i < NUM_BXDFS; ++i ) {
        delete bxdfs[i];
    }
}

class CudaBxDFManager {
public:
    CudaBxDF** _bxdfs;
    CudaLambertianReflection* _bxdf0;
    CudaLambertianTransmission* _bxdf1;
    CudaSpecularReflection* _bxdf2;
    CudaSpecularTransmission* _bxdf3;
    CudaMicrofacetReflection* _bxdf4;
    CudaMicrofacetTransmission* _bxdf5;
    CudaFresnelBlend* _bxdf6;
    CudaFresnelSpecular* _bxdf7;

    CudaBxDFManager() = default;

    __host__
        CudaBxDFManager( int i ) {
        NOOR::malloc( (void**) &_bxdfs, NUM_BXDFS * sizeof( CudaBxDF* ) );
        NOOR::malloc( (void**) &_bxdf0, sizeof( CudaLambertianReflection ) );
        NOOR::malloc( (void**) &_bxdf1, sizeof( CudaLambertianTransmission ) );
        NOOR::malloc( (void**) &_bxdf2, sizeof( CudaSpecularReflection ) );
        NOOR::malloc( (void**) &_bxdf3, sizeof( CudaSpecularTransmission ) );
        NOOR::malloc( (void**) &_bxdf4, sizeof( CudaMicrofacetReflection ) );
        NOOR::malloc( (void**) &_bxdf5, sizeof( CudaMicrofacetTransmission ) );
        NOOR::malloc( (void**) &_bxdf6, sizeof( CudaFresnelBlend ) );
        NOOR::malloc( (void**) &_bxdf7, sizeof( CudaFresnelSpecular ) );
        setup_bxdfs << < 1, 1 >> > ( _bxdfs,
                                     _bxdf0,
                                     _bxdf1,
                                     _bxdf2,
                                     _bxdf3,
                                     _bxdf4,
                                     _bxdf5,
                                     _bxdf6,
                                     _bxdf7
                                     );
    }
    __host__
        void free() {
        //free_bxdfs << < 1, 1 >> > ( _bxdfs );
        cudaFree( _bxdf0 );
        cudaFree( _bxdf1 );
        cudaFree( _bxdf2 );
        cudaFree( _bxdf3 );
        cudaFree( _bxdf4 );
        cudaFree( _bxdf5 );
        cudaFree( _bxdf6 );
        cudaFree( _bxdf7 );
        cudaFree( _bxdfs );
    }
};

__constant__
CudaBxDFManager _constant_bxdf_manager;
#endif /* CUDABXDF_CUH */

//class CudaShadowCatcher: public CudaBxDF {
//public:
//    __device__
//        CudaShadowCatcher() :
//        CudaBxDF( BxDFType( BSDF_REFLECTION | BSDF_DIFFUSE ) ) {}
//
//    __device__
//        float Pdf( const float3 &wo, const float3 &wi ) const {
//        return 1.f;
//    }
//
//    __device__
//    float3 f( const float3 &wo, const float3 &wi ) const {
//        return _skydome_manager.evaluate( -1.f*wo, false ) / AbsCosTheta( wi );
//    }
//
//    __device__
//        float3 Sample_f(
//        const float3 &wo
//        , float3 &wi
//        , const float2 &u
//        , float &pdf
//        , BxDFType &sampledType
//        ) const {
//        pdf = 1.f;
//        wi = NOOR::uniformSampleHemisphere( u );
//        return f( wo, wi );
//    }
//};