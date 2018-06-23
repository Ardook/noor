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
static bool SameHemisphere( const float3 &a, const float3 &b ) {
    return a.z * b.z > 0;
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
    wt = -eta * wi + ( eta * cosThetaI - cosThetaT ) * n;
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
    BSDF_CONDUCTOR = 1 << 5,
    BSDF_DIELECTRIC = 1 << 6,
    BSDF_NOOP = 1 << 7,
    BSDF_ALL = BSDF_DIFFUSE | BSDF_GLOSSY | BSDF_SPECULAR |
    BSDF_REFLECTION | BSDF_TRANSMISSION |
    BSDF_CONDUCTOR | BSDF_DIELECTRIC | BSDF_NOOP
};

enum BxDFIndex {
    //reflections
    ShadowCatcher = 0,
    LambertReflection,
    SpecularReflectionNoOp,
    SpecularReflectionDielectric,
    MicrofacetReflectionDielectric,
    MicrofacetReflectionConductor,

    // transmissions
    LambertTransmission,
    SpecularTransmission,
    MicrofacetTransmission,

    // multi-lobe
    FresnelBlend,
    FresnelSpecular,
    ClearCoat,
    NUM_BXDFS
};

class CudaBxDF {
protected:
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
        return ( ( _type & BSDF_SPECULAR ) != 0 );
    }
    __device__
        bool isConductor() const {
        return ( ( _type & BSDF_CONDUCTOR ) != 0 );
    }
    __device__
        bool isDielectric() const {
        return ( ( _type & BSDF_DIELECTRIC ) != 0 );
    }
    __device__
        int getIndex() const {
        return _index;
    }
    __device__
        CudaTrowbridgeReitz factoryDistribution( const CudaIntersection& I ) const {
        return CudaTrowbridgeReitz( _material_manager.getRoughness( I ) );
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
        float2 roughness( const CudaIntersection& I )const {
        return _material_manager.getRoughness( I );
    }

    __device__
        CudaFresnel factoryFresnel( const CudaIntersection& I ) const {
        const float3 etaI = make_float3( 1.f );
        if ( isDielectric() ) {
            const float3 etaT = _material_manager.getIorDielectric( I );
            I.setEta( etaT.x );
            return CudaFresnel( etaI, etaT );
        } else if ( isConductor() ) {
            return CudaFresnel( etaI, _material_manager.getIorConductor( I ), _material_manager.getK( I ) );
        } else
            return CudaFresnel();
    }

};
class CudaShadowCatcher : public CudaBxDF {
public:
    __device__
        CudaShadowCatcher() :
        CudaBxDF( BxDFType( BSDF_REFLECTION | BSDF_TRANSMISSION | BSDF_DIFFUSE ),
                  ShadowCatcher ) {}


    __device__
        float3 f( const CudaIntersection& I,
                  const float3 &wo,
                  const float3 &wi ) const {
        const CudaONB onb( I._shading._n, I._shading._dpdu, I._shading._dpdv );
        return _skydome_manager.evaluate( -onb.toWorld( wo ), false ) / AbsCosTheta( wi );
    }

    __device__
        float Pdf( const CudaIntersection& I,
                   const float3 &wo,
                   const float3 &wi ) const {
        return 1.f;
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
        pdf = 1.f;
        wi = NOOR::uniformSampleHemisphere( u, LEFT_HANDED );
        return f( I, wo, wi );
    }
    __device__
        void Sample_f(
        const CudaIntersection& I,
        const float3 &wo,
        float3 &wi,
        const float2 &u
        ) const {
        wi = NOOR::uniformSampleHemisphere( u, LEFT_HANDED );
    }
};

class CudaLambertianReflection : public CudaBxDF {
public:
    __device__
        CudaLambertianReflection() :
        CudaBxDF( BxDFType( BSDF_REFLECTION | BSDF_DIFFUSE | BSDF_DIELECTRIC ),
                  LambertReflection ) {}
    __device__
        float3 f( const CudaIntersection& I,
                  const float3 &wo,
                  const float3 &wi ) const {
        return R( I )*NOOR_invPI;
    }
    __device__
        float Pdf( const CudaIntersection& I,
                   const float3 &wo, const float3 &wi ) const {
        return SameHemisphere( wo, wi ) ? AbsCosTheta( wi ) * NOOR_invPI : 0.0f;
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
        void Sample_f(
        const CudaIntersection& I,
        const float3 &wo,
        float3 &wi,
        const float2 &u
        ) const {
        // Cosine-sample the hemisphere, flipping the direction if necessary
        // default left handed since we are in bxdf coordinate system
        wi = NOOR::cosineSampleHemisphere( u );
        if ( wo.z < 0 ) wi.z *= -1.0f;
    }

};

class CudaLambertianTransmission : public CudaBxDF {
public:
    __device__
        CudaLambertianTransmission() :
        CudaBxDF( BxDFType( BSDF_TRANSMISSION | BSDF_DIFFUSE | BSDF_DIELECTRIC ),
                  LambertTransmission ) {}
    __device__
        float3 f( const CudaIntersection& I,
                  const float3 &wo,
                  const float3 &wi ) const {
        return T( I )*NOOR_invPI;
    }

    __device__
        float Pdf( const CudaIntersection& I,
                   const float3 &wo, const float3 &wi ) const {
        return !SameHemisphere( wo, wi ) ? AbsCosTheta( wi ) * NOOR_invPI : 0.0f;
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
        void Sample_f(
        const CudaIntersection& I,
        const float3 &wo,
        float3 &wi,
        const float2 &u
        ) const {
        // Cosine-sample the hemisphere, flipping the direction if necessary
        // default left handed since we are in bxdf coordinate system
        wi = NOOR::cosineSampleHemisphere( u );
        if ( wo.z > 0.0f ) wi.z *= -1.0f;
    }

};

class CudaSpecularReflection : public CudaBxDF {
public:
    __device__
        CudaSpecularReflection( BxDFType type = BSDF_NOOP ) :
        CudaBxDF( BxDFType( type | BSDF_REFLECTION | BSDF_SPECULAR ),
                  type == BSDF_NOOP ?
                  SpecularReflectionNoOp :
                  SpecularReflectionDielectric
        ) {}
    __device__
        float3 f( const CudaIntersection& I,
                  const float3 &wo,
                  const float3 &wi ) const {
        return make_float3( 0.f );
    }

    __device__
        float Pdf( const CudaIntersection& I,
                   const float3 &wo, const float3 &wi ) const {
        return 0;
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
        return fresnel.evaluate( CosTheta( wi ) ) * R( I ) / AbsCosTheta( wi );
    }

    __device__
        void Sample_f(
        const CudaIntersection& I,
        const float3 &wo,
        float3 &wi,
        const float2 &u
        ) const {
        // Compute perfect specular reflection direction
        wi = make_float3( -wo.x, -wo.y, wo.z );
    }

};

class CudaSpecularTransmission : public CudaBxDF {
public:
    __device__
        CudaSpecularTransmission() :
        CudaBxDF( BxDFType( BSDF_TRANSMISSION | BSDF_SPECULAR |
                  BSDF_DIELECTRIC ), SpecularTransmission ) {}
    __device__
        float3 f( const CudaIntersection& I,
                  const float3 &wo,
                  const float3 &wi ) const {
        return make_float3( 0.f );
    }

    __device__
        float Pdf( const CudaIntersection& I,
                   const float3 &wo, const float3 &wi ) const {
        return 0;
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
        void Sample_f(
        const CudaIntersection& I,
        const float3 &wo,
        float3 &wi,
        const float2 &u
        ) const {
        const CudaFresnel fresnel = factoryFresnel( I );
        const bool entering = CosTheta( wo ) > 0.0f;
        const float etaI = entering ? fresnel._etaI.x : fresnel._etaT.x;
        const float etaT = entering ? fresnel._etaT.x : fresnel._etaI.x;

        // Compute ray direction for specular transmission
        if ( !Refract( wo, NOOR::faceforward( make_float3( 0, 0, 1 ), wo ), etaI / etaT, wi ) ) {
            wi = make_float3( -wo.x, -wo.y, -wo.z );
        }
    }
};

class CudaMicrofacetReflection : public CudaBxDF {
public:
    __device__
        CudaMicrofacetReflection( BxDFType type = BSDF_DIELECTRIC )
        : CudaBxDF( BxDFType( type | BSDF_REFLECTION | BSDF_GLOSSY ),
                    type == BSDF_DIELECTRIC ?
                    MicrofacetReflectionDielectric :
                    MicrofacetReflectionConductor
        ) {}

    __device__
        float3 f( const CudaIntersection& I,
                  const float3 &wo,
                  const float3 &wi ) const {
        if ( !SameHemisphere( wo, wi ) ) {
            return _constant_spec._black;
        }
        const float cosThetaO = AbsCosTheta( wo );
        const float cosThetaI = AbsCosTheta( wi );
        // Handle degenerate cases for microfacet reflection
        if ( cosThetaI == 0.0f || cosThetaO == 0.0f ) {
            return _constant_spec._black;
        }
        float3 wh = normalize( wi + wo );
        if ( NOOR::isBlack( wh ) ) {
            return _constant_spec._black;
        }
        wh *= NOOR::sign( wh.z );
        const CudaFresnel fresnel = factoryFresnel( I );
        const float3 F = fresnel.evaluate( dot( wo, wh ) );
        const CudaTrowbridgeReitz _distribution = factoryDistribution( I );
        const float3 result = F * S( I ) * _distribution.D( wh ) * _distribution.G( wo, wi ) /
            ( 4.0f * cosThetaI * cosThetaO );
        return result;
    }

    __device__
        float Pdf( const CudaIntersection& I,
                   const float3 &wo,
                   const float3 &wi ) const {
        if ( !SameHemisphere( wo, wi ) ) {
            return 0.0f;
        }
        const float3 wh = NOOR::normalize( wo + wi );
        const CudaTrowbridgeReitz _distribution = factoryDistribution( I );
        return _distribution.Pdf( wo, wh ) / ( 4.0f * dot( wo, wh ) );
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
        if ( wo.z == 0.0f ) return _constant_spec._black;
        const CudaTrowbridgeReitz _distribution = factoryDistribution( I );
        const float3 wh = _distribution.Sample_wh( wo, u );
        wi = Reflect( wo, wh );
        if ( !SameHemisphere( wo, wi ) ) {
            pdf = 0.0f;
            return _constant_spec._black;
        }
        pdf = _distribution.Pdf( wo, wh ) / ( 4.0f * dot( wo, wh ) );
        return f( I, wo, wi );
    }

    __device__
        void Sample_f(
        const CudaIntersection& I,
        const float3 &wo,
        float3 &wi,
        const float2 &u
        ) const {
        const CudaTrowbridgeReitz _distribution = factoryDistribution( I );
        const float3 wh = _distribution.Sample_wh( wo, u );
        wi = Reflect( wo, wh );
    }
};

class CudaMicrofacetTransmission : public CudaBxDF {
public:
    __device__
        CudaMicrofacetTransmission() :
        CudaBxDF( BxDFType( BSDF_TRANSMISSION | BSDF_GLOSSY | BSDF_DIELECTRIC ),
                  MicrofacetTransmission ) {}

    __device__
        float3 f( const CudaIntersection& I,
                  const float3 &wo,
                  const float3 &wi ) const {
        if ( SameHemisphere( wo, wi ) ) return _constant_spec._black;
        const float cosThetaO = CosTheta( wo );
        const float cosThetaI = CosTheta( wi );
        if ( cosThetaI == 0.0f || cosThetaO == 0.0f ) return _constant_spec._black;

        const CudaFresnel fresnel = factoryFresnel( I );
        float eta = CosTheta( wo ) > 0.0f ? ( fresnel._etaT / fresnel._etaI ).x :
            ( fresnel._etaI / fresnel._etaT ).x;
        float3 wh = NOOR::normalize( wo + wi * eta );
        wh *= NOOR::sign( wh.z );
        const float F = fresnel.evaluate( dot( wo, wh ) ).x;
        const float sqrtDenom = dot( wo, wh ) + eta * dot( wi, wh );
        const CudaTrowbridgeReitz _distribution = factoryDistribution( I );

        const float3 result = ( 1.f - F ) * T( I ) *
            fabsf( _distribution.D( wh ) * _distribution.G( wo, wi ) * eta * eta *
                   NOOR::absDot( wi, wh ) * NOOR::absDot( wo, wh ) /
                   ( cosThetaI * cosThetaO * sqrtDenom * sqrtDenom ) );
        return result;
    }

    __device__
        float Pdf( const CudaIntersection& I,
                   const float3 &wo,
                   const float3 &wi ) const {
        if ( SameHemisphere( wo, wi ) ) return 0;
        const CudaFresnel fresnel = factoryFresnel( I );
        float eta = CosTheta( wo ) > 0 ? ( fresnel._etaT / fresnel._etaI ).x :
            ( fresnel._etaI / fresnel._etaT ).x;
        float3 wh = NOOR::normalize( wo + wi *eta );
        //wh *= NOOR::sign( wh.z );
        const float sqrtDenom = dot( wo, wh ) + eta * dot( wi, wh );
        const float dwh_dwi =
            fabsf( ( eta * eta * dot( wi, wh ) ) / ( sqrtDenom * sqrtDenom ) );
        const CudaTrowbridgeReitz _distribution = factoryDistribution( I );
        return _distribution.Pdf( wo, wh ) * dwh_dwi;
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
        const CudaTrowbridgeReitz _distribution = factoryDistribution( I );
        float3 wh = _distribution.Sample_wh( wo, u );
        const CudaFresnel fresnel = factoryFresnel( I );
        const float eta = CosTheta( wo ) > 0 ? ( fresnel._etaI / fresnel._etaT ).x :
            ( fresnel._etaT / fresnel._etaI ).x;
        if ( !Refract( wo, wh, eta, wi ) ) {
            wi = reflect( wo, wh );
            wh *= NOOR::sign( wh.z );
            const float F = fresnel.evaluate( dot( wi, wh ) ).x;
            const float cosThetaO = CosTheta( wo );
            const float cosThetaI = CosTheta( wi );
            const float3 result = F * S( I ) * _distribution.D( wh ) * _distribution.G( wo, wi ) /
                ( 4.0f * cosThetaI * cosThetaO );
            pdf = _distribution.Pdf( wo, wh ) / ( 4.0f * dot( wo, wh ) );
            return result;
        }
        pdf = Pdf( I, wo, wi );
        return f( I, wo, wi );
    }

    __device__
        void Sample_f(
        const CudaIntersection& I,
        const float3 &wo,
        float3 &wi,
        const float2 &u
        ) const {
        const CudaTrowbridgeReitz _distribution = factoryDistribution( I );
        float3 wh = _distribution.Sample_wh( wo, u );
        const CudaFresnel fresnel = factoryFresnel( I );
        const float eta = CosTheta( wo ) > 0 ? ( fresnel._etaI / fresnel._etaT ).x :
            ( fresnel._etaT / fresnel._etaI ).x;
        if ( !Refract( wo, wh, eta, wi ) ) {
            wi = reflect( wo, wh );
        }
    }

};

class CudaFresnelBlend : public CudaBxDF {
public:
    __device__
        CudaFresnelBlend() :
        CudaBxDF( BxDFType( BSDF_REFLECTION | BSDF_GLOSSY | BSDF_DIELECTRIC ),
                  FresnelBlend ) {}
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
        const CudaTrowbridgeReitz _distribution = factoryDistribution( I );

        const float3 specular =
            _distribution.D( wh ) /
            ( 4.f * NOOR::absDot( wi, wh ) * fmaxf( AbsCosTheta( wi ), AbsCosTheta( wo ) ) ) *
            SchlickFresnel( I, dot( wi, wh ) );
        return diffuse + specular;
    }

    __device__
        float Pdf( const CudaIntersection& I,
                   const float3 &wo, const float3 &wi ) const {
        if ( !SameHemisphere( wo, wi ) ) return 0.f;
        const float3 wh = NOOR::normalize( wo + wi );
        const CudaTrowbridgeReitz _distribution = factoryDistribution( I );

        const float pdf_wh = _distribution.Pdf( wo, wh );
        return .5f * ( AbsCosTheta( wi ) * NOOR_invPI + pdf_wh / ( 4.f * dot( wo, wh ) ) );
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
            wi.z *= NOOR::sign( wi.z );
        } else {
            lu.x = fminf( 2.f * ( lu.x - .5f ), NOOR_ONE_MINUS_EPSILON );
            const CudaTrowbridgeReitz _distribution = factoryDistribution( I );

            // Sample microfacet orientation $\wh$ and reflected direction $\wi$
            const float3 wh = _distribution.Sample_wh( wo, lu );
            wi = Reflect( wo, wh );
            if ( !SameHemisphere( wo, wi ) ) return make_float3( 0.f );
        }
        pdf = Pdf( I, wo, wi );
        return f( I, wo, wi );
    }


    __device__
        void Sample_f(
        const CudaIntersection& I,
        const float3 &wo,
        float3 &wi,
        const float2 &u
        ) const {
        float2 lu = u;
        if ( lu.x < .5f ) {
            lu.x = fminf( 2.f * lu.x, NOOR_ONE_MINUS_EPSILON );
            // Cosine-sample the hemisphere, flipping the direction if necessary
            wi = NOOR::cosineSampleHemisphere( lu );
            //if ( wo.z < 0 ) wi.z *= -1.f;
            wi.z *= NOOR::sign( wi.z );
        } else {
            lu.x = fminf( 2.f * ( lu.x - .5f ), NOOR_ONE_MINUS_EPSILON );
            const CudaTrowbridgeReitz _distribution = factoryDistribution( I );

            // Sample microfacet orientation $\wh$ and reflected direction $\wi$
            const float3 wh = _distribution.Sample_wh( wo, lu );
            wi = Reflect( wo, wh );
        }
    }
};

class CudaFresnelSpecular : public CudaBxDF {
public:
    __device__
        CudaFresnelSpecular() :
        CudaBxDF( BxDFType( BSDF_REFLECTION | BSDF_TRANSMISSION |
                  BSDF_SPECULAR | BSDF_DIELECTRIC ), FresnelSpecular ) {}

    __device__
        float3 f( const CudaIntersection& I,
                  const float3 &wo,
                  const float3 &wi ) const {
        return make_float3( 0.f );
    }


    __device__
        float Pdf( const CudaIntersection& I,
                   const float3 &wo, const float3 &wi ) const {
        return 0.0f;
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
        void Sample_f(
        const CudaIntersection& I,
        const float3 &wo,
        float3 &wi,
        const float2 &u
        ) const {
        const CudaFresnel fresnel = factoryFresnel( I );
        float F = fresnel.evaluate( CosTheta( wo ) ).x;
        if ( u.x < F ) {
            // Compute specular reflection for FresnelSpecular
            // Compute perfect specular reflection direction
            wi = make_float3( -wo.x, -wo.y, wo.z );
        } else {
            // Compute specular transmission for FresnelSpecular
            // Figure out which eta is incident and which is transmitted
            const bool entering = CosTheta( wo ) > 0.0f;
            const float eta = entering ? ( fresnel._etaI / fresnel._etaT ).x :
                ( fresnel._etaT / fresnel._etaI ).x;

            // Compute ray direction for specular transmission
            if ( !Refract( wo, NOOR::faceforward( make_float3( 0, 0, 1 ), wo ), eta, wi ) ) {
                wi = make_float3( -wo.x, -wo.y, wo.z );
            }
        }
    }
};


class CudaClearCoat : public CudaBxDF {
    CudaMicrofacetReflection _substrate;
    CudaSpecularReflection _coat;
public:
    __device__
        CudaClearCoat() :
        CudaBxDF( BxDFType( BSDF_GLOSSY | BSDF_REFLECTION | BSDF_SPECULAR ), ClearCoat ),
        _substrate( CudaMicrofacetReflection( BSDF_CONDUCTOR ) ),
        _coat( CudaSpecularReflection( BSDF_DIELECTRIC ) )
    {}
    __device__
        float absorption( const CudaIntersection& I,
                          const float3& wo,
                          const float3& wi ) const {
        const float sigma = _material_manager.getCoatSigma( I );
        const float thickness = _material_manager.getCoatThickness( I );
        float sigmaA = sigma * thickness;
        if ( sigmaA != 0 ) {
            return expf( -sigmaA * ( 1.f / AbsCosTheta( wo ) + 1.f / AbsCosTheta( wi ) ) );
        }
        return 1.f;
    }
    __device__
        float3 f(
        const CudaIntersection& I
        , const float3 &wo
        , const float3 &wi
        ) const {
        const CudaFresnel fresnel = _coat.factoryFresnel( I );
        const float eta = ( fresnel._etaI / fresnel._etaT ).x;
        const float F0 = fresnel.evaluate( CosTheta( wo ) ).x;
        const float F1 = fresnel.evaluate( CosTheta( wi ) ).x;
        const float3 n = make_float3( 0, 0, 1 );
        float3 wo_c, wi_c;
        Refract( wo, n, eta, wo_c );
        Refract( wi, n, eta, wi_c );
        wo_c *= -1;
        wi_c *= -1;
        const float3 result = _substrate.f( I, wo_c, wi_c ) * ( 1.f - F0 ) * ( 1.f - F1 );
        return result * absorption( I, wo_c, wi_c );
    }

    __device__
        float Pdf(
        const CudaIntersection& I
        , const float3 &wo
        , const float3 &wi
        ) const {
        const CudaFresnel fresnel = _coat.factoryFresnel( I );
        const float eta = ( fresnel._etaI / fresnel._etaT ).x;
        const float F = fresnel.evaluate( CosTheta( wo ) ).x;
        const float3 n = make_float3( 0, 0, 1 );

        float3 wo_c, wi_c;
        Refract( wo, n, eta, wo_c );
        Refract( wi, n, eta, wi_c );
        wo_c *= -1;
        wi_c *= -1;

        const float w = _material_manager.getCoatWeight( I );
        const float coatPdf = F*w / ( F*w + ( 1 - F ) * ( 1 - w ) );
        const float substratePdf = _substrate.Pdf( I, wo_c, wi_c ) * ( 1.f - coatPdf );
        return substratePdf + coatPdf;
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
        const CudaFresnel fresnel = _coat.factoryFresnel( I );
        const float eta = ( fresnel._etaI / fresnel._etaT ).x;
        const float F0 = fresnel.evaluate( CosTheta( wo ) ).x;
        const float w = _material_manager.getCoatWeight( I );
        const float probSpecular = F0*w / ( F0*w + ( 1 - F0 ) * ( 1 - w ) );
        const float probSubstrate = 1.f - probSpecular;
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
            result = _coat.Sample_f( I, wo, wi, sample, pdf, sampledType );
            pdf *= probSpecular;
        } else {
            float3 wi_c, wo_c;
            float3 n = make_float3( 0, 0, 1 );
            // refract in
            Refract( wo, n, eta, wo_c );
            result = _substrate.Sample_f( I, -wo_c, wi_c, sample, pdf, sampledType );
            float F1 = fresnel.evaluate( CosTheta( -wi_c ) ).x;
            // refract out
            Refract( -wi_c, -n, 1.f / eta, wi );
            if ( F1 == 1.f ) {
                pdf = 0.f;
                return _constant_spec._black;
            }
            result *= absorption( I, wo_c, wi_c ) * ( 1.f - F0 )*( 1.f - F1 );

            // wi_c refracts out of the surface cosine factor
            result /= AbsCosTheta( wi_c );

            pdf *= probSubstrate;
        }
        return result;
    }

    __device__
        void Sample_f(
        const CudaIntersection& I,
        const float3 &wo,
        float3 &wi,
        const float2 &u
        ) const {
        const CudaFresnel fresnel = _coat.factoryFresnel( I );
        const float eta = ( fresnel._etaI / fresnel._etaT ).x;
        const float F0 = fresnel.evaluate( CosTheta( wo ) ).x;
        const float w = _material_manager.getCoatWeight( I );
        const float probSpecular = F0*w / ( F0*w + ( 1 - F0 ) * ( 1 - w ) );
        bool choseSpecular = false;
        float2 sample( u );
        if ( sample.y < probSpecular ) {
            sample.y /= probSpecular;
            choseSpecular = true;
        } else {
            sample.y = ( sample.y - probSpecular ) / ( 1.f - probSpecular );
            choseSpecular = false;
        }
        if ( choseSpecular ) {
            _coat.Sample_f( I, wo, wi, sample );
        } else {
            float3 wi_c, wo_c;
            float3 n = make_float3( 0, 0, 1 );
            // refract in
            Refract( wo, n, eta, wo_c );
            _substrate.Sample_f( I, -wo_c, wi_c, sample );
            // refract out
            if ( !Refract( -wi_c, -n, 1.f / eta, wi ) )
                _coat.Sample_f( I, wo, wi, sample );
        }
    }
};

__global__
void setup_bxdfs( CudaBxDF** bxdfs ) {
    bxdfs[ShadowCatcher] = new CudaShadowCatcher();
    bxdfs[LambertReflection] = new CudaLambertianReflection();
    bxdfs[SpecularReflectionNoOp] = new CudaSpecularReflection( BSDF_NOOP );
    bxdfs[SpecularReflectionDielectric] = new CudaSpecularReflection( BSDF_DIELECTRIC );
    bxdfs[MicrofacetReflectionDielectric] = new CudaMicrofacetReflection( BSDF_DIELECTRIC );
    bxdfs[MicrofacetReflectionConductor] = new CudaMicrofacetReflection( BSDF_CONDUCTOR );
    bxdfs[LambertTransmission] = new CudaLambertianTransmission();
    bxdfs[SpecularTransmission] = new CudaSpecularTransmission();
    bxdfs[MicrofacetTransmission] = new CudaMicrofacetTransmission();
    bxdfs[FresnelBlend] = new CudaFresnelBlend();
    bxdfs[FresnelSpecular] = new CudaFresnelSpecular();
    bxdfs[ClearCoat] = new CudaClearCoat();
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

    CudaBxDFManager() = default;

    __host__
        CudaBxDFManager( int i ) {
        checkNoorErrors( NOOR::malloc( &_bxdfs, NUM_BXDFS * sizeof( CudaBxDF* ) ) );
        setup_bxdfs << < 1, 1 >> > ( _bxdfs );
        checkNoorErrors( cudaDeviceSynchronize() );
        checkNoorErrors( cudaPeekAtLastError() );
    }

    __host__
        void free() {
        free_bxdfs << < 1, 1 >> > ( _bxdfs );
        checkNoorErrors( cudaFree( _bxdfs ) );
    }
};

__constant__
CudaBxDFManager _bxdf_manager;
#endif /* CUDABXDF_CUH */

