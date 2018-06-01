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
    BSDF_CONDUCTOR = 1 << 5,
    BSDF_DIELECTRIC = 1 << 6,
    BSDF_NOOP = 1 << 7,
    BSDF_ALL = BSDF_DIFFUSE | BSDF_GLOSSY | BSDF_SPECULAR |
    BSDF_REFLECTION | BSDF_TRANSMISSION |
    BSDF_CONDUCTOR | BSDF_DIELECTRIC | BSDF_NOOP
};

enum BxDFIndex {
    //reflections
    LambertReflection = 0,
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
    NUM_BXDFS
};

class CudaBxDF {
public:
    BxDFType _type;
    BxDFIndex _index;
    CudaTrowbridgeReitz _distribution;
    bool _distribution_is_set{ false };
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
    /*__device__
        bool isMictosurface() const {
        return (
            _index == MicrofacetReflectionDielectric ||
            _index == MicrofacetReflectionConductor ||
            _index == MicrofacetTransmission );
    }*/
    __device__
        bool isConductor() const {
        return ( _type & BSDF_CONDUCTOR );
    }
    __device__
        void factoryDistribution( const CudaIntersection& I ) {
        if ( !_distribution_is_set ) {
            _distribution = CudaTrowbridgeReitz( _material_manager.getRoughness( I ) );
            _distribution_is_set = true;
        }
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
        if ( _type & BSDF_CONDUCTOR )
            return CudaFresnel( etaI, etaT, k( I ) );
        else if ( _type & BSDF_DIELECTRIC ) {
            return CudaFresnel( etaI, etaT );
        } else {
            return CudaFresnel();
        }
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
        CudaBxDF( BxDFType( BSDF_REFLECTION | BSDF_TRANSMISSION |
                  BSDF_SPECULAR | BSDF_DIELECTRIC ), FresnelSpecular ) {}

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
        const float3 specular =
            _distribution.D( wh ) /
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
            const float3 wh = _distribution.Sample_wh( wo, lu );
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
        float pdf_wh = _distribution.Pdf( wo, wh );
        return .5f * ( AbsCosTheta( wi ) * NOOR_invPI + pdf_wh / ( 4.f * dot( wo, wh ) ) );
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
        float cosThetaO = AbsCosTheta( wo ), cosThetaI = AbsCosTheta( wi );
        // Handle degenerate cases for microfacet reflection
        if ( cosThetaI == 0.0f || cosThetaO == 0.0f ) return _constant_spec._black;
        float3 wh = wi + wo;
        if ( wh.x == 0.0f && wh.y == 0 && wh.z == 0.0f ) return _constant_spec._black;
        wh = NOOR::normalize( wh );
        const CudaFresnel fresnel = factoryFresnel( I );
        const float3 F = fresnel.evaluate( dot( wo, wh ) );
        return R( I ) * _distribution.D( wh ) * _distribution.G( wo, wi ) * F /
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
        const float3 wh = _distribution.Sample_wh( wo, u );
        wi = Reflect( wo, wh );
        if ( !SameHemisphere( wo, wi ) ) {
            pdf = 0.0f;
            return _constant_spec._black;
        }

        // Compute PDF of _wi_ for microfacet reflection
        pdf = _distribution.Pdf( wo, wh ) / ( 4.0f * dot( wo, wh ) );
        return f( I, wo, wi );
    }
    __device__
        float Pdf( const CudaIntersection& I,
                   const float3 &wo, const float3 &wi ) const {
        if ( !SameHemisphere( wo, wi ) ) {
            return 0.0f;
        }
        const float3 wh = NOOR::normalize( wo + wi );
        return _distribution.Pdf( wo, wh ) / ( 4.0f * dot( wo, wh ) );
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
        const float3 result = ( make_float3( 1.f ) - F ) * T( I ) *
            fabsf( _distribution.D( wh ) * _distribution.G( wo, wi ) * eta * eta *
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
        const float3 wh = _distribution.Sample_wh( wo, u );
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
        return _distribution.Pdf( wo, wh ) * dwh_dwi;
    }
};


__global__
void setup_bxdfs( CudaBxDF** bxdfs,
                  // reflections
                  CudaLambertianReflection* _lambertReflection,
                  CudaSpecularReflection* _specularReflectionNoOp,
                  CudaSpecularReflection* _specularReflectionDielectric,
                  CudaMicrofacetReflection* _microfacetReflectionDielectric,
                  CudaMicrofacetReflection* _microfacetReflectionConductor,
                  // transmissions
                  CudaLambertianTransmission* _lambertTransmission,
                  CudaSpecularTransmission* _specularTransmission,
                  CudaMicrofacetTransmission* _microfacetTransmission,
                  // multi-lobes
                  CudaFresnelBlend* _fresnelBlend,
                  CudaFresnelSpecular* _fresnelSpecular
) {
    _lambertReflection = new CudaLambertianReflection();
    bxdfs[LambertReflection] = _lambertReflection;

    _specularReflectionNoOp = new CudaSpecularReflection( BSDF_NOOP );
    bxdfs[SpecularReflectionNoOp] = _specularReflectionNoOp;

    _specularReflectionDielectric = new CudaSpecularReflection( BSDF_DIELECTRIC );
    bxdfs[SpecularReflectionDielectric] = _specularReflectionDielectric;

    _microfacetReflectionDielectric = new CudaMicrofacetReflection( BSDF_DIELECTRIC );
    bxdfs[MicrofacetReflectionDielectric] = _microfacetReflectionDielectric;

    _microfacetReflectionConductor = new CudaMicrofacetReflection( BSDF_CONDUCTOR );
    bxdfs[MicrofacetReflectionConductor] = _microfacetReflectionConductor;

    _lambertTransmission = new CudaLambertianTransmission();
    bxdfs[LambertTransmission] = _lambertTransmission;

    _specularTransmission = new CudaSpecularTransmission();
    bxdfs[SpecularTransmission] = _specularTransmission;

    _microfacetTransmission = new CudaMicrofacetTransmission();
    bxdfs[MicrofacetTransmission] = _microfacetTransmission;

    _fresnelBlend = new CudaFresnelBlend();
    bxdfs[FresnelBlend] = _fresnelBlend;

    _fresnelSpecular = new CudaFresnelSpecular();
    bxdfs[FresnelSpecular] = _fresnelSpecular;
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

    // reflections
    CudaLambertianReflection* _lambertReflection;
    CudaSpecularReflection* _specularReflectionNoOp;
    CudaSpecularReflection* _specularReflectionDielectric;
    CudaMicrofacetReflection* _microfacetReflectionDielectric;
    CudaMicrofacetReflection* _microfacetReflectionConductor;
    // transmissions
    CudaLambertianTransmission* _lambertTransmission;
    CudaSpecularTransmission* _specularTransmission;
    CudaMicrofacetTransmission* _microfacetTransmission;
    // multi-lobes
    CudaFresnelBlend* _fresnelBlend;
    CudaFresnelSpecular* _fresnelSpecular;

    CudaBxDFManager() = default;

    __host__
        CudaBxDFManager( int i ) {
        NOOR::malloc( (void**) &_bxdfs, NUM_BXDFS * sizeof( CudaBxDF* ) );
        setup_bxdfs << < 1, 1 >> > ( _bxdfs,
                                     // reflections
                                     _lambertReflection,
                                     _specularReflectionNoOp,
                                     _specularReflectionDielectric,
                                     _microfacetReflectionDielectric,
                                     _microfacetReflectionConductor,
                                     // transmissions
                                     _lambertTransmission,
                                     _specularTransmission,
                                     _microfacetTransmission,
                                     // multi-lobes
                                     _fresnelBlend,
                                     _fresnelSpecular
                                     );
    }

    __host__
        void free() {
        free_bxdfs << < 1, 1 >> > ( _bxdfs );
        cudaFree( _bxdfs );
    }
};

__constant__
CudaBxDFManager _bxdf_manager;
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