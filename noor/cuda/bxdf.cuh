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
    BSDF_ALL = BSDF_DIFFUSE | BSDF_GLOSSY | BSDF_SPECULAR | BSDF_REFLECTION |
    BSDF_TRANSMISSION,
};

class CudaBxDF {
protected:
    BxDFType _type{ BSDF_ALL };
public:
    __device__
        CudaBxDF( BxDFType type ) :_type( type ) {}
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
};

template<typename BXDF>
class CudaScaledBxDF : public CudaBxDF {
public:
    // ScaledBxDF Public Methods
    __device__
        CudaScaledBxDF( const BXDF &bxdf, const float3 &scale ) :
        CudaBxDF( BxDFType( bxdf->type ) ),
        _bxdf( bxdf ), _scale( scale ) {}
    __device__
        float3 f( const float3 &wo, const float3 &wi ) const {
        return _scale*_bxdf.f( wo, wi );
    }
    __device__
        float3 Sample_f( const float3 &wo, float3 &wi, const float2 &u, float &pdf, BxDFType& sampledType ) const {
        return _scale*_bxdf.Sample_f( wo, wi, u, pdf, sampledType );
    }
    __device__
        float Pdf( const float3 &wo, const float3 &wi ) const {
        return _bxdf.Pdf( wo, wi );
    }
private:
    const BXDF &_bxdf;
    float3 _scale;
};

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

class CudaLambertianReflection : public CudaBxDF {
    float3 _R;
public:
    // LambertianReflection Public Methods
    __device__
        CudaLambertianReflection( const float3 &R ) :
        CudaBxDF( BxDFType( BSDF_REFLECTION | BSDF_DIFFUSE ) )
        , _R( R ) {}
    __device__
        float3 f( const float3 &wo, const float3 &wi ) const { return _R*NOOR_invPI; }
    __device__
        float3 Sample_f(
        const float3 &wo
        , float3 &wi
        , const float2 &u
        , float &pdf
        , BxDFType &sampledType
        ) const {
        sampledType = _type;
        // Cosine-sample the hemisphere, flipping the direction if necessary
        // default left handed since we are in bxdf coordinate system
        wi = NOOR::cosineSampleHemisphere( u );
        if ( wo.z < 0 ) wi.z *= -1.0f;
        pdf = Pdf( wo, wi );
        return f( wo, wi );
    }
    __device__
        float Pdf( const float3 &wo, const float3 &wi ) const {
        return SameHemisphere( wo, wi ) ? AbsCosTheta( wi ) * NOOR_invPI : 0.0f;
    }
};

class CudaLambertianTransmission : public CudaBxDF {
    float3 _T;
public:
    // LambertianTransmission Public Methods
    __device__
        CudaLambertianTransmission( const float3 &T ) :
        CudaBxDF( BxDFType( BSDF_TRANSMISSION | BSDF_DIFFUSE ) )
        , _T( T ) {}
    __device__
        float3 f( const float3 &wo, const float3 &wi ) const { return _T * NOOR_invPI; }
    __device__
        float3 Sample_f(
        const float3 &wo
        , float3 &wi
        , const float2 &u
        , float &pdf
        , BxDFType &sampledType
        ) const {
        sampledType = _type;
        // Cosine-sample the hemisphere, flipping the direction if necessary
        // default left handed since we are in bxdf coordinate system
        wi = NOOR::cosineSampleHemisphere( u );
        if ( wo.z > 0.0f ) wi.z *= -1.0f;
        pdf = Pdf( wo, wi );
        return f( wo, wi );
    }
    __device__
        float Pdf( const float3 &wo, const float3 &wi ) const {
        return !SameHemisphere( wo, wi ) ? AbsCosTheta( wi ) * NOOR_invPI : 0.0f;
    }
};

class CudaOrenNayar : public CudaBxDF {
    float3 _R;
    float _A, _B;
public:
    // OrenNayar Public Methods
    __device__
        CudaOrenNayar( const float3 &R, float sigma ) :
        CudaBxDF( BxDFType( BSDF_REFLECTION | BSDF_DIFFUSE ) )
        , _R( R ) {
        sigma = NOOR::deg2rad( sigma );
        const float sigma2 = sigma * sigma;
        _A = 1.0f - ( sigma2 / ( 2.0f * ( sigma2 + 0.33f ) ) );
        _B = 0.45f * sigma2 / ( sigma2 + 0.09f );
    }
    __device__
        float3 f( const float3 &wo, const float3 &wi ) const {
        const float sinThetaI = SinTheta( wi );
        const float sinThetaO = SinTheta( wo );
        // Compute cosine term of Oren-Nayar model
        float maxCos = 0;
        if ( sinThetaI > 1e-4f && sinThetaO > 1e-4f ) {
            const float sinPhiI = SinPhi( wi ), cosPhiI = CosPhi( wi );
            const float sinPhiO = SinPhi( wo ), cosPhiO = CosPhi( wo );
            const float dCos = cosPhiI * cosPhiO + sinPhiI * sinPhiO;
            maxCos = fmaxf( 0.0f, dCos );
        }

        // Compute sine and tangent terms of Oren-Nayar model
        float sinAlpha, tanBeta;
        if ( AbsCosTheta( wi ) > AbsCosTheta( wo ) ) {
            sinAlpha = sinThetaO;
            tanBeta = sinThetaI / AbsCosTheta( wi );
        } else {
            sinAlpha = sinThetaI;
            tanBeta = sinThetaO / AbsCosTheta( wo );
        }
        return _R * NOOR_invPI * ( _A + _B * maxCos * sinAlpha * tanBeta );
    }
    __device__
        float3 Sample_f(
        const float3 &wo
        , float3 &wi
        , const float2 &u
        , float &pdf
        , BxDFType &sampledType
        ) const {
        sampledType = _type;
        // Cosine-sample the hemisphere, flipping the direction if necessary
        wi = NOOR::cosineSampleHemisphere( u );
        if ( wo.z < 0 ) wi.z *= -1.0f;
        pdf = Pdf( wo, wi );
        return f( wo, wi );
    }
    __device__
        float Pdf( const float3 &wo, const float3 &wi ) const {
        return SameHemisphere( wo, wi ) ? AbsCosTheta( wi ) * NOOR_invPI : 0.0f;
    }
};

class CudaFresnelSpecular : public CudaBxDF {
    // FresnelSpecular Private Data
    float3 _S, _T;
    CudaFresnelDielectric _fresnel;
public:
    // FresnelSpecular Public Methods
    __device__
        CudaFresnelSpecular( const float3 &R, const float3 &T, const CudaFresnelDielectric& fresnel ) :
        CudaBxDF( BxDFType( BSDF_REFLECTION | BSDF_TRANSMISSION | BSDF_SPECULAR ) )
        , _S( R )
        , _T( T )
        , _fresnel( fresnel ) {}

    __device__
        float3 f( const float3 &wo, const float3 &wi ) const { return make_float3( 0.f ); }

    __device__
        float3 Sample_f(
        const float3 &wo
        , float3 &wi
        , const float2 &u
        , float &pdf
        , BxDFType &sampledType
        ) const {
        float F = _fresnel.evaluate( CosTheta( wo ) ).x;
        if ( u.x < F ) {
            // Compute specular reflection for _FresnelSpecular_
            // Compute perfect specular reflection direction
            wi = make_float3( -wo.x, -wo.y, wo.z );
            pdf = F;
            sampledType = BxDFType( BSDF_REFLECTION | BSDF_SPECULAR );
            return F * _S / AbsCosTheta( wi );
        } else {
            // Compute specular transmission for _FresnelSpecular_
            // Figure out which eta is incident and which is transmitted
            const bool entering = CosTheta( wo ) > 0.0f;
            const float eta = entering ? _fresnel._etaI / _fresnel._etaT : _fresnel._etaT / _fresnel._etaI;

            // Compute ray direction for specular transmission
            if ( !Refract( wo, NOOR::faceforward( make_float3( 0, 0, 1 ), wo ), eta, wi ) ) {
                return _constant_spec._black;
            }
            // Account for non-symmetry with transmission to different medium
            pdf = 1.0f - F;
            sampledType = BxDFType( BSDF_TRANSMISSION | BSDF_SPECULAR );
            return _T * eta * eta * ( 1.0f - F ) / AbsCosTheta( wi );
        }
    }
    __device__
        float Pdf( const float3 &wo, const float3 &wi ) const { return 0.0f; }
};

template<class Fresnel>
class CudaSpecularReflection : public CudaBxDF {
public:
    // SpecularReflection Private Data
    float3 _S;
    Fresnel _fresnel;
    // SpecularReflection Public Methods
    __device__
        CudaSpecularReflection( const float3 &S, const Fresnel& fresnel ) :
        CudaBxDF( BxDFType( BSDF_REFLECTION | BSDF_SPECULAR ) )
        , _S( S )
        , _fresnel( fresnel ) {}
    __device__
        float3 f( const float3 &wo, const float3 &wi ) const { return make_float3( 0.f ); }
    __device__
        float3 Sample_f(
        const float3 &wo
        , float3 &wi
        , const float2 &u
        , float &pdf
        , BxDFType &sampledType
        ) const {
        sampledType = _type;
        // Compute perfect specular reflection direction
        wi = make_float3( -wo.x, -wo.y, wo.z );
        pdf = 1.0f;
        return _fresnel.evaluate( CosTheta( wi ) ) * _S / AbsCosTheta( wi );
    }
    __device__
        float Pdf( const float3 &wo, const float3 &wi ) const { return 0; }

};

class CudaSpecularTransmission : public CudaBxDF {
    // SpecularTransmission Private Data
    float3 _T;
    CudaFresnelDielectric _fresnel;
public:
    // SpecularTransmission Public Methods
    __device__
        CudaSpecularTransmission( const float3 &T, const float3& etaI, const float3& etaT ) :
        CudaBxDF( BxDFType( BSDF_TRANSMISSION | BSDF_SPECULAR ) )
        , _T( T )
        , _fresnel( etaI, etaT ) {}
    __device__
        float3 f( const float3 &wo, const float3 &wi ) const { return make_float3( 0.f ); }
    __device__
        float3 Sample_f(
        const float3 &wo
        , float3 &wi
        , const float2 &u
        , float &pdf
        , BxDFType &sampledType
        ) const {
        sampledType = _type;
        // Figure out which $\eta$ is incident and which is transmitted
        const bool entering = CosTheta( wo ) > 0.0f;
        const float etaI = entering ? _fresnel._etaI : _fresnel._etaT;
        const float etaT = entering ? _fresnel._etaT : _fresnel._etaI;

        // Compute ray direction for specular transmission
        if ( !Refract( wo, NOOR::faceforward( make_float3( 0, 0, 1 ), wo ), etaI / etaT, wi ) ) {
            return _constant_spec._black;
        }
        pdf = 1.0f;
        float3 ft = _T * ( make_float3( 1.f ) - _fresnel.evaluate( CosTheta( wi ) ) );
        // Account for non-symmetry with transmission to different medium
        ft *= ( etaI * etaI ) / ( etaT * etaT );
        return ft / AbsCosTheta( wi );
    }
    __device__
        float Pdf( const float3 &wo, const float3 &wi ) const { return 0; }

};

template<class Distribution>
class CudaFresnelBlend : public CudaBxDF {
public:
    // FresnelBlend Private Data
    const float3 _R, _S;
    Distribution _distribution;
    // FresnelBlend Public Methods
    __device__
        CudaFresnelBlend( const float3 &R, const float3 &S, const Distribution& distribution ) :
        CudaBxDF( BxDFType( BSDF_REFLECTION | BSDF_GLOSSY ) )
        , _R( R )
        , _S( S )
        , _distribution( distribution ) {}
    __device__
        float3 f( const float3 &wo, const float3 &wi ) const {
        const float3 diffuse = ( 28.f / ( 23.f * NOOR_PI ) ) * _R * ( make_float3( 1.f ) - _S ) *
            ( 1.f - NOOR::pow5f( 1.f - .5f * AbsCosTheta( wi ) ) ) *
            ( 1.f - NOOR::pow5f( 1.f - .5f * AbsCosTheta( wo ) ) );
        float3 wh = wi + wo;
        if ( wh.x == 0 && wh.y == 0 && wh.z == 0 ) return make_float3( 0 );
        wh = NOOR::normalize( wh );
        const float3 specular =
            _distribution.D( wh ) /
            ( 4.f * NOOR::absDot( wi, wh ) * fmaxf( AbsCosTheta( wi ), AbsCosTheta( wo ) ) ) *
            SchlickFresnel( dot( wi, wh ) );
        return diffuse + specular;
    }
    __device__
        float3 SchlickFresnel( float cosTheta ) const {
        return _S + NOOR::pow5f( 1.f - cosTheta ) * ( make_float3( 1.f ) - _S );
    }
    __device__
        float3 Sample_f(
        const float3 &wo
        , float3 &wi
        , const float2 &u
        , float &pdf
        , BxDFType &sampledType
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
        pdf = Pdf( wo, wi );
        return f( wo, wi );
    }
    __device__
        float Pdf( const float3 &wo, const float3 &wi ) const {
        if ( !SameHemisphere( wo, wi ) ) return 0.f;
        float3 wh = NOOR::normalize( wo + wi );
        float pdf_wh = _distribution.Pdf( wo, wh );
        return .5f * ( AbsCosTheta( wi ) * NOOR_invPI + pdf_wh / ( 4.f * dot( wo, wh ) ) );
    }

};

__forceinline__ __device__
float SchlickWeight( float cosTheta ) {
    float m = clamp( 1.f - cosTheta, 0.f, 1.f );
    return ( m * m ) * ( m * m ) * m;
}

__forceinline__ __device__
float3 sheen( const float3 &wo, const float3 &wi ) {
    float3 wh = wi + wo;
    if ( wh.x == 0 && wh.y == 0 && wh.z == 0 ) return make_float3( 0.f );
    wh = normalize( wh );
    float cosThetaD = dot( wi, wh );
    const float3 R = make_float3( 1.f );
    return R * SchlickWeight( cosThetaD );
}

template<class Distribution, class Fresnel>
class CudaMicrofacetReflection : public CudaBxDF {
public:
    // MicrofacetReflection Private Data
    float3 _R;
    Distribution _distribution;
    Fresnel _fresnel;
    // MicrofacetReflection Public Methods
    __device__
        CudaMicrofacetReflection(
        const float3 &R
        , const Distribution& distribution
        , const Fresnel& fresnel
        )
        : CudaBxDF( BxDFType( BSDF_REFLECTION | BSDF_GLOSSY ) ),
        _R( R )
        , _distribution( distribution )
        , _fresnel( fresnel ) {}

    __device__
        float3 f( const float3 &wo, const float3 &wi ) const {
        float cosThetaO = AbsCosTheta( wo ), cosThetaI = AbsCosTheta( wi );
        // Handle degenerate cases for microfacet reflection
        if ( cosThetaI == 0.0f || cosThetaO == 0.0f ) return _constant_spec._black;
        float3 wh = wi + wo;
        if ( wh.x == 0.0f && wh.y == 0 && wh.z == 0.0f ) return _constant_spec._black;
        wh = NOOR::normalize( wh );
        const float3 F = _fresnel.evaluate( dot( wo, wh ) );
        return sheen( wo, wi ) + _R * _distribution.D( wh ) * _distribution.G( wo, wi ) * F /
            ( 4.0f * cosThetaI * cosThetaO );
    }

    __device__
        float3 Sample_f(
        const float3 &wo
        , float3 &wi
        , const float2 &u
        , float &pdf
        , BxDFType &sampledType
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
        return f( wo, wi );
    }
    __device__
        float Pdf( const float3 &wo, const float3 &wi ) const {
        if ( !SameHemisphere( wo, wi ) ) {
            return 0.0f;
        }
        const float3 wh = NOOR::normalize( wo + wi );
        return _distribution.Pdf( wo, wh ) / ( 4.0f * dot( wo, wh ) );
    }
};

template<typename Distribution, typename Fresnel>
class CudaMicrofacetTransmission : public CudaBxDF {
public:
    // MicrofacetReflection Private Data
    float3 _T;
    Distribution _distribution;
    Fresnel _fresnel;
    // MicrofacetReflection Public Methods
    __device__
        CudaMicrofacetTransmission( const float3 &T
                                    , const Distribution& distribution
                                    , const Fresnel& fresnel
        ) :
        CudaBxDF( BxDFType( BSDF_TRANSMISSION | BSDF_GLOSSY ) )
        , _T( T )
        , _distribution( distribution )
        , _fresnel( fresnel ) {}

    __device__
        float3 f( const float3 &wo, const float3 &wi ) const {
        if ( SameHemisphere( wo, wi ) ) return _constant_spec._black;  // transmission only

        const float cosThetaO = CosTheta( wo );
        const float cosThetaI = CosTheta( wi );
        if ( cosThetaI == 0.0f || cosThetaO == 0.0f ) return _constant_spec._black;

        // Compute $\wh$ from $\wo$ and $\wi$ for microfacet transmission
        float eta = CosTheta( wo ) > 0.0f ? ( _fresnel._etaT / _fresnel._etaI ) : ( _fresnel._etaI / _fresnel._etaT );
        float3 wh = NOOR::normalize( wo + wi * eta );
        wh *= NOOR::sign( wh.z );

        const float3 F = _fresnel.evaluate( dot( wo, wh ) );

        const float sqrtDenom = dot( wo, wh ) + eta * dot( wi, wh );
        const float factor = 1.f / eta;
        const float3 result = sheen( wo, wi ) + ( make_float3( 1.f ) - F ) * _T *
            fabsf( _distribution.D( wh ) * _distribution.G( wo, wi ) * eta * eta *
                   NOOR::absDot( wi, wh ) * NOOR::absDot( wo, wh ) * factor * factor /
                   ( cosThetaI * cosThetaO * sqrtDenom * sqrtDenom ) );
        return result;
    }
    __device__
        float3 Sample_f(
        const float3 &wo
        , float3 &wi
        , const float2 &u
        , float &pdf
        , BxDFType &sampledType
        ) const {
        sampledType = _type;
        if ( wo.z == 0 ) return _constant_spec._black;
        float3 wh = _distribution.Sample_wh( wo, u );
        const float eta = CosTheta( wo ) > 0 ? ( _fresnel._etaI / _fresnel._etaT ) : ( _fresnel._etaT / _fresnel._etaI );
        if ( !Refract( wo, wh, eta, wi ) ) return _constant_spec._black;
        pdf = Pdf( wo, wi );
        return f( wo, wi );
    }
    __device__
        float Pdf( const float3 &wo, const float3 &wi ) const {
        if ( SameHemisphere( wo, wi ) ) return 0;
        const float eta = CosTheta( wo ) > 0 ? ( _fresnel._etaT / _fresnel._etaI ) : ( _fresnel._etaI / _fresnel._etaT );
        const float3 wh = NOOR::normalize( wo + wi * eta );

        const float sqrtDenom = dot( wo, wh ) + eta * dot( wi, wh );
        const float dwh_dwi =
            fabsf( ( eta * eta * dot( wi, wh ) ) / ( sqrtDenom * sqrtDenom ) );
        return _distribution.Pdf( wo, wh ) * dwh_dwi;
    }
};

template class CudaSpecularReflection<CudaFresnelNoOp>;
template class CudaSpecularReflection<CudaFresnelDielectric>;
template class CudaMicrofacetReflection<CudaTrowbridgeReitzDistribution, CudaFresnelDielectric>;
template class CudaMicrofacetTransmission<CudaTrowbridgeReitzDistribution, CudaFresnelDielectric>;
template class CudaMicrofacetReflection<CudaTrowbridgeReitzDistribution, CudaFresnelConductor>;
template class CudaFresnelBlend<CudaTrowbridgeReitzDistribution>;

using LambertionReflectionBxDF = CudaLambertianReflection;
using LambertionTransmissionBxDF = CudaLambertianTransmission;
using MirrorReflectionBxDF = CudaSpecularReflection<CudaFresnelNoOp>;
using SpecularReflectionBxDF = CudaSpecularReflection<CudaFresnelDielectric>;
using DielectricReflectionBxDF = CudaMicrofacetReflection<CudaTrowbridgeReitzDistribution, CudaFresnelDielectric>;
using DielectricTransmissionBxDF = CudaMicrofacetTransmission<CudaTrowbridgeReitzDistribution, CudaFresnelDielectric>;
using ConductorReflectionBxDF = CudaMicrofacetReflection<CudaTrowbridgeReitzDistribution, CudaFresnelConductor>;
using FresnelBlend = CudaFresnelBlend<CudaTrowbridgeReitzDistribution>;

#endif /* CUDABXDF_CUH */
