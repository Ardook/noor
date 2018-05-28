#include "path_tracer.cuh"
#include "bxdf.cuh"


__device__ __forceinline__
CudaFresnelSpecular::CudaFresnelSpecular( const float3 &R, const float3 &T, const CudaFresnelDielectric& fresnel ) :
	CudaBxDF( BxDFType( BSDF_REFLECTION | BSDF_TRANSMISSION | BSDF_SPECULAR ) )
	, _S( R )
	, _T( T )
	, _fresnel( fresnel ) {}

__device__ __forceinline__
float3 CudaFresnelSpecular::f( const float3 &wo, const float3 &wi ) const { return make_float3( 0.f ); }

__device__ 
float3 CudaFresnelSpecular::Sample_f(
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
		// Figure out which $\eta$ is incident and which is transmitted
		const bool entering = CosTheta( wo ) > 0.0f;
		const float eta = entering ? _fresnel._etaI / _fresnel._etaT : _fresnel._etaT / _fresnel._etaI;

		// Compute ray direction for specular transmission
		if ( !Refract( wo, NOOR::faceforward( make_float3( 0, 0, 1 ), wo ), eta, wi ) ) {
			return make_float3( 0.f );
		}
		// Account for non-symmetry with transmission to different medium
		pdf = 1.0f - F;
		sampledType = BxDFType( BSDF_TRANSMISSION | BSDF_SPECULAR );
		return _T *eta * eta * ( 1.0f - F ) / AbsCosTheta( wi );
	}
}

__device__ __forceinline__
float CudaFresnelSpecular::Pdf( const float3 &wo, const float3 &wi ) const { return 0.0f; }

template<class Fresnel>
__device__ __forceinline__
CudaSpecularReflection<Fresnel>::CudaSpecularReflection( const float3 &S, const Fresnel& fresnel ) :
	CudaBxDF( BxDFType( BSDF_REFLECTION | BSDF_SPECULAR ) )
	, _S( S )
	, _fresnel( fresnel ) {}

template<class Fresnel>
__device__ __forceinline__
float3 CudaSpecularReflection<Fresnel>::f( const float3 &wo, const float3 &wi ) const { return make_float3( 0.f ); }

template<class Fresnel>
__device__ __forceinline__
float3 CudaSpecularReflection<Fresnel>::Sample_f(
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

template<class Fresnel>
__device__ __forceinline__
float CudaSpecularReflection<Fresnel>::Pdf( const float3 &wo, const float3 &wi ) const { return 0; }

// SpecularTransmission Public Methods
__device__ __forceinline__
CudaSpecularTransmission::CudaSpecularTransmission( const float3 &T, const float3& etaI, const float3& etaT ) :
	CudaBxDF( BxDFType( BSDF_TRANSMISSION | BSDF_SPECULAR ) )
	, _T( T )
	, _fresnel( etaI, etaT ) {}
__device__ __forceinline__
float3 CudaSpecularTransmission::f( const float3 &wo, const float3 &wi ) const { return make_float3( 0.f ); }
__device__ __forceinline__
float3 CudaSpecularTransmission::Sample_f(
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
		return make_float3( 0.f );
	}
	pdf = 1.0f;
	const float3 ft = _T * ( make_float3( 1.f ) - _fresnel.evaluate( CosTheta( wi ) ) );
	// Account for non-symmetry with transmission to different medium
	return ft / AbsCosTheta( wi );
}
__device__ __forceinline__
float CudaSpecularTransmission::Pdf( const float3 &wo, const float3 &wi ) const { return 0; }

// FresnelBlend Public Methods
template<typename Distribution>
__device__ __forceinline__
CudaFresnelBlend<Distribution>::CudaFresnelBlend(
	const float3 &R,
	const float3 &S,
	const Distribution& distribution ) :
	CudaBxDF( BxDFType( BSDF_REFLECTION | BSDF_GLOSSY ) )
	, _R( R )
	, _S( S )
	, _distribution( distribution ) {}

template<typename Distribution>
__device__ __forceinline__
float3 CudaFresnelBlend<Distribution>::f(
	const float3 &wo,
	const float3 &wi ) const {
	const float3 diffuse = ( 28.f / ( 23.f * NOOR_PI ) ) * _R * ( make_float3( 1.f ) - _S ) *
		( 1.f - NOOR::pow5f( 1.f - .5f * AbsCosTheta( wi ) ) ) *
		( 1.f - NOOR::pow5f( 1.f - .5f * AbsCosTheta( wo ) ) );
	float3 wh = wi + wo;
	if ( wh.x == 0 && wh.y == 0 && wh.z == 0 ) return make_float3( 0 );
	wh = NOOR::normalize( wh );
	const float3 specular =
		_distribution.D( wh ) /
		( 4.f * NOOR::AbsDot( wi, wh ) * fmaxf( AbsCosTheta( wi ), AbsCosTheta( wo ) ) ) *
		SchlickFresnel( dot( wi, wh ) );
	return diffuse + specular;
}

template<typename Distribution>
__device__ __forceinline__
float3 CudaFresnelBlend<Distribution>::SchlickFresnel( float cosTheta ) const {
	return _S + NOOR::pow5f( 1.f - cosTheta ) * ( make_float3( 1.f ) - _S );
}
template<typename Distribution>
__device__ __forceinline__
float3 CudaFresnelBlend<Distribution>::Sample_f(
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

template<typename Distribution>
__device__ __forceinline__
float CudaFresnelBlend<Distribution>::Pdf( const float3 &wo, const float3 &wi ) const {
	if ( !SameHemisphere( wo, wi ) ) return 0.f;
	float3 wh = NOOR::normalize( wo + wi );
	float pdf_wh = _distribution.Pdf( wo, wh );
	return .5f * ( AbsCosTheta( wi ) * NOOR_invPI + pdf_wh / ( 4.f * dot( wo, wh ) ) );
}


template<typename Distribution, typename Fresnel>
__device__ __forceinline__
CudaMicrofacetReflection<Distribution, Fresnel>::CudaMicrofacetReflection(
	const float3 &R
	, const Distribution& distribution
	, const Fresnel& fresnel
)
	: CudaBxDF( BxDFType( BSDF_REFLECTION | BSDF_GLOSSY ) ),
	_R( R )
	, _distribution( distribution )
	, _fresnel( fresnel ) {}

template<typename Distribution, typename Fresnel>
__device__ __forceinline__
float3 CudaMicrofacetReflection<Distribution, Fresnel>::f(
	const float3 &wo,
	const float3 &wi ) const {
	float cosThetaO = AbsCosTheta( wo ), cosThetaI = AbsCosTheta( wi );
	// Handle degenerate cases for microfacet reflection
	if ( cosThetaI == 0.0f || cosThetaO == 0.0f ) return make_float3( 0.f );
	float3 wh = wi + wo;
	if ( wh.x == 0.0f && wh.y == 0 && wh.z == 0.0f ) return make_float3( 0.f );
	wh = NOOR::normalize( wh );
	const float3 F = _fresnel.evaluate( dot( wo, wh ) );
	return _R * _distribution.D( wh ) * _distribution.G( wo, wi ) * F /
		( 4.0f * cosThetaI * cosThetaO );
}

template<typename Distribution, typename Fresnel>
__device__ __forceinline__
float3 CudaMicrofacetReflection<Distribution, Fresnel>::Sample_f(
	const float3 &wo
	, float3 &wi
	, const float2 &u
	, float &pdf
	, BxDFType &sampledType
) const {
	sampledType = _type;
	// Sample microfacet orientation $\wh$ and reflected direction $\wi$
	if ( wo.z == 0.0f ) return make_float3( 0.f );
	const float3 wh = _distribution.Sample_wh( wo, u );
	wi = Reflect( wo, wh );
	if ( !SameHemisphere( wo, wi ) ) {
		pdf = 0.0f;
		return make_float3( 0.f );
	}

	// Compute PDF of _wi_ for microfacet reflection
	pdf = _distribution.Pdf( wo, wh ) / ( 4.0f * dot( wo, wh ) );
	return f( wo, wi );
}

template<typename Distribution, typename Fresnel>
__device__ __forceinline__
float CudaMicrofacetReflection<Distribution, Fresnel>::Pdf(
	const float3 &wo,
	const float3 &wi ) const {
	if ( !SameHemisphere( wo, wi ) ) {
		return 0.0f;
	}
	const float3 wh = NOOR::normalize( wo + wi );
	return _distribution.Pdf( wo, wh ) / ( 4.0f * dot( wo, wh ) );
}

template<typename Distribution, typename Fresnel>
__device__ __forceinline__
CudaMicrofacetTransmission<Distribution, Fresnel>::CudaMicrofacetTransmission(
	const float3 &T
	, const Distribution& distribution
	, const Fresnel& fresnel
) :
	CudaBxDF( BxDFType( BSDF_TRANSMISSION | BSDF_GLOSSY ) )
	, _T( T )
	, _distribution( distribution )
	, _fresnel( fresnel ) {}

template<typename Distribution, typename Fresnel>
__device__ __forceinline__
float3 CudaMicrofacetTransmission<Distribution, Fresnel>::f(
	const float3 &wo,
	const float3 &wi ) const {
	if ( SameHemisphere( wo, wi ) ) return make_float3( 0.f );  // transmission only

	const float cosThetaO = CosTheta( wo );
	const float cosThetaI = CosTheta( wi );
	if ( cosThetaI == 0.0f || cosThetaO == 0.0f ) return make_float3( 0.f );

	// Compute $\wh$ from $\wo$ and $\wi$ for microfacet transmission
	float eta = CosTheta( wo ) > 0.0f ? ( _fresnel._etaT / _fresnel._etaI ) : ( _fresnel._etaI / _fresnel._etaT );
	float3 wh = NOOR::normalize( wo + wi * eta );
	wh *= NOOR::sign( wh.z );

	const float3 F = _fresnel.evaluate( dot( wo, wh ) );

	const float sqrtDenom = dot( wo, wh ) + eta * dot( wi, wh );
	const float factor = 1.f / eta;
	const float3 result = ( make_float3( 1.f ) - F ) * _T *
		fabsf( _distribution.D( wh ) * _distribution.G( wo, wi ) * eta * eta *
			   NOOR::AbsDot( wi, wh ) * NOOR::AbsDot( wo, wh ) * factor * factor /
			   ( cosThetaI * cosThetaO * sqrtDenom * sqrtDenom ) );
	return result;
}
template<typename Distribution, typename Fresnel>
__device__ __forceinline__
float3 CudaMicrofacetTransmission<Distribution, Fresnel>::Sample_f(
	const float3 &wo
	, float3 &wi
	, const float2 &u
	, float &pdf
	, BxDFType &sampledType
) const {
	sampledType = _type;
	if ( wo.z == 0 ) return make_float3( 0.f );
	float3 wh = _distribution.Sample_wh( wo, u );
	const float eta = CosTheta( wo ) > 0 ? ( _fresnel._etaI / _fresnel._etaT ) : ( _fresnel._etaT / _fresnel._etaI );
	if ( !Refract( wo, wh, eta, wi ) ) return make_float3( 0.f );
	pdf = Pdf( wo, wi );
	return f( wo, wi );
}
template<typename Distribution, typename Fresnel>
__device__ __forceinline__
float CudaMicrofacetTransmission<Distribution, Fresnel>::Pdf( const float3 &wo, const float3 &wi ) const {
	if ( SameHemisphere( wo, wi ) ) return 0;
	const float eta = CosTheta( wo ) > 0 ? ( _fresnel._etaT / _fresnel._etaI ) : ( _fresnel._etaI / _fresnel._etaT );
	const float3 wh = NOOR::normalize( wo + wi * eta );

	const float sqrtDenom = dot( wo, wh ) + eta * dot( wi, wh );
	const float dwh_dwi =
		fabsf( ( eta * eta * dot( wi, wh ) ) / ( sqrtDenom * sqrtDenom ) );
	return _distribution.Pdf( wo, wh ) * dwh_dwi;
}

template<typename SubstrateBXDF, typename CoatingBXDF>
__device__ __forceinline__
CudaRoughCoating< SubstrateBXDF, CoatingBXDF>::CudaRoughCoating(
	const SubstrateBXDF& substrate,
	const CoatingBXDF& coating,
	const float3& sigma,
	const float3& etaI,
	const float3& etaT,
	float thickness
) : CudaBxDF( BxDFType( BSDF_REFLECTION | BSDF_GLOSSY ) ),
_substrate( substrate ),
_coating( coating ),
_sigma( sigma ),
_thickness( thickness ),
_eta( etaI.x / etaT.x ),
_invEta( etaT.x / etaI.x ) {
	float avgSigma = ( _sigma.x + _sigma.y + _sigma.z ) / 3.f;
	float avgAbsorption = expf( -2.f*avgSigma*_thickness );
	_specularSamplingWeight = 1.0f / ( avgAbsorption + 1.0f );
}
template<typename SubstrateBXDF, typename CoatingBXDF>
__device__ __forceinline__
float3 CudaRoughCoating< SubstrateBXDF, CoatingBXDF>::f(
	const float3 &wo
	, const float3 &wi
	, bool hasSpecular
	, bool hasNested
) const {
	if ( !SameHemisphere( wo, wi ) ) return make_float3( 0.f );
	float3 result = make_float3( 0.f );
	if ( hasSpecular ) {
		result += _coating.f( wo, wi );
	}
	if ( hasNested ) {
		float F0 = _coating._fresnel.evaluate( wo.z ).x;
		float F1 = _coating._fresnel.evaluate( wi.z ).x;
		if ( F0 == 1.f || F1 == 1.f ) return result;
		float3 wi_c, wo_c;
		Refract( wi, make_float3( 0, 0, 1 ), _eta, wi_c );
		Refract( wo, make_float3( 0, 0, 1 ), _eta, wo_c );
		float3 nested = _substrate.f( -wo_c, -wi_c )*( 1.f - F0 )*( 1.f - F1 );
		float3 sigmaA = _sigma * _thickness;
		if ( !NOOR::isBlack( sigmaA ) ) {
			nested *= NOOR::exp3f( -sigmaA * ( 1.f / AbsCosTheta( wo_c ) + 1.f / AbsCosTheta( wi_c ) ) );
		}
		result += nested;// / AbsCosTheta( wi_c );// *_invEta*_invEta / AbsCosTheta( wi_c );
	}
	return result;
}
template<typename SubstrateBXDF, typename CoatingBXDF>
__device__ __forceinline__
float CudaRoughCoating< SubstrateBXDF, CoatingBXDF>::Pdf(
	const float3 &wo
	, const float3 &wi
	, bool hasSpecular
	, bool hasNested
) const {
	if ( !SameHemisphere( wo, wi ) ) return 0;
	float probNested, probSpecular;

	if ( hasNested && hasSpecular ) {
		probSpecular = .5f;
		probNested = 1.f - probSpecular;
		probSpecular = ( probSpecular*_specularSamplingWeight ) /
			( probSpecular*_specularSamplingWeight + probNested *
			( 1.f - _specularSamplingWeight ) );
		probNested = 1.f - probSpecular;
	} else {
		probNested = probSpecular = 1.f;
	}
	float result = 0.f;
	const float3 h = normalize( wo + wi );// *NOOR::sign( CosTheta( wo ) );
	if ( hasSpecular ) {
		result = _coating.Pdf( wo, wi ) * probSpecular;
	}
	if ( hasNested ) {
		float3 wi_c, wo_c;
		Refract( wi, make_float3( 0, 0, 1 ), _eta, wi_c );
		Refract( wo, make_float3( 0, 0, 1 ), _eta, wo_c );
		result += _substrate.Pdf( -wo_c, -wi_c ) * probNested;
	}
	return result;
}

template<typename SubstrateBXDF, typename CoatingBXDF>
__device__ __forceinline__
float3 CudaRoughCoating< SubstrateBXDF, CoatingBXDF>::Sample_f(
	const float3 &wo
	, float3 &wi
	, const float2 &u
	, float &pdf
	, BxDFType &sampledType
) const {


	if ( wo.z == 0.0f ) return make_float3( 0.f );
	float	probSpecular = .5f;
	float	probNested = 1.f - probSpecular;
	probSpecular = ( probSpecular*_specularSamplingWeight ) /
		( probSpecular*_specularSamplingWeight + probNested *
		( 1 - _specularSamplingWeight ) );
	probNested = 1.f - probSpecular;
	bool choseSpecular;
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
		result = _coating.Sample_f( wo, wi, sample, pdf, sampledType );
		if ( NOOR::isBlack( result ) || pdf == 0 ) {
			pdf = 0.f;
			return make_float3( 0.f );
		}
		sampledType = BxDFType( BSDF_REFLECTION | BSDF_GLOSSY );
	} else {
		float3 wi_c, wo_c;
		Refract( wo, make_float3( 0, 0, 1 ), _eta, wo_c );
		result = _substrate.Sample_f( -1.0f*wo_c, wi_c, sample, pdf, sampledType );
		if ( NOOR::isBlack( result ) || pdf == 0.f ) return make_float3( 0.f );
		if ( !Refract( -1.0f*wi_c, make_float3( 0, 0, -1 ), _invEta, wi ) ) {
			pdf = 0.f;
			return make_float3( 0.f );
		}
	}
	pdf = choseSpecular ?
		probSpecular * Pdf( wo, wi, true, false ) :
		probNested *   Pdf( wo, wi, false, true );
	return choseSpecular ?
		f( wo, wi, true, false ) :
		f( wo, wi, false, true );
}
template class CudaSpecularReflection<CudaFresnelNoOp>;
template class CudaFresnelBlend<CudaTrowbridgeReitzDistribution>;
template class CudaMicrofacetReflection<CudaTrowbridgeReitzDistribution, CudaFresnelDielectric>;
template class CudaMicrofacetReflection<CudaTrowbridgeReitzDistribution, CudaFresnelConductor>;
template class CudaMicrofacetTransmission<CudaTrowbridgeReitzDistribution, CudaFresnelDielectric>;
template class CudaRoughCoating<ConductorReflectionBxDF, DielectricReflectionBxDF >;
