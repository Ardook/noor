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
#ifndef MICROSURFACE_CUH
#define MICROSURFACE_CUH
#define M_PI			3.14159265358979323846f	/* pi */
#define INV_M_PI		0.31830988618379067153f /* 1/pi */
#define M_PI_2			1.57079632679489661923f	/* pi/2 */
#define SQRT_M_PI		1.77245385090551602729f /* sqrt(pi) */
#define SQRT_2			1.41421356237309504880f /* sqrt(2) */
#define INV_SQRT_M_PI	0.56418958354775628694f /* 1/sqrt(pi) */
#define INV_2_SQRT_M_PI	0.28209479177387814347f /* 0.5/sqrt(pi) */
#define INV_SQRT_2_M_PI 0.3989422804014326779f /* 1/sqrt(2*pi) */
#define INV_SQRT_2		0.7071067811865475244f /* 1/sqrt(2) */
__forceinline__ __device__
float  abgam( double x ) {
	const float g0 = 1.f / 12.f;
	const float g1 = 1.f / 30.f;
	const float g2 = 53.f / 210.f;
	const float g3 = 195.f / 371.f;
	const float g4 = 22999.f / 22737.f;
	const float g5 = 29944523.f / 19733142.f;
	const float g6 = 109535241009.f / 48264275462.f;
	return  0.5f*logf( M_PI_2 ) - x + ( x - 0.5f )*logf( x )
		+ g0 / ( x + g1 / ( x + g2 / ( x + g3 / ( x + g4 /
		( x + g5 / ( x + g6 / x ) ) ) ) ) );
}

__forceinline__ __device__
float  gamma( double x ) {
	return expf( abgam( x + 5.f ) ) / ( x*( x + 1.f )*( x + 2.f )*( x + 3.f )*( x + 4.f ) );
}

__forceinline__ __device__
float  beta( float m, float n ) {
	return ( gamma( m )*gamma( n ) / gamma( m + n ) );
}
// build orthonormal basis (Building an Orthonormal Basis from a 3D Unit Vector Without Normalization, [Frisvad2012])
__forceinline__ __device__
void buildOrthonormalBasis( float3& omega_1, float3& omega_2, const float3& omega_3 ) {
	if ( omega_3.z < -0.9999999f ) {
		omega_1 = make_float3( 0.0f, -1.0f, 0.0f );
		omega_2 = make_float3( -1.0f, 0.0f, 0.0f );
	} else {
		const float a = 1.0f / ( 1.0f + omega_3.z );
		const float b = -omega_3.x*omega_3.y*a;
		omega_1 = make_float3( 1.0f - omega_3.x*omega_3.x*a, b, -omega_3.x );
		omega_2 = make_float3( b, 1.0f - omega_3.y*omega_3.y*a, -omega_3.y );
	}
}
__forceinline__ __device__
bool IsFiniteNumber( float x ) {
	return ( x <= FLT_MAX && x >= -FLT_MAX );
}
__forceinline__ __device__
bool IsFiniteNumber( const float3& x ) {
	return ( IsFiniteNumber( x.x ) && IsFiniteNumber( x.y ) && IsFiniteNumber( x.z ) );
}
/************* MICROSURFACE HEIGHT DISTRIBUTION *************/

/* API */
class MicrosurfaceHeight {
public:
	// height PDF	
	__device__
		float P1( const float h ) const;
	// height CDF	
	__device__
		float C1( const float h ) const;
	// inverse of the height CDF
	__device__
		float invC1( const float U ) const;
};

/* Uniform height distribution in [-1, 1] */
class MicrosurfaceHeightUniform : public MicrosurfaceHeight {
public:
	// height PDF	
	__device__
		float P1( const float h ) const {
		const float value = ( h >= -1.0f && h <= 1.0f ) ? 0.5f : 0.0f;
		return value;
	}

	// height CDF	
	__device__
		float C1( const float h ) const {
		const float value = fminf( 1.0f, fmaxf( 0.0f, 0.5f*( h + 1.0f ) ) );
		return value;
	}

	// inverse of the height CDF
	__device__
		float invC1( const float U ) const {
		const float h = fmaxf( -1.0f, fminf( 1.0f, 2.0f*U - 1.0f ) );
		return h;
	}
};

/* Gaussian height distribution N(0,1) */
class MicrosurfaceHeightGaussian : public MicrosurfaceHeight {
public:
	// height PDF	
	__device__
		float P1( const float h ) const {
		const float value = INV_SQRT_2_M_PI * expf( -0.5f * h*h );
		return value;
	}

	// height CDF	
	__device__
		float C1( const float h ) const {
		const float value = 0.5f + 0.5f * (float) NOOR::Erf( INV_SQRT_2*h );
		return value;
	}

	// inverse of the height CDF
	__device__
		float invC1( const float U ) const {
		const float h = sqrtf( 2.0f ) * NOOR::ErfInv( 2.0f*U - 1.0f );
		return h;
	}
};

/* Beckmann slope distribution */
class MicrosurfaceSlopeBeckmann {
	float _alpha_x, _alpha_y;
public:
	__device__
		MicrosurfaceSlopeBeckmann( float alpha_x = 1.0f, float alpha_y = 1.0f )
		:
		_alpha_x( alpha_x ),
		_alpha_y( alpha_y ) {}
	// distribution of normals (NDF)	
	__device__
		float D( const float3& wm ) const {
		if ( wm.z <= 0.0f )
			return 0.0f;

		// slope of wm
		const float slope_x = -wm.x / wm.z;
		const float slope_y = -wm.y / wm.z;

		// value
		const float value = P22( slope_x, slope_y ) / ( wm.z*wm.z*wm.z*wm.z );
		return value;
	}

	// distribution of visible normals (VNDF)
	__device__
		float D_wi( const float3& wi, const float3& wm ) const {
		if ( wm.z <= 0.0f )
			return 0.0f;

		// normalization coefficient
		const float projectedarea = projectedArea( wi );
		if ( projectedarea == 0 )
			return 0;
		const float c = 1.0f / projectedarea;

		// value
		const float value = c * fmaxf( 0.0f, dot( wi, wm ) ) * D( wm );
		return value;
	}

	// sample the VNDF
	__device__
		float3 sampleD_wi( const float3& wi, const float U1, const float U2 ) const {

		// stretch to match configuration with alpha=1.0	
		const float3 wi_11 = NOOR::normalize( make_float3( _alpha_x * wi.x, _alpha_y * wi.y, wi.z ) );

		// sample visible slope with alpha=1.0
		float2 slope_11 = sampleP22_11( acosf( wi_11.z ), U1, U2 );

		// align with view direction
		const float phi = atan2( wi_11.y, wi_11.x );
		float2 slope = make_float2( cosf( phi )*slope_11.x - sinf( phi )*slope_11.y, sinf( phi )*slope_11.x + cos( phi )*slope_11.y );

		// stretch back
		slope.x *= _alpha_x;
		slope.y *= _alpha_y;

		// if numerical instability
		if ( ( slope.x != slope.x ) || isinf( slope.x ) ) {
			if ( wi.z > 0 ) return make_float3( 0.0f, 0.0f, 1.0f );
			else return NOOR::normalize( make_float3( wi.x, wi.y, 0.0f ) );
		}

		// compute normal
		const float3 wm = NOOR::normalize( make_float3( -slope.x, -slope.y, 1.0f ) );
		return wm;
	}

	// projected roughness in wi
	__device__
		float alpha_i( const float3& wi ) const {
		const float invSinTheta2 = 1.0f / ( 1.0f - wi.z*wi.z );
		const float cosPhi2 = wi.x*wi.x*invSinTheta2;
		const float sinPhi2 = wi.y*wi.y*invSinTheta2;
		const float alpha_i = sqrtf( cosPhi2*_alpha_x*_alpha_x + sinPhi2*_alpha_y*_alpha_y );
		return alpha_i;
	}

	// distribution of slopes
	__device__
		float P22( float slope_x, float slope_y ) const {
		const float value = 1.0f / ( M_PI * _alpha_x * _alpha_y ) * expf( -slope_x*slope_x / ( _alpha_x*_alpha_x ) - slope_y*slope_y / ( _alpha_y*_alpha_y ) );
		return value;
	}

	// Smith's Lambda function
	__device__
		float Lambda( const float3& wi ) const {
		if ( wi.z > 0.9999f )
			return 0.0f;
		if ( wi.z < -0.9999f )
			return -1.0f;

		// a
		const float theta_i = acosf( wi.z );
		const float a = 1.0f / tanf( theta_i ) / alpha_i( wi );

		// value
		const float value = 0.5f*( (float) NOOR::Erf( a ) - 1.0f ) + INV_2_SQRT_M_PI / a * expf( -a*a );

		return value;
	}

	// projected area towards incident direction
	__device__
		float projectedArea( const float3& wi ) const {
		if ( wi.z > 0.9999f )
			return 1.0f;
		if ( wi.z < -0.9999f )
			return 0.0f;

		// a
		const float alphai = alpha_i( wi );
		const float theta_i = acosf( wi.z );
		const float a = 1.0f / tanf( theta_i ) / alphai;

		// value
		const float value = 0.5f*( (float) NOOR::Erf( a ) + 1.0f )*wi.z + INV_2_SQRT_M_PI * alphai * sinf( theta_i ) * expf( -a*a );

		return value;
	}

	// sample the distribution of visible slopes with alpha=1.0
	__device__
		float2 sampleP22_11( float theta_i, float U, float U_2 ) const {
		float2 slope;

		if ( theta_i < 0.0001f ) {
			const float r = sqrtf( -logf( U ) );
			const float phi = 6.28318530718f * U_2;
			slope.x = r * cosf( phi );
			slope.y = r * sinf( phi );
			return slope;
		}

		// constant
		const float sin_theta_i = sinf( theta_i );
		const float cos_theta_i = cosf( theta_i );

		// slope associated to theta_i
		const float slope_i = cos_theta_i / sin_theta_i;

		// projected area
		const float a = cos_theta_i / sin_theta_i;
		const float projectedarea = 0.5f*( (float) NOOR::Erf( a ) + 1.0f )*cos_theta_i + INV_2_SQRT_M_PI * sin_theta_i * expf( -a*a );
		if ( projectedarea < 0.0001f || projectedarea != projectedarea )
			return make_float2( 0.f, 0.f );
		// VNDF normalization factor
		const float c = 1.0f / projectedarea;

		// search 
		float erf_min = -0.9999f;
		float erf_max = fmaxf( erf_min, (float) NOOR::Erf( slope_i ) );
		float erf_current = 0.5f * ( erf_min + erf_max );

		while ( erf_max - erf_min > 0.00001f ) {
			if ( !( erf_current >= erf_min && erf_current <= erf_max ) )
				erf_current = 0.5f * ( erf_min + erf_max );

			// evaluate slope
			const float slope = NOOR::ErfInv( erf_current );

			// CDF
			const float CDF = ( slope >= slope_i ) ? 1.0f : c * ( INV_2_SQRT_M_PI*sin_theta_i*expf( -slope*slope ) + cos_theta_i*( 0.5f + 0.5f*(float) NOOR::Erf( slope ) ) );
			const float diff = CDF - U;

			// test estimate
			if ( abs( diff ) < 0.00001f )
				break;

			// update bounds
			if ( diff > 0.0f ) {
				if ( erf_max == erf_current )
					break;
				erf_max = erf_current;
			} else {
				if ( erf_min == erf_current )
					break;
				erf_min = erf_current;
			}

			// update estimate
			const float derivative = 0.5f*c*cos_theta_i - 0.5f*c*sin_theta_i * slope;
			erf_current -= diff / derivative;
		}

		slope.x = NOOR::ErfInv( fminf( erf_max, fmaxf( erf_min, erf_current ) ) );
		slope.y = NOOR::ErfInv( 2.0f*U_2 - 1.0f );
		return slope;
	}
};

/* GGX slope distribution */
class MicrosurfaceSlopeGGX {
public:
	float _alpha_x, _alpha_y;
	__device__
		MicrosurfaceSlopeGGX( float alpha_x, float alpha_y )
		:
		//_alpha_x( RoughnessToAlpha(alpha_x)  ),
		//_alpha_y( RoughnessToAlpha(alpha_y)  ) {
		_alpha_x( alpha_x ),
		_alpha_y( alpha_y ) {
	}

	__forceinline__ __device__ __host__
		static float RoughnessToAlpha( float roughness ) {
		roughness = fmaxf( roughness, 1e-3f );
		float x = logf( roughness );
		return 1.62142f + 0.819955f * x + 0.1734f * x * x + 0.0171201f * x * x * x +
			0.000640711f * x * x * x * x;
	}

	// distribution of normals (NDF)	
	__device__
		float D( const float3& wm ) const {
		if ( wm.z <= 0.0f )
			return 0.0f;

		// slope of wm
		const float slope_x = -wm.x / wm.z;
		const float slope_y = -wm.y / wm.z;

		// value
		const float value = P22( slope_x, slope_y ) / ( wm.z*wm.z*wm.z*wm.z );
		return value;
	}

	// distribution of visible normals (VNDF)
	__device__
		float D_wi( const float3& wi, const float3& wm ) const {
		if ( wm.z <= 0.0f )
			return 0.0f;

		// normalization coefficient
		const float projectedarea = projectedArea( wi );
		if ( projectedarea == 0 )
			return 0;
		const float c = 1.0f / projectedarea;

		// value
		const float value = c * fmaxf( 0.0f, dot( wi, wm ) ) * D( wm );
		return value;
	}

	// sample the VNDF
	__device__
		float3 sampleD_wi( const float3& wi, const float U1, const float U2 ) const {

		// stretch to match configuration with alpha=1.0	
		const float3 wi_11 = NOOR::normalize( make_float3( _alpha_x * wi.x, _alpha_y * wi.y, wi.z ) );

		// sample visible slope with alpha=1.0
		float2 slope_11 = sampleP22_11( acosf( wi_11.z ), U1, U2 );

		// align with view direction
		const float phi = atan2( wi_11.y, wi_11.x );
		float2 slope = make_float2( cosf( phi )*slope_11.x - sinf( phi )*slope_11.y, sinf( phi )*slope_11.x + cos( phi )*slope_11.y );

		// stretch back
		slope.x *= _alpha_x;
		slope.y *= _alpha_y;

		// if numerical instability
		if ( ( slope.x != slope.x ) || isinf( slope.x ) ) {
			if ( wi.z > 0 ) return make_float3( 0.0f, 0.0f, 1.0f );
			else return NOOR::normalize( make_float3( wi.x, wi.y, 0.0f ) );
		}

		// compute normal
		const float3 wm = NOOR::normalize( make_float3( -slope.x, -slope.y, 1.0f ) );
		return wm;
	}

	// projected roughness in wi
	__device__
		float alpha_i( const float3& wi ) const {
		const float invSinTheta2 = 1.0f / ( 1.0f - wi.z*wi.z );
		const float cosPhi2 = wi.x*wi.x*invSinTheta2;
		const float sinPhi2 = wi.y*wi.y*invSinTheta2;
		const float alpha_i = sqrtf( cosPhi2*_alpha_x*_alpha_x + sinPhi2*_alpha_y*_alpha_y );
		return alpha_i;
	}

	// distribution of slopes
	__device__
		float P22( float slope_x, float slope_y ) const {
		const float tmp = 1.0f + slope_x*slope_x / ( _alpha_x*_alpha_x ) + slope_y*slope_y / ( _alpha_y*_alpha_y );
		const float value = 1.0f / ( M_PI * _alpha_x * _alpha_y ) / ( tmp * tmp );
		return value;
	}

	// Smith's Lambda function
	__device__
		float Lambda( const float3& wi ) const {
		if ( wi.z > 0.9999f )
			return 0.0f;
		if ( wi.z < -0.9999f )
			return -1.0f;

		// a
		const float theta_i = acosf( wi.z );
		const float a = 1.0f / tanf( theta_i ) / alpha_i( wi );

		// value
		const float value = 0.5f*( -1.0f + NOOR::sign( a ) * sqrtf( 1.f + 1.f / ( a*a ) ) );

		return value;
	}

	// projected area towards incident direction
	__device__
		float projectedArea( const float3& wi ) const {
		if ( wi.z > 0.9999f )
			return 1.0f;
		if ( wi.z < -0.9999f )
			return 0.0f;

		// a
		const float theta_i = acosf( wi.z );
		const float sin_theta_i = sinf( theta_i );

		const float alphai = alpha_i( wi );

		// value
		const float value = 0.5f * ( wi.z + sqrtf( wi.z*wi.z + sin_theta_i*sin_theta_i*alphai*alphai ) );

		return value;
	}

	// sample the distribution of visible slopes with alpha=1.0
	__device__
		float2 sampleP22_11( float theta_i, float U, float U_2 ) const {
		float2 slope;

		if ( theta_i < 0.0001f ) {
			const float r = sqrtf( U / ( 1.0f - U ) );
			const float phi = 6.28318530718f * U_2;
			slope.x = r * cosf( phi );
			slope.y = r * sinf( phi );
			return slope;
		}

		// constant
		const float sin_theta_i = sinf( theta_i );
		const float cos_theta_i = cosf( theta_i );
		const float tan_theta_i = sin_theta_i / cos_theta_i;

		// slope associated to theta_i
		//const float slope_i = cos_theta_i / sin_theta_i;

		// projected area
		const float projectedarea = 0.5f * ( cos_theta_i + 1.0f );
		if ( projectedarea < 0.0001f || projectedarea != projectedarea )
			return make_float2( 0.f, 0.f );
		// normalization coefficient
		const float c = 1.0f / projectedarea;

		const float A = 2.0f*U / cos_theta_i / c - 1.0f;
		const float B = tan_theta_i;
		const float tmp = 1.0f / ( A*A - 1.0f );

		const float D = sqrtf( fmaxf( 0.0f, B*B*tmp*tmp - ( A*A - B*B )*tmp ) );
		const float slope_x_1 = B*tmp - D;
		const float slope_x_2 = B*tmp + D;
		slope.x = ( A < 0.0f || slope_x_2 > 1.0f / tan_theta_i ) ? slope_x_1 : slope_x_2;

		float U2;
		float S;
		if ( U_2 > 0.5f ) {
			S = 1.0f;
			U2 = 2.0f*( U_2 - 0.5f );
		} else {
			S = -1.0f;
			U2 = 2.0f*( 0.5f - U_2 );
		}
		const float z = ( U2*( U2*( U2*0.27385f - 0.73369f ) + 0.46341f ) ) / ( U2*( U2*( U2*0.093073f + 0.309420f ) - 1.000000f ) + 0.597999f );
		slope.y = S * z * sqrtf( 1.0f + slope.x*slope.x );

		return slope;
	}
};


template<typename Height, typename Slope>
class Microsurface : public CudaBxDF {
public:
	// height distribution
	const Height& _microsurfaceheight;
	// slope distribution
	const Slope& _microsurfaceslope;
	const CudaRNG& _rng;
	int _maxScattering;

public:
	__device__
		Microsurface(
		const Height& height,
		const Slope& slope,
		const CudaRNG& rng,
		BxDFType type,
		int maxScattering = 10
		) : CudaBxDF( BxDFType( type ) ),
		_microsurfaceheight( height ),
		_microsurfaceslope( slope ),
		_rng( rng ),
		_maxScattering( maxScattering ) {}

public:
	// masking function
	__device__
		float G_1( const float3& wi ) const {
		if ( wi.z > 0.9999f )
			return 1.0f;
		if ( wi.z <= 0.0f )
			return 0.0f;

		// Lambda
		const float Lambda = _microsurfaceslope.Lambda( wi );
		// value
		const float value = 1.0f / ( 1.0f + Lambda );
		return value;
	}

	// masking function at height h0
	__device__
		float G_1( const float3& wi, float h0 ) const {
		if ( wi.z > 0.9999f )
			return 1.0f;
		if ( wi.z <= 0.0f )
			return 0.0f;

		// height CDF
		const float C1_h0 = _microsurfaceheight.C1( h0 );
		// Lambda
		const float Lambda = _microsurfaceslope.Lambda( wi );
		// value
		const float value = powf( C1_h0, Lambda );
		return value;
	}

	// sample height in outgoing direction
	__device__
		float sampleHeight( const float3& wr, float hr, float U ) const {
		if ( wr.z > 0.9999f )
			return FLT_MAX;
		if ( wr.z < -0.9999f ) {
			const float value = _microsurfaceheight.invC1( U*_microsurfaceheight.C1( hr ) );
			return value;
		}
		if ( fabsf( wr.z ) < 0.0001f )
			return hr;

		// probability of intersection
		const float G_1_ = G_1( wr, hr );

		if ( U > 1.0f - G_1_ ) // leave the microsurface
			return FLT_MAX;

		const float h = _microsurfaceheight.invC1(
			_microsurfaceheight.C1( hr ) / powf( ( 1.0f - U ), 1.0f / _microsurfaceslope.Lambda( wr ) )
		);
		return h;
	}
};

template<typename Height, typename Slope>
/* Microsurface made of conductor material */
class MicrosurfaceConductor : public Microsurface<Height, Slope> {
	CudaFresnelConductor _fresnel;
	float3 _R;
public:
	__device__
		MicrosurfaceConductor(
		const Height& height,
		const Slope& slope,
		const CudaRNG& rng,
		const float3& R,
		const float3& etaI,
		const float3& etaT,
		const float3& k
		)
		: Microsurface<Height, Slope>( height, slope, rng, BxDFType( BSDF_REFLECTION | BSDF_GLOSSY ) ),
		_fresnel( CudaFresnelConductor( etaI, etaT, k ) ),
		_R( R ) {}

public:
	__device__
		float Pdf( const float3 &wi, const float3& wo ) const {
		if ( !SameHemisphere( wo, wi ) ) {
			return 0.0f;
		}
		// Calculate the reflection half-vector 
		const float3 wh = NOOR::normalize( wo + wi );
		return _microsurfaceslope.D( wh ) / ( 4.0f * wi.z * ( 1.0f + _microsurfaceslope.Lambda( wi ) ) ) + ( wo.z );
	}
	// evaluate local phase function 
	__device__
		float3 evalPhaseFunction( const float3& wi, const float3& wo ) const {
		// half vector 
		const float3 wh = normalize( wi + wo );
		if ( wh.z < 0.0f )
			return make_float3( 0.0f );

		const float3 F = _fresnel.evaluate( dot( wi, wh ) );
		// value
		const float3 value = F * _microsurfaceslope.D_wi( wi, wh ) / ( 4.0f*dot( wi, wh ) );
		return value;
	}

	// sample local phase function
	__device__
		float3 samplePhaseFunction( const float3& wi, float3& weight ) const {
		const float U1 = _rng();
		const float U2 = _rng();

		const float3 wm = _microsurfaceslope.sampleD_wi( wi, U1, U2 );

		// reflect
		const float3 wo = -1.0f*wi + 2.0f * wm * dot( wi, wm );
		weight = _fresnel.evaluate( dot( wi, wm ) );
		return wo;
	}
	__device__
		float3 evalSingleScattering( const float3& wi, const float3& wo ) const {
			// half-vector
		const float3 wh = NOOR::normalize( wi + wo );
		const float D = _microsurfaceslope.D( wh );
		// masking-shadowing 
		const float G2 = 1.0f / ( 1.0f + _microsurfaceslope.Lambda( wi ) + _microsurfaceslope.Lambda( wo ) );
		const float3 F = _fresnel.evaluate( dot( wi, wh ) );
		// BRDF * cos
		const float3 value = F * D * G2 / ( 4.0f * wi.z );
		return value;
	}
	// evaluate BSDF with a random walk (stochastic but unbiased)
	__device__
		float3 f(
		const float3& wi
		, const float3& wo
		) const {
		if ( wo.z < 0 )
			return make_float3( 0.f );
		// init
		float3 wr = -1.f*wi;
		float hr = 1.0f + _microsurfaceheight.invC1( 0.999f );
		const float3 singleScattering = evalSingleScattering( wi, wo );
		float3 multiScattering = make_float3( 0.0f );
		float3 throughput = make_float3( 1.0f );
		// random walk
		int scatteringOrder = 0;
		while ( scatteringOrder <= _maxScattering ) {
			// next height
			const float U = _rng();
			hr = sampleHeight( wr, hr, U );

			// leave the microsurface?
			if ( hr == FLT_MAX )
				break;
			else
				++scatteringOrder;
			if ( scatteringOrder > 1 ) {
				// next event estimation
				const float3 phasefunction = evalPhaseFunction( -wr, wo );
				const float shadowing = G_1( wo, hr );
				const float3 I = throughput * phasefunction * shadowing;

				if ( IsFiniteNumber( I ) && ( scatteringOrder == _maxScattering ) )
					multiScattering += I;
			}

			// next direction
			float3 weight;
			wr = samplePhaseFunction( -wr, weight );
			throughput *= weight;
			// if NaN (should not happen, just in case)
			if ( ( hr != hr ) || ( wr.z != wr.z ) )
				return make_float3(0.0f);
		}
		return ( .5f * singleScattering + multiScattering ) * _R;
	}
	// sample BSDF with a random walk
	// scatteringOrder is set to the number of bounces computed for this sample
	__device__
		float3 sample_f(
		const float3& wi
		, float3& wo
		, const float2& u
		, float& pdf
		, BxDFType &sampledType
		) const {
		sampledType = _type;
		float3 wr = -1.f*wi;
		float hr = 1.0f + _microsurfaceheight.invC1( 0.999f );

		// random walk
		int scatteringOrder = 0;
		float3 throughput = make_float3( 1.0f );
		pdf = 0.0f;
		while ( true ) {
			// next height
			float U = _rng();
			hr = sampleHeight( wr, hr, U );

			// leave the microsurface?
			if ( hr == FLT_MAX )
				break;
			else
				scatteringOrder++;
			float3 weight;
			// next direction
			wr = samplePhaseFunction( -wr, weight );
			throughput *= weight;

			// if NaN (should not happen, just in case)
			if ( ( hr != hr ) || ( wr.z != wr.z ) ) {
				wo = make_float3( 0, 0, 1 );
				return make_float3( 0 );
			}
			if ( scatteringOrder >= _maxScattering ) {
				wo = make_float3( 0, 0, 1 );
				return make_float3( 0 );
			}
		}
		wo = wr;
		pdf = 1.0f;// Pdf( wi, wo );// 1.0f;
		return _R*throughput;// _fresnel.evaluate( dot( wi, wo ) );
	}
};

template<typename Height, typename Slope>
/* Microsurface made of conductor material */
class MicrosurfaceDielectric : public Microsurface<Height, Slope> {
public:
	float _eta;
	float3 _R;
	float3 _T;
	CudaFresnelDielectric _fresnel;
public:
	__device__
		MicrosurfaceDielectric(
		const Height& height,
		const Slope& slope,
		const CudaRNG& rng,
		float eta,
		const float3& R,
		const float3& T
		)
		: Microsurface<Height, Slope>( height, slope, rng,
									   BxDFType( BSDF_REFLECTION | BSDF_TRANSMISSION | BSDF_GLOSSY ) ),
		_fresnel( CudaFresnelDielectric( 1.0f, eta ) ),
		_eta( eta ),
		_R( R ),
		_T( T ) {}

	__device__
		float3 Refract( const float3 &wi, const float3 &wm, float eta ) const {
		const float cos_theta_i = dot( wi, wm );
		const float cos_theta_t2 = 1.0f - ( 1.0f - cos_theta_i*cos_theta_i ) / ( eta*eta );
		const float cos_theta_t = -sqrtf( fmaxf( 0.0f, cos_theta_t2 ) );

		return wm * ( dot( wi, wm ) / eta + cos_theta_t ) - wi / eta;
	}

	__device__
		float Fresnel( const float3& wi, const float3& wm, float eta ) const {
		const float cos_theta_i = dot( wi, wm );
		const float cos_theta_t2 = 1.0f - ( 1.0f - cos_theta_i*cos_theta_i ) / ( eta*eta );

		// total internal reflection 
		if ( cos_theta_t2 <= 0.0f ) return 1.0f;

		const float cos_theta_t = sqrtf( cos_theta_t2 );
		const float Rs = ( cos_theta_i - eta * cos_theta_t ) / ( cos_theta_i + eta * cos_theta_t );
		const float Rp = ( eta * cos_theta_i - cos_theta_t ) / ( eta * cos_theta_i + cos_theta_t );
		const float F = 0.5f * ( Rs * Rs + Rp * Rp );
		return F;
	}
	
	__device__
		float evalPhaseFunction( const float3& wi, const float3& wo, bool wi_outside, bool wo_outside ) const {
		const float eta = wi_outside ? _eta : 1.0f / _eta;

		if ( wi_outside == wo_outside ) { // reflection
			// half vector 
			float3 wh = normalize( wi + wo );
			// value
			const float value = ( wi_outside ) ?
				( 0.25f * _microsurfaceslope.D_wi( wi, wh ) / dot( wi, wh ) * Fresnel( wi, wh, eta ) ) :
				( 0.25f * _microsurfaceslope.D_wi( -1.0f*wi, -wh ) / dot( -1.f*wi, -wh ) * Fresnel( -1.f*wi, -wh, eta ) );
			return value;
		} else { // transmission
			float3 wh = -normalize( wi + wo*eta );
			wh *= ( wi_outside ) ? ( NOOR::sign( wh.z ) ) : ( -NOOR::sign( wh.z ) );

			if ( dot( wh, wi ) < 0 )
				return 0;

			float value;
			if ( wi_outside ) {
				value = eta*eta * ( 1.0f - Fresnel( wi, wh, eta ) ) *
					_microsurfaceslope.D_wi( wi, wh ) * fmaxf( 0.0f, -dot( wo, wh ) ) *
					1.0f / powf( dot( wi, wh ) + eta*dot( wo, wh ), 2.0f );
			} else {
				value = eta*eta * ( 1.0f - Fresnel( -1.f*wi, -wh, eta ) ) *
					_microsurfaceslope.D_wi( -1.f*wi, -wh ) * fmaxf( 0.0f, -dot( -1.f*wo, -wh ) ) *
					1.0f / powf( dot( -1.f*wi, -wh ) + eta*dot( -1.f*wo, -wh ), 2.0f );
			}
			return value;
		}
	}

	__device__
		float3 samplePhaseFunction( const float3& wi, bool wi_outside, bool& wo_outside, float& weight ) const {
		const float U1 = _rng();
		const float U2 = _rng();

		const float eta = wi_outside ? _eta : 1.0f / _eta;

		float3 wm = wi_outside ? ( _microsurfaceslope.sampleD_wi( wi, U1, U2 ) ) :
								 ( -_microsurfaceslope.sampleD_wi( -1.f*wi, U1, U2 ) );

		const float F = Fresnel( wi, wm, eta );

		if ( _rng() < F ) { //reflect
			weight = F;
			const float3 wo = -1.f*wi + 2.0f * wm * dot( wi, wm ); 
			return wo;
		} else { //refract
			weight = 1.f - F;
			wo_outside = !wi_outside;
			const float3 wo = Refract( wi, wm, eta );
			return normalize( wo );
		}
	}

	__device__
		float evalSingleScattering( const float3& wi, const float3& wo ) const {
		//bool wi_outside = true;
		bool wo_outside = wo.z > 0;

		const float eta = _eta;

		if ( wo_outside ) { // reflection
			// D
			const float3 wh = normalize( float3( wi + wo ) );
			const float D = _microsurfaceslope.D( wh );

			// masking shadowing
			const float Lambda_i = _microsurfaceslope.Lambda( wi );
			const float Lambda_o = _microsurfaceslope.Lambda( wo );
			const float G2 = 1.0f / ( 1.0f + Lambda_i + Lambda_o );

			// BRDF
			const float value = Fresnel( wi, wh, eta ) * D * G2 / ( 4.0f * wi.z );
			return value;
		} else { // refraction
			// D
			float3 wh = -normalize( wi + wo*eta );
			if ( eta < 1.0f )
				wh = -wh;
			const float D = _microsurfaceslope.D( wh );

			// G2
			const float Lambda_i = _microsurfaceslope.Lambda( wi );
			const float Lambda_o = _microsurfaceslope.Lambda( -1.f*wo );
			const float G2 = (float) beta( 1.0f + Lambda_i, 1.0f + Lambda_o );

			// BSDF
			const float value = fmaxf( 0.0f, dot( wi, wh ) ) * fmaxf( 0.0f, -dot( wo, wh ) ) *
				1.0f / wi.z * eta*eta * ( 1.0f - Fresnel( wi, wh, eta ) ) *
				G2 * D / powf( dot( wi, wh ) + eta*dot( wo, wh ), 2.0f );
			return value;
		}
	}

	// evaluate BSDF with a random walk (stochastic but unbiased)
	__device__
		float3 f( const float3& wi, const float3& wo ) const {
		// init
		bool wi_outside = CosTheta( wi ) > 0;
		float3 wr = -1.0f*wi;
		float hr = 1.0f + _microsurfaceheight.invC1( 0.999f );
		float multiScattering = 0.0f;
		// random walk
		int scatteringOrder = 0;
		while ( scatteringOrder < _maxScattering ) {
			// next height
			float U = _rng();
			hr = ( wi_outside ) ? sampleHeight( wr, hr, U ) : -sampleHeight( -wr, -hr, U );

			// leave the microsurface?
			if ( hr == FLT_MAX || hr == -FLT_MAX )
				break;
			else
				scatteringOrder++;
			// next event estimation
			const float phasefunction = evalPhaseFunction( -wr, wo, wi_outside, ( wo.z > 0 ) );
			const float shadowing = ( wo.z > 0 ) ? G_1( wo, hr ) : G_1( -1.f*wo, -hr );
			const float I = phasefunction * shadowing;

			if ( !isinf( I ))
				multiScattering += I;
			float w;
			// next direction
			wr = samplePhaseFunction( -wr, wi_outside, wi_outside, w );

			// if NaN (should not happen, just in case)
			if ( isinf( hr ) || NOOR::isinf3( wr ) )
				return make_float3(0.f);
		}
		wi_outside = ( CosTheta( wi ) > 0 );
		const bool wo_outside = ( CosTheta( wo ) > 0 );
		if ( ( wi_outside != wo_outside ) || ( !wi_outside && !wo_outside ) ) {
			return multiScattering*_T;
		} else{
			return multiScattering*_R;
		}
	}
		
	// sample final BSDF with a random walk
	// scatteringOrder is set to the number of bounces computed for this sample
	__device__
		float3 sample_f(
		const float3& wi
		, float3& wo
		, const float2& u
		, float& pdf
		, BxDFType& sampledType
		) const {
		bool wi_outside = CosTheta( wi ) > 0;
		// init
		float3 wr = wi_outside?-1.0f*wi:wi;
		wi_outside = true;
		float hr = 1.0f + _microsurfaceheight.invC1( 0.999f );
		// random walk
		int scatteringOrder = 0;
		float energy = 1.0f;
		while ( true ) {
			// next height
			float U = _rng();
			hr = ( wi_outside ) ? sampleHeight( wr, hr, U ) : -sampleHeight( -wr, -hr, U );

			// leave the microsurface?
			if ( hr == FLT_MAX || hr == -FLT_MAX )
				break;
			else
				scatteringOrder++;
			if ( scatteringOrder >= _maxScattering ) {
				wo = make_float3( 0, 0, 1 );
				return make_float3( 0.0f );
			}
			// next direction
			float weight;
			wr = samplePhaseFunction( -wr, wi_outside, wi_outside, weight );
			energy *= weight;
			// if NaN (should not happen, just in case)
			if ( isinf( hr ) || NOOR::isinf3( wr ) ) {
				wo = make_float3( 0, 0, 1 );
				return make_float3( 0.0f );
			}
		}

		wi_outside = ( CosTheta( wi ) > 0 );
		wo = wi_outside?wr:-wr;
		pdf = 1.f;// Pdf( wi, wo );
		const bool wo_outside = ( CosTheta( wo ) > 0 );
		if ( (wi_outside != wo_outside) || (!wi_outside && !wo_outside) ) {
			return energy*_T;
		} else{
			return energy*_R;
		} 
	}
	__device__
		float Pdf( const float3 &wi, const float3& wo ) const {
		//bool reflective = ( wi.z*wo.z > 0.0f );
		//const float eta = CosTheta( wi ) > 0 ? 1.f / _eta : _eta;
		//float3 wh = NOOR::normalize( wi + ( reflective ? wo : wo*eta ) );
		///*if ( wh.z < 0.0f )
		//	wh = -wh;*/
		//float fresnel;
		//float3 r_wi = wi;// ( wi.z < 0.0f ) ? -1.0f*wi : wi;
		//if ( wi.z*wo.z > 0.0f ) {
		//	fresnel = _fresnel.evaluate( dot( r_wi, wh ) ).x;
		//} else {
		//	fresnel = 1.0f - _fresnel.evaluate( dot( r_wi, wh ) ).x;
		//}
		//float pdf = fresnel * fmaxf( 0.0f, dot( r_wi, wh ) ) * _microsurfaceslope.D( wh ) / ( ( 1.0f + _microsurfaceslope.Lambda( r_wi ) ) * r_wi.z );// +( wo.z );
		//return fresnel;

		bool	reflect = CosTheta( wi ) * CosTheta( wo ) > 0;

		float3 wh;

		if ( reflect ) {
			/* Calculate the reflection half-vector */
			wh = normalize( wo + wi );
		} else {
			/* Zero probability if this component was not requested */
			/* Calculate the transmission half-vector */
			float eta = CosTheta( wi ) > 0 ? _eta : 1.0f/_eta;

			wh = normalize( wi + wo*eta );
		}

		/* Ensure that the half-vector points into the
		same hemisphere as the macrosurface normal */
		wh *= NOOR::sign( CosTheta( wh ) );

		float s = 1.f;// NOOR::sign( CosTheta( wi ) );
		float3 lwi = s*wi;// ( s*bRec.wi.x, s*bRec.wi.y, s*bRec.wi.z );
		const float lambda = _microsurfaceslope.Lambda( lwi );
		float prob = fmaxf( 0.0f, dot( wh, lwi ) ) * _microsurfaceslope.D( wh ) / ( 1.0f + lambda ) / CosTheta( lwi );

		float F = Fresnel( lwi, wh , _eta );
		prob *= reflect ? F : ( 1 - F );

		// single-scattering PDF + diffuse 
		// otherwise too many fireflies due to lack of multiple-scattering PDF
		// (MIS works even if the PDF is wrong and not normalized)
		return fabsf( prob * s ) + CosTheta( wo );

	}
};

/* Microsurface made of conductor material */
template<typename Height, typename Slope>
class MicrosurfaceDiffuse : public Microsurface<Height, Slope> {
public:
	__device__
		MicrosurfaceDiffuse( const CudaRNG& rng,
							 const Height& height,
							 const Slope& slope )
		: Microsurface<Height, Slope>( rng, height, slope ) {}

public:
	__device__
		float evalPhaseFunction( const float3& wi, const float3& wo ) const {
		const float U1 = _rng();
		const float U2 = _rng();
		float3 wm = _microsurfaceslope.sampleD_wi( wi, U1, U2 );

		// value
		const float value = 1.0f / M_PI * fmaxf( 0.0f, dot( wo, wm ) );
		return value;
	}

	__device__
		float3 samplePhaseFunction( const float3& wi ) const {
		const float U1 = _rng();
		const float U2 = _rng();
		const float U3 = _rng();
		const float U4 = _rng();

		float3 wm = _microsurfaceslope.sampleD_wi( wi, U1, U2 );

		// sample diffuse reflection
		float3 w1, w2;
		buildOrthonormalBasis( w1, w2, wm );

		float r1 = 2.0f*U3 - 1.0f;
		float r2 = 2.0f*U4 - 1.0f;

		// concentric map code from
		// http://psgraphics.blogspot.ch/2011/01/improved-code-for-concentric-map.html
		float phi, r;
		if ( r1 == 0 && r2 == 0 ) {
			r = phi = 0;
		} else if ( r1*r1 > r2*r2 ) {
			r = r1;
			phi = ( M_PI / 4.0f ) * ( r2 / r1 );
		} else {
			r = r2;
			phi = ( M_PI / 2.0f ) - ( r1 / r2 ) * ( M_PI / 4.0f );
		}
		float x = r*cosf( phi );
		float y = r*sinf( phi );
		float z = sqrtf( fmaxf( 0.0f, 1.0f - x*x - y*y ) );
		float3 wo = x*w1 + y*w2 + z*wm;

		return wo;
	}

	// stochastic evaluation  
	// Heitz and Dupuy 2015
	// Implementing a Simple Anisotropic Rough Diffuse Material with Stochastic Evaluation
	__device__
		float evalSingleScattering( const float3& wi, const float3& wo ) const {
		// sample visible microfacet
		const float U1 = _rng();
		const float U2 = _rng();
		const float3 wm = _microsurfaceslope.sampleD_wi( wi, U1, U2 );

		// shadowing given masking
		const float Lambda_i = _microsurfaceslope.Lambda( wi );
		const float Lambda_o = _microsurfaceslope.Lambda( wo );
		float G2_given_G1 = ( 1.0f + Lambda_i ) / ( 1.0f + Lambda_i + Lambda_o );

		// evaluate diffuse and shadowing given masking
		const float value = 1.0f / (float) M_PI * fmaxf( 0.0f, dot( wm, wo ) ) * G2_given_G1;

		return value;
	}

};



#endif /* MICROSURFACE_CUH */

