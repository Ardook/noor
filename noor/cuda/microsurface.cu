#include "microsurface.cuh"
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

/************* MICROSURFACE HEIGHT DISTRIBUTION *************/

__forceinline__ __device__
float MicrosurfaceHeightUniform::P1( const float h ) const {
	const float value = ( h >= -1.0f && h <= 1.0f ) ? 0.5f : 0.0f;
	return value;
}

__forceinline__ __device__
float MicrosurfaceHeightUniform::C1( const float h ) const {
	const float value = fminf( 1.0f, fmaxf( 0.0f, 0.5f*( h + 1.0f ) ) );
	return value;
}

__forceinline__ __device__
float MicrosurfaceHeightUniform::invC1( const float U ) const {
	const float h = fmaxf( -1.0f, fminf( 1.0f, 2.0f*U - 1.0f ) );
	return h;
}

__forceinline__ __device__
float MicrosurfaceHeightGaussian::P1( const float h ) const {
	const float value = INV_SQRT_2_M_PI * expf( -0.5f * h*h );
	return value;
}

__forceinline__ __device__
float MicrosurfaceHeightGaussian::C1( const float h ) const {
	const float value = 0.5f + 0.5f * (float) NOOR::Erf( INV_SQRT_2*h );
	return value;
}

__forceinline__ __device__
float MicrosurfaceHeightGaussian::invC1( const float U ) const {
	const float h = sqrtf(2.0f) * NOOR::ErfInv( 2.0f*U - 1.0f );
	return h;
}


/************* MICROSURFACE SLOPE DISTRIBUTION *************/

__forceinline__ __device__
float MicrosurfaceSlope::D( const float3& wm ) const {
	if ( wm.z <= 0.0f )
		return 0.0f;

	// slope of wm
	const float slope_x = -wm.x / wm.z;
	const float slope_y = -wm.y / wm.z;

	// value
	const float value = P22( slope_x, slope_y ) / ( wm.z*wm.z*wm.z*wm.z );
	return value;
}

__forceinline__ __device__
float MicrosurfaceSlope::D_wi( const float3& wi, const float3& wm ) const {
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

__forceinline__ __device__
float3 MicrosurfaceSlope::sampleD_wi( const float3& wi, const float U1, const float U2 ) const {

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

__forceinline__ __device__
float MicrosurfaceSlope::alpha_i( const float3& wi ) const {
	const float invSinTheta2 = 1.0f / ( 1.0f - wi.z*wi.z );
	const float cosPhi2 = wi.x*wi.x*invSinTheta2;
	const float sinPhi2 = wi.y*wi.y*invSinTheta2;
	const float alpha_i = sqrtf( cosPhi2*_alpha_x*_alpha_x + sinPhi2*_alpha_y*_alpha_y );
	return alpha_i;
}

__forceinline__ __device__
float MicrosurfaceSlopeBeckmann::P22( const float slope_x, const float slope_y ) const {
	const float value = 1.0f / ( M_PI * _alpha_x * _alpha_y ) * expf( -slope_x*slope_x / ( _alpha_x*_alpha_x ) - slope_y*slope_y / ( _alpha_y*_alpha_y ) );
	return value;
}

__forceinline__ __device__
float MicrosurfaceSlopeBeckmann::Lambda( const float3& wi ) const {
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

__forceinline__ __device__
float MicrosurfaceSlopeBeckmann::projectedArea( const float3& wi ) const {
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

__forceinline__ __device__
float2 MicrosurfaceSlopeBeckmann::sampleP22_11( const float theta_i, const float U, const float U_2 ) const {
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

__forceinline__ __device__
float MicrosurfaceSlopeGGX::P22( const float slope_x, const float slope_y ) const {
	const float tmp = 1.0f + slope_x*slope_x / ( _alpha_x*_alpha_x ) + slope_y*slope_y / ( _alpha_y*_alpha_y );
	const float value = 1.0f / ( M_PI * _alpha_x * _alpha_y ) / ( tmp * tmp );
	return value;
}

__forceinline__ __device__
float sign( float n ) {
	return n < 0.f ? -1.f : 1.f;
}
__forceinline__ __device__
float MicrosurfaceSlopeGGX::Lambda( const float3& wi ) const {
	if ( wi.z > 0.9999f )
		return 0.0f;
	if ( wi.z < -0.9999f )
		return -1.0f;

	// a
	const float theta_i = acosf( wi.z );
	const float a = 1.0f / tanf( theta_i ) / alpha_i( wi );

	// value
	const float value = 0.5f*( -1.0f + sign( a ) * sqrtf( 1 + 1 / ( a*a ) ) );

	return value;
}

__forceinline__ __device__
float MicrosurfaceSlopeGGX::projectedArea( const float3& wi ) const {
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

__forceinline__ __device__
float2 MicrosurfaceSlopeGGX::sampleP22_11( const float theta_i, const float U, const float U_2 ) const {
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


/************* MICROSURFACE *************/

template<typename Height, typename Slope>
__forceinline__ __device__
float Microsurface<Height, Slope>::G_1( const float3& wi ) const {
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

template<typename Height, typename Slope>
__forceinline__ __device__
float Microsurface<Height, Slope>::G_1( const float3& wi, const float h0 ) const {
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

template<typename Height, typename Slope>
__forceinline__ __device__
float Microsurface<Height, Slope>::sampleHeight( const float3& wr, const float hr, const float U ) const {
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

template<typename Height, typename Slope>
__forceinline__ __device__
float3 Microsurface<Height, Slope>::sample( const float3& wi, int& scatteringOrder ) const {
	// init
	float3 wr = -wi;
	float hr = 1.0f + _microsurfaceheight.invC1( 0.999f );

	// random walk
	scatteringOrder = 0;
	while ( true ) {
		// next height
		float U = _rng();
		hr = sampleHeight( wr, hr, U );

		// leave the microsurface?
		if ( hr == FLT_MAX )
			break;
		else
			scatteringOrder++;

		// next direction
		wr = samplePhaseFunction( -wr );

		// if NaN (should not happen, just in case)
		if ( ( hr != hr ) || ( wr.z != wr.z ) )
			return float3( 0, 0, 1 );
	}

	return wr;
}

template<typename Height, typename Slope>
__forceinline__ __device__
float Microsurface<Height, Slope>::eval( const float3& wi, const float3& wo, const int scatteringOrder ) const {
	if ( wo.z < 0 )
		return 0;
	// init
	float3 wr = -wi;
	float hr = 1.0f + _microsurfaceheight.invC1( 0.999f );

	float sum = 0;

	// random walk
	int current_scatteringOrder = 0;
	while ( scatteringOrder == 0 || current_scatteringOrder <= scatteringOrder ) {
		// next height
		float U = _rng();
		hr = sampleHeight( wr, hr, U );

		// leave the microsurface?
		if ( hr == FLT_MAX )
			break;
		else
			current_scatteringOrder++;

		// next event estimation
		float phasefunction = evalPhaseFunction( -wr, wo );
		float shadowing = G_1( wo, hr );
		float I = phasefunction * shadowing;

		if ( isinf( I ) && ( scatteringOrder == 0 || current_scatteringOrder == scatteringOrder ) )
			sum += I;

		// next direction
		wr = samplePhaseFunction( -wr );

		// if NaN (should not happen, just in case)
		if ( ( hr != hr ) || ( wr.z != wr.z ) )
			return 0.0f;
	}

	return sum;
}

template<typename Height, typename Slope>
__forceinline__ __device__
float MicrosurfaceConductor<Height, Slope>::evalPhaseFunction( const float3& wi, const float3& wo ) const {
	// half vector 
	const float3 wh = normalize( wi + wo );
	if ( wh.z < 0.0f )
		return 0.0f;

	// value
	const float value = 0.25f * _microsurfaceslope.D_wi( wi, wh ) / dot( wi, wh );
	return value;
}

template<typename Height, typename Slope>
__forceinline__ __device__
float3 MicrosurfaceConductor<Height, Slope>::samplePhaseFunction( const float3& wi ) const {
	const float U1 = _rng();
	const float U2 = _rng();

	float3 wm = _microsurfaceslope.sampleD_wi( wi, U1, U2 );

	// reflect
	const float3 wo = -wi + 2.0f * wm * dot( wi, wm );

	return wo;
}

template<typename Height, typename Slope>
__forceinline__ __device__
float MicrosurfaceConductor<Height, Slope>::evalSingleScattering( const float3& wi, const float3& wo ) const {
	// half-vector
	const float3 wh = normalize( wi + wo );
	const float D = _microsurfaceslope.D( wh );

	// masking-shadowing 
	const float G2 = 1.0f / ( 1.0f + _microsurfaceslope.Lambda( wi ) + _microsurfaceslope.Lambda( wo ) );

	// BRDF * cos
	const float value = D * G2 / ( 4.0f * wi.z );

	return value;
}

template<typename Height, typename Slope>
__forceinline__ __device__
float3 MicrosurfaceDielectric<Height, Slope>::refract( const float3 &wi, const float3 &wm, const float eta ) const {
	const float cos_theta_i = dot( wi, wm );
	const float cos_theta_t2 = 1.0f - ( 1.0f - cos_theta_i*cos_theta_i ) / ( eta*eta );
	const float cos_theta_t = -sqrtf( fmaxf( 0.0f, cos_theta_t2 ) );

	return wm * ( dot( wi, wm ) / eta + cos_theta_t ) - wi / eta;
}


template<typename Height, typename Slope>
__forceinline__ __device__
float MicrosurfaceDielectric<Height, Slope>::Fresnel( const float3& wi, const float3& wm, const float eta ) const {
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

template<typename Height, typename Slope>
// wrapper (only for the API and testing)
__forceinline__ __device__
float MicrosurfaceDielectric<Height, Slope>::evalPhaseFunction( const float3& wi, const float3& wo ) const {
	return evalPhaseFunction( wi, wo, true, true ) + evalPhaseFunction( wi, wo, true, false );
}

template<typename Height, typename Slope>
__forceinline__ __device__
float MicrosurfaceDielectric<Height, Slope>::evalPhaseFunction( const float3& wi, const float3& wo, const bool wi_outside, const bool wo_outside ) const {
	const float eta = wi_outside ? _eta : 1.0f / _eta;

	if ( wi_outside == wo_outside ) // reflection
	{
		// half vector 
		const float3 wh = normalize( wi + wo );
		// value
		const float value = ( wi_outside ) ?
			( 0.25f * _microsurfaceslope.D_wi( wi, wh ) / dot( wi, wh ) * Fresnel( wi, wh, eta ) ) :
			( 0.25f * _microsurfaceslope.D_wi( -wi, -wh ) / dot( -wi, -wh ) * Fresnel( -wi, -wh, eta ) );
		return value;
	} else // transmission
	{
		float3 wh = -normalize( wi + wo*eta );
		wh *= ( wi_outside ) ? ( sign( wh.z ) ) : ( -sign( wh.z ) );

		if ( dot( wh, wi ) < 0 )
			return 0;

		float value;
		if ( wi_outside ) {
			value = eta*eta * ( 1.0f - Fresnel( wi, wh, eta ) ) *
				_microsurfaceslope.D_wi( wi, wh ) * fmaxf( 0.0f, -dot( wo, wh ) ) *
				1.0f / powf( dot( wi, wh ) + eta*dot( wo, wh ), 2.0f );
		} else {
			value = eta*eta * ( 1.0f - Fresnel( -wi, -wh, eta ) ) *
				_microsurfaceslope.D_wi( -wi, -wh ) * fmaxf( 0.0f, -dot( -wo, -wh ) ) *
				1.0f / powf( dot( -wi, -wh ) + eta*dot( -wo, -wh ), 2.0f );
		}

		return value;
	}
}

template<typename Height, typename Slope>
__forceinline__ __device__
float3 MicrosurfaceDielectric<Height, Slope>::samplePhaseFunction( const float3& wi ) const {
	bool wo_outside;
	return samplePhaseFunction( wi, true, wo_outside );
}

template<typename Height, typename Slope>
__forceinline__ __device__
float3 MicrosurfaceDielectric<Height, Slope>::samplePhaseFunction( const float3& wi, const bool wi_outside, bool& wo_outside ) const {
	const float U1 = _rng();
	const float U2 = _rng();

	const float eta = wi_outside ? _eta : 1.0f / _eta;

	float3 wm = wi_outside ? ( _microsurfaceslope.sampleD_wi( wi, U1, U2 ) ) :
		( -_microsurfaceslope.sampleD_wi( -wi, U1, U2 ) );

	const float F = Fresnel( wi, wm, eta );

	if ( _rng() < F ) {
		const float3 wo = -wi + 2.0f * wm * dot( wi, wm ); // reflect
		return wo;
	} else {
		wo_outside = !wi_outside;
		const float3 wo = refract( wi, wm, eta );
		return normalize( wo );
	}
}

template<typename Height, typename Slope>
__forceinline__ __device__
float MicrosurfaceDielectric<Height, Slope>::evalSingleScattering( const float3& wi, const float3& wo ) const {
	bool wi_outside = true;
	bool wo_outside = wo.z > 0;

	const float eta = _eta;

	if ( wo_outside ) // reflection
	{
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
	} else // refraction
	{
		// D
		float3 wh = -normalize( wi + wo*eta );
		if ( eta<1.0f )
			wh = -wh;
		const float D = _microsurfaceslope.D( wh );

		// G2
		const float Lambda_i = _microsurfaceslope.Lambda( wi );
		const float Lambda_o = _microsurfaceslope.Lambda( -wo );
		const float G2 = (float) beta( 1.0f + Lambda_i, 1.0f + Lambda_o );

		// BSDF
		const float value = fmaxf( 0.0f, dot( wi, wh ) ) * fmaxf( 0.0f, -dot( wo, wh ) ) *
			1.0f / wi.z * eta*eta * ( 1.0f - Fresnel( wi, wh, eta ) ) *
			G2 * D / powf( dot( wi, wh ) + eta*dot( wo, wh ), 2.0f );
		return value;
	}
}

template<typename Height, typename Slope>
__forceinline__ __device__
float MicrosurfaceDielectric<Height, Slope>::eval( const float3& wi, const float3& wo, const int scatteringOrder ) const {
	// init
	float3 wr = -wi;
	float hr = 1.0f + _microsurfaceheight.invC1( 0.999f );
	bool outside = true;

	float sum = 0.0f;

	// random walk
	int current_scatteringOrder = 0;
	while ( scatteringOrder == 0 || current_scatteringOrder <= scatteringOrder ) {
		// next height
		float U = _rng();
		hr = ( outside ) ? sampleHeight( wr, hr, U ) : -sampleHeight( -wr, -hr, U );

		// leave the microsurface?
		if ( hr == FLT_MAX || hr == -FLT_MAX )
			break;
		else
			current_scatteringOrder++;

		// next event estimation
		float phasefunction = evalPhaseFunction( -wr, wo, outside, ( wo.z>0 ) );
		float shadowing = ( wo.z>0 ) ? G_1( wo, hr ) : G_1( -wo, -hr );
		float I = phasefunction * shadowing;

		if ( isinf( I ) && ( scatteringOrder == 0 || current_scatteringOrder == scatteringOrder ) )
			sum += I;

		// next direction
		wr = samplePhaseFunction( -wr, outside, outside );

		// if NaN (should not happen, just in case)
		if ( ( hr != hr ) || ( wr.z != wr.z ) )
			return 0.0f;
	}

	return sum;
}

template<typename Height, typename Slope>
__forceinline__ __device__
float3 MicrosurfaceDielectric<Height, Slope>::sample( const float3& wi, int& scatteringOrder ) const {
	// init
	float3 wr = -wi;
	float hr = 1.0f + _microsurfaceheight.invC1( 0.999f );
	bool outside = true;

	// random walk
	scatteringOrder = 0;
	while ( true ) {
		// next height
		float U = _rng();
		hr = ( outside ) ? sampleHeight( wr, hr, U ) : -sampleHeight( -wr, -hr, U );

		// leave the microsurface?
		if ( hr == FLT_MAX || hr == -FLT_MAX )
			break;
		else
			scatteringOrder++;

		// next direction
		wr = samplePhaseFunction( -wr, outside, outside );

		// if NaN (should not happen, just in case)
		if ( ( hr != hr ) || ( wr.z != wr.z ) )
			return float3( 0, 0, 1 );
	}

	return wr;
}

template<typename Height, typename Slope>
__forceinline__ __device__
float MicrosurfaceDiffuse<Height, Slope>::evalPhaseFunction( const float3& wi, const float3& wo ) const {
	const float U1 = _rng();
	const float U2 = _rng();
	float3 wm = _microsurfaceslope.sampleD_wi( wi, U1, U2 );

	// value
	const float value = 1.0f / M_PI * fmaxf( 0.0f, dot( wo, wm ) );
	return value;
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


template<typename Height, typename Slope>
__forceinline__ __device__
float3 MicrosurfaceDiffuse<Height, Slope>::samplePhaseFunction( const float3& wi ) const {
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
template<typename Height, typename Slope>
__forceinline__ __device__
float MicrosurfaceDiffuse<Height, Slope>::evalSingleScattering( const float3& wi, const float3& wo ) const {
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

