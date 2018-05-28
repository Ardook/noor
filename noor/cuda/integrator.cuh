/*
MIT License

Copyright (c) 2015-2017 Ardavan Kanani

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
#ifndef CUDAINTEGRATOR_CUH
#define CUDAINTEGRATOR_CUH
template<class BSDF>
__forceinline__ __device__
float3 direct(
	const BSDF& bsdf
	, const CudaRay& ray
	, const CudaIntersection& I
	, const CudaRNG& rng
) {
	return _light_manager.direct( bsdf, ray, rng, I );
}

__forceinline__ __device__
float3 scatter(
	CudaRay& ray
	, CudaIntersection& I
	, const CudaRNG& rng
	, const float3& wo
	, float3& wi
	, float& pdf
) {
	BxDFType type = BSDF_ALL;
	BxDFType sampledType;
	if ( I._material_type & GLASS ) {
		const CudaRoughGlassBSDF bsdf = factoryRoughGlassBSDF( I );
		return sample_f( bsdf, I, rng, wo, wi, pdf, type, sampledType );
	} else if ( I._material_type & DIFFUSE ) {
		const CudaDiffuseBSDF bsdf = factoryDiffuseBSDF( I );
		return sample_f( bsdf, I, rng, wo, wi, pdf, type, sampledType );
	} else if ( I._material_type & GLOSSY ) {
		const CudaGlossyBSDF bsdf = factoryGlossyBSDF( I );
		return sample_f( bsdf, I, rng, wo, wi, pdf, type, sampledType );
	} else if ( I._material_type & METAL ) {
		const CudaMetalBSDF bsdf = factoryMetalBSDF( I );
		return sample_f( bsdf, I, rng, wo, wi, pdf, type, sampledType );
	} else if ( I._material_type & MIRROR ) {
		const CudaMirrorBSDF bsdf = factoryMirrorBSDF( I );
		return sample_f( bsdf, I, rng, wo, wi, pdf, type, sampledType );
	} else
		return _constant_spec._black;
}
__forceinline__ __device__
float3 direct(
	CudaRay& ray
	, CudaIntersection& I
	, const CudaRNG& rng
) {
	if ( I._material_type & GLASS ) {
		const CudaRoughGlassBSDF bsdf = factoryRoughGlassBSDF( I );
		return direct( bsdf, ray, I, rng );
	} else if ( I._material_type & DIFFUSE ) {
		const CudaDiffuseBSDF bsdf = factoryDiffuseBSDF( I );
		return direct( bsdf, ray, I, rng );
	} else if ( I._material_type & GLOSSY ) {
		const CudaGlossyBSDF bsdf = factoryGlossyBSDF( I );
		return direct( bsdf, ray, I, rng );
	} else if ( I._material_type & METAL ) {
		const CudaMetalBSDF bsdf = factoryMetalBSDF( I );
		return direct( bsdf, ray, I, rng );
	} else if ( I._material_type & MIRROR ) {
		const CudaMirrorBSDF bsdf = factoryMirrorBSDF( I );
		return direct( bsdf, ray, I, rng );
	} else
		return _constant_spec._black;
}
__forceinline__ __device__
float3 Li(
	CudaRay& ray
	, CudaIntersection& I
	, const CudaRNG& rng
) {
	float3 L = _constant_spec._black;
	float3 beta = _constant_spec._white;
	bool specular_bounce = false;
	for ( unsigned char bounce = 0; bounce < _constant_spec._bounces; ++bounce ) {
		if ( intersect( ray, I ) ) {
			if ( bounce == 0 ) {
				I._lookAt = make_float4( I._p, 1.0f );
			}
			//if ( I._backface == true ) {
			//	break;
			//}
			// terminate if light is hit
			if ( I._material_type == EMITTER && ( bounce == 0 || specular_bounce ) ) {
				L += beta * _material_manager.getEmmitance( I );
				break;
			}
			/*if ( I._material_type & BUMP ) {
				bump( I );
			}*/
			//if ( !I.isSpecularBounce() ) {
			//}
			float pdf;
			float3 wi;
			const float3 wo = -1.0f*ray.dir();
			const float3 f = scatter( ray, I, rng, wo, wi, pdf );
			if ( NOOR::isBlack( f ) || pdf == 0.0f ) { 
				break; 
			}
			beta *= f * NOOR::AbsDot(wi, I._shading._n) / pdf;
			L += beta* direct( ray, I, rng );
			ray = I.spawnRay( wi );

			//// Russian Roulette 
			//if ( bounce >= _constant_spec._rr ) {
			//	const float q = fmaxf( 0.05f, 1.0f - NOOR::max3f( beta ) );
			//	if ( rng() < q ) break;
			//	beta /= 1.0f - q;
			//}
			//specular_bounce = I.isSpecularBounce();
		} else { // no intersection 
			if ( _constant_spec.is_sky_light_enabled() && ( bounce == 0 || I.isSpecularBounce() ) ) {
				L += beta*_skydome_manager.sample( ray.dir() );
			}
			break;
		}
	}
	return L;
}
#endif /* CUDAINTEGRATOR_CUH */
