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

#ifndef CUDALIGHTIMP_CUH
#define CUDALIGHTIMP_CUH

__forceinline__ __device__
float3 CudaLight::Le( const float3& w ) const {
    return _constant_spec._black;
}
//-------------- Area Light ----------------
__forceinline__ __device__
float3 CudaAreaLight::Le( const float3& w ) const {
    if ( _shape._type != SPHERE )
        return ( _two_sided || dot( w, _shape._n ) < 0 ) ?
        _ke :
        _constant_spec._black;
    else
        return _ke;
}

__forceinline__ __device__
float3 CudaAreaLight::sample_Le( const CudaIntersection& I,
                                 CudaRay& ray,
                                 float3& n,
                                 float& pdfPos,
                                 float& pdfDir )const {
    // Sample a point on the area light's _Shape_, _pShape_
    float3 p;
    _shape.sample( I, p, pdfPos, &n );

    // Sample a cosine-weighted outgoing direction _w_ for area light
    float3 w;
    const float2 u = make_float2( I._rng(), I._rng() );
    if ( _two_sided ) {
        // Choose a side to sample and then remap u[0] to [0,1] before
        // applying cosine-weighted hemisphere sampling for the chosen side.
        if ( u.x < .5 ) {
            w = NOOR::cosineSampleHemisphere( u, RIGHT_HANDED );
        } else {
            w = NOOR::cosineSampleHemisphere( u, RIGHT_HANDED );
            w.z *= -1;
        }
        pdfDir = 0.5f * NOOR::cosineHemispherePdf( fabsf( w.z ) );
    } else {
        w = NOOR::cosineSampleHemisphere( u, RIGHT_HANDED );
        pdfDir = NOOR::cosineHemispherePdf( w.y );
    }

    float3 v1, v2;
    NOOR::coordinateSystem( n, v1, v2 );
    w = w.x * v1 + w.y * v2 + w.z * n;
    ray = CudaRay( p, w );
    return Le( w );
}

__forceinline__ __device__
void CudaAreaLight::pdf_Le( const CudaRay& ray,
                            const float3& n,
                            float& pdfPos,
                            float& pdfDir ) const {
    pdfPos = _shape.pdf();
    pdfDir = _two_sided ? 
        .5f * NOOR::cosineHemispherePdf( NOOR::absDot( n, ray.getDir() ) )
        : NOOR::cosineHemispherePdf( dot( n, ray.getDir() ) );

}
__forceinline__ __device__
float CudaAreaLight::pdf_Li(
    const CudaIntersection& I,
    const float3& wi
) const {
    return _shape.pdf( I, wi );
}

__forceinline__ __device__
float3 CudaAreaLight::sample_Li(
    const CudaIntersection& I,
    CudaLightRecord& Lr,
    float& pdf
) const {
    float3 p;
    _shape.sample( I, p, pdf );
    if ( pdf == 0.f ) return _constant_spec._black;
    Lr._vis = CudaVisibility( I.getP(), p );
    return  Le( Lr._vis._wi );
}

//-------------- Infinite Light ----------------
__forceinline__ __device__
float3 CudaInfiniteLight::setPower() const {
    return NOOR_PI * _constant_spec._wr2 *
        _skydome_manager.evaluateLuminance();
}
__forceinline__ __device__
float3 CudaInfiniteLight::Le( const float3& w ) const {
    return ( _constant_spec.is_sky_light_enabled() ) ?
        _skydome_manager.evaluate( w ) :
        _constant_spec._black;
}
__forceinline__ __device__
float3 CudaInfiniteLight::sample_Le( const CudaIntersection& I,
                                 CudaRay& ray,
                                 float3& n,
                                 float& pdfPos,
                                 float& pdfDir )const {
    // Find $(u,v)$ sample coordinates in infinite light texture
    float mapPdf;
    float3 d = -1.f * _skydome_manager.importance_sample_dir( I._rng, mapPdf );
    n = d;

    // Compute origin for infinite light sample ray
    float3 v1, v2;
    NOOR::coordinateSystem( d, v1, v2 );
    const float2 u = make_float2(I._rng(), I._rng());
    const float2 cd = NOOR::concentricSampleDisk( u );
    const float3 worldCenter = make_float3( 0 );
    const float r = _constant_spec._world_radius;
    const float3 pDisk = worldCenter + r * ( cd.x * v1 + cd.y * v2 );
    ray = CudaRay( pDisk + r * d, -d );

    // Compute _InfiniteAreaLight_ ray PDFs
    pdfDir = mapPdf;
    pdfPos = 1.f / ( NOOR_PI * _constant_spec._wr2 );
    return Le( d );
}

__forceinline__ __device__
void CudaInfiniteLight::pdf_Le( const CudaRay& ray,
                            const float3& n,
                            float& pdfPos,
                            float& pdfDir ) const {
    const float3 d = -1.f*ray.getDir();
    const float theta = NOOR::sphericalTheta( d ), phi = NOOR::sphericalPhi( d );
    pdfDir = _skydome_manager.Pdf( make_float2( phi * NOOR_inv2PI, theta * NOOR_invPI ) )
        / ( 2.0f * NOOR_PI * NOOR_PI * sinf(theta) );
    pdfPos = 1.f / ( NOOR_PI * _constant_spec._wr2 );
}

__forceinline__ __device__
float CudaInfiniteLight::pdf_Li(
    const CudaIntersection& I,
    const float3& wi
) const {
    if ( !_constant_spec.is_sky_light_enabled() ) return 0.f;
    const float theta = NOOR::sphericalTheta( wi );
    const float phi = NOOR::sphericalPhi( wi );
    const float sinTheta = sinf( theta );
    if ( sinTheta == 0.f ) return 0.f;
    return _skydome_manager.Pdf( make_float2( phi*NOOR_inv2PI, theta*NOOR_invPI ) )
        / ( 2.0f * NOOR_PI * NOOR_PI * sinTheta );
}

__forceinline__ __device__
float3 CudaInfiniteLight::sample_Li(
    const CudaIntersection& I,
    CudaLightRecord& Lr,
    float& pdf
) const {
    Lr._vis._wi = _skydome_manager.importance_sample_dir( I._rng, pdf );
    if ( isinf( pdf ) || isnan( pdf ) ) pdf = 0.f;
    Lr._vis = CudaVisibility( I.getP(), I.getP() + 
                              2.f * _constant_spec._world_radius * 
                              Lr._vis._wi );
    return Le( Lr._vis._wi );
}


//-------------- Point Light ----------------
__forceinline__ __device__
float3 CudaPointLight::sample_Li(
    const CudaIntersection& I,
    CudaLightRecord& Lr,
    float& pdf
) const {
    pdf = 1.0f;
    Lr._vis = CudaVisibility( I.getP(), _position );
    const float dist2 = NOOR::length2( Lr._vis._to - Lr._vis._from );
    if ( dist2 == 0.0f ) {
        pdf = 0.0f;
        return _constant_spec._black;
    }
    return _ke / dist2;
}

//-------------- Spot Light ----------------
__forceinline__ __device__
float3 CudaSpotLight::sample_Li(
    const CudaIntersection& I,
    CudaLightRecord& Lr,
    float& pdf
) const {
    pdf = 1.0f;
    Lr._vis = CudaVisibility( I.getP(), _position );
    const float dist2 = NOOR::length2( Lr._vis._to - Lr._vis._from );
    if ( dist2 == 0.0f ) {
        pdf = 0.0f;
        return _constant_spec._black;
    } else {
        const float fo = falloff( Lr._vis._wi );
        if ( fo == 0.0f ) {
            pdf = 0.0f;
            return _constant_spec._black;
        }
        return _ke * fo / ( dist2 );
    }
}

//-------------- Distant Light ----------------
__forceinline__ __device__
float3 CudaDistantLight::sample_Li(
    const CudaIntersection& I,
    CudaLightRecord& Lr,
    float& pdf
) const {
    pdf = 1.0f;
    Lr._vis = CudaVisibility( I.getP(), I.getP() + 
                              2.f * _constant_spec._world_radius * _direction );
    return _ke;
}


__global__
void setup_lights(
    CudaLight** lights,
    CudaAreaLight* area_lights,
    int num_area_lights,
    CudaPointLight* point_lights,
    int num_point_lights,
    CudaSpotLight* spot_lights,
    int num_spot_lights,
    CudaDistantLight* distant_lights,
    int num_distant_lights,
    CudaInfiniteLight* infinite_lights,
    int num_infinite_lights
) {
    int curr = 0;
    int i = 0;
    for ( ; i < num_area_lights; ++i ) {
        area_lights[i]._shape._light_idx = curr;
        area_lights[i].setPower();
        lights[curr++] = (CudaLight*)&area_lights[i];
    }
    for ( i = 0; i < num_point_lights; ++i ) {
        point_lights[i].setPower();
        lights[curr++] = (CudaLight*)&point_lights[i];
    }
    for ( i = 0; i < num_spot_lights; ++i ) {
        spot_lights[i].setPower();
        lights[curr++] = (CudaLight*)&spot_lights[i];
    }
    for ( i = 0; i < num_distant_lights; ++i ) {
        distant_lights[i].setPower();
        lights[curr++] = (CudaLight*)&distant_lights[i];
    }
    for ( i = 0; i < num_infinite_lights; ++i ) {
        infinite_lights[i].setPower();
        lights[curr++] = (CudaLight*)&infinite_lights[i];
    }
}

struct CudaLightManager {
    CudaLight** _lights;
    CudaAreaLight* _area_lights;
    CudaPointLight* _point_lights;
    CudaSpotLight* _spot_lights;
    CudaDistantLight* _distant_lights;
    CudaInfiniteLight* _infinite_lights;

    int _num_area_lights;
    int _num_point_lights;
    int _num_spot_lights;
    int _num_distant_lights;
    int _num_infinite_lights;
    int _num_lights;

    CudaLightManager() = default;
    __host__
        CudaLightManager( const CudaPayload* payload ) {
        _num_lights = 0;
        _num_area_lights = static_cast<int>( payload->_num_area_lights );
        _num_lights += _num_area_lights;
        _num_point_lights = static_cast<int>( payload->_num_point_lights );
        _num_lights += _num_point_lights;
        _num_distant_lights = static_cast<int>( payload->_num_distant_lights );
        _num_lights += _num_distant_lights;
        _num_spot_lights = static_cast<int>( payload->_num_spot_lights );
        _num_lights += _num_spot_lights;
        _num_infinite_lights = static_cast<int>( payload->_num_infinite_lights );
        _num_lights += _num_infinite_lights;

        if ( _num_area_lights != 0 ) {
            checkNoorErrors( NOOR::malloc( &_area_lights,
                             _num_area_lights * sizeof( CudaAreaLight ) ) );
            checkNoorErrors( NOOR::memcopy( _area_lights,
                (void*)&payload->_area_light_data[0],
                             _num_area_lights * sizeof( CudaAreaLight ) ) );
        }

        if ( _num_point_lights != 0 ) {
            checkNoorErrors( NOOR::malloc( &_point_lights,
                             _num_point_lights * sizeof( CudaPointLight ) ) );
            checkNoorErrors( NOOR::memcopy( _point_lights,
                (void*)&payload->_point_light_data[0],
                             _num_point_lights * sizeof( CudaPointLight ) ) );
        }

        if ( _num_spot_lights != 0 ) {
            checkNoorErrors( NOOR::malloc( &_spot_lights,
                             _num_spot_lights * sizeof( CudaSpotLight ) ) );
            checkNoorErrors( NOOR::memcopy( _spot_lights,
                (void*)&payload->_spot_light_data[0],
                             _num_spot_lights * sizeof( CudaSpotLight ) ) );
        }

        if ( _num_distant_lights != 0 ) {
            checkNoorErrors( NOOR::malloc( &_distant_lights,
                             _num_distant_lights * sizeof( CudaDistantLight ) ) );
            checkNoorErrors( NOOR::memcopy( _distant_lights,
                (void*)&payload->_distant_light_data[0],
                             _num_distant_lights * sizeof( CudaDistantLight ) ) );
        }

        if ( _num_infinite_lights != 0 ) {
            checkNoorErrors( NOOR::malloc( &_infinite_lights,
                             _num_infinite_lights * sizeof( CudaInfiniteLight ) ) );
            checkNoorErrors( NOOR::memcopy( _infinite_lights,
                (void*)&payload->_infinite_light_data[0], sizeof( CudaInfiniteLight ) ) );
        }

        checkNoorErrors( NOOR::malloc( &_lights, _num_lights * sizeof( CudaLight* ) ) );

        dim3 blockSize( 1, 1, 1 );
        dim3 gridSize( 1, 1, 1 );
        setup_lights << < blockSize, gridSize >> > (
            _lights,
            _area_lights,
            _num_area_lights,
            _point_lights,
            _num_point_lights,
            _spot_lights,
            _num_spot_lights,
            _distant_lights,
            _num_distant_lights,
            _infinite_lights,
            _num_infinite_lights
            );
        checkNoorErrors( cudaDeviceSynchronize() );
        checkNoorErrors( cudaPeekAtLastError() );
    }
    __host__
        void free() {
        if ( _num_area_lights > 0 ) {
            checkNoorErrors( cudaFree( _area_lights ) );
        }
        if ( _num_point_lights > 0 ) {
            checkNoorErrors( cudaFree( _point_lights ) );
        }
        if ( _num_spot_lights > 0 ) {
            checkNoorErrors( cudaFree( _spot_lights ) );
        }
        if ( _num_distant_lights > 0 ) {
            checkNoorErrors( cudaFree( _distant_lights ) );
        }
        if ( _num_infinite_lights > 0 ) {
            checkNoorErrors( cudaFree( _infinite_lights ) );
        }
        if ( _num_lights > 0 ) {
            checkNoorErrors( cudaFree( _lights ) );
        }
    }

    __device__
        bool intersect( const CudaRay& ray, int light_idx ) const {
        if ( _lights[light_idx]->isInfiniteLight() &&
             _constant_spec.is_sky_light_enabled() ) {
            ray.setTmax( 2.0f*_constant_spec._world_radius );
            return true;
        }
        if ( _lights[light_idx]->isDeltaLight() ) return false;
        const CudaAreaLight* light = (CudaAreaLight*)_lights[light_idx];
        return( light->_shape.intersect( ray ) );
    }

    __device__
        bool isDeltaLight( int light_idx ) const {
        const CudaAreaLight* light = (CudaAreaLight*)_lights[light_idx];
        return( light->isDeltaLight() );
    }

    __device__
        float3 sample_Li(
        const CudaIntersection& I,
        CudaLightRecord& Lr,
        float& pdf
        ) const {
        pdf = 0;
        const int num_lights = _constant_spec.is_sky_light_enabled() ?
            _num_lights :
            _num_lights - 1;
        if ( num_lights == 0 ) return _constant_spec._black;
        const int light_idx = num_lights * I._rng();
        Lr._light_idx = light_idx;

        float3 Ld = _constant_spec._black;
        switch ( _lights[light_idx]->_type ) {
            case Area:
                Ld = ( (const CudaAreaLight*)
                       _lights[light_idx] )->sample_Li( I, Lr, pdf );
                break;
            case Point:
                Ld = ( (const CudaPointLight*)
                       _lights[light_idx] )->sample_Li( I, Lr, pdf );
                break;
            case Spot:
                Ld = ( (const CudaSpotLight*)
                       _lights[light_idx] )->sample_Li( I, Lr, pdf );
                break;
            case Distant:
                Ld = ( (const CudaDistantLight*)
                       _lights[light_idx] )->sample_Li( I, Lr, pdf );
                break;
            case Infinite:
                Ld = ( (const CudaInfiniteLight*)
                       _lights[light_idx] )->sample_Li( I, Lr, pdf );
                break;
            default:
                break;
        }
        if ( isinf( pdf ) ) pdf = 0.f;
        return Ld * num_lights;
    }

    __device__
        float pdf_Li(
        const CudaIntersection& I,
        const float3& wi,
        int light_idx
        ) const {
        float pdf = 0.f;
        switch ( _lights[light_idx]->_type ) {
            case Area:
                pdf = ( (const CudaAreaLight*)
                        _lights[light_idx] )->pdf_Li( I, wi );
                break;
            case Point:
                pdf = ( (const CudaPointLight*)
                        _lights[light_idx] )->pdf_Li( I, wi );
                break;
            case Spot:
                pdf = ( (const CudaSpotLight*)
                        _lights[light_idx] )->pdf_Li( I, wi );
                break;
            case Distant:
                pdf = ( (const CudaDistantLight*)
                        _lights[light_idx] )->pdf_Li( I, wi );
                break;
            case Infinite:
                pdf = ( (const CudaInfiniteLight*)
                        _lights[light_idx] )->pdf_Li( I, wi );
                break;
            default:
                break;
        }
        return isinf( pdf ) ? 0.f : pdf;
    }

    // emitted light in the direction of wi
    __device__
        float3 Le(
        const float3& wi,
        int light_idx
        ) const {
        switch ( _lights[light_idx]->_type ) {
            case Area:
                return ( (const CudaAreaLight*)_lights[light_idx] )->Le( wi );
            case Point:
                return ( (const CudaPointLight*)_lights[light_idx] )->Le( wi );
            case Spot:
                return ( (const CudaSpotLight*)_lights[light_idx] )->Le( wi );
            case Distant:
                return ( (const CudaDistantLight*)_lights[light_idx] )->Le( wi );
            case Infinite:
                return ( (const CudaInfiniteLight*)_lights[light_idx] )->Le( wi );
            default:
                return _constant_spec._black;
        }
    }
};

__constant__
CudaLightManager _light_manager;
#endif /* CUDALIGHT_CU */
