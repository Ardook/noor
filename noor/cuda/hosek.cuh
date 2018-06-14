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

/* Cuda port of the Hosek & Wilkie Sun Sky Model. http://cgg.mff.cuni.cz/projects/SkylightModelling/ */

#ifndef CUDAHOSEKSKY_CUH
#define CUDAHOSEKSKY_CUH 

typedef float CudaArHosekSkyModelConfiguration[9];

struct CudaArHosekSkyModelState {
    CudaArHosekSkyModelConfiguration _configs[3];
    NOOR::noor_float3            _radiances;
    float                        _turbidity;
    float                        _solar_radius;
    NOOR::noor_float3            _emission_correction_factor_sky;
    NOOR::noor_float3            _emission_correction_factor_sun;
    float                        _albedo;
    float                        _elevation;
};

class CudaHosekSky {
public:
    CudaArHosekSkyModelState _state_r;
    CudaArHosekSkyModelState _state_g;
    CudaArHosekSkyModelState _state_b;
    float3 _solar_dir;
    float3 _ground_albedo;
    float _solar_scale;
    float _solar_radius;
    CudaHosekSky() = default;
    __device__
        float arhosek_tristim_skymodel_radiance(
        const CudaArHosekSkyModelState& state,
        float                  theta,
        float                  gamma,
        int                    channel
        ) const {
        return
            ArHosekSkyModel_GetRadianceInternal(
            state._configs[channel],
            theta,
            gamma
            )
            * state._radiances[channel];
    }
    __device__
        float ArHosekSkyModel_GetRadianceInternal(
        const CudaArHosekSkyModelConfiguration&  configuration,
        float                        theta,
        float                        gamma
        ) const {
        const float cos_gamma = cosf( gamma );
        const float cos_gamma_2 = cos_gamma*cos_gamma;
        const float cos_theta = cosf( theta );
        const float expM = expf( configuration[4] * gamma );
        const float rayM = cos_gamma_2;
        const float mieM = ( 1.0f + cos_gamma_2 ) / powf( ( 1.0f + configuration[8] * configuration[8] - 2.0f*configuration[8] * cos_gamma ), 1.5 );
        const float zenith = sqrtf( cos_theta );

        return ( 1.0f + configuration[0] * expf( configuration[1] / ( cos_theta + 0.01f ) ) ) *
            ( configuration[2] + configuration[3] * expM + configuration[5] * rayM + configuration[6] * mieM + configuration[7] * zenith );
    }
    __device__
        float3 querySkyModel( const float3& v ) const {
        if ( v.y < 0.f ) return make_float3( 0 );
        const float scale = .025f;
        const float theta = clamp( NOOR::sphericalTheta( v, RIGHT_HANDED ), 0.f, NOOR_PI_over_2 );
        const float gamma = acosf( clamp( dot( v, _solar_dir ), -1.f, 1.f ) );
        float3 rgb;
        rgb.x = arhosek_tristim_skymodel_radiance( _state_r, theta, gamma, 0 );
        rgb.y = arhosek_tristim_skymodel_radiance( _state_g, theta, gamma, 1 );
        rgb.z = arhosek_tristim_skymodel_radiance( _state_b, theta, gamma, 2 );
        if ( gamma < _solar_radius ) {
            return _solar_scale* rgb;
        }
        return rgb * scale;
    }
};
#ifdef __CUDACC__
__constant__ CudaHosekSky _constant_hosek_sky;
#endif
#endif /* CUDAHOSEKSKY_CUH */
