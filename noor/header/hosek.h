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
#ifndef HOSEK_H
#define HOSEK_H 
class HosekSky {
    friend class Scene;
    Scene& _scene;
    CudaHosekSky _cuda_hosek_sky;
    float _turbidity{ 3.5f };
    float _scale{ 250.f };
    float _solar_radius{ 2.f };
    float _sun_lon{ NOOR_PI_over_4 };
    float _sun_lat{ NOOR_PI_over_4 };
    float3 _ground_albedo{ make_float3( 1.0f ) };
    float3 _solar_dir;
    bool _outofsync{ true };

public:
    HosekSky( Scene& scene ) :
        _scene( scene )
    {
        update();
    }

    HosekSky( const HosekSky& sky ) = default;
    HosekSky& operator=( const HosekSky& sky ) = default;


    void motion() {
        if ( _scene._mouse->buttonPressed( GLFW_MOUSE_BUTTON_LEFT, GLFW_MOD_ALT) ){
            _sun_lon += NOOR_2PI * _scene._mouse->_dt.x;
            _sun_lat += NOOR_PI_over_2 * _scene._mouse->_dt.y;
            _sun_lon = _sun_lon <= 0.0f ? NOOR_2PI : _sun_lon;
            _sun_lat = glm::clamp( _sun_lat, 0.0f, NOOR_PI_over_2 );
            update();
        }
    }

    void update() {
        const float elevation = NOOR_PI_over_2 - _sun_lat;
        arhosek_rgb_skymodelstate_alloc_update( 
            (ArHosekSkyModelState*) &_cuda_hosek_sky._state_r, 
            _turbidity, 
            _ground_albedo.x, 
            elevation );
        arhosek_rgb_skymodelstate_alloc_update( 
            (ArHosekSkyModelState*) &_cuda_hosek_sky._state_g, 
            _turbidity, 
            _ground_albedo.y, 
            elevation );
        arhosek_rgb_skymodelstate_alloc_update( 
            (ArHosekSkyModelState*) &_cuda_hosek_sky._state_b, 
            _turbidity, 
            _ground_albedo.z, 
            elevation );
        _cuda_hosek_sky._solar_scale = _scale;
        _cuda_hosek_sky._solar_dir = NOOR::sphericalDirection( _sun_lat, _sun_lon );
        _cuda_hosek_sky._solar_radius = _solar_radius * _cuda_hosek_sky._state_r._solar_radius;
        _cuda_hosek_sky._ground_albedo = _ground_albedo;
        _outofsync = true;
    }

    void updateCudaHosek() {
        if ( _outofsync ) {
            update_cuda_hosek();
            _outofsync = false;
        }
    }
};
#endif // HOSEK_CUH
