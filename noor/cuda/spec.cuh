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
#ifndef SPEC_CUH
#define SPEC_CUH

#define LIGHT_AREA_MASK 0x0001
#define LIGHT_POINT_MASK 0x0002
#define LIGHT_SPOT_MASK 0x0004
#define LIGHT_DISTANT_MASK 0x0008
#define LIGHT_SUN_MASK 0x0010
#define LIGHT_SKY_MASK 0x0020

static constexpr short NOT_LIGHT_AREA_MASK = ~LIGHT_AREA_MASK;
static constexpr short NOT_LIGHT_POINT_MASK = ~LIGHT_POINT_MASK;
static constexpr short NOT_LIGHT_SPOT_MASK = ~LIGHT_SPOT_MASK;
static constexpr short NOT_LIGHT_DISTANT_MASK = ~LIGHT_DISTANT_MASK;
static constexpr short NOT_LIGHT_SUN_MASK = ~LIGHT_SUN_MASK;
static constexpr short NOT_LIGHT_SKY_MASK = ~LIGHT_SKY_MASK;

class CudaSpec {
public:
    float3 _white;
    float3 _black;
    float _reflection_bias;
    float _shadow_bias;
    float _world_radius;
    unsigned int _bvh_root_node;
    unsigned short _lighting_type;
    unsigned short _bounces;
    unsigned short _rr;
    unsigned short _bvh_height;
    unsigned short _gpuID;
    SkydomeType _skydome_type;
    bool _debug_sky;
    bool _mis;
    mutable bool _outofsync;
#ifndef __CUDACC__
    __host__
        CudaSpec() :_lighting_type( 0 )
        , _outofsync( true )
        , _debug_sky( false )
        , _mis( false ) {}
#endif
    __host__
        void setOutOfSync( bool flag )const {
        _outofsync = flag;
    }

    __host__ __device__
        void enable_area_light() {
        _lighting_type |= LIGHT_AREA_MASK;
        _outofsync = true;
    }

    __host__ __device__
        void enable_point_light() {
        _lighting_type |= LIGHT_POINT_MASK;
        _outofsync = true;
    }

    __host__ __device__
        void enable_spot_light() {
        _lighting_type |= LIGHT_SPOT_MASK;
        _outofsync = true;
    }

    __host__ __device__
        void enable_distant_light() {
        _lighting_type |= LIGHT_DISTANT_MASK;
        _outofsync = true;
    }

    __host__ __device__
        void enable_mis() {
        _mis = true;
        _outofsync = true;
    }

    __host__ __device__
        void disable_mis() {
        _mis = false;
        _outofsync = true;
    }

    __host__ __device__
        void enable_debug_sky() {
        _debug_sky = true;
        _outofsync = true;
    }

    __host__ __device__
        void disable_debug_sky() {
        _debug_sky = false;
        _outofsync = true;
    }

    __host__ __device__
        void enable_sky_light() {
        _lighting_type |= LIGHT_SKY_MASK;
        _outofsync = true;
    }

    __host__ __device__
        void disable_sky_light() {
        _lighting_type &= NOT_LIGHT_SKY_MASK;
        _outofsync = true;
    }

    __host__ __device__
        void disable_area_light() {
        _lighting_type &= NOT_LIGHT_AREA_MASK;
        _outofsync = true;
    }

    __host__ __device__
        void disable_distant_light() {
        _lighting_type &= NOT_LIGHT_DISTANT_MASK;
        _outofsync = true;
    }

    __host__ __device__
        void disable_point_light() {
        _lighting_type &= NOT_LIGHT_POINT_MASK;
        _outofsync = true;
    }

    __host__ __device__
        void disable_spot_light() {
        _lighting_type &= NOT_LIGHT_SPOT_MASK;
        _outofsync = true;
    }

    __host__ __device__
        bool is_mis_enabled() const {
        return ( _mis == true );
    }
    __host__ __device__
        bool is_area_light_enabled() const {
        return ( ( _lighting_type & LIGHT_AREA_MASK ) != 0 );
    }

    __host__ __device__
        bool is_sun_light_enabled() const {
        return ( ( _lighting_type & LIGHT_SUN_MASK ) != 0 );
    }

    __host__ __device__
        bool is_distant_light_enabled() const {
        return ( ( _lighting_type & LIGHT_DISTANT_MASK ) != 0 );
    }

    __host__ __device__
        bool is_point_light_enabled() const {
        return ( ( _lighting_type & LIGHT_POINT_MASK ) != 0 );
    }

    __host__ __device__
        bool is_spot_light_enabled() const {
        return ( ( _lighting_type & LIGHT_SPOT_MASK ) != 0 );
    }

    __host__ __device__
        bool is_sky_light_enabled() const {
        return ( ( _lighting_type & LIGHT_SKY_MASK ) != 0 );
    }
};

#ifdef __CUDACC__
__constant__ CudaSpec	_constant_spec;
#endif

#endif /* SPEC_CUH */