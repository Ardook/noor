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

#ifndef CUDALIGHT_CUH
#define CUDALIGHT_CUH
class CudaIntersection;
class CudaRNG;
class CudaRay;
class CudaLightRecord;

// LightFlags Declarations
enum LightType {
    Point = 1 << 0,
    Spot = 1 << 1,
    Distant = 1 << 2,
    Area = 1 << 3,
    Infinite = 1 << 4,
    Delta = Point | Spot | Distant
};

struct CudaLight {
    float3 _ke;
    float3 _power;
    LightType _type;

    __host__ __device__
        CudaLight( const float3& ke, LightType t ) :
        _ke( ke )
        , _type( t ) {}
    __device__
        bool isDeltaLight() const {
        return ( ( _type & Delta ) != 0 );
    }
    __device__
        bool isInfiniteLight() const {
        return ( _type == Infinite );
    }
    __device__
        float3 Le( const float3& w ) const;

    __device__
        float3 power() const {
        return _power;
    }

    __device__
        float pdf_Li(
        const CudaIntersection& I,
        const float3& wi
        ) const {
        return 0.0f;
    }
};

class CudaAreaLight : public CudaLight {
public:
    CudaShape _shape;
    bool _two_sided;

    __host__
        CudaAreaLight(
        const CudaShape& shape,
        const float3& ke,
        bool two_sided = false
        ) : CudaLight( ke, Area ),
        _shape( shape ),
        _two_sided( two_sided ) {}

#ifdef __CUDACC__
    __device__
        float3 setPower()const {
        return ( _two_sided ? 2 : 1 ) * _ke * _shape._area * NOOR_PI;
    }

    __device__
        float3 Le( const float3& w ) const;
    __device__
        float3 sample_Le( const CudaIntersection& I,
                          CudaRay& ray,
                          float3& n,
                          float& pdfPos,
                          float& pdfDir )const;
    __device__
        void pdf_Le( const CudaRay& ray,
                     const float3& n,
                     float& pdfPos,
                     float& pdfDir ) const;
    __device__
        float pdf_Li(
        const CudaIntersection& I,
        const float3& wi
        ) const;
    __device__
        float3 sample_Li(
        const CudaIntersection& I,
        CudaLightRecord& Lr,
        float& pdf
        ) const;
#endif
};

class CudaInfiniteLight : public CudaLight {
public:
    __host__ __device__
        CudaInfiniteLight(
        float world_radius
        ) : CudaLight( make_float3( 0 ), Infinite ) {}

#ifdef __CUDACC__
    __device__
        float3 setPower() const;
    __device__
        float3 Le( const float3& w ) const;
    __device__
        float3 sample_Le( const CudaIntersection& I,
                          CudaRay& ray,
                          float3& n,
                          float& pdfPos,
                          float& pdfDir )const;
    __device__
        void pdf_Le( const CudaRay& ray,
                     const float3& n,
                     float& pdfPos,
                     float& pdfDir ) const;
    __device__
        float pdf_Li(
        const CudaIntersection& I,
        const float3& wi
        ) const;
    __device__
        float3 sample_Li(
        const CudaIntersection& I,
        CudaLightRecord& Lr,
        float& pdf
        ) const;
#endif
};
class CudaPointLight : public CudaLight {
public:
public:
    float3 _position;

    __host__ __device__
        CudaPointLight(
        const float3& position
        , const float3& ke
        ) : CudaLight( ke, Point )
        , _position( position ) {}

#ifdef __CUDACC__
    __device__
        float3 setPower() const {
        return NOOR_4PI * _ke;
    }
    __device__
        float3 sample_Li(
        const CudaIntersection& I,
        CudaLightRecord& Lr,
        float& pdf
        ) const;
#endif
};

class CudaSpotLight : public CudaLight {
public:
    float3 _position;
    float3 _direction;
    float _cosinner;
    float _cosouter;

    __host__ __device__
        CudaSpotLight(
        const float3& position
        , const float3& direction
        , const float3& ke
        , float cosinner
        , float cosouter
        ) : CudaLight( ke, Spot )
        , _position( position )
        , _direction( direction )
        , _cosinner( cosinner )
        , _cosouter( cosouter ) {}

#ifdef __CUDACC__
    __device__
        float3 setPower() const {
        return _ke * NOOR_2PI * ( 1 - .5f * ( _cosinner + _cosouter ) ); 
    }
    __device__
        float falloff( const float3& w ) const {
        float costheta = dot( _direction, w );
        if ( costheta < _cosouter ) return 0;
        if ( costheta >= _cosinner ) return 1;
        // Compute falloff inside spotlight cone
        const float delta = ( costheta - _cosouter ) / ( _cosinner - _cosouter );
        return ( delta * delta ) * ( delta * delta );
    }
    __device__
        float3 sample_Li(
        const CudaIntersection& I,
        CudaLightRecord& Lr,
        float& pdf
        ) const;
#endif
};

class CudaDistantLight : public CudaLight {
public:
    float3 _direction;
    __host__ __device__
        CudaDistantLight(
        const float3& direction
        , const float3& ke
        ) : CudaLight( ke, Distant )
        , _direction( normalize( direction ) )
    {}
#ifdef __CUDACC__
    __device__
        float3 setPower()const {
        return _ke * NOOR_PI * _constant_spec._wr2;
    }
    __device__ float3 sample_Li(
        const CudaIntersection& I,
        CudaLightRecord& Lr,
        float& pdf
    ) const;
#endif
};


#endif /* CUDALIGHT_CUH */
