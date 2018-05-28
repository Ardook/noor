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
#ifndef CUDARAY_CUH
#define CUDARAY_CUH
class CudaRay {
    float3 _origin{ 0, 0, 0 };
    float3 _dir{ 0, 0, 0 };
    float3 _invDir{ 0, 0, 0 };

    float3 _origin_dx{ 0, 0, 0 };
    float3 _origin_dy{ 0, 0, 0 };
    float3 _dir_dx{ 0, 0, 0 };
    float3 _dir_dy{ 0, 0, 0 };

    NOOR::noor_int3 _posneg;
    mutable float _tmax{ NOOR_INF };

protected:
    bool _isDifferential{ false };
public:
    __device__
        CudaRay(
        const float3& origin,
        const float3& dir,
        float tmax = NOOR_INF,
        bool isDifferential = false
        ) :
        _origin( origin ),
        _dir( dir ),
        _tmax( tmax ),
        _isDifferential( isDifferential ) {
        update();
    }

    __device__
        CudaRay(
        const float3& origin,
        const float3& dir,
        const float3& origin_dx,
        const float3& origin_dy,
        const float3& dir_dx,
        const float3& dir_dy,
        float tmax = NOOR_INF
        ) :
        _origin( origin ),
        _dir( dir ),
        _tmax( tmax ),
        _isDifferential( true ),
        _origin_dx( origin_dx ),
        _origin_dy( origin_dy ),
        _dir_dx( dir_dx ),
        _dir_dy( dir_dy ) {
        update();
        //scaleDifferentials( .25f );
    }

    __device__
        void setOrigin( const float3& origin ) {
        _origin = origin;
    }
    __device__
        void setDir( const float3& dir ) {
        _dir = dir;
        update();
    }
    __device__
        void setTmax( float tmax ) const {
        _tmax = tmax;
    }
    __device__
        bool isDifferential() const {
        return _isDifferential;
    }
    __device__
        float getTmax() const {
        return _tmax;
    }
    __device__
        const NOOR::noor_int3& getPosneg() const {
        return _posneg;
    }
    __device__
        const float3& getOrigin() const {
        return _origin;
    }
    __device__
        const float3& getDir() const {
        return _dir;
    }
    __device__
        float getOrigin( uint axis ) const {
        return ( axis == 0 ? _origin.x : ( axis == 1 ? _origin.y : _origin.z ) );
    }
    __device__
        float getDir( uint axis ) const {
        return ( axis == 0 ? _dir.x : ( axis == 1 ? _dir.y : _dir.z ) );
    }
    __device__
        const float3& getInvDir() const {
        return _invDir;
    }
    __device__
        void setOriginDx( const float3& origin_dx ) {
        _origin_dx = origin_dx;
    }
    __device__
        void setOriginDy( const float3& origin_dy ) {
        _origin_dy = origin_dy;
    }
    __device__
        void setDirDx( const float3& dir_dx ) {
        _dir_dx = dir_dx;
    }
    __device__
        void setDirDy( const float3& dir_dy ) {
        _dir_dy = dir_dy;
    }
    __device__
        const float3& getOriginDx()const {
        return _origin_dx;
    }
    __device__
        const float3& getOriginDy()const {
        return _origin_dy;
    }
    __device__
        const float3& getDirDx()const {
        return _dir_dx;
    }
    __device__
        const float3& getDirDy()const {
        return _dir_dy;
    }

    __device__
        float3 pointAtParameter( float t ) const {
        return _origin + t*_dir;
    }
    __device__
        void transform( const CudaTransform& T ) {
        _origin = T.transformPoint( _origin );
        _dir = normalize( T.transformVector( _dir ) );
        update();
        if ( _isDifferential ) transformDifferentials( T );
    }
    __device__
        void transformDifferentials( const CudaTransform& T ) {
        _origin_dx = T.transformPoint( _origin_dx );
        _origin_dy = T.transformPoint( _origin_dy );
        _dir_dx = T.transformVector( _dir_dx );
        _dir_dy = T.transformVector( _dir_dy );
    }
    __device__
        CudaRay transformToObject( uint instance_idx ) const {
        CudaTransform T = _transform_manager.getWorldToObjectTransformation( instance_idx );
        const float3 origin = T.transformPoint( _origin );
        const float3 dir = T.transformVector( _dir );
        return CudaRay( origin, dir, _tmax );
    }
    __device__
        void scaleDifferentials( float s ) {
        _origin_dx = ( 1.f - s )*getOrigin() + s* _origin_dx;
        _origin_dy = ( 1.f - s )*getOrigin() + s* _origin_dy;

        _dir_dx = ( 1.f - s )*getDir() + s* _dir_dx;
        _dir_dy = ( 1.f - s )*getDir() + s* _dir_dy;
    }

    __device__
        void scaleDifferentials( const float2& s ) {
        _origin_dx = ( 1.f - s.x )*getOrigin() + s.x* _origin_dx;
        _origin_dy = ( 1.f - s.y )*getOrigin() + s.y* _origin_dy;

        _dir_dx = ( 1.f - s.x )*getDir() + s.x* _dir_dx;
        _dir_dy = ( 1.f - s.y )*getDir() + s.y* _dir_dy;
    }

private:
    __device__
        void update() {
        _invDir = 1.f / _dir;
        _posneg[0] = ( _dir.x > 0 ? 0 : 1 );
        _posneg[1] = ( _dir.y > 0 ? 0 : 1 );
        _posneg[2] = ( _dir.z > 0 ? 0 : 1 );
    }
};

#endif /* CUDARAY_CUH */

