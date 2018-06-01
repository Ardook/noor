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
#ifndef CUDAINTERSECTION_CUH
#define CUDAINTERSECTION_CUH
class CudaVisibility {
public:
    float3 _from;
    float3 _to;
    float3 _wi;
    float _dist;

    __device__
        CudaVisibility() {}

    __device__
        CudaVisibility( const float3 from, const float3 to ) :
        _from( from ),
        _to( to ),
        _wi( to - from ) {
        _dist = length( _wi );
        _wi /= _dist;
    }
};

class CudaLightRecord {
public:
    int _light_idx{ -1 };
    CudaVisibility _vis;

    __device__
        CudaLightRecord() {}
    __device__
        CudaLightRecord( int light_idx, const CudaVisibility& vis ) :
        _light_idx( light_idx ),
        _vis( vis ) {}
};

class CudaIntersection {
    struct GeometryFrame {
        float3 _dpdu{ 0.0f, 0.0f, 0.0f };
        float3 _dpdv{ 0.0f, 0.0f, 0.0f };
    };
    struct DifferentialFrame {
        float3 _dpdx{ 0.0f, 0.0f, 0.0f };
        float3 _dpdy{ 0.0f, 0.0f, 0.0f };

        float _dudx{ 0.0f };
        float _dudy{ 0.0f };
        float _dvdx{ 0.0f };
        float _dvdy{ 0.0f };

        float3 _dndu{ 0.0f, 0.0f, 0.0f };
        float3 _dndv{ 0.0f, 0.0f, 0.0f };
        __device__
            void reset() {
            _dudx = _dudy = _dvdx = _dvdy = 0;
        }
    };
public:
    struct ShadingFrame {
        float3 _n{ 0.0f, 0.0f, 0.0f };
        float3 _dpdu{ 0.0f, 0.0f, 0.0f };
        float3 _dpdv{ 0.0f, 0.0f, 0.0f };
    };
    GeometryFrame _geometry;
    ShadingFrame _shading;
    DifferentialFrame _differential;
    float3 _n{ 0.0f, 0.0f, 0.0f };
    float3 _p{ 0.0f, 0.0f, 0.0f };
    float3 _wo{ 0.0f, 0.0f, 0.0f };
    float2 _uv{ 0.0f, 0.0f };
    float _u{ 0.0f };
    float _v{ 0.0f };
    uint _tri_idx{ 0u };
    uint _mat_idx{ 0u };
    uint _material_type{ DIFFUSE };
    uint _ins_idx{ 0u };
    int _tid;
    mutable bool _specular_bounce{ false };
    mutable float _eta{ 1.0f };

    __device__
        CudaRay spawnRay( const CudaRay& ray, const float3& wi ) const {
        const bool isDifferential = isGlossyBounce() && ray.isDifferential();
        if ( isDifferential ) {
            if ( dot( wi, _n ) >= 0.0f ) {
                const float3& origin = _p + _constant_spec._reflection_bias * _n;
                const float3& dir = wi;
                const float3 origin_dx = ( origin + _differential._dpdx );
                const float3 origin_dy = ( origin + _differential._dpdy );

                const float3 dndx = _differential._dndu * _differential._dudx + _differential._dndv * _differential._dvdx;
                const float3 dndy = _differential._dndu * _differential._dudy + _differential._dndv * _differential._dvdy;
                const float3 dwodx = -( ray.getDirDx() + _wo );
                const float3 dwody = -( ray.getDirDy() + _wo );
                const float dDNdx = dot( dwodx, _shading._n ) + dot( _wo, dndx );
                const float dDNdy = dot( dwody, _shading._n ) + dot( _wo, dndy );

                const float3 dir_dx = wi - dwodx + 2.f *( dot( _wo, _shading._n ) * dndx + dDNdx * _shading._n );
                const float3 dir_dy = wi - dwody + 2.f *( dot( _wo, _shading._n ) * dndy + dDNdy * _shading._n );
                return CudaRay( origin, dir, origin_dx, origin_dy, dir_dx, dir_dy );
            } else {
                const float3& origin = _p - _constant_spec._reflection_bias * _n;
                const float3& dir = wi;
                const float3 origin_dx = ( origin + _differential._dpdx );
                const float3 origin_dy = ( origin + _differential._dpdy );

                const float3 w = -1.f*_wo;
                const float eta = dot( _wo, _shading._n ) > 0 ? 1.f / _eta : _eta;

                const float3 dndx = _differential._dndu * _differential._dudx +
                    _differential._dndv * _differential._dvdx;

                const float3 dndy = _differential._dndv * _differential._dudy +
                    _differential._dndv * _differential._dvdy;

                const float3 dwodx = -( ray.getDirDx() + _wo );
                const float3 dwody = -( ray.getDirDy() + _wo );
                const float dDNdx = dot( dwodx, _shading._n ) + dot( _wo, dndx );
                const float dDNdy = dot( dwody, _shading._n ) + dot( _wo, dndy );
                const float mu = eta * dot( w, _shading._n ) - dot( wi, _shading._n );
                const float dmudx =
                    ( eta - ( eta * eta * dot( w, _shading._n ) ) / dot( wi, _shading._n ) ) * dDNdx;
                const float dmudy =
                    ( eta - ( eta * eta * dot( w, _shading._n ) ) / dot( wi, _shading._n ) ) * dDNdy;

                const float3 dir_dx = wi + eta * dwodx - ( mu * dndx + dmudx *_shading._n );
                const float3 dir_dy = wi + eta * dwody - ( mu * dndy + dmudy *_shading._n );
                return CudaRay( origin, dir, origin_dx, origin_dy, dir_dx, dir_dy );
            }
        } else {
            return dot( wi, _n ) >= 0.0f ?
                CudaRay( _p + _constant_spec._reflection_bias * _n, wi ) :
                CudaRay( _p - _constant_spec._reflection_bias * _n, wi );
        }
    }

    __device__
        CudaRay spawnShadowRay( const float3& dir, float tmax = NOOR_INF ) const {
        return dot( dir, _n ) >= 0.0f ?
            CudaRay( _p + _constant_spec._reflection_bias * _n, dir, tmax ) :
            CudaRay( _p - _constant_spec._reflection_bias * _n, dir, tmax );
    }

    __device__
        CudaRay spawnShadowRay( const CudaVisibility& v ) const {
        const float3 p = dot( v._wi, _n ) >= 0.0f ?
            v._from + _constant_spec._reflection_bias * _n :
            v._from - _constant_spec._reflection_bias * _n;
        float dist = length( p - v._to );
        dist -= dist * _constant_spec._shadow_bias;
        return CudaRay( p, v._wi, dist );
    }

    __device__
        void setEta( float eta ) const {
        _eta = eta;
    }

    __device__
        void setSpecularBounce( bool b ) const {
        _specular_bounce = b;
    }
    __device__
        bool isBumped() const {
        return ( ( _material_type & BUMP ) != 0 );
    }
    __device__
        bool isShadowCatcher() const {
        return ( ( _material_type & SHADOW ) != 0 );
    }
    __device__
        bool isSpecularBounce() const {
        return _specular_bounce;
    }
    __device__
        bool isTransparentBounce() const {
        return ( ( _material_type & NOOR_TRANSPARENT ) != 0 );
    }
    __device__
        bool isGlossyBounce() const {
        return ( ( _material_type & NOOR_GLOSSY ) != 0 );
    }
    __device__
        float3 getVertex( uint attr_idx ) const {
        const float4 v = _mesh_manager.getVertex( attr_idx );
        return make_float3( v );
    }
    __device__
        float3 getVertexNormal( uint attr_idx ) const {
        const float4 n = _mesh_manager.getVertexNormal( attr_idx );
        return make_float3( n );
    }
    __device__
        float2 getUV( uint attr_idx ) const {
        return _mesh_manager.getUV( attr_idx );
    }
    __device__
        void updateIntersection( const CudaRay& ray ) {
        if ( _material_type & MESHLIGHT ) return;
        _specular_bounce = ( _material_type & NOOR_SPECULAR );
        const uint4 attr_idx = _mesh_manager.getAttrIndex( _tri_idx );
        _mat_idx = attr_idx.w;
        if ( _material_type & EMITTER ) return;
        _wo = normalize( -1.0f*ray.getDir() );
        computeTangentSpace( ray, attr_idx );
        computeDifferential( ray );
    }

    __device__
        void computeTangentSpace(
        const CudaRay& ray,
        const uint4& attr_idx
        ) {
        const float2 uv0 = getUV( attr_idx.x );
        const float2 uv1 = getUV( attr_idx.y );
        const float2 uv2 = getUV( attr_idx.z );

        const float3 v0 = getVertex( attr_idx.x );
        const float3 v1 = getVertex( attr_idx.y );
        const float3 v2 = getVertex( attr_idx.z );

        const float3 n0 = getVertexNormal( attr_idx.x );
        const float3 n1 = getVertexNormal( attr_idx.y );
        const float3 n2 = getVertexNormal( attr_idx.z );

        const CudaTransform T = _transform_manager.getNormalTransformation( _ins_idx );
        const float w = 1.0f - _u - _v;
        _uv = w*uv0 + _u*uv1 + _v*uv2;
        _shading._n = NOOR::normalize( T.transformNormal( w*n0 + _u*n1 + _v*n2 ) );

        const CudaTransform S = _transform_manager.getObjectToWorldTransformation( _ins_idx );
        const float3 dp01 = S.transformVector( v1 - v0 );
        const float3 dp02 = S.transformVector( v2 - v0 );
        _n = NOOR::normalize( cross( dp01, dp02 ) );

        const float2 duv01 = uv1 - uv0;
        const float2 duv02 = uv2 - uv0;
        NOOR::Matrix2x2 A{ duv01, duv02 };
        const float determinant = A.determinant();
        const bool degenerateUV = fabsf( determinant ) < 1e-8;
        if ( !degenerateUV ) {
            A.inverse();
            _geometry._dpdu = ( A[0][0] * dp01 + A[0][1] * dp02 );
            _geometry._dpdv = ( A[1][0] * dp01 + A[1][1] * dp02 );
            if ( NOOR::length2( cross( _geometry._dpdu, _geometry._dpdv ) ) == 0.0f ) {
                NOOR::coordinateSystem( _n, _geometry._dpdu, _geometry._dpdv );
            }
        } else {
            NOOR::coordinateSystem( _n, _geometry._dpdu, _geometry._dpdv );
        }
        const float3& n = _shading._n;
        _shading._dpdu = _geometry._dpdu;
        _shading._dpdv = _geometry._dpdv;
        // Gram-Schmidt
        _shading._dpdu = NOOR::normalize( _shading._dpdu - n*dot( n, _shading._dpdu ) );
        const float sign = dot( cross( n, _shading._dpdu ), _shading._dpdv ) < 0.0f ? -1.0f : 1.0f;
        _shading._dpdv = NOOR::normalize( sign*cross( n, _shading._dpdu ) );

        // Compute deltas for triangle partial derivatives of normal
        const float3 dn01 = n1 - n0;
        const float3 dn02 = n2 - n0;
        if ( !degenerateUV ) {
            _differential._dndu = ( A[0][0] * dn01 + A[0][1] * dn02 );
            _differential._dndv = ( A[1][0] * dn01 + A[1][1] * dn02 );
        } else {
            const float3 dn = NOOR::normalize( T.transformNormal( cross( dn01, dn02 ) ) );
            if ( NOOR::length2( dn ) == 0 ) {
                _differential._dndu = _differential._dndv = make_float3( 0.0f );
            } else {
                NOOR::coordinateSystem( dn, _differential._dndu, _differential._dndv );
            }
        }
    }

    __device__
        void computeDifferential( const CudaRay& ray ) {
        _differential.reset();
        if ( !ray.isDifferential() ) {
            return;
        }
        const float d = dot( _n, _p );
        const float tx = -( dot( _n, ray.getOriginDx() ) - d ) / dot( _n, ray.getDirDx() );
        if ( isinf( tx ) || isnan( tx ) ) {
            return;
        }
        const float3 px = ray.getOriginDx() + tx * ray.getDirDx();

        const float ty = -( dot( _n, ray.getOriginDy() ) - d ) / dot( _n, ray.getDirDy() );
        if ( isinf( ty ) || isnan( ty ) ) {
            return;
        }
        const float3 py = ray.getOriginDy() + ty * ray.getDirDy();

        _differential._dpdx = px - _p;
        _differential._dpdy = py - _p;

        int2 dim;
        if ( fabsf( _n.x ) > fabsf( _n.y ) && fabsf( _n.x ) > fabsf( _n.z ) ) {
            dim.x = 1;
            dim.y = 2;
        } else if ( fabsf( _n.y ) > fabsf( _n.z ) ) {
            dim.x = 0;
            dim.y = 2;
        } else {
            dim.x = 0;
            dim.y = 1;
        }
        const NOOR::noor_float3 dpdu( _geometry._dpdu );
        const NOOR::noor_float3 dpdv( _geometry._dpdv );

        const NOOR::noor_float3 dpdx( _differential._dpdx );
        const NOOR::noor_float3 dpdy( _differential._dpdy );

        const NOOR::Matrix2x2 A{ make_float2( dpdu[dim.x], dpdv[dim.x] ),
                                 make_float2( dpdu[dim.y], dpdv[dim.y] ) };
        const NOOR::noor_float2 Bx{ make_float2( dpdx[dim.x], dpdx[dim.y] ) };
        const NOOR::noor_float2 By{ make_float2( dpdy[dim.x], dpdy[dim.y] ) };

        NOOR::solve2x2( A, Bx, _differential._dudx, _differential._dvdx );
        NOOR::solve2x2( A, By, _differential._dudy, _differential._dvdy );
    }
};

#endif /* CUDAINTERSECTION_CUH */