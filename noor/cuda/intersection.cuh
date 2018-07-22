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

struct ShadingFrame {
    float3 _n{ 0.0f, 0.0f, 0.0f };
    float3 _dpdu{ 0.0f, 0.0f, 0.0f };
    float3 _dpdv{ 0.0f, 0.0f, 0.0f };
};
struct GeometryFrame {
    float3 _p{ 0.0f, 0.0f, 0.0f };
    float3 _n{ 0.0f, 0.0f, 0.0f };
};
struct CudaIntersectionRecord {
    ShadingFrame _shading;
    GeometryFrame _geometry;

    float3 _wo{ 0.0f, 0.0f, 0.0f };
    float3 _wi{ 0.0f, 0.0f, 0.0f };
    float2 _bc{ 0.0f, 0.0f };
    uint _tri_idx{ 0 };
    uint _ins_idx{ 0 };
    uint _tid{ 0 };
    MaterialType _material_type{ DIFFUSE };

    CudaIntersectionRecord() = default;

    __device__
        CudaIntersectionRecord( uint tid ) : _tid( tid ) {}
};

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

class  CudaIntersection {
    struct DifferentialFrame {
        float3 _dpdx{ 0.0f, 0.0f, 0.0f };
        float3 _dpdy{ 0.0f, 0.0f, 0.0f };

        float3 _dndu{ 0.0f, 0.0f, 0.0f };
        float3 _dndv{ 0.0f, 0.0f, 0.0f };

        float _dudx{ 0.0f };
        float _dudy{ 0.0f };
        float _dvdx{ 0.0f };
        float _dvdy{ 0.0f };
    };

public:
    const CudaRNG& _rng;
    DifferentialFrame _differential;
    float2 _uv{ 0.0f, 0.0f };
    uint _mat_idx{ 0u };
    mutable float _eta{ 1.0f };
    CudaIntersectionRecord* _rec;

    __device__
        CudaIntersection( const CudaRay& ray,
                          const CudaRNG& rng,
                          CudaIntersectionRecord* rec ) :
        _rng( rng ),
        _rec( rec )
    {
        updateIntersection( ray );
    }

    __device__
        void spawnRay( CudaRay& ray ) const {
        const bool isDifferential = isGlossy() && ray.isDifferential();
        if ( isDifferential ) {
            if ( dot( _rec->_wi, _rec->_geometry._n ) >= 0.0f ) {
                const float3& origin = _rec->_geometry._p + 
                    _constant_spec._reflection_bias * _rec->_geometry._n;
                const float3& dir = _rec->_wi;
                const float3 origin_dx = ( origin + _differential._dpdx );
                const float3 origin_dy = ( origin + _differential._dpdy );

                const float3 dndx = _differential._dndu * _differential._dudx +
                    _differential._dndv * _differential._dvdx;
                const float3 dndy = _differential._dndu * _differential._dudy +
                    _differential._dndv * _differential._dvdy;
                const float3 dwodx = -( ray.getDirDx() + _rec->_wo );
                const float3 dwody = -( ray.getDirDy() + _rec->_wo );
                const float dDNdx = dot( dwodx, _rec->_shading._n ) + 
                    dot( _rec->_wo, dndx );
                const float dDNdy = dot( dwody, _rec->_shading._n ) + 
                    dot( _rec->_wo, dndy );

                const float3 dir_dx = _rec->_wi - dwodx + 
                    2.f *( dot( _rec->_wo, _rec->_shading._n ) * 
                           dndx + dDNdx * _rec->_shading._n );
                const float3 dir_dy = _rec->_wi - dwody + 
                    2.f *( dot( _rec->_wo, _rec->_shading._n ) * 
                           dndy + dDNdy * _rec->_shading._n );
                ray = CudaRay( origin, dir, origin_dx, origin_dy, dir_dx, dir_dy );
            } else {
                const float3& origin = _rec->_geometry._p - 
                    _constant_spec._reflection_bias * _rec->_geometry._n;
                const float3& dir = _rec->_wi;
                const float3 origin_dx = ( origin + _differential._dpdx );
                const float3 origin_dy = ( origin + _differential._dpdy );

                const float3 w = -1.f*_rec->_wo;
                const float eta = dot( _rec->_wo, _rec->_shading._n ) > 0 ? 
                    1.f / _eta : _eta;

                const float3 dndx = _differential._dndu * _differential._dudx +
                    _differential._dndv * _differential._dvdx;

                const float3 dndy = _differential._dndv * _differential._dudy +
                    _differential._dndv * _differential._dvdy;

                const float3 dwodx = -( ray.getDirDx() + _rec->_wo );
                const float3 dwody = -( ray.getDirDy() + _rec->_wo );
                const float dDNdx = dot( dwodx, _rec->_shading._n ) + 
                    dot( _rec->_wo, dndx );
                const float dDNdy = dot( dwody, _rec->_shading._n ) + 
                    dot( _rec->_wo, dndy );
                const float mu = eta * dot( w, _rec->_shading._n ) - 
                    dot( _rec->_wi, _rec->_shading._n );
                const float dmudx =
                    ( eta - ( eta * eta * dot( w, _rec->_shading._n ) ) / 
                      dot( _rec->_wi, _rec->_shading._n ) ) * dDNdx;
                const float dmudy =
                    ( eta - ( eta * eta * dot( w, _rec->_shading._n ) ) / 
                      dot( _rec->_wi, _rec->_shading._n ) ) * dDNdy;

                const float3 dir_dx = _rec->_wi + eta * dwodx - 
                    ( mu * dndx + dmudx *_rec->_shading._n );
                const float3 dir_dy = _rec->_wi + eta * dwody - 
                    ( mu * dndy + dmudy *_rec->_shading._n );
                ray = CudaRay( origin, dir, origin_dx, origin_dy, dir_dx, dir_dy );
            }
            ray.scaleDifferentials( .25f );
        } else {
            ray = dot( _rec->_wi, _rec->_geometry._n ) >= 0.0f ?
                CudaRay( _rec->_geometry._p + _constant_spec._reflection_bias * 
                         _rec->_geometry._n, _rec->_wi ) :
                CudaRay( _rec->_geometry._p - _constant_spec._reflection_bias * 
                         _rec->_geometry._n, _rec->_wi );
        }
    }

    __device__
        CudaRay spawnShadowRay( const float3& dir, float tmax = NOOR_INF ) const {
        return dot( dir, _rec->_geometry._n ) >= 0.0f ?
            CudaRay( _rec->_geometry._p + _constant_spec._reflection_bias * 
                     _rec->_geometry._n, dir, tmax ) :
            CudaRay( _rec->_geometry._p - _constant_spec._reflection_bias * 
                     _rec->_geometry._n, dir, tmax );
    }

    __device__
        CudaRay spawnShadowRay( const CudaVisibility& v ) const {
        const float3 p = dot( v._wi, _rec->_geometry._n ) >= 0.0f ?
            v._from + _constant_spec._reflection_bias * _rec->_geometry._n :
            v._from - _constant_spec._reflection_bias * _rec->_geometry._n;
        const float dist = length( p - v._to );
        //dist -= dist * _constant_spec._shadow_bias;
        return CudaRay( p, v._wi, dist );
    }
    __device__
        int getInsIdx() const {
        return _rec->_ins_idx;
    }
    __device__
        const float3& getWo() const {
        return _rec->_wo;
    }
    __device__
        float3& getWi() {
        return _rec->_wi;
    }
    __device__
        const float3& getP() const {
        return _rec->_geometry._p;
    }
    __device__
        const float3& getGn() const {
        return _rec->_geometry._n;
    }
    __device__
        const float3& getSn() const {
        return _rec->_shading._n;
    }
    __device__
        const float3& getSdpdu() const {
        return _rec->_shading._dpdu;
    }
    __device__
        const float3& getSdpdv() const {
        return _rec->_shading._dpdv;
    }
    __device__
        ShadingFrame& getShadingFrame() {
        return _rec->_shading;
    }
    __device__
        uint getTid() const {
        return _rec->_tid;
    }
    __device__
        void setEta( float eta ) const {
        _eta = eta;
    }
    __device__
        void setMaterialType( MaterialType material_type ) {
        _rec->_material_type = material_type;
    }
    __device__
        MaterialType getMaterialType() const {
        return (MaterialType)( _rec->_material_type & NOOR_NO_BUMP_ALPHA );
    }
    __device__
        bool isBumped() const {
        return ( _rec->_material_type & BUMP );
    }
    __device__
        bool isEmitter() const {
        return ( _rec->_material_type & NOOR_EMITTER );
    }
    __device__
        bool isGlass() const {
        return ( _rec->_material_type & NOOR_GLASS );
    }
    __device__
        bool isMeshLight() const {
        return ( _rec->_material_type & MESHLIGHT );
    }
    __device__
        bool isShadowCatcher() const {
        return ( _rec->_material_type & SHADOWCATCHER );
    }
    __device__
        bool isSpecular() const {
        return ( _rec->_material_type & NOOR_SPECULAR );
    }
    __device__
        bool isTransparent() const {
        return ( _rec->_material_type & NOOR_TRANSPARENT );
    }
    __device__
        bool isGlossy() const {
        return ( _rec->_material_type & NOOR_GLOSSY );
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
        _rec->_geometry._p = ray.pointAtParameter( ray.getTmax() );
        _rec->_wo = normalize( -1.0f*ray.getDir() );
        computeTangentSpace( ray );
    }

    __device__
        void computeTangentSpace( const CudaRay& ray ) {
        const uint4 attr_idx = _mesh_manager.getAttrIndex( _rec->_tri_idx );
        _mat_idx = attr_idx.w;
        if ( _rec->_material_type & ( MESHLIGHT | EMITTER ) ) return;
        const float2 uv0 = getUV( attr_idx.x );
        const float2 uv1 = getUV( attr_idx.y );
        const float2 uv2 = getUV( attr_idx.z );

        const float3 v0 = getVertex( attr_idx.x );
        const float3 v1 = getVertex( attr_idx.y );
        const float3 v2 = getVertex( attr_idx.z );

        const float3 n0 = getVertexNormal( attr_idx.x );
        const float3 n1 = getVertexNormal( attr_idx.y );
        const float3 n2 = getVertexNormal( attr_idx.z );

        const CudaTransform T = 
            _transform_manager.getNormalTransformation( _rec->_ins_idx );
        const float w = 1.0f - _rec->_bc.x - _rec->_bc.y;
        _uv = w*uv0 + _rec->_bc.x*uv1 + _rec->_bc.y*uv2;
        _rec->_shading._n = 
            NOOR::normalize( T.transformNormal( w*n0 + _rec->_bc.x*n1 + 
                             _rec->_bc.y*n2 ) );

        const CudaTransform S = 
            _transform_manager.getObjectToWorldTransformation( _rec->_ins_idx );
        const float3 dp01 = S.transformVector( v1 - v0 );
        const float3 dp02 = S.transformVector( v2 - v0 );
        _rec->_geometry._n = NOOR::normalize( cross( dp01, dp02 ) );

        const float2 duv01 = uv1 - uv0;
        const float2 duv02 = uv2 - uv0;
        NOOR::Matrix2x2 A{ duv01, duv02 };
        const float determinant = A.determinant();
        const bool degenerateUV = fabsf( determinant ) < 1e-8;
        float3 dpdu, dpdv;
        if ( !degenerateUV ) {
            A.inverse();
            dpdu = ( A[0][0] * dp01 + A[0][1] * dp02 );
            dpdv = ( A[1][0] * dp01 + A[1][1] * dp02 );
            if ( NOOR::length2( cross( dpdu, dpdv ) ) == 0.0f ) {
                NOOR::coordinateSystem( _rec->_geometry._n, dpdu, dpdv );
            }
        } else {
            NOOR::coordinateSystem( _rec->_geometry._n, dpdu, dpdv );
        }
        const float3& n = _rec->_shading._n;
        _rec->_shading._dpdu = dpdu;
        _rec->_shading._dpdv = dpdv;

        // Gram-Schmidt
        _rec->_shading._dpdu = 
            NOOR::normalize( _rec->_shading._dpdu - 
                                                n * dot( n, _rec->_shading._dpdu ) );
        const float sign = dot( cross( n, _rec->_shading._dpdu ), 
                                _rec->_shading._dpdv ) < 0.0f ? -1.0f : 1.0f;
        _rec->_shading._dpdv = 
            NOOR::normalize( sign*cross( n, _rec->_shading._dpdu ) );

        if ( !ray.isDifferential() ) return;
        // Compute deltas for triangle partial derivatives of normal
        const float3 dn01 = n1 - n0;
        const float3 dn02 = n2 - n0;
        if ( !degenerateUV ) {
            _differential._dndu = ( A[0][0] * dn01 + A[0][1] * dn02 );
            _differential._dndv = ( A[1][0] * dn01 + A[1][1] * dn02 );
        } else {
            const float3 dn = 
                NOOR::normalize( T.transformNormal( cross( dn01, dn02 ) ) );
            if ( NOOR::length2( dn ) == 0 ) {
                _differential._dndu = _differential._dndv = make_float3( 0.0f );
            } else {
                NOOR::coordinateSystem( dn, _differential._dndu, _differential._dndv );
            }
        }
        computeDifferential( ray, dpdu, dpdv );
    }

    __device__
        void computeDifferential( const CudaRay& ray, 
                                  const float3& _dpdu, 
                                  const float3& _dpdv ) {
        if ( !ray.isDifferential() )  return; 
        const float d = dot( _rec->_geometry._n, _rec->_geometry._p );
        const float tx = -( dot( _rec->_geometry._n, ray.getOriginDx() ) - d ) / 
            dot( _rec->_geometry._n, ray.getDirDx() );
        if ( isinf( tx ) || isnan( tx ) ) return; 
        const float3 px = ray.getOriginDx() + tx * ray.getDirDx();

        const float ty = -( dot( _rec->_geometry._n, ray.getOriginDy() ) - d ) / 
            dot( _rec->_geometry._n, ray.getDirDy() );
        if ( isinf( ty ) || isnan( ty ) ) return; 
        const float3 py = ray.getOriginDy() + ty * ray.getDirDy();

        _differential._dpdx = px - _rec->_geometry._p;
        _differential._dpdy = py - _rec->_geometry._p;

        int2 dim;
        if ( fabsf( _rec->_geometry._n.x ) > fabsf( _rec->_geometry._n.y ) && 
             fabsf( _rec->_geometry._n.x ) > fabsf( _rec->_geometry._n.z ) ) {
            dim.x = 1;
            dim.y = 2;
        } else if ( fabsf( _rec->_geometry._n.y ) > fabsf( _rec->_geometry._n.z ) ) {
            dim.x = 0;
            dim.y = 2;
        } else {
            dim.x = 0;
            dim.y = 1;
        }
        const NOOR::noor_float3 dpdu( _dpdu );
        const NOOR::noor_float3 dpdv( _dpdv );

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