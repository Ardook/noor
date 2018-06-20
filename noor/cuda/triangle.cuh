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
#ifndef CUDATRIANGLE_CUH
#define CUDATRIANGLE_CUH

class CudaTriangle {
public:
    __device__
        uint axis()  const {
        return _ax_nu_nv_nd.x;
    }
    __device__
        float n_u() const {
        return __uint_as_float( _ax_nu_nv_nd.y );
    }
    __device__
        float n_v() const {
        return __uint_as_float( _ax_nu_nv_nd.z );
    }
    __device__
        float n_d() const {
        return __uint_as_float( _ax_nu_nv_nd.w );
    }
    __device__
        float b_nu() const {
        return __uint_as_float( _bnu_bnv_au_av.x );
    }
    __device__
        float b_nv() const {
        return __uint_as_float( _bnu_bnv_au_av.y );
    }
    __device__
        float a_u()	const {
        return __uint_as_float( _bnu_bnv_au_av.z );
    }
    __device__
        float a_v()	const {
        return __uint_as_float( _bnu_bnv_au_av.w );
    }
    __device__
        float c_nu() const {
        return __uint_as_float( _cnu_cnv_deg_mat.x );
    }
    __device__
        float c_nv() const {
        return __uint_as_float( _cnu_cnv_deg_mat.y );
    }
    __device__
        bool isDegenerate() const {
        return  _cnu_cnv_deg_mat.z == 1;
    }
    __device__
        bool notOpaque() const {
        return  _cnu_cnv_deg_mat.w & ALPHA;
    }
    __device__
        MaterialType materialType() const {
        return  (MaterialType)_cnu_cnv_deg_mat.w;
    }

    __device__
        bool intersect(
        const CudaRay& ray,
        CudaIntersection& I,
        uint tri_idx
        ) {
        _ax_nu_nv_nd = _mesh_manager.getWaldAxNuNvNd( tri_idx );
        _cnu_cnv_deg_mat = _mesh_manager.getWaldCnuCnvPadPad( tri_idx );
        if ( isDegenerate() ) { return false; }
        const NOOR::noor_uint4 modulo{ make_uint4( 1u,2u,0u,1u ) };
        const uint k = axis();
        const uint ku = modulo[k + 0u];
        const uint kv = modulo[k + 1u];

        const float ro_u = ray.getOrigin( ku );
        const float ro_v = ray.getOrigin( kv );
        const float ro_k = ray.getOrigin( k );

        const float rd_u = ray.getDir( ku );
        const float rd_v = ray.getDir( kv );
        const float rd_k = ray.getDir( k );

        const float nd = n_d();
        const float nu = n_u();
        const float nv = n_v();

        const float denom = rd_k + nu*rd_u + nv*rd_v;
        if ( denom == 0 ) return false;
        const float t = ( nd - ro_k - nu*ro_u - nv*ro_v ) / denom;

        if ( t < 0.0f || t > ray.getTmax() ) {
            return false;
        }

        _bnu_bnv_au_av = _mesh_manager.getWaldBnuBnvAuAv( tri_idx );

        const float hu = ro_u + t*rd_u - a_u();
        const float hv = ro_v + t*rd_v - a_v();

        const float u = hv*b_nu() + hu*b_nv();
        const float v = hu*c_nu() + hv*c_nv();
        bool hit = ( u >= 0.0f && v >= 0.0f && u + v <= 1.0f );
        if ( hit ) {
            if ( notOpaque() ) {
                const uint4 attr_idx = _mesh_manager.getAttrIndex( tri_idx );
                const float2 uv0 = _mesh_manager.getUV( attr_idx.x );
                const float2 uv1 = _mesh_manager.getUV( attr_idx.y );
                const float2 uv2 = _mesh_manager.getUV( attr_idx.z );
                const float2 uv = ( 1.0f - u - v )*uv0 + u*uv1 + v*uv2;
                const float alpha = _material_manager.getAlpha( uv, attr_idx.w );
                if ( alpha <= 0.0001f ) {
                    return false;
                }
            }
            I._u = u;
            I._v = v;
            I._tri_idx = tri_idx;
            I.setMaterialType( materialType() );
            ray.setTmax( t );
        }
        return hit;
    }

    __device__
        bool intersect(
        const CudaRay& ray,
        uint tri_idx
        ) {
        _ax_nu_nv_nd = _mesh_manager.getWaldAxNuNvNd( tri_idx );
        _cnu_cnv_deg_mat = _mesh_manager.getWaldCnuCnvPadPad( tri_idx );
        if ( isDegenerate() ) return false;
        const NOOR::noor_uint4 modulo{ make_uint4( 1u,2u,0u,1u ) };
        const uint k = axis();
        const uint ku = modulo[k + 0u];
        const uint kv = modulo[k + 1u];

        const float ro_u = ray.getOrigin( ku );
        const float ro_v = ray.getOrigin( kv );
        const float ro_k = ray.getOrigin( k );

        const float rd_u = ray.getDir( ku );
        const float rd_v = ray.getDir( kv );
        const float rd_k = ray.getDir( k );

        const float nd = n_d();
        const float nu = n_u();
        const float nv = n_v();

        const float denom = rd_k + nu*rd_u + nv*rd_v;
        if ( denom == 0 )
            return false;
        const float t = ( nd - ro_k - nu*ro_u - nv*ro_v ) / denom;

        if ( t < 0.0f || t > ray.getTmax() ) {
            return false;
        }

        _bnu_bnv_au_av = _mesh_manager.getWaldBnuBnvAuAv( tri_idx );

        const float hu = ro_u + t*rd_u - a_u();
        const float hv = ro_v + t*rd_v - a_v();

        const float u = hv*b_nu() + hu*b_nv();
        const float v = hu*c_nu() + hv*c_nv();
        bool hit = ( u >= 0.0f && v >= 0.0f && u + v <= 1.0f );
        if ( hit ) {
            if ( notOpaque() ) {
                const uint4 attr_idx = _mesh_manager.getAttrIndex( tri_idx );
                const float2 uv0 = _mesh_manager.getUV( attr_idx.x );
                const float2 uv1 = _mesh_manager.getUV( attr_idx.y );
                const float2 uv2 = _mesh_manager.getUV( attr_idx.z );
                const float2 uv = ( 1.0f - u - v )*uv0 + u*uv1 + v*uv2;
                const float alpha = _material_manager.getAlpha( uv, attr_idx.w );
                if ( alpha <= 0.0001f ) {
                    return false;
                }
            }
        }
        return hit;
    }
    uint4 _ax_nu_nv_nd, _bnu_bnv_au_av, _cnu_cnv_deg_mat;
};

#endif /* CUDATRIANGLE_CUH */