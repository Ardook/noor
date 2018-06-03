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
#ifndef CUDAMESH_CUH
#define CUDAMESH_CUH

struct CudaMeshManager {
    // per vertex attributes
    float4* _vertices;
    float4* _normals;
    float2* _uvs;

    // per vertex attribute indices
    // x:v0, y:v1, z:v2
    // v0, v1, v2 CCW
    uint4* _attributes;

    // Wald triangle
    uint4* _wald_ax_nu_nv_nd;
    uint4* _wald_bnu_bnv_au_av;
    uint4* _wald_cnu_cnv_pad_pad;

    // bvh nodes
    uint4* _bvh_min_right_start;
    uint4* _bvh_max_axis_count;

    cudaTextureObject_t _vertices_texobj;
    cudaTextureObject_t _normals_texobj;
    cudaTextureObject_t _uvs_texobj;

    cudaTextureObject_t _face_normals_texobj;
    cudaTextureObject_t _face_tangents_texobj;
    cudaTextureObject_t _face_bitangents_texobj;

    cudaTextureObject_t _attributes_texobj;

    cudaTextureObject_t _wald_ax_nu_nv_nd_texobj;
    cudaTextureObject_t _wald_bnu_bnv_au_av_texobj;
    cudaTextureObject_t _wald_cnu_cnv_pad_pad_texobj;
    cudaTextureObject_t _bvh_min_right_start_texobj;
    cudaTextureObject_t _bvh_max_axis_count_texobj;

    CudaMeshManager() = default;
    __host__
        CudaMeshManager( const CudaPayload* payload ) {
        checkNoorErrors(NOOR::malloc(  &_vertices, payload->_n_size_bytes ));
        checkNoorErrors(NOOR::malloc(  &_normals, payload->_n_size_bytes ));
        checkNoorErrors(NOOR::malloc(  &_uvs, payload->_uv_size_bytes ));

        checkNoorErrors(NOOR::malloc(  &_attributes, payload->_ai_size_bytes ));

        checkNoorErrors(NOOR::malloc(  &_wald_ax_nu_nv_nd, payload->_wald_size_bytes ));
        checkNoorErrors(NOOR::malloc(  &_wald_bnu_bnv_au_av, payload->_wald_size_bytes ));
        checkNoorErrors(NOOR::malloc(  &_wald_cnu_cnv_pad_pad, payload->_wald_size_bytes ));
        checkNoorErrors(NOOR::malloc(  &_bvh_min_right_start, payload->_bvh_size_bytes ));
        checkNoorErrors(NOOR::malloc(  &_bvh_max_axis_count, payload->_bvh_size_bytes ));

        // copy
        checkNoorErrors(NOOR::memcopy( _vertices, (void*) &payload->_vertices[0], payload->_n_size_bytes ));
        checkNoorErrors(NOOR::memcopy( _normals, (void*) &payload->_normals[0], payload->_n_size_bytes ));
        checkNoorErrors(NOOR::memcopy( _uvs, (void*) &payload->_uvs[0], payload->_uv_size_bytes ));

        checkNoorErrors(NOOR::memcopy( _attributes, (void*) &payload->_attribute_indices[0], payload->_ai_size_bytes ));

        checkNoorErrors(NOOR::memcopy( _wald_ax_nu_nv_nd, (void*) &payload->_wald_ax_nu_nv_nd[0], payload->_wald_size_bytes ));
        checkNoorErrors(NOOR::memcopy( _wald_bnu_bnv_au_av, (void*) &payload->_wald_bnu_bnv_au_av[0], payload->_wald_size_bytes ));
        checkNoorErrors(NOOR::memcopy( _wald_cnu_cnv_pad_pad, (void*) &payload->_wald_cnu_cnv_deg_mat[0], payload->_wald_size_bytes ));
        checkNoorErrors(NOOR::memcopy( _bvh_min_right_start, (void*) &payload->_bvh_min_right_start_nodes[0], payload->_bvh_size_bytes ));
        checkNoorErrors(NOOR::memcopy( _bvh_max_axis_count, (void*) &payload->_bvh_max_axis_count_nodes[0], payload->_bvh_size_bytes ));

        // create texture object
        checkNoorErrors(NOOR::create_1d_texobj( &_vertices_texobj, _vertices, payload->_n_size_bytes, NOOR::_float4_channelDesc ));
        checkNoorErrors(NOOR::create_1d_texobj( &_normals_texobj, _normals, payload->_n_size_bytes, NOOR::_float4_channelDesc ));
        checkNoorErrors(NOOR::create_1d_texobj( &_uvs_texobj, _uvs, payload->_uv_size_bytes, NOOR::_float2_channelDesc ));

        checkNoorErrors(NOOR::create_1d_texobj( &_attributes_texobj, _attributes, payload->_ai_size_bytes, NOOR::_uint4_channelDesc ));

        checkNoorErrors(NOOR::create_1d_texobj( &_wald_ax_nu_nv_nd_texobj, _wald_ax_nu_nv_nd, payload->_wald_size_bytes, NOOR::_uint4_channelDesc ));
        checkNoorErrors(NOOR::create_1d_texobj( &_wald_bnu_bnv_au_av_texobj, _wald_bnu_bnv_au_av, payload->_wald_size_bytes, NOOR::_uint4_channelDesc ));
        checkNoorErrors(NOOR::create_1d_texobj( &_wald_cnu_cnv_pad_pad_texobj, _wald_cnu_cnv_pad_pad, payload->_wald_size_bytes, NOOR::_uint4_channelDesc ));
        checkNoorErrors(NOOR::create_1d_texobj( &_bvh_min_right_start_texobj, _bvh_min_right_start, payload->_bvh_size_bytes, NOOR::_uint4_channelDesc ));
        checkNoorErrors(NOOR::create_1d_texobj( &_bvh_max_axis_count_texobj, _bvh_max_axis_count, payload->_bvh_size_bytes, NOOR::_uint4_channelDesc ));
    }

    __host__
        void free() {
        checkNoorErrors( cudaDestroyTextureObject( _vertices_texobj ));
        checkNoorErrors( cudaDestroyTextureObject( _normals_texobj ) );
        checkNoorErrors( cudaDestroyTextureObject( _uvs_texobj ) );

        checkNoorErrors( cudaDestroyTextureObject( _attributes_texobj ) );

        checkNoorErrors( cudaDestroyTextureObject( _wald_ax_nu_nv_nd_texobj ) );
        checkNoorErrors( cudaDestroyTextureObject( _wald_bnu_bnv_au_av_texobj ) );
        checkNoorErrors( cudaDestroyTextureObject( _wald_cnu_cnv_pad_pad_texobj ) );
        checkNoorErrors( cudaDestroyTextureObject( _bvh_min_right_start_texobj ) );
        checkNoorErrors( cudaDestroyTextureObject( _bvh_max_axis_count_texobj ) );

        checkNoorErrors(cudaFree( _vertices ));
        checkNoorErrors(cudaFree( _normals ));
        checkNoorErrors(cudaFree( _uvs ));

        checkNoorErrors(cudaFree( _attributes ));

        checkNoorErrors(cudaFree( _wald_ax_nu_nv_nd ));
        checkNoorErrors(cudaFree( _wald_bnu_bnv_au_av ));
        checkNoorErrors(cudaFree( _wald_cnu_cnv_pad_pad ));
        checkNoorErrors(cudaFree( _bvh_min_right_start ));
        checkNoorErrors(cudaFree( _bvh_max_axis_count ));
    }

    __device__
        uint4 getAttrIndex( uint tri_idx )const {
        return tex1Dfetch<uint4>( _attributes_texobj, tri_idx );
    }
    __device__
        float4 getVertex( uint attr_idx )const {
        return tex1Dfetch<float4>( _vertices_texobj, attr_idx );
    }
    __device__
        float4 getVertexNormal( uint attr_idx )const {
        return tex1Dfetch<float4>( _normals_texobj, attr_idx );
    }
    __device__
        float2 getUV( const uint attr_idx ) const {
        const float2 uv = tex1Dfetch<float2>( _uvs_texobj, attr_idx );
        return make_float2( uv.x, 1.0f - uv.y );
    }
    __device__
        uint4 getBvhMaxAxisCount( uint node_idx ) const {
        return tex1Dfetch<uint4>( _bvh_max_axis_count_texobj, node_idx );
    }
    __device__
        uint4 getBvhMinRightStart( uint node_idx ) const {
        return tex1Dfetch<uint4>( _bvh_min_right_start_texobj, node_idx );
    }
    __device__
        uint4 getWaldAxNuNvNd( uint tri_idx ) const {
        return tex1Dfetch<uint4>( _wald_ax_nu_nv_nd_texobj, tri_idx );
    }
    __device__
        uint4 getWaldCnuCnvPadPad( uint tri_idx ) const {
        return tex1Dfetch<uint4>( _wald_cnu_cnv_pad_pad_texobj, tri_idx );
    }
    __device__
        uint4 getWaldBnuBnvAuAv( uint tri_idx ) const {
        return tex1Dfetch<uint4>( _wald_bnu_bnv_au_av_texobj, tri_idx );
    }
};

__constant__
CudaMeshManager _mesh_manager;
#endif /* CUDAMESH_CUH */