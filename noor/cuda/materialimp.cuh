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
#ifndef CUDAMATERIALIMP_CUH
#define CUDAMATERIALIMP_CUH

class CudaMaterialManager {
    CudaMaterial* _materials;
    int _num_materials;
public:
    CudaMaterialManager() = default;
    __host__
        CudaMaterialManager( const CudaPayload* payload ) {
        _num_materials = (int) payload->_materials.size();
        NOOR::malloc( (void**) &_materials, payload->_mat_size_bytes );
        NOOR::memcopy( _materials, (void*) &payload->_materials[0], payload->_mat_size_bytes );
    }
    __host__
        void free() {
        cudaFree( _materials );
    }
    __device__
        const CudaMaterial* getMaterial( uint mat_idx ) const {
        return &_materials[mat_idx];
    }
    __device__
        uint getType( uint mat_idx ) const {
        const CudaMaterial* m = &_materials[mat_idx];
        return m->_type;
    }
    __device__
        const float2& getBumpUVScale( const CudaIntersection& I ) const {
        const CudaMaterial* m = &_materials[I._mat_idx];
        return m->_bump_uvscale;
    }
    __device__
        const float2& getRoughnessUVScale( const CudaIntersection& I ) const {
        const CudaMaterial* m = &_materials[I._mat_idx];
        return m->_roughness_uvscale;
    }
    __device__
        const float2& getDiffuseUVScale( const CudaIntersection& I ) const {
        const CudaMaterial* m = &_materials[I._mat_idx];
        return m->_diffuse_uvscale;
    }
    __device__
        const float2& getSpecularUVScale( const CudaIntersection& I ) const {
        const CudaMaterial* m = &_materials[I._mat_idx];
        return m->_specular_uvscale;
    }
    __device__
        const float2& getTransparencyUVScale( const CudaIntersection& I ) const {
        const CudaMaterial* m = &_materials[I._mat_idx];
        return m->_transparency_uvscale;
    }
    __device__
        const float2& getEmittanceUVScale( const CudaIntersection& I ) const {
        const CudaMaterial* m = &_materials[I._mat_idx];
        return m->_emittance_uvscale;
    }
    __device__
        float getBumpFactor( const CudaIntersection& I ) const {
        const CudaMaterial* m = &_materials[I._mat_idx];
        return m->_bumpfactor;
    }
    __device__
        float getCoatWeight( const CudaIntersection& I ) const {
        const CudaMaterial* m = &_materials[I._mat_idx];
        return m->_coat_weight;
    }
    __device__
        float getCoatRoughness( const CudaIntersection& I ) const {
        const CudaMaterial* m = &_materials[I._mat_idx];
        return m->_coat_roughness;
    }
    __device__
        float getCoatThickness( const CudaIntersection& I ) const {
        const CudaMaterial* m = &_materials[I._mat_idx];
        return m->_coat_thickness;
    }
    __device__
        float getCoatSigma( const CudaIntersection& I ) const {
        const CudaMaterial* m = &_materials[I._mat_idx];
        return m->_coat_sigma;
    }
    __device__
        float getCoatIOR( const CudaIntersection& I ) const {
        const CudaMaterial* m = &_materials[I._mat_idx];
        return m->_coat_ior;
    }
    __device__
        float getBump( const CudaIntersection& I, float du = 0, float dv = 0 ) const {
        const float2& uvscale = getBumpUVScale( I );
        const CudaTexture& tex = getBumpTexture( I._mat_idx );
        return	tex.evaluateGrad<float>( I, uvscale, du, dv );
    }
    __device__
        float3 getDiffuse( const CudaIntersection& I ) const {
        const float2& uvscale = getDiffuseUVScale( I );
        const CudaTexture& tex = getDiffuseTexture( I._mat_idx );
        return make_float3( tex.evaluateGrad<float4>( I, uvscale ) );
    }
    __device__
        float3 getEmmitance( const CudaIntersection& I ) const {
        const float2& uvscale = getEmittanceUVScale( I );
        const CudaTexture& tex = getEmmitanceTexture( I._mat_idx );
        return make_float3( tex.evaluateGrad<float4>( I, uvscale ) );
    }
    __device__
        float3 getSpecular( const CudaIntersection& I ) const {
        const float2& uvscale = getSpecularUVScale( I );
        const CudaTexture& tex = getSpecularTexture( I._mat_idx );
        return make_float3( tex.evaluateGrad<float4>( I, uvscale ) );
    }
    __device__
        float2 getRoughness( const CudaIntersection& I ) const {
        const float2& uvscale = getRoughnessUVScale( I );
        const CudaTexture& tex = getRoughnessTexture( I._mat_idx );
        const float4 roughness = tex.evaluate<float4>( I, uvscale );
        return make_float2( roughness.x, roughness.y );
    }
    __device__
        float getAlpha( const float2& uv, uint mat_idx ) const {
        const CudaMaterial* m = &_materials[mat_idx];
        const CudaTexture& tex = getTransparencyTexture( mat_idx );
        return	tex.evaluate<float4>( uv, m->_transparency_uvscale ).x;
    }
    __device__
        float3 getReflection( const CudaIntersection& I ) const {
        const float2& uvscale = make_float2( 1.0f );
        const CudaTexture& tex = getReflectionTexture( I._mat_idx );
        return make_float3( tex.evaluateGrad<float4>( I, uvscale ) );
    }
    __device__
        float3 getTransmission( const CudaIntersection& I ) const {
        const float2& uvscale = make_float2( 1.0f );
        const CudaTexture& tex = getTransmissionTexture( I._mat_idx );
        return make_float3( tex.evaluateGrad<float4>( I, uvscale ) );
    }
    __device__
        float3 getIor( const CudaIntersection& I ) const {
        const float2& uvscale = make_float2( 1.0f );
        const CudaTexture& tex = getIorTexture( I._mat_idx );
        const float3 ior = make_float3( tex.evaluateGrad<float4>( I, uvscale ) );
        I.setEta( ior.x );
        return ior;
    }
    __device__
        float3 getK( const CudaIntersection& I ) const {
        const float2& uvscale = make_float2( 1.0f );
        const CudaTexture& tex = getConductorKTexture( I._mat_idx );
        return	make_float3( tex.evaluateGrad<float4>( I, uvscale ) );
    }
private:
    __device__
        const CudaTexture& getBumpTexture( uint mat_idx ) const {
        const CudaMaterial* m = &_materials[mat_idx];
        return _texture_manager.getTexture( m->_bump_tex_idx );
    }
    __device__
        const CudaTexture& getDiffuseTexture( uint mat_idx ) const {
        const CudaMaterial* m = &_materials[mat_idx];
        return _texture_manager.getTexture( m->_diffuse_tex_idx );
    }
    __device__
        const CudaTexture& getEmmitanceTexture( uint mat_idx ) const {
        const CudaMaterial* m = &_materials[mat_idx];
        return _texture_manager.getTexture( m->_emittance_tex_idx );
    }
    __device__
        const CudaTexture& getSpecularTexture( uint mat_idx ) const {
        const CudaMaterial* m = &_materials[mat_idx];
        return _texture_manager.getTexture( m->_specular_tex_idx );
    }
    __device__
        const CudaTexture& getReflectionTexture( uint mat_idx ) const {
        const CudaMaterial* m = &_materials[mat_idx];
        return _texture_manager.getTexture( m->_reflection_tex_idx );
    }
    __device__
        const CudaTexture& getRoughnessTexture( uint mat_idx ) const {
        const CudaMaterial* m = &_materials[mat_idx];
        return _texture_manager.getTexture( m->_roughness_tex_idx );
    }
    __device__
        const CudaTexture& getTransparencyTexture( uint mat_idx ) const {
        const CudaMaterial* m = &_materials[mat_idx];
        return _texture_manager.getTexture( m->_transparency_tex_idx );
    }
    __device__
        const CudaTexture& getTransmissionTexture( uint mat_idx ) const {
        const CudaMaterial* m = &_materials[mat_idx];
        return _texture_manager.getTexture( m->_transmission_tex_idx );
    }
    __device__
        const CudaTexture& getIorTexture( uint mat_idx ) const {
        const CudaMaterial* m = &_materials[mat_idx];
        return _texture_manager.getTexture( m->_ior_tex_idx );
    }
    __device__
        const CudaTexture& getConductorKTexture( uint mat_idx ) const {
        const CudaMaterial* m = &_materials[mat_idx];
        return _texture_manager.getTexture( m->_k_tex_idx );
    }
};
__constant__
CudaMaterialManager _material_manager;

#endif /* CUDAMATERIAL_CU */