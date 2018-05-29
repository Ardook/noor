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
#ifndef OBJLOADER_H
#define OBJLOADER_H
#include <glm.h>
#include "assetLoader.h"

class ObjLoader : public AssetLoader {
    GLMmodel* _m;
    glm::vec3 getVertex( glm::uint32 tri_idx, glm::uint8 v ) const {
        glm::uint32 vertex_idx = 3u * _m->triangles[tri_idx].vindices[v];
        return glm::make_vec3( &_m->vertices[vertex_idx] );
    }
public:
    ObjLoader() = default;
    ObjLoader( const std::string& filename ) { load( filename ); }
    ~ObjLoader() override { if ( _m ) glmDelete( _m ); }

    void load( const std::string& filename ) override;

    glm::uint32 getNumMeshes() const override { return 1; }
    glm::uint32 getNumTriangles() const override { return _m->numtriangles; }
    glm::uint32 getNumVertices() const override { return _m->numvertices; }
    glm::uint32 getNumNormals()  const override { return _m->numnormals; }
    glm::uint32 getNumTangents()  const override { return 0; }
    glm::uint32 getNumBinormals()  const override { return 0; }
    glm::uint32 getNumUVs() const override { return _m->numtexcoords; }

    int getNumMaterials() const override { return _m->nummaterials; }
    int getNumMaterialGroups() const override { return _m->numgroups; }
    int getNumCameras() const override { return 0; }

    void getMeshes( std::vector<Mesh>& meshes ) const override;
    void getVertices( std::vector<glm::vec4>& vertices, std::vector<glm::uvec3>& vindices ) const override;
    void getNormals( std::vector<glm::vec4>& normals, std::vector<glm::uvec3>& nindices ) const override;
    void getUVs( std::vector<glm::vec2>& uvs, std::vector<glm::uvec3>& uvindices ) const override;
    void getTangents( std::vector<glm::vec4>& tangents, std::vector<glm::uvec3>& tindices ) const override { return; }
    void getBinormals( std::vector<glm::vec4>& binormals, std::vector<glm::uvec3>& bindices ) const override { return; }
    void getLights( std::vector<CudaAreaLight>& area_light_data
                    , std::vector<CudaPointLight>& point_light_data
                    , std::vector<CudaSpotLight>& spot_light_data
                    , std::vector<CudaDistantLight>& distant_light_data
                    , float world_radius = 0 ) const override;

    void getCamera( glm::vec3& eye, glm::vec3& lookAt, glm::vec3& up, float& fov, float& lens_radius, float& focal_length, float& orthozoom ) const override {
        return;
    }
    glm::vec3 getMaterialDiffuse( int mat_idx ) const override {
        return glm::make_vec3( _m->materials[mat_idx].diffuse );
    }

    glm::vec3 getMaterialSpecular( int mat_idx ) const override {
        return glm::make_vec3( _m->materials[mat_idx].specular );
    }

    glm::vec3 getMaterialEmittance( int mat_idx ) const override {
        return glm::make_vec3( _m->materials[mat_idx].emmissive );
    }

    glm::vec3 getMaterialTransmission( int mat_idx ) const override {
        return glm::make_vec3( _m->materials[mat_idx].transmittance );
    }

    MaterialType getMaterialType( int mat_idx ) const override {
        if ( _m->materials[mat_idx].mat_type == 3 ) // mirror
            return MIRROR;
        else if ( _m->materials[mat_idx].mat_type == 1 ) // glossy
            return GLOSSY;
        else if ( _m->materials[mat_idx].mat_type == 7 ) // glass
            return GLASS;
        else if ( _m->materials[mat_idx].mat_type == 5 ) // light
            return EMITTER;
        else if ( _m->materials[mat_idx].mat_type == 6 ) // metal
            return METAL;
        else // 2,4,0 are diffuse
            return DIFFUSE;
    }

    glm::vec3 getMaterialTransparency( int mat_idx ) const override {
        return glm::vec3( _m->materials[mat_idx].transparency );
    }

    glm::vec2 getMaterialRoughness( int mat_idx ) const override {
        return glm::vec2( 1.0f / _m->materials[mat_idx].shininess );
    }

    glm::vec3 getMaterialIor( int mat_idx ) const override {
        return glm::vec3( _m->materials[mat_idx].refraction );
    }
    glm::vec3 getMaterialK( int mat_idx ) const override {
        return glm::vec3( 1.0f );
    }

    std::string getMaterial_map_roughness( int mat_idx, glm::vec2& uvscale ) const override {
        return "";
    }
    std::string getMaterial_map_metalness( int mat_idx, glm::vec2& uvscale ) const override {
        return "";
    }

    std::string getMaterial_map_kd( int mat_idx, glm::vec2& uvscale ) const override {
        uvscale = glm::vec2( _m->materials[mat_idx].uvscale[0], _m->materials[mat_idx].uvscale[1] );
        if ( _m->materials[mat_idx].map_Kd )
            return std::string( _m->materials[mat_idx].map_Kd );
        else
            return "";
    }

    std::string getMaterial_map_ks( int mat_idx, glm::vec2& uvscale ) const override {
        uvscale = glm::vec2( 1.0f );
        if ( _m->materials[mat_idx].map_Ks )
            return std::string( _m->materials[mat_idx].map_Ks );
        else
            return "";
    }

    std::string getMaterial_map_ka( int mat_idx, glm::vec2& uvscale ) const override {
        uvscale = glm::vec2( 1.0f );
        if ( _m->materials[mat_idx].map_Ka )
            return std::string( _m->materials[mat_idx].map_Ka );
        else
            return "";
    }

    std::string getMaterial_map_ke( int mat_idx, glm::vec2& uvscale ) const override {
        uvscale = glm::vec2( 1.0f );
        if ( _m->materials[mat_idx].map_Ke )
            return std::string( _m->materials[mat_idx].map_Ke );
        else
            return "";
    }

    std::string getMaterial_map_d( int mat_idx, glm::vec2& uvscale ) const override {
        uvscale = glm::vec2( 1.0f );
        if ( _m->materials[mat_idx].map_d )
            return std::string( _m->materials[mat_idx].map_d );
        else
            return "";
    }

    std::string getMaterial_map_bump( int mat_idx, glm::vec2& uvscale, float& bumpfactor ) const override {
        uvscale = glm::vec2( 1.0f );
        bumpfactor = _m->materials[mat_idx].bumpfactor;
        if ( _m->materials[mat_idx].map_bump )
            return std::string( _m->materials[mat_idx].map_bump );
        else
            return "";
    }
};
#endif /* OBJLOADER_H */