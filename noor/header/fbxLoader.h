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
#ifndef FBXLOADER_H
#define FBXLOADER_H
#include <fbxsdk.h>
#include "assetLoader.h"
#include <set>
class FBXLoader : public AssetLoader {
    FbxManager* _fbx_manager{ nullptr };
    FbxScene* _fbx_scene{ nullptr };
    FbxCamera* _fbx_camera{ nullptr };
    std::vector<FbxMesh*> _fbx_meshes;
    std::vector<FbxLight*> _fbx_lights;

    std::vector<CudaAreaLight> _area_lights;
    std::vector<CudaPointLight> _point_lights;
    std::vector<CudaDistantLight> _distant_lights;
    std::vector<CudaSpotLight> _spot_lights;

    std::vector<Mesh> _meshes;
    std::unordered_map<FbxMesh*, std::list<FbxNode*>> _mesh2nodes;

    glm::uint32 _num_meshes = 0;
    glm::uint32 _num_triangles = 0;
    glm::uint32 _num_triangles_ass = 0;
    glm::uint32 _num_vertices = 0;
    glm::uint32 _num_normals = 0;
    glm::uint32 _num_binormals = 0;
    glm::uint32 _num_tangents = 0;
    glm::uint32 _num_uvs = 0;

    int _num_materials = 0;
    int _num_cameras = 0;

    mutable std::unordered_set<glm::uint32> _tangent_lookup;
    mutable std::unordered_set<glm::uint32> _normal_lookup;
    mutable std::unordered_set<glm::uint32> _binormal_lookup;
    mutable std::unordered_set<glm::uint32> _uv_lookup;

    bool initialize();
    bool loadScene( const std::string& filename );
    void processSceneGraph( FbxNode* node );
    void readNormal( const FbxMesh* mesh
                     , glm::uint32 vertexIndex
                     , glm::uint32 vertexCounter
                     , glm::uint32& normalIndex
                     , glm::uint32 meshIndex
                     , std::vector<glm::vec4>& normal ) const;
    void readBinormal( const FbxMesh* mesh
                       , glm::uint32 vertexIndex
                       , glm::uint32 vertexCounter
                       , glm::uint32& binormalIndex
                       , glm::uint32 meshIndex
                       , std::vector<glm::vec4>& binormal ) const;
    void readTangent( const FbxMesh* mesh
                      , glm::uint32 vertexIndex
                      , glm::uint32 vertexCounter
                      , glm::uint32& tangentIndex
                      , glm::uint32 meshIndex
                      , std::vector<glm::vec4>& tangent ) const;
    void readUV( const FbxMesh* mesh
                 , glm::uint32 vertexIndex
                 , int inTextureUVIndex
                 , glm::uint32& uvIndex
                 , glm::uint32 meshIndex
                 , std::vector<glm::vec2>& outUV ) const;

    void processMaterialGroups();

    void getLights();
    void getAreaLight( const FbxLight* light, std::vector<CudaAreaLight>& area_lights ) const;
    void getPointLight( const FbxLight* light, std::vector<CudaPointLight>& point_lights ) const;
    void getDistantLight( const FbxLight* light, std::vector<CudaDistantLight>& distant_light ) const;
    void getSpotLight( const FbxLight* light, std::vector<CudaSpotLight>& spot_lights ) const;

    std::string getMaterialName( int mat_index ) const;
    glm::uint32 getNumTriangles( const FbxMesh* mesh ) const;
    glm::uint32 getNumVertices( const FbxMesh* mesh ) const;
    glm::uint32 getNumNormals( const FbxMesh* mesh )  const;
    glm::uint32 getNumTangents( const FbxMesh* mesh )  const;
    glm::uint32 getNumBinormals( const FbxMesh* mesh )  const;
    glm::uint32 getNumUVs( const FbxMesh* mesh ) const;

    glm::vec4 fbx2glm( const FbxVector4& v ) const;
    glm::mat4 getTransformation( const FbxNode* node ) const;
    glm::mat4 getTransformation( const FbxMesh* mesh ) const;
    glm::mat4 getTransformation( const FbxLight* light ) const;
    glm::vec4 fbx2glm_vector( const FbxVector4& v ) const {
        glm::vec4 result;
        result.x = static_cast<float>( v[0] );
        result.y = static_cast<float>( v[1] );
        result.z = static_cast<float>( v[2] );
        result.w = static_cast<float>( 0.0f );
        return result;
    }

    glm::vec4 fbx2glm_point( const FbxVector4& v ) const {
        glm::vec4 result;
        result.x = static_cast<float>( v[0] );
        result.y = static_cast<float>( v[1] );
        result.z = static_cast<float>( v[2] );
        result.w = static_cast<float>( 1.0f );
        return result;
    }

    glm::vec2 fbx2glm_vector( const FbxVector2& v ) const {
        glm::vec2 result;
        result.x = static_cast<float>( v[0] );
        result.y = static_cast<float>( v[1] );
        return result;
    }

    glm::vec3 fbx2glm_vector( const FbxDouble3& v ) const {
        glm::vec3 result;
        result.x = static_cast<float>( v[0] );
        result.y = static_cast<float>( v[1] );
        result.z = static_cast<float>( v[2] );
        return result;
    }

public:
    FBXLoader() = default;
    FBXLoader( const std::string& filename ) { load( filename ); }
    ~FBXLoader() override {
        if ( _fbx_scene )
            _fbx_scene->Destroy();
        if ( _fbx_manager )
            _fbx_manager->Destroy();
    }
    void load( const std::string& filename ) override;
    glm::uint32 getNumMeshes() const override { return _num_meshes; }
    glm::uint32 getNumTriangles() const override { return _num_triangles; }
    glm::uint32 getNumVertices() const override { return _num_vertices; }
    glm::uint32 getNumNormals() const override { return _num_normals; }
    glm::uint32 getNumBinormals() const override { return _num_binormals; }
    glm::uint32 getNumTangents() const override { return _num_tangents; }
    glm::uint32 getNumUVs() const override { return _num_uvs; }
    int getNumMaterials() const override { return _num_materials; }
    int getNumCameras() const { return _num_cameras; }

    void getMeshes( std::vector<Mesh>& meshes ) const override;
    void getVertices( std::vector<glm::vec4>& vertices, std::vector<glm::uvec3>& vindices ) const override;
    void getNormals( std::vector<glm::vec4>& normals, std::vector<glm::uvec3>& nindices ) const override;
    void getUVs( std::vector<glm::vec2>& uvs, std::vector<glm::uvec3>& uvindices ) const override;
    void getTangents( std::vector<glm::vec4>& tangents, std::vector<glm::uvec3>& tindices ) const override;
    void getBinormals( std::vector<glm::vec4>& binormals, std::vector<glm::uvec3>& bindices ) const override;
    void getCamera( glm::vec3& eye, glm::vec3& lookAt, glm::vec3& up, float& fov, float& lens_radius, float& focal_length, float& orthozoom ) const override;
    void getLights( std::vector<CudaAreaLight>& area_lights
                    , std::vector<CudaPointLight>& point_lights
                    , std::vector<CudaSpotLight>& spot_lights
                    , std::vector<CudaDistantLight>& distant_lights ) const override;

    glm::vec3 getMaterialDiffuse( int mat_idx ) const override;
    glm::vec3 getMaterialSpecular( int mat_idx ) const override;
    glm::vec3 getMaterialEmittance( int mat_idx ) const override;
    glm::vec3 getMaterialTransmission( int mat_idx ) const override;
    glm::vec3 getMaterialIor( int mat_idx ) const;
    glm::vec3 getMaterialK( int mat_idx ) const;
    glm::vec3 getMaterialTransparency( int mat_idx ) const override;
    glm::vec2 getMaterialRoughness( int mat_idx ) const override;

    float getMaterialCoatingRoughness( int mat_idx ) const override;
    float getMaterialCoatingWeight( int mat_idx ) const override;
    float getMaterialCoatingThickness( int mat_idx ) const override;
    float getMaterialCoatingSigma( int mat_idx ) const override;
    float getMaterialCoatingIOR( int mat_idx ) const override;
    float getMaterialEmitterScale( int mat_idx ) const override;

    MaterialType getMaterialType( int mat_idx ) const override;

    std::string getMaterial_map_roughness( int mat_idx, glm::vec2& uvscale ) const override;
    std::string getMaterial_map_metalness( int mat_idx, glm::vec2& uvscale ) const override;
    std::string getMaterial_map_kd( int mat_idx, glm::vec2& uvscale ) const override;
    std::string getMaterial_map_ks( int mat_idx, glm::vec2& uvscale ) const override;
    std::string getMaterial_map_ka( int mat_idx, glm::vec2& uvscale ) const override;
    std::string getMaterial_map_ke( int mat_idx, glm::vec2& uvscale ) const override;
    std::string getMaterial_map_d( int mat_idx, glm::vec2& uvscale ) const override;
    std::string getMaterial_map_bump( int mat_idx, glm::vec2& uvscale, float& bumpfactor ) const override;
};

#endif /* FBXLOADER_H*/