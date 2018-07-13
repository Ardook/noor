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
#ifndef ASSET_LOADER_H
#define ASSET_LOADER_H
#include "path_tracer.cuh"

struct MaterialGroup {
    std::string _name;
    int _material_index;
    glm::uint32 _num_triangles;
    std::vector<glm::uint32> _triangle_indices;

    MaterialGroup() = default;
    MaterialGroup(
        const std::string& name
        , glm::uint32 num_triangles
        , int material_index
    ) :
        _name( name )
        , _material_index( material_index )
        , _num_triangles( num_triangles ) {}
};

struct Mesh {
    std::pair<glm::uint32, glm::uint32> _start_count{ std::make_pair( 0, 0 ) };
    BBox _instance_bbox;
    BBox _instance_centbox;

    BBox _bbox;
    BBox _centbox;

    glm::mat4 _to_global{ glm::mat4( 1.f ) };
    glm::mat4 _to_local{ glm::mat4( 1.f ) };
    glm::mat3 _normal_transform{ glm::mat4( 1.f ) };
    int _instance{ -1 };
    int _mat_idx{ 0 };
    int _light_idx{ -1 };
    bool _is_light{ false };

    Mesh() = default;
    Mesh(
        glm::uint32 start,
        glm::uint32 count,
        const glm::mat4& to_global,
        int instance = -1,
        int mat_idx = 0
    ) :
        _start_count( std::make_pair( start, count ) ),
        _to_global( to_global ),
        _to_local( glm::inverse( to_global ) ),
        _instance( instance ),
        _mat_idx( mat_idx ),
        _is_light( false ) {
        _normal_transform = glm::mat3( to_global );
        _normal_transform = glm::transpose( glm::inverse( _normal_transform ) );
    }
    Mesh( const CudaShape& shape, int light_idx ) :
        _light_idx( light_idx ),
        _is_light( true ) {
        float3 lmin, lmax;
        shape.getBounds( lmin, lmax );
        glm::vec3 gmin( lmin.x, lmin.y, lmin.z );
        glm::vec3 gmax( lmax.x, lmax.y, lmax.z );
        _instance_bbox.setMin( gmin ); _instance_bbox.setMax( gmax );
        _instance_centbox.setMin( _instance_bbox.centroid() );
        _instance_centbox.setMax( _instance_bbox.centroid() );

        _bbox = _instance_bbox;
        _centbox = _instance_centbox;
    }

    void setToGlobal( const glm::mat4& toGlobal ) {
        _to_global = toGlobal;
        _to_local = glm::inverse( _to_global );
        _normal_transform = glm::mat3( _to_global );
        _normal_transform = glm::transpose( glm::inverse( _normal_transform ) );
    }
    bool isInstance()const { return _instance != -1; }
    bool isLight()const { return _is_light; }
};

class AssetLoader {
protected:
    std::vector<MaterialGroup> _material_groups;
public:
    AssetLoader() = default;
    virtual ~AssetLoader() = default;

    virtual void load( const std::string& filename ) = 0;
    virtual glm::uint32 getNumMeshes() const = 0;
    virtual glm::uint32 getNumTriangles() const = 0;
    virtual glm::uint32 getNumVertices() const = 0;
    virtual glm::uint32 getNumNormals()  const = 0;
    virtual glm::uint32 getNumTangents()  const = 0;
    virtual glm::uint32 getNumBinormals()  const = 0;
    virtual glm::uint32 getNumUVs() const = 0;

    virtual void getCamera( glm::vec3& eye, glm::vec3& lookAt, glm::vec3& up, float& fov, float& lens_radius, float& focal_length, float& orthozoom ) const = 0;
    virtual void getMeshes( std::vector<Mesh>& meshes ) const = 0;
    virtual void getVertices( std::vector<glm::vec4>& vertices, std::vector<glm::uvec3>& vindices ) const = 0;
    virtual void getNormals( std::vector<glm::vec4>& normals, std::vector<glm::uvec3>& nindices ) const = 0;
    virtual void getTangents( std::vector<glm::vec4>& tangents, std::vector<glm::uvec3>& tindices ) const = 0;
    virtual void getBinormals( std::vector<glm::vec4>& binormals, std::vector<glm::uvec3>& bindices ) const = 0;
    virtual void getUVs( std::vector<glm::vec2>& uvs, std::vector<glm::uvec3>& uvindices ) const = 0;
    virtual void getLights( std::vector<CudaAreaLight>& area_light_data
                            , std::vector<CudaPointLight>& point_light_data
                            , std::vector<CudaSpotLight>& spot_light_data
                            , std::vector<CudaDistantLight>& distant_light_data ) const = 0;
    virtual int getNumMaterials() const = 0;
    virtual int getNumCameras() const = 0;
    virtual int getNumMaterialGroups() const { return static_cast<int>( _material_groups.size() ); }

    virtual glm::vec3 getMaterialDiffuse( int mat_idx ) const = 0;
    virtual glm::vec3 getMaterialSpecular( int mat_idx ) const = 0;
    virtual glm::vec3 getMaterialTransmission( int mat_idx ) const = 0;
    virtual glm::vec3 getMaterialEmittance( int mat_idx ) const = 0;
    virtual glm::vec3 getMaterialIor( int mat_idx ) const = 0;
    virtual glm::vec3 getMaterialK( int mat_idx ) const = 0;
    virtual glm::vec3 getMaterialTransparency( int mat_idx ) const = 0;
    virtual glm::vec2 getMaterialRoughness( int mat_idx ) const = 0;

    virtual float getMaterialCoatingRoughness( int mat_idx ) const { return 0.001f; }
    virtual float getMaterialCoatingWeight( int mat_idx ) const { return 1.0f; }
    virtual float getMaterialCoatingThickness( int mat_idx ) const { return 1.0f; }
    virtual float getMaterialCoatingSigma( int mat_idx ) const { return 0.0f; }
    virtual float getMaterialCoatingIOR( int mat_idx ) const { return 1.5f; }
    virtual float getMaterialEmitterScale( int mat_idx ) const { return 1.f; }

    virtual MaterialType getMaterialType( int mat_idx ) const = 0;

    virtual std::string getMaterial_map_roughness( int mat_idx, glm::vec2& uvscale ) const = 0;
    virtual std::string getMaterial_map_metalness( int mat_idx, glm::vec2& uvscale ) const = 0;
    virtual std::string getMaterial_map_kd( int mat_idx, glm::vec2& uvscale ) const = 0;
    virtual std::string getMaterial_map_ks( int mat_idx, glm::vec2& uvscale ) const = 0;
    virtual std::string getMaterial_map_ka( int mat_idx, glm::vec2& uvscale ) const = 0;
    virtual std::string getMaterial_map_ke( int mat_idx, glm::vec2& uvscale ) const = 0;
    virtual std::string getMaterial_map_d( int mat_idx, glm::vec2& uvscale ) const = 0;
    virtual std::string getMaterial_map_bump( int mat_idx, glm::vec2& uvscale, float& bumpfactor ) const = 0;

    const MaterialGroup& getMaterialGroup( int group_idx ) const { return _material_groups[group_idx]; }
};

#endif /* ASSET_LOADER_H */