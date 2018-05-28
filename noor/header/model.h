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

#ifndef MODEL_H
#define MODEL_H

#include "assetLoader.h"
class CudaAreaLight;
class Stat;

class Model {
    friend class BVH;
    std::string _model_file;
    std::string _texture_dir;
    std::unique_ptr<AssetLoader> _loader;

    glm::uint32 _num_triangles = 0;
    glm::uint32 _num_meshes = 0;
    glm::uint32 _num_materials = 0;
    glm::uint32 _num_vertices = 0;
    glm::uint32 _num_normals = 0;
    glm::uint32 _num_uvs = 0;
    glm::uint32 _num_groups = 0;
    glm::uint32 _num_area_lights = 0;
    glm::uint32 _num_point_lights = 0;
    glm::uint32 _num_distant_lights = 0;
    glm::uint32 _num_spot_lights = 0;
    glm::uint32 _num_cameras = 0;

    std::vector<glm::uint32> _triangle_indices;
    std::vector<glm::uint32> _mesh_indices;
    std::vector<glm::uvec3>  _vindices;
    std::vector<glm::uvec3>  _nindices;
    std::vector<glm::uvec3>  _uvindices;

    std::vector<glm::vec4>  _vertices;
    std::vector<glm::vec4>  _normals;
    std::vector<glm::vec2>  _uvs;

    std::vector<glm::vec4>  _face_normals;
    std::vector<glm::vec4>  _face_tangents;
    std::vector<glm::vec4>  _face_bitangents;

    std::vector<glm::uint16> _material_indices;
    std::vector<glm::uint16> _map_Ka_indices;
    std::vector<glm::uint16> _map_Ke_indices;
    std::vector<glm::uint16> _map_Kd_indices;
    std::vector<glm::uint16> _map_Ks_indices;
    std::vector<glm::uint16> _map_d_indices;
    std::vector<glm::uint16> _map_bump_indices;


    std::vector<ImageTexture> _textures;
    std::vector<CudaMaterial> _materials;
    std::vector<Mesh> _meshes;
    std::vector<BBox> _tri_bboxes;
    BBox _scene_bbox;
    BBox _scene_centbox;
    Stat& _stat;
    const Spec& _spec;
    std::unique_ptr<BVH> _bvh;
public:
    glm::vec3 _eye;
    glm::vec3 _lookAt;
    glm::vec3 _up;
    float _fov;
    float _orthozoom;
    std::vector<CudaAreaLight> _area_lights;
    std::vector<CudaPointLight> _point_lights;
    std::vector<CudaDistantLight> _distant_lights;
    std::vector<CudaSpotLight> _spot_lights;
    std::vector<CudaShape> _shapes;

    Model( const Spec& spec, Stat& stat );
    ~Model() = default;

    void load();
    const BBox& getSceneBBox() const { return _scene_bbox; }
    const BBox& getSceneCentBox() const { return _scene_centbox; }
    const BBox& getMeshBBox( glm::uint32 mesh_idx ) const { return _meshes[mesh_idx]._bbox; }
    const BBox& getMeshCentBox( glm::uint32 mesh_idx ) const { return _meshes[mesh_idx]._centbox; }
    const BBox& getInstanceMeshBBox( glm::uint32 mesh_idx ) const { return _meshes[mesh_idx]._instance_bbox; }
    const BBox& getInstanceMeshCentBox( glm::uint32 mesh_idx ) const { return _meshes[mesh_idx]._instance_centbox; }
    const BBox& getTriBBox( glm::uint32 tri_idx ) const { return _tri_bboxes[tri_idx]; }
    bool isMeshLight( glm::uint32 mesh_idx ) const { return _meshes[mesh_idx].isLight(); }
    glm::uint32 getNumTriangles() const { return _num_triangles; }
    glm::uint32 getNumMeshes() const { return _num_meshes; }
    void buildWaldTriangles( std::unique_ptr<CudaPayload>& payload ) const;
    void loadCudaPayload( std::unique_ptr<CudaPayload>& payload );
private:
    void computeBBoxes();
    void loadMaterial();
    void loadCamera();
    void loadLights();
    std::unique_ptr<AssetLoader> loaderFactory( const std::string& filename );
    const glm::vec4& getVertex( glm::uint32 t, glm::uint8 v ) const {
        glm::uint32 vi = _vindices[t][v];
        return  _vertices[vi];
    }

    const glm::vec4& getNormal( glm::uint32 t, glm::uint8 v ) const {
        glm::uint32 ni = _nindices[t][v];
        return _normals[ni];
    }

    const glm::vec2& getUV( glm::uint32 t, glm::uint8 v ) const {
        glm::uint32 uvi = _uvindices[t][v];
        return _uvs[uvi];
    }

    glm::vec4 computeFaceNormal(
        const glm::vec3& v0
        , const glm::vec3& v1
        , const glm::vec3& v2
    ) const {
        return glm::vec4( glm::normalize( glm::cross( v1 - v0, v2 - v0 ) ), 0 );
    }

};



#endif /* MODEL_H */
