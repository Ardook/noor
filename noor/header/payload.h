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

#ifndef PAYLOAD_H
#define PAYLOAD_H
class CudaPayload {
public:
    glm::uint32 _num_triangles;

    // per vertex attribute indices
    // x:v0, y:v1, z:v2
    // (v0 -> v1 -> v2) CCW
    std::vector<glm::uvec4>  _attribute_indices;

    // per vertex attributes
    std::vector<glm::vec4>  _vertices;
    std::vector<glm::vec4>  _normals;
    std::vector<glm::vec2>  _uvs;
    std::vector<glm::vec4>  _colors;

    size_t _ai_size_bytes;
    size_t _n_size_bytes;
    size_t _fn_size_bytes;
    size_t _uv_size_bytes;
    size_t _ti_size_bytes;

    // normal transformations per mesh/object/instance
    std::vector<glm::vec4> _normal_row0;
    std::vector<glm::vec4> _normal_row1;
    std::vector<glm::vec4> _normal_row2;

    // world to object transformations per mesh/object/instance
    std::vector<glm::vec4> _world_to_object_row0;
    std::vector<glm::vec4> _world_to_object_row1;
    std::vector<glm::vec4> _world_to_object_row2;

    // object to World transformations per mesh/object/instance
    std::vector<glm::vec4> _object_to_world_row0;
    std::vector<glm::vec4> _object_to_world_row1;
    std::vector<glm::vec4> _object_to_world_row2;
    size_t _transforms_size_bytes;

    // Wald triangles
    std::vector<glm::uvec4>  _wald_ax_nu_nv_nd;
    std::vector<glm::uvec4>  _wald_bnu_bnv_au_av;
    std::vector<glm::uvec4>  _wald_cnu_cnv_deg_mat;
    size_t _wald_size_bytes;

    // BVH
    glm::uint32 _bvh_root_node;
    std::vector<glm::uvec4> _bvh_min_right_start_nodes;
    std::vector<glm::uvec4> _bvh_max_axis_count_nodes;
    size_t _bvh_size_bytes;

    // textures
    std::vector<glm::uvec4> _texture_indices;
    size_t _txi_size_bytes;

    std::vector<ImageTexture> _textures;
    //ImageTexture _env_texture;

    std::vector<glm::uint32> _element_array;
    // area lights
    std::vector<CudaAreaLight> _area_light_data;
    std::vector<CudaShape> _shapes;
    int _num_area_lights;

    // point lights
    std::vector<CudaPointLight> _point_light_data;
    int _num_point_lights;

    // spot lights
    std::vector<CudaSpotLight> _spot_light_data;
    int _num_spot_lights;

    // distant lights 
    std::vector<CudaDistantLight> _distant_light_data;
    int _num_distant_lights;

    // infinite light 
    std::vector<CudaInfiniteLight> _infinite_light_data;
    int _num_infinite_lights;

    int _num_lights{ 0 };
    // cuda materials
    std::vector<CudaMaterial>  _materials;
    size_t _mat_size_bytes;

    glm::vec3 getVertex( glm::uint32 t, glm::uint8 v ) const {
        glm::uint32 vi = _element_array[3 * t + v];
        return glm::vec3( _vertices[vi] );
    }

    glm::vec3 getNormal( glm::uint32 t, glm::uint8 v ) const {
        glm::uint32 ni = _element_array[3 * t + v];
        return glm::vec3( _normals[ni] );
    }

    glm::vec2 getUV( glm::uint32 t, glm::uint8 v ) const {
        glm::uint32 uvi = _element_array[3 * t + v];
        return  _uvs[uvi];
    }

    CudaPayload() = default;
};
#endif /* PAYLOAD_H */