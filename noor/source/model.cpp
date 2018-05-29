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
#include "pch.h"
#include "model.h"
#include "bvh.h"
#include "triangle.h"
#include "objLoader.h"
#include "fbxLoader.h"
using V_N_UV_Lookup =
std::unordered_map<glm::uint32
    , std::unordered_map<glm::uint32
    , std::unordered_map<glm::uint32, glm::uint32> > >;

static inline bool isDuplicate(
    V_N_UV_Lookup& U
    , glm::uint32 vi
    , glm::uint32 ni
    , glm::uint32 uvi
) {
    if ( U.find( vi ) != U.end() &&
         U[vi].find( ni ) != U[vi].end() &&
         U[vi][ni].find( uvi ) != U[vi][ni].end() )
        return true;
    else
        return false;
}

Model::Model( const Spec& spec, Stat& stat ) :
    _spec( spec )
    , _model_file( spec._model_spec._model_filename )
    , _texture_dir( "" )
    , _loader( loaderFactory( _model_file ) )
    , _stat( stat ) {
    load();
}

// create loader based on the file type/extension (obj, FBX, and etc.)
std::unique_ptr<AssetLoader> Model::loaderFactory( const std::string& filename ) {
    const std::experimental::filesystem::path p( filename );
    const std::string file_extension( ( p.extension() ).string() );
    if ( file_extension == ".obj" ) {
        _texture_dir = std::string( p.parent_path().string() + R"(\)" );
        return std::make_unique<ObjLoader>( filename );
    } else if ( file_extension == ".fbx" ) {
        _texture_dir = "";
        return std::make_unique<FBXLoader>( filename );
    } else return nullptr;
}

void Model::computeBBoxes() {
    _tri_bboxes.reserve( _num_triangles );
    for ( auto & mesh : _meshes ) {
        if ( !mesh.isInstance() ) {
            const glm::uint32 start = mesh._start_count.first;
            const glm::uint32 count = start + mesh._start_count.second;
            BBox mesh_bbox;
            BBox mesh_centbox;
            for ( glm::uint32 tri_idx = start; tri_idx < count; ++tri_idx ) {
                const glm::vec4 v0 = getVertex( tri_idx, 0 );
                const glm::vec4 v1 = getVertex( tri_idx, 1 );
                const glm::vec4 v2 = getVertex( tri_idx, 2 );

                // triangle bbox used in BVH build
                BBox tri_bbox = BBox( v0, v1, v2 );
                _tri_bboxes.emplace_back( tri_bbox );

                mesh_bbox.merge( tri_bbox );
                mesh_centbox.include( tri_bbox.centroid() );
            }
            mesh._bbox = mesh_bbox;
            mesh._centbox = mesh_centbox;
        }
    }
    for ( auto & mesh : _meshes ) {
        if ( mesh.isInstance() ) {
            mesh._bbox = _meshes[mesh._instance]._bbox;
            mesh._centbox = _meshes[mesh._instance]._centbox;
        }
        mesh._instance_bbox = mesh._to_global * mesh._bbox;
        mesh._instance_centbox = mesh._to_global * mesh._centbox;

        _scene_bbox.merge( mesh._instance_bbox );
        _scene_centbox.include( mesh._instance_bbox.centroid() );
    }
}

void Model::loadCamera() {
    float lens_radius, focal_length;
    if ( _loader->getNumCameras() != 0 )
        _loader->getCamera( _eye, _lookAt, _up, _fov, lens_radius, focal_length, _orthozoom );
    else {
        _lookAt = _scene_bbox.centroid();
        _eye = _lookAt + glm::vec3( 0.f, 0.f, 3.0f*_scene_bbox.radius() );
        _up = glm::vec3( 0.f, 1.f, 0.f );
        _fov = NOOR_PI_over_4;
        _orthozoom = _scene_bbox.radius();
    }
}
void Model::loadLights() {
    const float world_radius = _scene_bbox.radius();
    _loader->getLights( _area_lights,
                        _point_lights,
                        _spot_lights,
                        _distant_lights,
                        world_radius );
}
void Model::load() {
    if ( _loader == nullptr ) {
        std::string msg = "File " + std::string( __FILE__ ) + " LINE " + std::to_string( __LINE__ ) + "\n";
        msg += "Error: loader hasn't been initialized";
        std::cerr << msg << std::endl;
        exit( EXIT_FAILURE );
    }
    printf( "loading model\n" );
    Timer timer;
    _num_triangles = _loader->getNumTriangles();
    _num_vertices = _loader->getNumVertices();
    _num_normals = _loader->getNumNormals();
    _num_uvs = _loader->getNumUVs();
    _num_materials = _loader->getNumMaterials();
    _num_groups = _loader->getNumMaterialGroups();
    _num_meshes = _loader->getNumMeshes();

    _loader->getMeshes( _meshes );
    _loader->getVertices( _vertices, _vindices );
    _loader->getNormals( _normals, _nindices );
    _loader->getUVs( _uvs, _uvindices );

    computeBBoxes();
    loadMaterial();
    loadLights();
    loadCamera();
    _loader.reset( nullptr );
    _stat._process_model_duration = timer.elapsed();
    // end Model load

    // start BVH build
    printf( "building bvh\n" );
    timer.start();
    _bvh = std::make_unique<BVH>( *this, _spec._bvh_spec, _stat );
    _stat._process_bvh_duration = timer.elapsed();
    // end BVH build

    _stat._num_area_lights = static_cast<glm::uint32>( _area_lights.size() );
    _stat._num_point_lights = static_cast<glm::uint32>( _point_lights.size() );
    _stat._num_spot_lights = static_cast<glm::uint32>( _spot_lights.size() );
    _stat._num_distant_lights = static_cast<glm::uint32>( _distant_lights.size() );
    _stat._num_infinite_lights = 1;
    _stat._num_triangles = _num_triangles;
    _stat._num_meshes = _num_meshes;
    _stat._num_vertices = _num_vertices;
    _stat._num_normals = _num_normals;
    _stat._num_tex_coordinates = _num_uvs;
    _stat._num_materials = _num_materials;
    _stat._num_material_groups = _num_groups;
}

void Model::loadMaterial() {
    CudaMaterial mt;
    _materials.reserve( _num_materials + 1 );
    if ( _num_materials == 0 ) {
        _materials.push_back( mt );
    }
    _textures.emplace_back( make_float3( .5f ) );
    std::map<std::string, int> name2index;
    for ( unsigned int mat_idx = 0; mat_idx < _num_materials; ++mat_idx ) {
        const float3 diffuse = V2F3( _loader->getMaterialDiffuse( mat_idx ) );
        const float3 specular = V2F3( _loader->getMaterialSpecular( mat_idx ) );
        const float3 transmission = V2F3( _loader->getMaterialTransmission( mat_idx ) );
        const float3 emittance = V2F3( _loader->getMaterialEmittance( mat_idx ) );

        const float3 transparency = V2F3( _loader->getMaterialTransparency( mat_idx ) );
        const float3 ior = V2F3( _loader->getMaterialIor( mat_idx ) );
        const float3 k = V2F3( _loader->getMaterialK( mat_idx ) );
        const float2 roughness = V2F2( _loader->getMaterialRoughness( mat_idx ) );

        mt._coat_weight = _loader->getMaterialCoatingWeight( mat_idx );
        mt._coat_roughness = _loader->getMaterialCoatingRoughness( mat_idx );
        mt._coat_thickness = _loader->getMaterialCoatingThickness( mat_idx );
        mt._coat_sigma = _loader->getMaterialCoatingSigma( mat_idx );
        mt._coat_ior = _loader->getMaterialCoatingIOR( mat_idx );

        mt._type = _loader->getMaterialType( mat_idx );
        _textures.emplace_back( transmission );
        mt._transmission_tex_idx = static_cast<int>( _textures.size() - 1 );
        _textures.emplace_back( ior );
        mt._ior_tex_idx = static_cast<int>( _textures.size() - 1 );
        _textures.emplace_back( k );
        mt._k_tex_idx = static_cast<int>( _textures.size() - 1 );

        // diffuse
        glm::vec2 uvscale( 1.0f );
        std::string texfile = _loader->getMaterial_map_kd( mat_idx, uvscale );
        if ( !texfile.empty() ) {
            if ( name2index.find( texfile ) == name2index.end() ) {
                _textures.emplace_back( _texture_dir + texfile );
                name2index[texfile] = static_cast<int>( _textures.size() - 1 );
            }
            mt._diffuse_tex_idx = name2index[texfile];
            mt._diffuse_uvscale = V2F2( uvscale );
        } else {
            _textures.emplace_back( diffuse );
            mt._diffuse_tex_idx = static_cast<int>( _textures.size() - 1 );
        }
        // roughness
        texfile = _loader->getMaterial_map_roughness( mat_idx, uvscale );
        if ( !texfile.empty() ) {
            if ( name2index.find( texfile ) == name2index.end() ) {
                _textures.emplace_back( _texture_dir + texfile );
                name2index[texfile] = static_cast<int>( _textures.size() - 1 );
            }
            mt._roughness_tex_idx = name2index[texfile];
            mt._roughness_uvscale = V2F2( uvscale );
        } else {
            float3 r = make_float3( roughness.x, roughness.y, 1.f );
            _textures.emplace_back( r );
            mt._roughness_tex_idx = static_cast<int>( _textures.size() - 1 );
        }
        // metalness
        texfile = _loader->getMaterial_map_metalness( mat_idx, uvscale );
        if ( !texfile.empty() ) {
            if ( name2index.find( texfile ) == name2index.end() ) {
                _textures.emplace_back( _texture_dir + texfile );
                name2index[texfile] = static_cast<int>( _textures.size() - 1 );
            }
            mt._metalness_tex_idx = name2index[texfile];
            mt._metalness_uvscale = V2F2( uvscale );
        } else {
            float metalness = 1.0f;
            _textures.emplace_back( metalness );
            mt._metalness_tex_idx = static_cast<int>( _textures.size() - 1 );
        }
        // specular
        texfile = _loader->getMaterial_map_ks( mat_idx, uvscale );
        if ( !texfile.empty() ) {
            if ( name2index.find( texfile ) == name2index.end() ) {
                _textures.emplace_back( _texture_dir + texfile );
                name2index[texfile] = static_cast<int>( _textures.size() - 1 );
            }
            mt._specular_tex_idx = name2index[texfile];
            mt._specular_uvscale = V2F2( uvscale );
        } else {
            _textures.emplace_back( specular );
            mt._specular_tex_idx = static_cast<int>( _textures.size() - 1 );
        }
        // emittance
        texfile = _loader->getMaterial_map_ke( mat_idx, uvscale );
        if ( !texfile.empty() ) {
            if ( name2index.find( texfile ) == name2index.end() ) {
                _textures.emplace_back( _texture_dir + texfile );
                name2index[texfile] = static_cast<int>( _textures.size() - 1 );
            }
            mt._emittance_tex_idx = name2index[texfile];
            mt._emittance_uvscale = V2F2( uvscale );
        } else {
            _textures.emplace_back( emittance );
            mt._emittance_tex_idx = static_cast<int>( _textures.size() - 1 );
        }
        // transparency
        texfile = _loader->getMaterial_map_d( mat_idx, uvscale );
        if ( !texfile.empty() ) {
            if ( name2index.find( texfile ) == name2index.end() ) {
                _textures.emplace_back( _texture_dir + texfile, cudaFilterModePoint );
                name2index[texfile] = static_cast<int>( _textures.size() - 1 );
            }
            mt._transparency_tex_idx = name2index[texfile];
            mt._transparency_uvscale = V2F2( uvscale );
            mt._type = MaterialType( ALPHA | mt._type );
        } else {
            _textures.emplace_back( transparency );
            mt._transparency_tex_idx = static_cast<int>( _textures.size() - 1 );
        }
        // bump
        texfile = _loader->getMaterial_map_bump( mat_idx, uvscale, mt._bumpfactor );
        if ( !texfile.empty() ) {
            if ( name2index.find( texfile ) == name2index.end() ) {
                _textures.emplace_back( _texture_dir + texfile );
                name2index[texfile] = (int) _textures.size() - 1;
            }
            mt._bump_tex_idx = name2index[texfile];
            mt._bump_uvscale = V2F2( uvscale );
            mt._type = MaterialType( BUMP | mt._type );
        }
        _materials.push_back( mt );
    }
    // environment map HDR
    if ( _spec._model_spec._skydome_type == 0 )
        _textures.emplace_back( _spec._model_spec._hdr_filename, cudaFilterModeLinear, cudaAddressModeClamp );
    else if ( _spec._model_spec._skydome_type == 1 )
        _textures.emplace_back( skydome_res.x, skydome_res.y, cudaFilterModeLinear, cudaAddressModeClamp );
    else
        _textures.emplace_back( SKYDOME_COLOR );

    _material_indices.resize( _num_triangles );
    for ( int grp_idx = 0; grp_idx < _loader->getNumMaterialGroups(); ++grp_idx ) {
        const MaterialGroup& mg = _loader->getMaterialGroup( grp_idx );
        for ( glm::uint32 j = 0; j < mg._num_triangles; j++ ) {
            _material_indices[mg._triangle_indices[j]] = mg._material_index;
        }
    }
}


void Model::buildWaldTriangles( std::unique_ptr<CudaPayload>& payload ) const {
    payload->_wald_ax_nu_nv_nd.resize( _num_triangles );
    payload->_wald_bnu_bnv_au_av.resize( _num_triangles );
    payload->_wald_cnu_cnv_deg_mat.resize( _num_triangles );
    payload->_wald_size_bytes = _num_triangles * sizeof( glm::uvec4 );
    payload->_attribute_indices.resize( _num_triangles );
    payload->_ai_size_bytes = _num_triangles * sizeof( glm::uvec4 );

    glm::uint32 index = 0;
    auto buildWaldTriangle = [&]( glm::uint32 tri_idx ) {
        const glm::vec3 v0 = payload->getVertex( index, 0 );
        const glm::vec3 v1 = payload->getVertex( index, 1 );
        const glm::vec3 v2 = payload->getVertex( index, 2 );
        WaldTriangle wt( v0, v1, v2 );

        const glm::uint16 mi = _material_indices[tri_idx];
        payload->_wald_ax_nu_nv_nd[index] = wt.get_ax_nu_nv_nd();
        payload->_wald_bnu_bnv_au_av[index] = wt.get_bnu_bnv_au_av();
        payload->_wald_cnu_cnv_deg_mat[index] = wt.get_cnu_cnv_deg_mat();
        payload->_wald_cnu_cnv_deg_mat[index].w = _materials[mi]._type;
        const glm::uint32 v0i = payload->_element_array[3 * index + 0];
        const glm::uint32 v1i = payload->_element_array[3 * index + 1];
        const glm::uint32 v2i = payload->_element_array[3 * index + 2];
        payload->_attribute_indices[index] = glm::uvec4( v0i, v1i, v2i, mi );
        ++index;
    };
    std::for_each( _triangle_indices.begin(), _triangle_indices.end(), buildWaldTriangle );
}

void Model::loadCudaPayload( std::unique_ptr<CudaPayload>& payload ) {
    _bvh->loadCudaPayload( payload );
    printf( "loading model payload\n" );
    Timer timer;
    payload->_num_triangles = _num_triangles;
    payload->_ti_size_bytes = _num_triangles * sizeof( glm::uint32 );

    const glm::uint32 element_array_count = 3u * _num_triangles;
    payload->_vertices.reserve( element_array_count );
    payload->_normals.reserve( element_array_count );
    payload->_uvs.reserve( element_array_count );
    payload->_colors.reserve( element_array_count );
    payload->_element_array.reserve( element_array_count );

    // lookup table used to decide whether to split a vertex
    V_N_UV_Lookup U;

    glm::uint32 vi, ni, ti, uvi;
    glm::uint32 index = 0u;
    for ( glm::uint32 i = 0; i < _num_triangles; ++i ) {
        glm::uint32 tri_idx = _triangle_indices[i];
        for ( glm::uint8 j = 0; j < 3; j++ ) {
            vi = _vindices[tri_idx][j];
            ni = _nindices[tri_idx][j];
            uvi = _uvindices[tri_idx][j];
            ti = 3 * tri_idx + j;
            // avoid vertex split/duplication
            if ( !isDuplicate( U, vi, ni, uvi ) ) {
                payload->_vertices.push_back( _vertices[vi] );
                payload->_normals.push_back( _normals[ni] );
                payload->_uvs.push_back( _uvs[uvi] );
                payload->_colors.push_back( glm::vec4( 1 ) );
                payload->_element_array.push_back( index );
                U[vi][ni][uvi] = index;
                ++index;
            } else {
                payload->_element_array.push_back( U[vi][ni][uvi] );
            }
        }
    }

    payload->_vertices.shrink_to_fit();
    payload->_normals.shrink_to_fit();
    payload->_uvs.shrink_to_fit();
    payload->_colors.shrink_to_fit();
    payload->_element_array.shrink_to_fit();

    payload->_n_size_bytes = payload->_normals.size() * sizeof( glm::vec4 );
    payload->_uv_size_bytes = payload->_uvs.size() * sizeof( glm::vec2 );

    buildWaldTriangles( payload );

    // load cuda materials
    payload->_mat_size_bytes = _materials.size() * sizeof( CudaMaterial );
    payload->_materials.reserve( _num_materials + 1 );
    payload->_materials = std::move( _materials );
    // plus 1: the sky dome texture is saved as the last texture
    payload->_textures.reserve( _textures.size() + 1 );
    payload->_textures = std::move( _textures );

    payload->_num_area_lights = static_cast<int> ( _area_lights.size() );
    if ( payload->_num_area_lights > 0 ) {
        payload->_num_lights += payload->_num_area_lights;
        payload->_area_light_data.reserve( payload->_num_area_lights );
        payload->_area_light_data = std::move( _area_lights );
        payload->_shapes.reserve( payload->_num_area_lights );
        payload->_shapes = std::move( _shapes );
    }

    payload->_num_point_lights = static_cast<int> ( _point_lights.size() );
    if ( payload->_num_point_lights > 0 ) {
        payload->_num_lights += payload->_num_point_lights;
        payload->_point_light_data.reserve( payload->_num_point_lights );
        payload->_point_light_data = std::move( _point_lights );
    }

    payload->_num_distant_lights = static_cast<int> ( _distant_lights.size() );
    if ( payload->_num_distant_lights > 0 ) {
        payload->_num_lights += payload->_num_distant_lights;
        payload->_distant_light_data.reserve( payload->_num_distant_lights );
        payload->_distant_light_data = std::move( _distant_lights );
    }

    payload->_num_spot_lights = static_cast<int> ( _spot_lights.size() );
    if ( payload->_num_spot_lights > 0 ) {
        payload->_num_lights += payload->_num_spot_lights;
        payload->_spot_light_data.reserve( payload->_num_spot_lights );
        payload->_spot_light_data = std::move( _spot_lights );
    }

    payload->_num_infinite_lights = 1;
    payload->_num_lights += payload->_num_infinite_lights;
    payload->_infinite_light_data.reserve( payload->_num_infinite_lights );
    payload->_infinite_light_data.push_back( CudaInfiniteLight( _scene_bbox.radius() ) );

    _stat._model_payload_duration = timer.elapsed();
}