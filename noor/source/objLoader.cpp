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
#include "objLoader.h"

void ObjLoader::load( const std::string& filename ) {
    _m = glmReadOBJ( const_cast<char*>( filename.c_str() ) );
    if ( !_m ) {
        std::string msg = "File " + std::string( __FILE__ ) + " LINE " + std::to_string( __LINE__ ) + "\n";
        msg += "Unable to read/open model file " + filename;
        throw std::runtime_error( msg );
    }
    glm::vec3 scene_min, scene_max;
    glmUnitize( _m, 1.0f, &scene_min[0], &scene_max[0] );
    glmFacetNormals( _m );
    glmVertexNormals( _m, 30.0f );

    GLMgroup* g = _m->groups;
    _material_groups.reserve( getNumMaterialGroups() );
    int current = 0;
    while ( g ) {
        _material_groups.emplace_back( std::string( g->name ), g->numtriangles, g->material );
        _material_groups[current]._triangle_indices.reserve( g->numtriangles );
        std::copy(
            &g->triangles[0]
            , &g->triangles[g->numtriangles]
            , back_inserter( _material_groups[current]._triangle_indices ) );

        ++current;
        g = g->next;
    }
}

void ObjLoader::getLights( std::vector<CudaAreaLight>& area_light_data
                           , std::vector<CudaPointLight>& point_light_data
                           , std::vector<CudaSpotLight>& spot_light_data
                           , std::vector<CudaDistantLight>& distant_light_data
                           , float world_radius ) const {
    for ( int grp_idx = 0; grp_idx < getNumMaterialGroups(); ++grp_idx ) {
        const MaterialGroup& mg = getMaterialGroup( grp_idx );
        const std::string token( "light" );
        if ( NOOR::contains( mg._name, token ) ) {
            glm::uint32 tri_idx_0 = mg._triangle_indices[0];
            glm::uint32 mat_idx = mg._material_index;
            const glm::vec3 Ke = glm::vec3( getMaterialEmittance( mat_idx ) );

            const glm::vec3 base = getVertex( tri_idx_0, 1 );
            const glm::vec3 v1 = getVertex( tri_idx_0, 2 );
            const glm::vec3 v2 = getVertex( tri_idx_0, 0 );
            const glm::vec3 u = v1 - base;
            const glm::vec3 v = v2 - base;
            const glm::vec3 n = glm::normalize( glm::cross( u, v ) );

            CudaShape shape( V2F3( base ), V2F3( u ), V2F3( v ), V2F3( n ), QUAD );
            area_light_data.emplace_back( shape, V2F3( Ke ) );
        }
    }
}

void ObjLoader::getMeshes( std::vector<Mesh>& meshes ) const {
    meshes.reserve( 1 );
    meshes.emplace_back( 0, getNumTriangles(), glm::mat4( 1.0f ) );
}

void ObjLoader::getVertices( std::vector<glm::vec4>& vertices, std::vector<glm::uvec3>& vindices ) const {
    const glm::uint32 num_vertices = getNumVertices();
    vertices.reserve( num_vertices );

    for ( glm::uint32 i = 1; i <= num_vertices; ++i ) {
        vertices.emplace_back( glm::make_vec3( &_m->vertices[3 * i] ), 1 );
    }
    const glm::uint32 num_triangles = getNumTriangles();
    vindices.reserve( num_triangles );

    glm::uint32 v0, v1, v2;

    for ( glm::uint32 i = 0; i < num_triangles; ++i ) {
        // Vertex indices
        v0 = _m->triangles[i].vindices[0] - 1;
        v1 = _m->triangles[i].vindices[1] - 1;
        v2 = _m->triangles[i].vindices[2] - 1;
        vindices.emplace_back( v0, v1, v2 );
    }
}

void ObjLoader::getNormals( std::vector<glm::vec4>& normals, std::vector<glm::uvec3>& nindices ) const {
    glm::uint32 num_normals = getNumNormals();
    normals.reserve( num_normals );
    for ( glm::uint32 i = 1; i <= num_normals; ++i ) {
        normals.emplace_back( glm::make_vec3( &_m->normals[3 * i] ), 0 );
    }

    glm::uint32 num_triangles = getNumTriangles();
    nindices.reserve( num_triangles );

    glm::uint32 n0, n1, n2;
    for ( glm::uint32 i = 0; i < num_triangles; ++i ) {
        // Normal indices
        n0 = _m->triangles[i].nindices[0] - 1;
        n1 = _m->triangles[i].nindices[1] - 1;
        n2 = _m->triangles[i].nindices[2] - 1;
        nindices.emplace_back( n0, n1, n2 );
    }
}

void ObjLoader::getUVs( std::vector<glm::vec2>& uvs, std::vector<glm::uvec3>& uvindices ) const {
    glm::uint32 num_uvs = getNumUVs();
    uvs.reserve( num_uvs );

    // start at index 0 to include the default tex coordinate (0,0)
    for ( glm::uint32 i = 0; i < num_uvs; ++i ) {
        uvs.emplace_back( _m->texcoords[2 * i], _m->texcoords[2 * i + 1] );
    }

    glm::uint32 num_triangles = getNumTriangles();
    uvindices.reserve( num_triangles );

    glm::uint32 t0, t1, t2;
    for ( glm::uint32 i = 0; i < num_triangles; ++i ) {
        // UV indices
        t0 = _m->triangles[i].tindices[0];
        t1 = _m->triangles[i].tindices[1];
        t2 = _m->triangles[i].tindices[2];
        uvindices.emplace_back( t0, t1, t2 );
    }
}
