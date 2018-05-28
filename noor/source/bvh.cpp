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
#include "splitter.h"
#include "model.h"
#include "bvh.h"

BVH::BVH(
    Model& model
    , const BVHSpec& spec
    , Stat& stat
) :
    _model( model )
    , _spec( spec )
    , _stat( stat )
    , _triangle_indices( model._triangle_indices )
    , _mesh_indices( model._mesh_indices ) {
    _splitter = std::make_unique<Splitter>( *this, _spec );
    // BVH preprocess
    // generate array of triangle indices
    glm::uint32 n = 0;
    _triangle_indices.resize( model._num_triangles );
    std::generate_n( _triangle_indices.begin(), model._num_triangles, [&]() { return n++; } );

    // generate array of mesh/instance indices
    _mesh_indices.resize( model.getNumMeshes() ); n = 0;
    std::generate_n( _mesh_indices.begin(), model.getNumMeshes(), [&]() { return n++; } );

    build();
}

void BVH::loadCudaPayload( std::unique_ptr<CudaPayload>& payload ) {
    payload->_bvh_root_node = _bvh_root_node;
    payload->_bvh_size_bytes = _bvh_min_right_start_nodes.size() * sizeof( glm::uvec4 );
    payload->_bvh_min_right_start_nodes = std::move( _bvh_min_right_start_nodes );
    payload->_bvh_max_axis_count_nodes = std::move( _bvh_max_axis_count_nodes );

    payload->_world_to_object_row0.reserve( _model.getNumMeshes() );
    payload->_world_to_object_row1.reserve( _model.getNumMeshes() );
    payload->_world_to_object_row2.reserve( _model.getNumMeshes() );
    payload->_object_to_world_row0.reserve( _model.getNumMeshes() );
    payload->_object_to_world_row1.reserve( _model.getNumMeshes() );
    payload->_object_to_world_row2.reserve( _model.getNumMeshes() );
    payload->_normal_row0.reserve( _model.getNumMeshes() );
    payload->_normal_row1.reserve( _model.getNumMeshes() );
    payload->_normal_row2.reserve( _model.getNumMeshes() );
    payload->_transforms_size_bytes = _model.getNumMeshes() * sizeof( glm::vec4 );

    for ( glm::uint32 i = 0; i < _model.getNumMeshes(); ++i ) {
        const glm::mat4& to_local = _model._meshes[_mesh_indices[i]]._to_local;
        payload->_world_to_object_row0.push_back( glm::row( to_local, 0 ) );
        payload->_world_to_object_row1.push_back( glm::row( to_local, 1 ) );
        payload->_world_to_object_row2.push_back( glm::row( to_local, 2 ) );

        const glm::mat4& to_global = _model._meshes[_mesh_indices[i]]._to_global;
        payload->_object_to_world_row0.push_back( glm::row( to_global, 0 ) );
        payload->_object_to_world_row1.push_back( glm::row( to_global, 1 ) );
        payload->_object_to_world_row2.push_back( glm::row( to_global, 2 ) );

        const glm::mat3& normal_transform = _model._meshes[_mesh_indices[i]]._normal_transform;
        payload->_normal_row0.emplace_back( glm::vec4( glm::row( normal_transform, 0 ), 0 ) );
        payload->_normal_row1.emplace_back( glm::vec4( glm::row( normal_transform, 1 ), 0 ) );
        payload->_normal_row2.emplace_back( glm::vec4( glm::row( normal_transform, 2 ), 0 ) );
    }
}

glm::uint32 BVH::makeLeaf( const Bin& bin, glm::uint32 level ) {
    const glm::vec3& min = bin._bbox.getMin();
    const glm::vec3& max = bin._bbox.getMax();
    glm::uvec4 min_right_start = glm::uvec4(
        NOOR::float_as_uint( min.x )
        , NOOR::float_as_uint( min.y )
        , NOOR::float_as_uint( min.z )
        , 0
    );

    glm::uvec4 max_axis_count = glm::uvec4(
        NOOR::float_as_uint( max.x )
        , NOOR::float_as_uint( max.y )
        , NOOR::float_as_uint( max.z )
        , 0
    );
    const glm::uint32 start =
        ( bin._type == TRIBIN )
        ?
        static_cast<glm::uint32>( std::distance( _triangle_indices.begin(), bin._start ) )
        :
        static_cast<glm::uint32>( _bvh_max_axis_count_nodes.size() ) + 1;

    const glm::uint32 count = ( bin._type == TRIBIN ) ? TRI_LEAF_MASK | bin.count() : MESH_LEAF_MASK | bin.count();
    min_right_start.w = start;
    max_axis_count.w = count;

    _bvh_min_right_start_nodes.push_back( min_right_start );
    _bvh_max_axis_count_nodes.push_back( max_axis_count );

    const glm::uint32 result = static_cast<glm::uint32>( _bvh_max_axis_count_nodes.size() ) - 1;

    if ( bin._type == MESHBIN ) {
        glm::uint32 mesh_index = static_cast<glm::uint32>( std::distance( _mesh_indices.begin(), bin._start ) );
        for ( TriangleIndex i = bin._start; i != bin._end; ++i ) {
            const glm::vec3& min = _model.getInstanceMeshBBox( *i ).getMin();
            const glm::vec3& max = _model.getInstanceMeshBBox( *i ).getMax();
            glm::uvec4 min_right_start = glm::uvec4(
                NOOR::float_as_uint( min.x )
                , NOOR::float_as_uint( min.y )
                , NOOR::float_as_uint( min.z )
                , _bvh_mesh_start[*i]
            );
            glm::uvec4 max_axis_count = glm::uvec4(
                NOOR::float_as_uint( max.x )
                , NOOR::float_as_uint( max.y )
                , NOOR::float_as_uint( max.z )
                , MESH_INSTANCE_MASK | mesh_index
            );
            ++mesh_index;
            if ( _model.isMeshLight( *i ) )
                max_axis_count.w = LIGHT_NODE_MASK | _model._meshes[*i]._light_idx;
            _bvh_min_right_start_nodes.push_back( min_right_start );
            _bvh_max_axis_count_nodes.push_back( max_axis_count );
            //++level;
        }
    }
    if ( _stat._height < level ) {
        _stat._height = level;
    }
    if ( bin._type == TRIBIN ) {
        ++_stat._num_leafnodes;
        _stat._max_leaf_tris = std::max( _stat._max_leaf_tris, count ^ TRI_LEAF_MASK );
    }
    return result;
}

glm::uint32 BVH::makeInner( const Bin& bin, glm::uint8 axis, glm::uint32 level ) {
    const glm::vec3& min = bin._bbox.getMin();
    const glm::vec3& max = bin._bbox.getMax();

    glm::uvec4 min_right_start = glm::uvec4(
        NOOR::float_as_uint( min.x )
        , NOOR::float_as_uint( min.y )
        , NOOR::float_as_uint( min.z )
        , 0
    );

    glm::uvec4 max_axis_count = glm::uvec4(
        NOOR::float_as_uint( max.x )
        , NOOR::float_as_uint( max.y )
        , NOOR::float_as_uint( max.z )
        , 0
    );

    max_axis_count.w = ( bin._type == TRIBIN ) ? axis : MESH_INNER_MASK | axis;
    _bvh_min_right_start_nodes.push_back( min_right_start );
    _bvh_max_axis_count_nodes.push_back( max_axis_count );
    ++_stat._num_innernodes;
    return static_cast<glm::uint32>( _bvh_max_axis_count_nodes.size() ) - 1;
}

glm::uint32 BVH::recursiveBuild( Bin& bin, glm::uint32 level ) {
    const glm::uint32 bin_count = bin.count();
    if ( bin._type == TRIBIN ) {
        if ( level >= _spec._max_height || bin_count <= _spec._min_leaf_tris ) {
            return makeLeaf( bin, level );
        }
    } else {
        if ( bin_count < 2 ) {
            return makeLeaf( bin, level );
        }
    }

    const SAH object_sah = _splitter->findBestSplit( bin );
    const float leaf_cost = _spec._Ci * bin_count;
    if ( bin._type == TRIBIN ) {
        if ( !object_sah.isValid()
             ||
             ( leaf_cost < object_sah._sah_cost && bin_count <= _spec._max_leaf_tris )
             ) {
            return makeLeaf( bin, level );
        }
    } else {
        if ( !object_sah.isValid() )
            return makeLeaf( bin, level );
    }
    Bin lbin;
    Bin rbin;
    _splitter->split( bin, lbin, rbin, object_sah );

    const glm::uint32 result = makeInner( bin, object_sah._axis, level );
    const glm::uint32 left = recursiveBuild( lbin, level + 1 );
    const glm::uint32 right = recursiveBuild( rbin, level + 1 );
    _bvh_min_right_start_nodes[result].w = right;
    return result;
}

void BVH::build() {
    // number of meshes (sources + instances)
    glm::uint32 num_meshes = _model.getNumMeshes();
    // total number of triangles for all sources
    const glm::uint32 num_triangles = _model.getNumTriangles();
    // marker/index of the root node for each mesh (sources + instances)
    _bvh_mesh_start.resize( num_meshes );
    // estimate number of BVH nodes (top and bottom levels combined)
    const glm::uint32 node_count_estimate = 2 * ( num_triangles + num_meshes );
    _bvh_min_right_start_nodes.reserve( node_count_estimate );
    _bvh_max_axis_count_nodes.reserve( node_count_estimate );

    // i tracks number of instanced meshes
    glm::uint32 i = 0;
    TriangleIndex start, end;
    for ( ; i < num_meshes; ++i ) {
        if ( !_model._meshes[i].isInstance() ) {
            _bvh_mesh_start[i] = ( ( glm::uint32 )_bvh_min_right_start_nodes.size() );
            start = _triangle_indices.begin() + _model._meshes[i]._start_count.first;
            end = start + _model._meshes[i]._start_count.second;
            Bin root_bin = Bin( start, end, _model.getMeshBBox( i ), _model.getMeshCentBox( i ), TRIBIN );
            recursiveBuild( root_bin, 0 );
        }
    }
    for ( i = 0; i < num_meshes; ++i ) {
        if ( _model._meshes[i].isInstance() ) {
            _bvh_mesh_start[i] = _bvh_mesh_start[_model._meshes[i]._instance];
        }
    }
    // start building the top level BVH over all the bottom level ones
    // set the root node for the entire BVH (top and bottom)
    _bvh_root_node = static_cast<glm::uint32>( _bvh_min_right_start_nodes.size() );
    start = _mesh_indices.begin();
    end = _mesh_indices.end();
    // set the bin type to MESHBIN to trigger top level build
    Bin root_bin = Bin( start, end, _model.getSceneBBox(), _model.getSceneCentBox(), MESHBIN );
    // recursively build the top level BVH
    recursiveBuild( root_bin, _stat._height );
    // shrink in case we overestimated the number of BVH nodes
    _bvh_min_right_start_nodes.shrink_to_fit();
    _bvh_max_axis_count_nodes.shrink_to_fit();
    _stat._height += 1;
}