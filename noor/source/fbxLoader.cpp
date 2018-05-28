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
#include "fbxLoader.h"

bool FBXLoader::initialize() {
    _fbx_manager = FbxManager::Create();
    if ( !_fbx_manager ) {
        return false;
    }

    FbxIOSettings* fbxIOSettings = FbxIOSettings::Create( _fbx_manager, IOSROOT );
    _fbx_manager->SetIOSettings( fbxIOSettings );
    _fbx_scene = FbxScene::Create( _fbx_manager, "myScene" );
    if ( !_fbx_scene ) {
        return false;
    }
    return true;
}

bool FBXLoader::loadScene( const std::string& filename ) {
    FbxImporter* fbxImporter = FbxImporter::Create( _fbx_manager, "FbxImporter" );
    if ( !fbxImporter ) {
        return false;
    }
    if ( !fbxImporter->Initialize( filename.c_str(), -1, _fbx_manager->GetIOSettings() ) ) {
        return false;
    }
    if ( !fbxImporter->Import( _fbx_scene ) ) {
        return false;
    }
    fbxImporter->Destroy();
    return true;
}

void FBXLoader::processSceneGraph( FbxNode* node ) {
    if ( node->GetNodeAttribute() ) {
        switch ( node->GetNodeAttribute()->GetAttributeType() ) {
            case FbxNodeAttribute::eMesh:
                _mesh2nodes[node->GetMesh()].push_back( node );
                ++_num_meshes;
                break;
            case FbxNodeAttribute::eLight:
                _fbx_lights.push_back( FbxCast<FbxLight>( node->GetNodeAttribute() ) );
                break;
            case FbxNodeAttribute::eCamera:
                _fbx_camera = const_cast<FbxCamera*>( node->GetCamera() );
                ++_num_cameras;
                break;
            default:
                break;
        }
    }
    for ( int i = 0; i < node->GetChildCount(); ++i ) {
        processSceneGraph( node->GetChild( i ) );
    }
}

void FBXLoader::getCamera( glm::vec3& eye, glm::vec3& lookAt, glm::vec3& up, float& fov, float& lens_radius, float& focal_length, float& orthozoom ) const {
    if ( _fbx_camera != nullptr ) {
        eye = fbx2glm_vector( _fbx_camera->Position.Get() );
        lookAt = fbx2glm_vector( _fbx_camera->InterestPosition.Get() );
        up = fbx2glm_vector( _fbx_camera->UpVector.Get() );
        fov = NOOR::deg2rad( static_cast<float>( _fbx_camera->FieldOfView ) );
        orthozoom = static_cast<float>( _fbx_camera->OrthoZoom.Get() )*15.f;
    }
}

void FBXLoader::load( const std::string& filename ) {
    initialize();
    if ( !loadScene( filename ) ) std::cout << "error\n";
    processSceneGraph( _fbx_scene->GetRootNode() );
    getLights();
    glm::uint32 curr_mesh_start = 0;
    glm::uint32 curr_instance_id = 0;
    _meshes.reserve( _num_meshes );
    for ( auto l : _mesh2nodes ) {
        FbxMesh* fbxmesh = l.first;
        _fbx_meshes.push_back( fbxmesh );
        _num_triangles += getNumTriangles( fbxmesh );
        _num_vertices += getNumVertices( fbxmesh );
        _num_normals += getNumNormals( fbxmesh );
        _num_binormals += getNumBinormals( fbxmesh );
        _num_tangents += getNumTangents( fbxmesh );
        _num_uvs += getNumUVs( fbxmesh );

        // create partial meshes
        // bounding boxes are computed later in Model.cpp
        std::list<FbxNode*>& nodes = l.second;
        auto it = nodes.cbegin();
        _meshes.emplace_back( curr_mesh_start, getNumTriangles( fbxmesh ), getTransformation( *it ) );
        ++it;
        while ( it != nodes.cend() ) {
            _meshes.emplace_back( curr_mesh_start, getNumTriangles( fbxmesh ), getTransformation( *it ), curr_instance_id );
            ++it;
        }
        curr_mesh_start += getNumTriangles( fbxmesh );
        curr_instance_id += ( glm::uint32 )nodes.size();
    }
    for ( size_t i = 0; i < _area_lights.size(); ++i ) {
        _meshes.emplace_back( _area_lights[i]._shape, (int) i );
        _meshes[_meshes.size() - 1]._instance = curr_instance_id++;
    }
    processMaterialGroups();
}

void FBXLoader::getMeshes( std::vector<Mesh>& meshes ) const {
    meshes = std::move( _meshes );
}

void FBXLoader::processMaterialGroups() {
    _num_materials = _fbx_scene->GetMaterialCount();
    std::map<std::string, int> material_name2index;
    for ( int i = 0; i < _num_materials; ++i ) {
        FbxSurfaceMaterial* m = _fbx_scene->GetMaterial( i );
        material_name2index[m->GetName()] = i;
    }
    using TriMatPair = glm::uvec2;
    std::vector<TriMatPair> tri_mat_array( _num_triangles * 2 );
    glm::uint32 current = 0;

    for ( auto mesh : _fbx_meshes ) {
        FbxLayerElementArrayTemplate<int>* materialIndices = &( mesh->GetElementMaterial()->GetIndexArray() );
        for ( glm::uint32 i = 0; i < getNumTriangles( mesh ); ++i ) {
            FbxSurfaceMaterial* lMaterial = mesh->GetNode()->GetMaterial( materialIndices->GetAt( i ) );
            std::string name( lMaterial->GetName() );
            tri_mat_array[i + current] = glm::uvec2( i + current, material_name2index[name] );
        }
        current += getNumTriangles( mesh );
    }

    auto comp = []( const TriMatPair& a, const TriMatPair& b ) {
        return a.y < b.y;
    };
    std::sort( tri_mat_array.begin(), tri_mat_array.end(), comp );
    _material_groups.reserve( getNumMaterials() );

    std::vector<TriMatPair>::const_iterator left = tri_mat_array.begin();
    std::vector<TriMatPair>::const_iterator right = tri_mat_array.begin();

    while ( left != tri_mat_array.end() ) {
        const std::string mat_name( _fbx_scene->GetMaterial( ( *left ).y )->GetName() );
        while ( right != tri_mat_array.end() && ( *left ).y == ( *right ).y ) {
            std::advance( right, 1 );
        }

        const glm::uint32 num_tris = static_cast<glm::uint32>( std::distance( left, right ) );
        MaterialGroup mg( mat_name, num_tris, ( *left ).y );
        mg._triangle_indices.reserve( num_tris );
        while ( left != right ) {
            mg._triangle_indices.push_back( ( *left ).x );
            std::advance( left, 1 );
        }
        _material_groups.push_back( mg );
    }
    assert( _material_groups.size() );
}

void FBXLoader::getVertices(
    std::vector<glm::vec4>& vertices
    , std::vector<glm::uvec3>& vindices
) const {
    vertices.reserve( _num_vertices );
    vindices.reserve( _num_triangles );

    glm::vec4 p;
    glm::uint32 current = 0;
    for ( auto mesh : _fbx_meshes ) {
        for ( glm::uint32 i = 0; i < getNumVertices( mesh ); ++i ) {
            p = fbx2glm_point( mesh->GetControlPointAt( i ) );
            vertices.push_back( p );
        }
        glm::uint32 v0, v1, v2;
        for ( glm::uint32 i = 0; i < getNumTriangles( mesh ); ++i ) {
            v0 = mesh->GetPolygonVertex( i, 0 ) + current;
            v1 = mesh->GetPolygonVertex( i, 1 ) + current;
            v2 = mesh->GetPolygonVertex( i, 2 ) + current;
            vindices.emplace_back( v0, v1, v2 );
        }
        current += getNumVertices( mesh );
    }
}

void FBXLoader::getNormals(
    std::vector<glm::vec4>& normals
    , std::vector<glm::uvec3>& nindices
) const {
    normals.resize( _num_normals );
    nindices.reserve( _num_triangles );

    glm::uint32 current = 0;
    for ( auto mesh : _fbx_meshes ) {
        glm::uvec3 nindex;
        glm::uint32 vertexCounter = 0;
        for ( glm::uint32 i = 0; i < getNumTriangles( mesh ); ++i ) {
            for ( glm::uint32 j = 0; j < 3; ++j ) {
                glm::uint32 ctrlPointIndex = mesh->GetPolygonVertex( i, j );
                readNormal( mesh, ctrlPointIndex, vertexCounter, nindex[j], current, normals );
                ++vertexCounter;
            }
            nindices.push_back( nindex );
        }
        current += getNumNormals( mesh );
    }
}

void FBXLoader::getTangents(
    std::vector<glm::vec4>& tangents
    , std::vector<glm::uvec3>& tindices
) const {
    tindices.reserve( _num_triangles );
    tangents.resize( _num_tangents );

    glm::uint32 current = 0;
    for ( auto mesh : _fbx_meshes ) {
        glm::uvec3 tindex;
        glm::uint32 vertexCounter = 0;
        for ( glm::uint32 i = 0; i < getNumTriangles( mesh ); ++i ) {
            for ( glm::uint32 j = 0; j < 3; ++j ) {
                glm::uint32 ctrlPointIndex = mesh->GetPolygonVertex( i, j );
                readTangent( mesh, ctrlPointIndex, vertexCounter, tindex[j], current, tangents );
                ++vertexCounter;
            }
            tindices.push_back( tindex );
        }
        current += getNumTangents( mesh );
    }
}

void FBXLoader::getBinormals(
    std::vector<glm::vec4>& binormals
    , std::vector<glm::uvec3>& bindices
) const {
    bindices.reserve( _num_triangles );
    binormals.resize( _num_binormals );

    glm::uint32 current = 0;
    for ( auto mesh : _fbx_meshes ) {
        glm::uvec3 bindex;
        glm::uint32 vertexCounter = 0;
        for ( glm::uint32 i = 0; i < getNumTriangles( mesh ); ++i ) {
            for ( glm::uint32 j = 0; j < 3; ++j ) {
                glm::uint32 ctrlPointIndex = mesh->GetPolygonVertex( i, j );
                readBinormal( mesh, ctrlPointIndex, vertexCounter, bindex[j], current, binormals );
                ++vertexCounter;
            }
            bindices.push_back( bindex );
        }
        current += getNumBinormals( mesh );
    }
}

void FBXLoader::getUVs(
    std::vector<glm::vec2>& uvs
    , std::vector<glm::uvec3>& uvindices
) const {
    uvindices.reserve( _num_triangles );
    uvs.resize( _num_uvs );

    glm::uint32 current = 0;
    for ( auto mesh : _fbx_meshes ) {
        glm::uvec3 uvindex;
        for ( glm::uint32 i = 0; i < getNumTriangles( mesh ); ++i ) {
            for ( glm::uint32 j = 0; j < 3; ++j ) {
                glm::uint32 ctrlPointIndex = mesh->GetPolygonVertex( i, j );
                int uvIndex = mesh->GetTextureUVIndex( i, j );
                uvIndex = uvIndex < 0 ? 0 : uvIndex;
                readUV( mesh, ctrlPointIndex, uvIndex, uvindex[j], current, uvs );
            }
            uvindices.push_back( uvindex );
        }
        current += getNumUVs( mesh );
    }
    if ( _num_uvs == 0 ) {
        uvs.reserve( 1 );
        uvs.emplace_back( 0, 0 );
    }
}

void FBXLoader::readNormal(
    const FbxMesh* mesh
    , glm::uint32 vertexIndex
    , glm::uint32 vertexCounter
    , glm::uint32& normalIndex
    , glm::uint32 meshIndex
    , std::vector<glm::vec4>& normals
) const {
    if ( mesh->GetElementNormalCount() < 1 ) {
        throw std::exception( "Invalid Normal Number" );
    }
    glm::uint32 index = 0;
    const FbxGeometryElementNormal* normalElement = mesh->GetElementNormal( 0 );
    switch ( normalElement->GetMappingMode() ) {
        case FbxGeometryElement::eByControlPoint:
            switch ( normalElement->GetReferenceMode() ) {
                case FbxGeometryElement::eDirect:
                    index = static_cast<glm::uint32>( vertexIndex );
                    break;
                case FbxGeometryElement::eIndexToDirect:
                    index = static_cast<glm::uint32>( normalElement->GetIndexArray().GetAt( vertexIndex ) );
                    break;
                default:
                    throw std::exception( "Invalid Reference" );
            }
            break;

        case FbxGeometryElement::eByPolygonVertex:
            switch ( normalElement->GetReferenceMode() ) {
                case FbxGeometryElement::eDirect:
                    index = static_cast<glm::uint32>( vertexCounter );
                    break;
                case FbxGeometryElement::eIndexToDirect:
                    index = static_cast<glm::uint32>( normalElement->GetIndexArray().GetAt( vertexCounter ) );
                    break;

                default:
                    throw std::exception( "Invalid Reference" );
            }
            break;
    }
    const glm::uint32 currentIndex = meshIndex + index;
    std::unordered_set<glm::uint32>::iterator it = _normal_lookup.find( currentIndex );
    if ( it == _normal_lookup.end() ) {
        normals[currentIndex] = fbx2glm_vector( normalElement->GetDirectArray().GetAt( index ) );
        _normal_lookup.insert( currentIndex );
        normalIndex = index + meshIndex;
    } else {
        normalIndex = *it;
    }
}
void FBXLoader::readBinormal(
    const FbxMesh* mesh
    , glm::uint32 vertexIndex
    , glm::uint32 vertexCounter
    , glm::uint32& binormalIndex
    , glm::uint32 meshIndex
    , std::vector<glm::vec4>& binormals
) const {
    if ( mesh->GetElementBinormalCount() < 1 ) {
        throw std::exception( "Invalid Binormal Number" );
    }
    glm::uint32 index = 0;
    const FbxGeometryElementBinormal* binormalElement = mesh->GetElementBinormal( 0 );
    switch ( binormalElement->GetMappingMode() ) {
        case FbxGeometryElement::eByControlPoint:
            switch ( binormalElement->GetReferenceMode() ) {
                case FbxGeometryElement::eDirect:
                    index = static_cast<glm::uint32>( vertexIndex );
                    break;
                case FbxGeometryElement::eIndexToDirect:
                    index = static_cast<glm::uint32>( binormalElement->GetIndexArray().GetAt( vertexIndex ) );
                    break;
                default:
                    throw std::exception( "Invalid Reference" );
            }
            break;

        case FbxGeometryElement::eByPolygonVertex:
            switch ( binormalElement->GetReferenceMode() ) {
                case FbxGeometryElement::eDirect:
                    index = static_cast<glm::uint32>( vertexCounter );
                    break;
                case FbxGeometryElement::eIndexToDirect:
                    index = static_cast<glm::uint32>( binormalElement->GetIndexArray().GetAt( vertexCounter ) );
                    break;

                default:
                    throw std::exception( "Invalid Reference" );
            }
            break;
    }
    const glm::uint32 currentIndex = meshIndex + index;
    std::unordered_set<glm::uint32>::iterator it = _binormal_lookup.find( currentIndex );
    if ( it == _binormal_lookup.end() ) {
        binormals[currentIndex] = fbx2glm_vector( binormalElement->GetDirectArray().GetAt( index ) );
        _binormal_lookup.insert( currentIndex );
        binormalIndex = currentIndex;
    } else {
        binormalIndex = *it;
    }
}

void FBXLoader::readTangent(
    const FbxMesh* mesh
    , glm::uint32 vertexIndex
    , glm::uint32 vertexCounter
    , glm::uint32& tangentIndex
    , glm::uint32 meshIndex
    , std::vector<glm::vec4>& tangents
) const {
    if ( mesh->GetElementTangentCount() < 1 ) {
        throw std::exception( "Invalid Tangent Number" );
    }
    glm::uint32 index = 0;
    const FbxGeometryElementTangent* tangentElement = mesh->GetElementTangent( 0 );
    switch ( tangentElement->GetMappingMode() ) {
        case FbxGeometryElement::eByControlPoint:
            switch ( tangentElement->GetReferenceMode() ) {
                case FbxGeometryElement::eDirect:
                    index = static_cast<glm::uint32>( vertexIndex );
                    break;
                case FbxGeometryElement::eIndexToDirect:
                    index = static_cast<glm::uint32>( tangentElement->GetIndexArray().GetAt( vertexIndex ) );
                    break;
                default:
                    throw std::exception( "Invalid Reference" );
            }
            break;

        case FbxGeometryElement::eByPolygonVertex:
            switch ( tangentElement->GetReferenceMode() ) {
                case FbxGeometryElement::eDirect:
                    index = static_cast<glm::uint32>( vertexCounter );
                    break;
                case FbxGeometryElement::eIndexToDirect:
                    index = static_cast<glm::uint32>( tangentElement->GetIndexArray().GetAt( vertexCounter ) );
                    break;

                default:
                    throw std::exception( "Invalid Reference" );
            }
            break;
    }
    const glm::uint32 currentIndex = meshIndex + index;
    std::unordered_set<glm::uint32>::iterator it = _tangent_lookup.find( currentIndex );
    if ( it == _tangent_lookup.end() ) {
        tangents[currentIndex] = fbx2glm_vector( tangentElement->GetDirectArray().GetAt( index ) );
        _tangent_lookup.insert( currentIndex );
        tangentIndex = currentIndex;
    } else {
        tangentIndex = *it;
    }
}

void FBXLoader::readUV(
    const FbxMesh* mesh
    , glm::uint32 vertexIndex
    , int inTextureUVIndex
    , glm::uint32& uvIndex
    , glm::uint32 meshIndex
    , std::vector<glm::vec2>& uvs
) const {
    if ( mesh->GetElementUVCount() < 1 ) {
        return;
    }
    int index = 0;
    const FbxGeometryElementUV* uvElement = mesh->GetElementUV( 0 );
    switch ( uvElement->GetMappingMode() ) {
        case FbxGeometryElement::eByControlPoint:
            switch ( uvElement->GetReferenceMode() ) {
                case FbxGeometryElement::eDirect:
                    index = static_cast<glm::uint32>( vertexIndex );
                    break;
                case FbxGeometryElement::eIndexToDirect:
                    index = static_cast<glm::uint32>( uvElement->GetIndexArray().GetAt( vertexIndex ) );
                    break;
                default:
                    throw std::exception( "Invalid Reference" );
            }
            break;

        case FbxGeometryElement::eByPolygonVertex:
            switch ( uvElement->GetReferenceMode() ) {
                case FbxGeometryElement::eDirect:
                case FbxGeometryElement::eIndexToDirect:
                    index = inTextureUVIndex;
                    break;
                default:
                    throw std::exception( "Invalid Reference" );
            }
            break;
    }
    const glm::uint32 currentIndex = meshIndex + index;
    std::unordered_set<glm::uint32>::iterator it = _uv_lookup.find( currentIndex );
    if ( it == _uv_lookup.end() ) {
        uvs[currentIndex] = fbx2glm_vector( uvElement->GetDirectArray().GetAt( index ) );
        _uv_lookup.insert( currentIndex );
        uvIndex = currentIndex;
    } else {
        uvIndex = *it;
    }
}

void FBXLoader::getLights() {
    for ( auto light : _fbx_lights ) {
        switch ( light->LightType ) {
            case( FbxLight::eArea ):
                getAreaLight( light, _area_lights );
                ++_num_meshes;
                break;
            case( FbxLight::ePoint ):
                getPointLight( light, _point_lights );
                break;
            case( FbxLight::eDirectional ):
                // TODO: change to exact world radius
                getDistantLight( light, _distant_lights, 100.f );
                break;
            case( FbxLight::eSpot ):
                getSpotLight( light, _spot_lights );
                break;
            default:
                break;
        }
    }
}

void FBXLoader::getLights( std::vector<CudaAreaLight>& area_lights
                           , std::vector<CudaPointLight>& point_lights
                           , std::vector<CudaSpotLight>& spot_lights
                           , std::vector<CudaDistantLight>& distant_lights
                           , float world_radius ) const {
    area_lights = std::move( _area_lights );
    point_lights = std::move( _point_lights );
    spot_lights = std::move( _spot_lights );
    distant_lights = std::move( _distant_lights );
}

glm::vec4 FBXLoader::fbx2glm( const FbxVector4& v ) const {
    glm::vec4 result;
    result.x = static_cast<float>( v[0] );
    result.y = static_cast<float>( v[1] );
    result.z = static_cast<float>( v[2] );
    result.w = static_cast<float>( v[3] );
    return result;
}

glm::mat4 FBXLoader::getTransformation( const FbxMesh* mesh ) const {
    FbxNode* node = mesh->GetNode();
    const FbxAMatrix& global = node->EvaluateGlobalTransform();
    const glm::vec4 c0 = fbx2glm( global.GetRow( 0 ) );
    const glm::vec4 c1 = fbx2glm( global.GetRow( 1 ) );
    const glm::vec4 c2 = fbx2glm( global.GetRow( 2 ) );
    const glm::vec4 c3 = fbx2glm( global.GetRow( 3 ) );

    return glm::mat4( c0, c1, c2, c3 );
}

glm::mat4 FBXLoader::getTransformation( const FbxNode* node ) const {
    const FbxAMatrix& global = const_cast<FbxNode*>( node )->EvaluateGlobalTransform();
    const glm::vec4 c0 = fbx2glm( global.GetRow( 0 ) );
    const glm::vec4 c1 = fbx2glm( global.GetRow( 1 ) );
    const glm::vec4 c2 = fbx2glm( global.GetRow( 2 ) );
    const glm::vec4 c3 = fbx2glm( global.GetRow( 3 ) );

    return glm::mat4( c0, c1, c2, c3 );
}

glm::mat4 FBXLoader::getTransformation( const FbxLight* light ) const {
    FbxNode* node = light->GetNode();
    FbxDouble3 T = node->LclTranslation.Get();
    FbxDouble3 R = node->LclRotation.Get();
    FbxDouble3 S = node->LclScaling.Get();

    glm::vec3 t, r, s;
    t.x = static_cast<float>( T[0] );
    t.y = static_cast<float>( T[1] );
    t.z = static_cast<float>( T[2] );

    r.x = glm::radians( static_cast<float>( R[0] ) );
    r.y = glm::radians( static_cast<float>( R[1] ) );
    r.z = glm::radians( static_cast<float>( R[2] ) );

    s.x = static_cast<float>( S[0] );
    s.y = static_cast<float>( S[1] );
    s.z = static_cast<float>( S[2] );

    const glm::quat q( r );

    const glm::mat4 light_translate = glm::translate( t );
    const glm::mat4 light_rotate = glm::toMat4( q );
    const glm::mat4 light_scale = glm::scale( s );

    return light_translate * light_rotate * light_scale;

}
void FBXLoader::getAreaLight( const FbxLight* light,
                              std::vector<CudaAreaLight>& area_light_data ) const {
    AreaMeshLightType type = QUAD;
    FbxProperty lPropertyType = light->GetNode()->FindProperty( "Type" );
    if ( lPropertyType.IsValid() )
        type = static_cast<AreaMeshLightType>( lPropertyType.Get<FbxEnum>() );
    bool twoSided = false;
    lPropertyType = light->GetNode()->FindProperty( "TwoSided" );
    if ( lPropertyType.IsValid() )
        twoSided = static_cast<AreaMeshLightType>( lPropertyType.Get<FbxBool>() );
    const glm::mat4 trs = getTransformation( light );
    // default area light in Maya is a quad facing negative z
    // base is left bottom corner
    glm::vec3 base = trs * glm::vec4( -1, -1, 0, 1 );

    // u and v are the edge vectors
    glm::vec3 u = trs * glm::vec4( 2, 0, 0, 0 );
    glm::vec3 v = trs * glm::vec4( 0, 2, 0, 0 );

    // n is the normal 
    const glm::mat4 _normal_transform = glm::transpose( glm::inverse( glm::mat3( trs ) ) );
    glm::vec3 n = glm::normalize( _normal_transform * glm::vec4( 0, 0, -1, 0 ) );
    CudaShape shape( V2F3( base ), V2F3( u ), V2F3( v ), V2F3( n ), type );

    const FbxDouble3 color = light->Color.Get();
    glm::vec3 Ke;
    Ke.x = static_cast<float>( color[0] );
    Ke.y = static_cast<float>( color[1] );
    Ke.z = static_cast<float>( color[2] );

    const float intensity = static_cast<float>( light->Intensity.Get() ) / 100.0f;
    Ke *= intensity;
    area_light_data.emplace_back( shape, V2F3( Ke ), twoSided );
}

void FBXLoader::getPointLight( const FbxLight* light, std::vector<CudaPointLight>& point_light_data ) const {
    const glm::mat4 trs = getTransformation( light );

    glm::vec3 position( trs[3].x, trs[3].y, trs[3].z );

    const FbxDouble3 color = light->Color.Get();
    glm::vec3 Ke;
    Ke.x = static_cast<float>( color[0] );
    Ke.y = static_cast<float>( color[1] );
    Ke.z = static_cast<float>( color[2] );

    const float intensity = static_cast<float>( light->Intensity.Get() ) / 100.0f;
    Ke *= intensity;
    point_light_data.emplace_back( V2F3( position ), V2F3( Ke ) );
}

void FBXLoader::getSpotLight( const FbxLight* light, std::vector<CudaSpotLight>& spot_light_data ) const {
    const glm::mat4 trs = getTransformation( light );
    const glm::vec4 direction = glm::normalize( trs * glm::vec4( 0, 0, 1, 0 ) );

    glm::vec3 position( trs[3].x, trs[3].y, trs[3].z );

    const FbxDouble3 color = light->Color.Get();
    glm::vec3 Ke;
    Ke.x = static_cast<float>( color[0] );
    Ke.y = static_cast<float>( color[1] );
    Ke.z = static_cast<float>( color[2] );

    const float intensity = static_cast<float>( light->Intensity.Get() ) / 100.0f;
    Ke *= intensity;
    float innerdegrees = (float) light->InnerAngle;
    float outerdegrees = (float) light->OuterAngle;
    const float innerangle = glm::radians( static_cast<float>( light->InnerAngle ) );
    const float outerangle = glm::radians( static_cast<float>( light->OuterAngle ) );

    const float cosinner = cosf( innerangle / 2.0f );
    const float cosouter = cosf( outerangle / 2.0f );
    spot_light_data.emplace_back(
        V2F3( position )
        , V2F3( direction )
        , V2F3( Ke )
        , cosinner
        , cosouter
    );
}
void FBXLoader::getDistantLight( const FbxLight* light,
                                 std::vector<CudaDistantLight>& distant_light_data,
                                 float world_radius
) const {
    const glm::mat4 trs = getTransformation( light );
    const glm::vec4 direction = glm::normalize( trs * glm::vec4( 0, 0, 1, 0 ) );

    const FbxDouble3 color = light->Color.Get();
    glm::vec3 Ke;
    Ke.x = static_cast<float>( color[0] );
    Ke.y = static_cast<float>( color[1] );
    Ke.z = static_cast<float>( color[2] );

    const float intensity = static_cast<float>( light->Intensity.Get() ) / 100.0f;
    Ke *= intensity;
    const glm::vec3 world_center( 0.0f );
    distant_light_data.emplace_back( V2F3( direction ), V2F3( Ke ), V2F3( world_center ), world_radius );
}

glm::vec3 FBXLoader::getMaterialDiffuse( int mat_idx ) const {
    const FbxSurfaceLambert* material = FbxCast<FbxSurfaceLambert>( _fbx_scene->GetMaterial( mat_idx ) );
    if ( material != nullptr ) {
        const FbxDouble3 d = material->Diffuse;
        return glm::vec4(
            static_cast<float>( d[0] )
            , static_cast<float>( d[1] )
            , static_cast<float>( d[2] )
            , 1.0f
        );
    }
    return glm::vec3( 0.0f );
}

glm::vec3 FBXLoader::getMaterialSpecular( int mat_idx ) const {
    const FbxSurfacePhong* material = FbxCast<FbxSurfacePhong>( _fbx_scene->GetMaterial( mat_idx ) );
    if ( material != nullptr ) {
        const FbxDouble3 s = material->Specular;
        return glm::vec3(
            static_cast<float>( s[0] )
            , static_cast<float>( s[1] )
            , static_cast<float>( s[2] )
        );
    }
    return glm::vec3( 0.0f );
}
glm::vec3 FBXLoader::getMaterialReflection( int mat_idx ) const {
    const FbxSurfacePhong* material = FbxCast<FbxSurfacePhong>( _fbx_scene->GetMaterial( mat_idx ) );
    if ( material != nullptr ) {
        const FbxDouble3 s = material->Reflection;
        return glm::vec3(
            static_cast<float>( s[0] )
            , static_cast<float>( s[1] )
            , static_cast<float>( s[2] )
        );
    }
    return glm::vec3( 0.0f );
}

glm::vec3  FBXLoader::getMaterialEmittance( int mat_idx ) const {
    const FbxSurfaceLambert* material = FbxCast<FbxSurfaceLambert>( _fbx_scene->GetMaterial( mat_idx ) );
    if ( material != nullptr ) {
        const FbxDouble3 e = material->Emissive;
        return glm::vec3(
            static_cast<float>( e[0] )
            , static_cast<float>( e[1] )
            , static_cast<float>( e[2] )
        );
    }
    return glm::vec3( 0.0f );
}

glm::vec3 FBXLoader::getMaterialTransmission( int mat_idx ) const {
    const FbxSurfaceLambert* material = FbxCast<FbxSurfaceLambert>( _fbx_scene->GetMaterial( mat_idx ) );
    if ( material != nullptr ) {
        const FbxDouble3 t = material->TransparentColor;
        return glm::vec3(
            static_cast<float>( t[0] )
            , static_cast<float>( t[1] )
            , static_cast<float>( t[2] )
        );
    }
    return glm::vec3( 0.0f );
}

glm::vec3 FBXLoader::getMaterialTransparency( int mat_idx ) const {
    const FbxSurfaceLambert* material = FbxCast<FbxSurfaceLambert>( _fbx_scene->GetMaterial( mat_idx ) );
    if ( material != nullptr ) {
        const FbxDouble t = material->TransparencyFactor;
        return glm::vec3( static_cast<float>( t ) );
    }
    return glm::vec3( 0.0f );
}

float FBXLoader::getMaterialCoatingRoughness( int mat_idx ) const {
    FbxProperty lProperty = _fbx_scene->GetMaterial( mat_idx )->FindProperty( "Roughness" );
    if ( lProperty.IsValid() ) {
        const float roughness = static_cast<float>( lProperty.Get<FbxDouble>() );
        return roughness;
    }
    return 0.001f;
}

float FBXLoader::getMaterialCoatingWeight( int mat_idx ) const {
    FbxProperty lProperty = _fbx_scene->GetMaterial( mat_idx )->FindProperty( "Weight" );
    if ( lProperty.IsValid() ) {
        const float weight = static_cast<float>( lProperty.Get<FbxDouble>() );
        return weight;
    }
    return 1.0f;
}

float FBXLoader::getMaterialCoatingThickness( int mat_idx ) const {
    FbxProperty lProperty = _fbx_scene->GetMaterial( mat_idx )->FindProperty( "Thickness" );
    if ( lProperty.IsValid() ) {
        const float thickness = static_cast<float>( lProperty.Get<FbxDouble>() );
        return thickness;
    }
    return 1.0f;
}

float FBXLoader::getMaterialCoatingSigma( int mat_idx ) const {
    FbxProperty lProperty = _fbx_scene->GetMaterial( mat_idx )->FindProperty( "Sigma" );
    if ( lProperty.IsValid() ) {
        const float sigma = static_cast<float>( lProperty.Get<FbxDouble>() );
        return sigma;
    }
    return 0.0f;
}

float FBXLoader::getMaterialCoatingIOR( int mat_idx ) const {
    FbxProperty lProperty = _fbx_scene->GetMaterial( mat_idx )->FindProperty( "IOR" );
    if ( lProperty.IsValid() ) {
        const float ior = static_cast<float>( lProperty.Get<FbxDouble>() );
        return ior;
    }
    return 1.5f;
}

glm::vec2 FBXLoader::getMaterialRoughness( int mat_idx ) const {
    FbxProperty lPropertyU = _fbx_scene->GetMaterial( mat_idx )->FindProperty( "RoughnessU" );
    FbxProperty lPropertyV = _fbx_scene->GetMaterial( mat_idx )->FindProperty( "RoughnessV" );
    if ( lPropertyU.IsValid() && lPropertyV.IsValid() ) {
        const float roughnessU = static_cast<float>( lPropertyU.Get<FbxDouble>() );
        const float roughnessV = static_cast<float>( lPropertyV.Get<FbxDouble>() );
        return glm::vec2( roughnessU, roughnessV );
    }
    return glm::vec2( 0.0f );
}

glm::vec3 FBXLoader::getMaterialIor( int mat_idx ) const {
    FbxProperty lProperty = _fbx_scene->GetMaterial( mat_idx )->FindProperty( "Eta" );
    if ( lProperty.IsValid() ) {
        const FbxDouble3 e = lProperty.Get<FbxDouble3>();
        return glm::vec3(
            static_cast<float>( e[0] )
            , static_cast<float>( e[1] )
            , static_cast<float>( e[2] )
        );
    }
    return glm::vec3( 1.0f );
}

glm::vec3 FBXLoader::getMaterialK( int mat_idx ) const {
    FbxProperty lProperty = _fbx_scene->GetMaterial( mat_idx )->FindProperty( "K" );
    if ( lProperty.IsValid() ) {
        const FbxDouble3 k = lProperty.Get<FbxDouble3>();
        return glm::vec3(
            static_cast<float>( k[0] )
            , static_cast<float>( k[1] )
            , static_cast<float>( k[2] )
        );
    }
    return glm::vec3( 1.0f );
}

MaterialType FBXLoader::getMaterialType( int mat_idx ) const {
    FbxProperty lProperty = _fbx_scene->GetMaterial( mat_idx )->FindProperty( "Type" );
    if ( lProperty.IsValid() )
        return static_cast<MaterialType>( 1 << lProperty.Get<FbxEnum>() );
    return DIFFUSE;
}

std::string FBXLoader::getMaterial_map_roughness( int mat_idx, glm::vec2& uvscale ) const {
    const FbxSurfaceMaterial* material = _fbx_scene->GetMaterial( mat_idx );
    if ( material != nullptr ) {
        FbxProperty p = material->FindProperty( FbxSurfaceMaterial::sAmbient );
        if ( p.IsValid() ) {
            int n = p.GetSrcObjectCount<FbxFileTexture>();
            if ( n > 0 ) {
                FbxFileTexture* t = p.GetSrcObject<FbxFileTexture>( 0 );
                if ( t != nullptr ) {
                    uvscale.x = static_cast<float>( t->GetScaleU() );
                    uvscale.y = static_cast<float>( t->GetScaleV() );
                    return std::string( t->GetFileName() );
                }
            }
        }
    }
    uvscale = glm::vec2( 1.0f );
    return std::string( "" );
}

std::string FBXLoader::getMaterial_map_kd( int mat_idx, glm::vec2& uvscale ) const {
    FbxSurfaceMaterial* material = _fbx_scene->GetMaterial( mat_idx );
    if ( material != nullptr ) {
        FbxProperty p = material->FindProperty( FbxSurfaceMaterial::sDiffuse );
        int n = p.GetSrcObjectCount<FbxTexture>();
        if ( n > 0 ) {
            FbxFileTexture* t = p.GetSrcObject<FbxFileTexture>( 0 );
            if ( t != nullptr ) {
                uvscale.x = static_cast<float>( t->GetScaleU() );
                uvscale.y = static_cast<float>( t->GetScaleV() );
                return std::string( t->GetFileName() );
            }
        }
    }
    uvscale = glm::vec2( 1.0f );
    return std::string( "" );
}

std::string FBXLoader::getMaterial_map_ks( int mat_idx, glm::vec2& uvscale ) const {
    FbxSurfaceMaterial* material = _fbx_scene->GetMaterial( mat_idx );
    if ( material != nullptr ) {
        FbxProperty p = material->FindProperty( FbxSurfaceMaterial::sSpecular );
        int n = p.GetSrcObjectCount<FbxTexture>();
        if ( n > 0 ) {
            FbxFileTexture* t = p.GetSrcObject<FbxFileTexture>( 0 );
            if ( t != nullptr ) {
                uvscale.x = static_cast<float>( t->GetScaleU() );
                uvscale.y = static_cast<float>( t->GetScaleV() );
                return std::string( t->GetFileName() );
            }
        }
    }
    uvscale = glm::vec2( 1.0f );
    return std::string( "" );
}

std::string FBXLoader::getMaterial_map_ka( int mat_idx, glm::vec2& uvscale ) const {
    FbxSurfaceMaterial* material = _fbx_scene->GetMaterial( mat_idx );
    if ( material != nullptr ) {
        FbxProperty p = material->FindProperty( FbxSurfaceMaterial::sAmbient );
        int n = p.GetSrcObjectCount<FbxTexture>();
        if ( n > 0 ) {
            FbxFileTexture* t = p.GetSrcObject<FbxFileTexture>( 0 );
            if ( t != nullptr ) {
                uvscale.x = static_cast<float>( t->GetScaleU() );
                uvscale.y = static_cast<float>( t->GetScaleV() );
                return std::string( t->GetFileName() );
            }
        }
    }
    uvscale = glm::vec2( 1.0f );
    return std::string( "" );
}

std::string FBXLoader::getMaterial_map_ke( int mat_idx, glm::vec2& uvscale ) const {
    FbxSurfaceMaterial* material = _fbx_scene->GetMaterial( mat_idx );
    if ( material != nullptr ) {
        FbxProperty p = material->FindProperty( FbxSurfaceMaterial::sEmissive );
        int n = p.GetSrcObjectCount<FbxTexture>();
        if ( n > 0 ) {
            FbxFileTexture* t = p.GetSrcObject<FbxFileTexture>( 0 );
            if ( t != nullptr ) {
                uvscale.x = static_cast<float>( t->GetScaleU() );
                uvscale.y = static_cast<float>( t->GetScaleV() );
                return std::string( t->GetFileName() );
            }
        }
    }
    uvscale = glm::vec2( 1.0f );
    return std::string( "" );
}

std::string FBXLoader::getMaterial_map_d( int mat_idx, glm::vec2& uvscale ) const {
    FbxSurfaceMaterial* material = _fbx_scene->GetMaterial( mat_idx );
    if ( material != nullptr ) {
        FbxProperty p = material->FindProperty( FbxSurfaceMaterial::sTransparentColor );
        int n = p.GetSrcObjectCount<FbxTexture>();
        if ( n > 0 ) {
            FbxFileTexture* t = p.GetSrcObject<FbxFileTexture>( 0 );
            if ( t != nullptr ) {
                uvscale.x = static_cast<float>( t->GetScaleU() );
                uvscale.y = static_cast<float>( t->GetScaleV() );
                return std::string( t->GetFileName() );
            }
        }
    }
    uvscale = glm::vec2( 1.0f );
    return std::string( "" );
}

std::string FBXLoader::getMaterial_map_bump( int mat_idx, glm::vec2& uvscale, float& bumpfactor ) const {
    const FbxSurfaceLambert* material = FbxCast<FbxSurfaceLambert>( _fbx_scene->GetMaterial( mat_idx ) );
    if ( material != nullptr ) {
        bumpfactor = static_cast<float>( material->BumpFactor );
        FbxProperty p = material->FindProperty( FbxSurfaceMaterial::sBump );
        int n = p.GetSrcObjectCount<FbxTexture>();
        if ( n > 0 ) {
            FbxFileTexture* t = p.GetSrcObject<FbxFileTexture>( 0 );
            if ( t != nullptr ) {
                uvscale.x = static_cast<float>( t->GetScaleU() );
                uvscale.y = static_cast<float>( t->GetScaleV() );
                return std::string( t->GetFileName() );
            }
        }
    }
    uvscale = glm::vec2( 1.0f );
    bumpfactor = 1.0f;
    return std::string( "" );
}

std::string FBXLoader::getMaterialName( int mat_index )const {
    return std::string( _fbx_scene->GetMaterial( mat_index )->GetName() );
}

glm::uint32 FBXLoader::getNumTriangles( const FbxMesh* mesh ) const {
    return mesh->GetPolygonCount();
}

glm::uint32 FBXLoader::getNumVertices( const FbxMesh* mesh ) const {
    return mesh->GetControlPointsCount();
}

glm::uint32 FBXLoader::getNumNormals( const FbxMesh* mesh )  const {
    const FbxGeometryElementNormal* normalElement = mesh->GetElementNormal( 0 );
    if ( normalElement )
        return static_cast<glm::uint32>( normalElement->GetDirectArray().GetCount() );
    else
        return 0;
}

glm::uint32 FBXLoader::getNumTangents( const FbxMesh* mesh )  const {
    const FbxGeometryElementTangent* tangentElement = mesh->GetElementTangent( 0 );
    if ( tangentElement )
        return static_cast<glm::uint32>( tangentElement->GetDirectArray().GetCount() );
    else
        return 0;
}

glm::uint32 FBXLoader::getNumBinormals( const FbxMesh* mesh )  const {
    const FbxGeometryElementBinormal* binormalElement = mesh->GetElementBinormal( 0 );
    if ( binormalElement )
        return static_cast<glm::uint32>( binormalElement->GetDirectArray().GetCount() );
    else
        return 0;
}

glm::uint32 FBXLoader::getNumUVs( const FbxMesh* mesh ) const {
    const FbxGeometryElementUV* uvElement = mesh->GetElementUV( 0 );
    if ( uvElement )
        return static_cast<glm::uint32>( uvElement->GetDirectArray().GetCount() );
    else
        return 0;
}