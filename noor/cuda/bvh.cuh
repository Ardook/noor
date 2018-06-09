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
#ifndef CUDABVH_CUH
#define CUDABVH_CUH


class CudaBVHNode {
public:
    __device__
        void load( uint node_idx ) {
        _max_axis_count = _mesh_manager.getBvhMaxAxisCount( node_idx );
        _min_right_start = _mesh_manager.getBvhMinRightStart( node_idx );
    }

    __device__
        float3 get_min() const {
        float3 result;
        result.x = __uint_as_float( _min_right_start.x );
        result.y = __uint_as_float( _min_right_start.y );
        result.z = __uint_as_float( _min_right_start.z );
        return result;
    }
    __device__
        float3 get_max() const {
        float3 result;
        result.x = __uint_as_float( _max_axis_count.x );
        result.y = __uint_as_float( _max_axis_count.y );
        result.z = __uint_as_float( _max_axis_count.z );
        return result;
    }
    __device__
        uint get_count() const {
        return ( _max_axis_count.w & COUNT_MASK );
    }
    __device__
        uint get_tri_count() const {
        return ( _max_axis_count.w & TRI_COUNT_MASK );
    }
    __device__
        uint get_light_idx() const {
        return ( _max_axis_count.w & LIGHT_MASK );
    }
    __device__
        uint get_instance_id() const {
        return ( _max_axis_count.w & TRANS_MASK );
    }
    __device__
        uint get_axis() const {
        return ( _max_axis_count.w & AXIS_MASK );
    }
    __device__
        uint get_right() const {
        return _min_right_start.w;
    }
    __device__
        uint get_start() const {
        return _min_right_start.w;
    }

    __device__
        bool is_mesh_node() const {
        return ( ( MESH_NODE_MASK & _max_axis_count.w ) != 0 );
    }
    __device__
        bool is_tri_leaf() const {
        return ( ( TRI_LEAF_MASK & _max_axis_count.w ) != 0 );
    }
    __device__
        bool is_mesh_leaf() const {
        return ( ( MESH_LEAF_MASK & _max_axis_count.w ) != 0 );
    }
    __device__
        bool is_mesh_instance() const {
        return ( ( MESH_INSTANCE_MASK & _max_axis_count.w ) != 0 );
    }
    __device__
        bool is_mesh_inner() const {
        return ( ( MESH_INNER_MASK & _max_axis_count.w ) != 0 );
    }
    __device__
        bool is_light_node() const {
        return ( ( LIGHT_NODE_MASK & _max_axis_count.w ) != 0 );
    }
    __device__
        bool intersectBBox( const CudaRay& ray ) {
        CudaBBox bbox( get_min(), get_max() );
        return bbox.intersect( ray );
    }
    __device__
        bool intersectLeaf(
        const CudaRay& ray,
        CudaIntersection& intersection
        ) {
        const uint start = get_start();
        const uint count = start + get_tri_count();
        CudaTriangle tri;
        bool hit = false;
        for ( uint tri_idx = start; tri_idx < count; ++tri_idx ) {
            if ( tri.intersect( ray, intersection, tri_idx ) )
                hit = true;
        }
        return hit;
    }
    __device__
        bool intersectLeaf( const CudaRay& ray ) {
        const uint start = get_start();
        const uint count = start + get_tri_count();
        CudaTriangle tri;
        for ( uint tri_idx = start; tri_idx < count; ++tri_idx ) {
            if ( tri.intersect( ray, tri_idx ) ) {
                return true;
            }
        }
        return false;
    }
    uint4		_min_right_start;
    uint4		_max_axis_count;
};

__device__ __forceinline__
bool intersect( const CudaRay& ray, CudaIntersection& I ) {
    bool hit = false;
    uint currentNodeIndex;
    CudaBVHNode current_node;
    CudaStack<uint> stack( &shstack[_constant_spec._bvh_height * I.getTid()] );
    CudaRay lray = ray;
    uint ins_idx = 0;
    stack.push( _constant_spec._bvh_root_node );
    while ( !stack.isEmpty() ) {
        currentNodeIndex = stack.pop();
        current_node.load( currentNodeIndex );
        if ( current_node.is_mesh_node() ) {
            lray = ray;
        }
        if ( !current_node.intersectBBox( lray ) )continue;
        if ( current_node.is_light_node() ) {
            const int light_idx = current_node.get_light_idx();
            if ( _light_manager.intersect( ray, light_idx ) ) {
                hit = true;
                I._ins_idx = light_idx;
                I.setMaterialType( MaterialType( MESHLIGHT | EMITTER) );
            }
            continue;
        } else if ( current_node.is_mesh_instance() ) {
            ins_idx = current_node.get_instance_id();
            lray = ray.transformToObject( ins_idx );
            stack.push( current_node.get_right() );
        } else if ( current_node.is_mesh_leaf() ) {
            for ( uint i = 1; i <= current_node.get_count(); ++i ) {
                stack.push( currentNodeIndex + i );
            }
        } else if ( current_node.is_tri_leaf() ) {
            if ( current_node.intersectLeaf( lray, I ) ) {
                hit = true;
                ray.setTmax( lray.getTmax() );
                I._ins_idx = ins_idx;
            }
        } else {
            if ( lray.getPosneg()[current_node.get_axis()] ) {
                stack.push( currentNodeIndex + 1 );
                stack.push( current_node.get_right() );
            } else {
                stack.push( current_node.get_right() );
                stack.push( currentNodeIndex + 1 );
            }
        }
    }
    if ( hit ) {
        I._p = ray.pointAtParameter( ray.getTmax() );
        I.updateIntersection( ray );
        I._eta = _material_manager.getIorDielectric( I ).x;
    }
    return hit;
}

__device__ __forceinline__
bool intersectP( const CudaRay& ray, const CudaIntersection& I ) {
    uint currentNodeIndex;
    CudaBVHNode current_node;
    CudaStack<uint> stack( &shstack[_constant_spec._bvh_height * I.getTid()] );
    CudaRay lray = ray;
    stack.push( _constant_spec._bvh_root_node );
    while ( !stack.isEmpty() ) {
        currentNodeIndex = stack.pop();
        current_node.load( currentNodeIndex );
        if ( current_node.is_mesh_node() ) {
            lray = ray;
        }
        if ( current_node.intersectBBox( lray ) ) {
            if ( current_node.is_light_node() ) {
                const int index = current_node.get_light_idx();
                if ( _light_manager.intersect( ray, index ) ) {
                    return true;
                }
            } else if ( current_node.is_mesh_instance() ) {
                lray = ray.transformToObject( current_node.get_instance_id() );
                stack.push( current_node.get_right() );
            } else if ( current_node.is_mesh_leaf() ) {
                for ( uint i = 1; i <= current_node.get_count(); ++i ) {
                    stack.push( currentNodeIndex + i );
                }
            } else if ( current_node.is_tri_leaf() ) {
                if ( current_node.intersectLeaf( lray ) ) {
                    return true;
                }
            } else {
                if ( lray.getPosneg()[current_node.get_axis()] ) {
                    stack.push( currentNodeIndex + 1 );
                    stack.push( current_node.get_right() );
                } else {
                    stack.push( current_node.get_right() );
                    stack.push( currentNodeIndex + 1 );
                }
            }
        }
    }
    return false;
}


#endif /* CUDABVH_CUH */