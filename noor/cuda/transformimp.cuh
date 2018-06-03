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
**/
#ifndef CUDATRANSFORMIMP_CUH
#define CUDATRANSFORMIMP_CUH

struct CudaTransformManager {
    cudaTextureObject_t _normal_row0_texobj;
    cudaTextureObject_t _normal_row1_texobj;
    cudaTextureObject_t _normal_row2_texobj;

    cudaTextureObject_t _world_to_object_row0_texobj;
    cudaTextureObject_t _world_to_object_row1_texobj;
    cudaTextureObject_t _world_to_object_row2_texobj;

    cudaTextureObject_t _object_to_world_row0_texobj;
    cudaTextureObject_t _object_to_world_row1_texobj;
    cudaTextureObject_t _object_to_world_row2_texobj;

    float4* _normal_row0;
    float4* _normal_row1;
    float4* _normal_row2;

    float4* _world_to_object_row0;
    float4* _world_to_object_row1;
    float4* _world_to_object_row2;

    float4* _object_to_world_row0;
    float4* _object_to_world_row1;
    float4* _object_to_world_row2;

    CudaTransformManager() = default;
    __host__
        CudaTransformManager( const CudaPayload* payload ) {
        size_t size_bytes = payload->_transforms_size_bytes;
        checkNoorErrors(NOOR::malloc(  &_normal_row0, size_bytes ));
        checkNoorErrors(NOOR::malloc(  &_normal_row1, size_bytes ));
        checkNoorErrors(NOOR::malloc(  &_normal_row2, size_bytes ));

        checkNoorErrors(NOOR::malloc(  &_world_to_object_row0, size_bytes ));
        checkNoorErrors(NOOR::malloc(  &_world_to_object_row1, size_bytes ));
        checkNoorErrors(NOOR::malloc(  &_world_to_object_row2, size_bytes ));

        checkNoorErrors(NOOR::malloc(  &_object_to_world_row0, size_bytes ));
        checkNoorErrors(NOOR::malloc(  &_object_to_world_row1, size_bytes ));
        checkNoorErrors(NOOR::malloc(  &_object_to_world_row2, size_bytes ));

        // copy
        checkNoorErrors(NOOR::memcopy( _normal_row0, (void*) &payload->_normal_row0[0], size_bytes ));
        checkNoorErrors(NOOR::memcopy( _normal_row1, (void*) &payload->_normal_row1[0], size_bytes ));
        checkNoorErrors(NOOR::memcopy( _normal_row2, (void*) &payload->_normal_row2[0], size_bytes ));

        checkNoorErrors(NOOR::memcopy( _world_to_object_row0, (void*) &payload->_world_to_object_row0[0], size_bytes ));
        checkNoorErrors(NOOR::memcopy( _world_to_object_row1, (void*) &payload->_world_to_object_row1[0], size_bytes ));
        checkNoorErrors(NOOR::memcopy( _world_to_object_row2, (void*) &payload->_world_to_object_row2[0], size_bytes ));

        checkNoorErrors(NOOR::memcopy( _object_to_world_row0, (void*) &payload->_object_to_world_row0[0], size_bytes ));
        checkNoorErrors(NOOR::memcopy( _object_to_world_row1, (void*) &payload->_object_to_world_row1[0], size_bytes ));
        checkNoorErrors(NOOR::memcopy( _object_to_world_row2, (void*) &payload->_object_to_world_row2[0], size_bytes ));

        // create texture object
        checkNoorErrors(NOOR::create_1d_texobj( &_normal_row0_texobj, _normal_row0, size_bytes, NOOR::_float4_channelDesc ));
        checkNoorErrors(NOOR::create_1d_texobj( &_normal_row1_texobj, _normal_row1, size_bytes, NOOR::_float4_channelDesc ));
        checkNoorErrors(NOOR::create_1d_texobj( &_normal_row2_texobj, _normal_row2, size_bytes, NOOR::_float4_channelDesc ));

        checkNoorErrors(NOOR::create_1d_texobj( &_world_to_object_row0_texobj, _world_to_object_row0, size_bytes, NOOR::_float4_channelDesc ));
        checkNoorErrors(NOOR::create_1d_texobj( &_world_to_object_row1_texobj, _world_to_object_row1, size_bytes, NOOR::_float4_channelDesc ));
        checkNoorErrors(NOOR::create_1d_texobj( &_world_to_object_row2_texobj, _world_to_object_row2, size_bytes, NOOR::_float4_channelDesc ));

        checkNoorErrors(NOOR::create_1d_texobj( &_object_to_world_row0_texobj, _object_to_world_row0, size_bytes, NOOR::_float4_channelDesc ));
        checkNoorErrors(NOOR::create_1d_texobj( &_object_to_world_row1_texobj, _object_to_world_row1, size_bytes, NOOR::_float4_channelDesc ));
        checkNoorErrors(NOOR::create_1d_texobj( &_object_to_world_row2_texobj, _object_to_world_row2, size_bytes, NOOR::_float4_channelDesc ));
    }

    __host__
        void free() {
        checkNoorErrors( cudaDestroyTextureObject( _normal_row0_texobj ));
        checkNoorErrors( cudaDestroyTextureObject( _normal_row1_texobj ));
        checkNoorErrors( cudaDestroyTextureObject( _normal_row2_texobj ));

        checkNoorErrors( cudaDestroyTextureObject( _world_to_object_row0_texobj ));
        checkNoorErrors( cudaDestroyTextureObject( _world_to_object_row1_texobj ));
        checkNoorErrors( cudaDestroyTextureObject( _world_to_object_row2_texobj ));

        checkNoorErrors( cudaDestroyTextureObject( _object_to_world_row0_texobj ));
        checkNoorErrors( cudaDestroyTextureObject( _object_to_world_row1_texobj ));
        checkNoorErrors( cudaDestroyTextureObject( _object_to_world_row2_texobj ));

        checkNoorErrors(cudaFree( _normal_row0 ));
        checkNoorErrors(cudaFree( _normal_row1 ));
        checkNoorErrors(cudaFree( _normal_row2 ));

        checkNoorErrors(cudaFree( _world_to_object_row0 ));
        checkNoorErrors(cudaFree( _world_to_object_row1 ));
        checkNoorErrors(cudaFree( _world_to_object_row2 ));

        checkNoorErrors(cudaFree( _object_to_world_row0 ));
        checkNoorErrors(cudaFree( _object_to_world_row1 ));
        checkNoorErrors(cudaFree( _object_to_world_row2 ));
    }

    __device__
        void getWorldToObjectTransformation( CudaTransform& T, uint instance_idx ) const {
        const float4 row0 = tex1Dfetch<float4>( _world_to_object_row0_texobj, instance_idx );
        const float4 row1 = tex1Dfetch<float4>( _world_to_object_row1_texobj, instance_idx );
        const float4 row2 = tex1Dfetch<float4>( _world_to_object_row2_texobj, instance_idx );
        T = CudaTransform( row0, row1, row2 );
    }

    __device__
        void getObjectToWorldTransformation( CudaTransform& T, uint instance_idx ) const {
        const float4 row0 = tex1Dfetch<float4>( _object_to_world_row0_texobj, instance_idx );
        const float4 row1 = tex1Dfetch<float4>( _object_to_world_row1_texobj, instance_idx );
        const float4 row2 = tex1Dfetch<float4>( _object_to_world_row2_texobj, instance_idx );
        T = CudaTransform( row0, row1, row2 );
    }

    __device__
        CudaTransform getWorldToObjectTransformation( uint instance_idx ) const {
        const float4 row0 = tex1Dfetch<float4>( _world_to_object_row0_texobj, instance_idx );
        const float4 row1 = tex1Dfetch<float4>( _world_to_object_row1_texobj, instance_idx );
        const float4 row2 = tex1Dfetch<float4>( _world_to_object_row2_texobj, instance_idx );
        return CudaTransform( row0, row1, row2 );
    }

    __device__
        CudaTransform getObjectToWorldTransformation( uint instance_idx ) const {
        const float4 row0 = tex1Dfetch<float4>( _object_to_world_row0_texobj, instance_idx );
        const float4 row1 = tex1Dfetch<float4>( _object_to_world_row1_texobj, instance_idx );
        const float4 row2 = tex1Dfetch<float4>( _object_to_world_row2_texobj, instance_idx );
        return CudaTransform( row0, row1, row2 );
    }

    __device__
        CudaTransform getNormalTransformation( uint instance_idx ) const {
        const float3 row0 = make_float3( tex1Dfetch<float4>( _normal_row0_texobj, instance_idx ) );
        const float3 row1 = make_float3( tex1Dfetch<float4>( _normal_row1_texobj, instance_idx ) );
        const float3 row2 = make_float3( tex1Dfetch<float4>( _normal_row2_texobj, instance_idx ) );
        return CudaTransform( row0, row1, row2 );
    }
};
__constant__
CudaTransformManager _transform_manager;
#endif /* CUDATRANSFORM_CUH */
