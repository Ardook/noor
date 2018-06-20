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
#ifndef CUDARENDERMANAGER_CUH
#define CUDARENDERMANAGER_CUH

template<typename T>
struct mydeleter {
    void operator()( T* t ) {
        t->free();
    }
};
template<class T>
using myunique_ptr = std::unique_ptr< T, mydeleter<T> >;

struct CudaRenderDevice {
    int _gpuID;
    const CudaCamera& _host_camera;
    const CudaHosekSky& _host_hosek;
    const CudaSpec& _host_spec;
    myunique_ptr<CudaMeshManager> _host_mesh_manager;
    myunique_ptr<CudaMaterialManager> _host_material_manager;
    myunique_ptr<CudaTextureManager> _host_texture_manager;
    myunique_ptr<CudaLightManager> _host_light_manager;
    myunique_ptr<CudaTransformManager> _host_transform_manager;
    myunique_ptr<CudaSkyDomeManager> _host_skydome_manager;
    myunique_ptr<CudaBxDFManager> _host_bxdf_manager;
    myunique_ptr<CudaFrameBufferManager> _host_framebuffer_manager;

    void free() {
        checkNoorErrors( cudaSetDevice( _gpuID ) );
        _host_transform_manager.reset();
        _host_light_manager.reset();
        _host_mesh_manager.reset();
        _host_material_manager.reset();
        _host_skydome_manager.reset();
        _host_texture_manager.reset();
        _host_bxdf_manager.reset();
        _host_framebuffer_manager.reset();
        checkNoorErrors( cudaDeviceReset() );
    }
    CudaRenderDevice(
        const std::unique_ptr<CudaPayload>& payload,
        const CudaHosekSky& hosek,
        const CudaCamera& camera,
        const CudaSpec& spec,
        int gpuID
    ) :
        _gpuID( gpuID ),
        _host_hosek( hosek ),
        _host_camera( camera ),
        _host_spec( spec )
    {
        bool managed = spec._num_gpus > 1;
        printf( "GPU %d selected\n", _gpuID );
        checkNoorErrors( cudaSetDevice( _gpuID ) );
        _host_texture_manager = myunique_ptr<CudaTextureManager>( new CudaTextureManager( payload.get() ) );
        _host_mesh_manager = myunique_ptr<CudaMeshManager>( new CudaMeshManager( payload.get() ) );
        _host_material_manager = myunique_ptr<CudaMaterialManager>( new CudaMaterialManager( payload.get() ) );
        _host_light_manager = myunique_ptr<CudaLightManager>( new CudaLightManager( payload.get() ) );
        _host_transform_manager = myunique_ptr<CudaTransformManager>( new CudaTransformManager( payload.get() ) );
        _host_skydome_manager = myunique_ptr<CudaSkyDomeManager>( new CudaSkyDomeManager( _host_texture_manager->getEnvTexture(), _host_spec._skydome_type ) );
        _host_bxdf_manager = myunique_ptr<CudaBxDFManager>( new CudaBxDFManager( 1 ) );
        _host_framebuffer_manager = myunique_ptr<CudaFrameBufferManager>( new CudaFrameBufferManager( camera._w, camera._h, managed ) );
        update_spec();
        update_camera();
        update_hoseksky();

        checkNoorErrors( NOOR::memcopy_symbol( &_mesh_manager, _host_mesh_manager.get() ) );
        checkNoorErrors( NOOR::memcopy_symbol( &_material_manager, _host_material_manager.get() ) );
        checkNoorErrors( NOOR::memcopy_symbol( &_texture_manager, _host_texture_manager.get() ) );
        checkNoorErrors( NOOR::memcopy_symbol( &_light_manager, _host_light_manager.get() ) );
        checkNoorErrors( NOOR::memcopy_symbol( &_transform_manager, _host_transform_manager.get() ) );
        checkNoorErrors( NOOR::memcopy_symbol( &_skydome_manager, _host_skydome_manager.get() ) );
        checkNoorErrors( NOOR::memcopy_symbol( &_bxdf_manager, _host_bxdf_manager.get() ) );
        checkNoorErrors( NOOR::memcopy_symbol( &_framebuffer_manager, _host_framebuffer_manager.get() ) );
    }

    const float4* getBuffer() const {
        return _host_framebuffer_manager->_buffer;
    }

    void update_spec() {
        checkNoorErrors( cudaSetDevice( _gpuID ) );
        checkNoorErrors( NOOR::memcopy_symbol_async( &_constant_spec, &_host_spec ) );
    }

    void update_camera() {
        checkNoorErrors( cudaSetDevice( _gpuID ) );
        checkNoorErrors( NOOR::memcopy_symbol_async( &_constant_camera, &_host_camera ) );
    }

    void update_hoseksky() {
        checkNoorErrors( cudaSetDevice( _gpuID ) );
        checkNoorErrors( NOOR::memcopy_symbol_async( &_constant_hosek_sky, &_host_hosek ) );
        update_skydome();
    }

    void update_skydome() {
        checkNoorErrors( cudaSetDevice( _gpuID ) );
        if ( _host_spec._skydome_type == PHYSICAL ) _host_skydome_manager->update();
    }
};

__device__ __managed__
float4 _device_lookAt;

class CudaRenderManager {
    // Cuda array mapped to OpenGL texture
    cudaArray* _buffer_array;
    // Cuda to OpenGL mapping resource
    cudaGraphicsResource* _glResource;
    size_t _size_bytes;
    int _buffer_offset;
    myunique_ptr<CudaRenderDevice> _gpu[2];
public:
    int _w, _h;
    int _num_gpus;
    size_t _shmsize;
    float4 _host_lookAt{ make_float4( 0 ) };
    CudaRenderManager( const std::unique_ptr<CudaPayload>& payload,
                       const CudaHosekSky& hosek,
                       const CudaCamera& camera,
                       const CudaSpec& spec,
                       GLuint* textureID
    ) : _num_gpus( spec._num_gpus ),
        _w( camera._w ),
        _h( camera._h )
    {
        _shmsize = THREAD_N * spec._bvh_height * sizeof( uint );
        _size_bytes = _w * _h * sizeof( float4 ) / _num_gpus;
        _buffer_offset = _h / _num_gpus;
        for ( int i = 0; i < _num_gpus; ++i ) {
            checkNoorErrors( cudaSetDevice( i ) );
            _gpu[i] = myunique_ptr<CudaRenderDevice>( new CudaRenderDevice(
                payload, hosek,
                camera, spec,
                i ) );
        }
        checkNoorErrors( cudaSetDevice( 0 ) );
        glGenTextures( 1, textureID );
        glBindTexture( GL_TEXTURE_2D, *textureID );
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
        glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F, _w, _h, 0, GL_RGBA, GL_FLOAT, nullptr );
        checkNoorErrors( cudaGraphicsGLRegisterImage( &_glResource, *textureID,
                         GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard ) );
    }

    ~CudaRenderManager() {
        for ( int i = _num_gpus - 1; i >= 0; --i ) {
            _gpu[i].reset();
        }
    }

    void update() {
        checkNoorErrors( cudaSetDevice( 0 ) );
        checkNoorErrors( cudaGraphicsMapResources( 1, &_glResource, nullptr ) );
        checkNoorErrors( cudaGraphicsSubResourceGetMappedArray( &_buffer_array, _glResource, 0, 0 ) );
        for ( int i = 0; i < _num_gpus; ++i ) {
            checkNoorErrors( cudaMemcpyToArrayAsync( _buffer_array, 0, i*_buffer_offset, _gpu[i]->getBuffer(), _size_bytes, cudaMemcpyDeviceToDevice ) );
        }
        checkNoorErrors( cudaGraphicsUnmapResources( 1, &_glResource, nullptr ) );
    }
    void update_spec() {
        for ( int i = _num_gpus - 1; i >= 0; --i ) {
            _gpu[i]->update_spec();
        }
    }

    void update_camera() {
        for ( int i = _num_gpus - 1; i >= 0; --i ) {
            _gpu[i]->update_camera();
        }
    }

    void update_hoseksky() {
        for ( int i = _num_gpus - 1; i >= 0; --i ) {
            _gpu[i]->update_hoseksky();
        }
    }

    void update_skydome() {
        for ( int i = _num_gpus - 1; i >= 0; --i ) {
            _gpu[i]->update_skydome();
        }
    }

    void get_lookAt( float4& lookAt ) {
        lookAt = _host_lookAt;
    }
};
#endif /* CUDARENDERMANAGER_CUH */