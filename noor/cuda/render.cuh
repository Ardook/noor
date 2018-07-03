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
    CudaRenderTask _task;
    int _gpu_id;
    cudaStream_t _stream;

    void free() {
        checkNoorErrors( cudaSetDevice( _task._gpu_id ) );
        checkNoorErrors( cudaStreamDestroy( _stream ) );
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
        const CudaRenderTask& task
    ) :
        _host_hosek( hosek ),
        _host_camera( camera ),
        _host_spec( spec ),
        _task( task )
    {
        _gpu_id = _task._gpu_id;
        printf( "GPU %d selected\n", _gpu_id );
        checkNoorErrors( cudaSetDevice( _gpu_id ) );
        checkNoorErrors( cudaStreamCreate( &_stream ) );
        _host_texture_manager = myunique_ptr<CudaTextureManager>(
            new CudaTextureManager( payload.get() ) );
        _host_mesh_manager = myunique_ptr<CudaMeshManager>(
            new CudaMeshManager( payload.get() ) );
        _host_material_manager = myunique_ptr<CudaMaterialManager>(
            new CudaMaterialManager( payload.get() ) );
        _host_light_manager = myunique_ptr<CudaLightManager>(
            new CudaLightManager( payload.get() ) );
        _host_transform_manager = myunique_ptr<CudaTransformManager>(
            new CudaTransformManager( payload.get() ) );
        _host_skydome_manager = myunique_ptr<CudaSkyDomeManager>(
            new CudaSkyDomeManager( _host_texture_manager->getEnvTexture(),
            _host_spec._skydome_type ) );
        _host_bxdf_manager = myunique_ptr<CudaBxDFManager>(
            new CudaBxDFManager( 1 ) );
        _host_framebuffer_manager = myunique_ptr<CudaFrameBufferManager>(
            new CudaFrameBufferManager( _task ) );
        update_spec();
        update_camera();
        update_hoseksky();

        checkNoorErrors( NOOR::memcopy_symbol( &_mesh_manager,
                         _host_mesh_manager.get() ) );
        checkNoorErrors( NOOR::memcopy_symbol( &_material_manager,
                         _host_material_manager.get() ) );
        checkNoorErrors( NOOR::memcopy_symbol( &_texture_manager,
                         _host_texture_manager.get() ) );
        checkNoorErrors( NOOR::memcopy_symbol( &_light_manager,
                         _host_light_manager.get() ) );
        checkNoorErrors( NOOR::memcopy_symbol( &_transform_manager,
                         _host_transform_manager.get() ) );
        checkNoorErrors( NOOR::memcopy_symbol( &_skydome_manager,
                         _host_skydome_manager.get() ) );
        checkNoorErrors( NOOR::memcopy_symbol( &_bxdf_manager,
                         _host_bxdf_manager.get() ) );
        checkNoorErrors( NOOR::memcopy_symbol( &_framebuffer_manager,
                         _host_framebuffer_manager.get() ) );
    }

    const float4* getBuffer() const {
        return _host_framebuffer_manager->_buffer;
    }

    void update_spec() {
        checkNoorErrors( cudaSetDevice( _gpu_id ) );
        checkNoorErrors( NOOR::memcopy_symbol_async( &_constant_spec,
                         &_host_spec ) );
    }

    void update_camera() {
        checkNoorErrors( cudaSetDevice( _gpu_id ) );
        checkNoorErrors( NOOR::memcopy_symbol_async( &_constant_camera,
                         &_host_camera ) );
    }

    void update_hoseksky() {
        checkNoorErrors( cudaSetDevice( _gpu_id ) );
        checkNoorErrors( NOOR::memcopy_symbol_async( &_constant_hosek_sky,
                         &_host_hosek ) );
        update_skydome();
    }

    void update_skydome() {
        checkNoorErrors( cudaSetDevice( _gpu_id ) );
        if ( _host_spec._skydome_type == PHYSICAL )
            _host_skydome_manager->update();
    }
};

__device__ __managed__
float4 _device_lookAt;

class CudaRenderManager {
    // Cuda array mapped to OpenGL texture
    cudaArray* _buffer_array;
    // Cuda to OpenGL mapping resource
    cudaGraphicsResource* _glResource;
public:
    myunique_ptr<CudaRenderDevice> _gpu[2];
    int _num_gpus;
    size_t _shmsize;
    float4 _host_lookAt{ make_float4( 0 ) };
    CudaRenderManager( const std::unique_ptr<CudaPayload>& payload,
                       const CudaHosekSky& hosek,
                       const CudaCamera& camera,
                       const CudaSpec& spec,
                       GLuint* textureID,
                       float f = .5f
    ) : _num_gpus( spec._num_gpus ) {
        _shmsize = THREAD_N * spec._bvh_height * sizeof( uint );
        if ( _num_gpus == 1 ) f = 1;
        int job_size[] = { (int)( camera._h*f ), (int)( camera._h*( 1 - f ) ) };
        dim3 block( THREAD_W, THREAD_H, 1 );
        for ( int gpu_id = 0; gpu_id < _num_gpus; ++gpu_id ) {
            checkNoorErrors( cudaSetDevice( gpu_id ) );
            _gpu[gpu_id] = myunique_ptr<CudaRenderDevice>(
                new CudaRenderDevice(
                payload, hosek,
                camera, spec,
                CudaRenderTask( camera._w, job_size[gpu_id], gpu_id )
                )
                );
        }

        checkNoorErrors( cudaSetDevice( 0 ) );
        cudaStreamAttachMemAsync( _gpu[0]->_stream, &_device_lookAt );
        glGenTextures( 1, textureID );
        glBindTexture( GL_TEXTURE_2D, *textureID );
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
        glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F, camera._w, camera._h, 0,
                      GL_RGBA, GL_FLOAT, nullptr );
        checkNoorErrors( cudaGraphicsGLRegisterImage( &_glResource, *textureID,
                         GL_TEXTURE_2D,
                         cudaGraphicsMapFlagsWriteDiscard ) );
        checkNoorErrors( cudaGraphicsMapResources( 1, &_glResource, nullptr ) );
        checkNoorErrors( cudaGraphicsSubResourceGetMappedArray( &_buffer_array,
                         _glResource, 0, 0 ) );
        checkNoorErrors( cudaGraphicsUnmapResources( 1, &_glResource ) );
    }

    ~CudaRenderManager() {
        for ( int i = _num_gpus - 1; i >= 0; --i ) {
            _gpu[i].reset();
        }
    }

    void update() {
        checkNoorErrors( cudaDeviceSynchronize() );
        for ( int gpu_id = _num_gpus - 1; gpu_id >= 0; --gpu_id ) {
            int offset = gpu_id*_gpu[0]->_task._h;
            checkNoorErrors(
                cudaMemcpyToArrayAsync(
                _buffer_array,
                0,
                _gpu[0]->_task._h * gpu_id,
                _gpu[gpu_id]->getBuffer(),
                _gpu[gpu_id]->_task._size,
                //cudaMemcpyDeviceToDevice,
                cudaMemcpyDefault,
                _gpu[gpu_id]->_stream
            )
            );
        }
        _host_lookAt = _device_lookAt;
    }

    cudaStream_t getStream( int gpu_id )const {
        return _gpu[gpu_id]->_stream;
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