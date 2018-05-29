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

struct CudaRenderManager {
    int _gpuID;
    SkydomeType _skydome_type;
    CudaCamera _camera;
    CudaHosekSky& _host_hosek_sky;
    myunique_ptr<CudaMeshManager> _host_mesh_manager;
    myunique_ptr<CudaMaterialManager> _host_material_manager;
    myunique_ptr<CudaTextureManager> _host_texture_manager;
    myunique_ptr<CudaLightManager> _host_light_manager;
    myunique_ptr<CudaTransformManager> _host_transform_manager;
    myunique_ptr<CudaSkyDomeManager> _host_skydome_manager;
    myunique_ptr<CudaFrameBufferManager> _framebuffer_manager;

    ~CudaRenderManager() {
        _host_transform_manager.reset();
        _host_light_manager.reset();
        _host_mesh_manager.reset();
        _host_material_manager.reset();
        _host_skydome_manager.reset();
        _host_texture_manager.reset();
        _framebuffer_manager.reset();
        cudaDeviceReset();
    }
    CudaRenderManager( const std::unique_ptr<CudaPayload>& payload, CudaHosekSky& hosek_sky, int gpuID, SkydomeType skydome_type, GLuint* textureID, uint w, uint h ) :
        _gpuID( gpuID )
        , _skydome_type( skydome_type )
        , _host_hosek_sky( hosek_sky ) {
        cudaSetDevice( gpuID );
        printf( "GPU %d selected\n", gpuID );
        _host_texture_manager = myunique_ptr<CudaTextureManager>( new CudaTextureManager( payload.get() ) );
        _host_mesh_manager = myunique_ptr<CudaMeshManager>( new CudaMeshManager( payload.get() ) );
        _host_material_manager = myunique_ptr<CudaMaterialManager>( new CudaMaterialManager( payload.get() ) );
        _host_light_manager = myunique_ptr<CudaLightManager>( new CudaLightManager( payload.get() ) );
        _host_transform_manager = myunique_ptr<CudaTransformManager>( new CudaTransformManager( payload.get() ) );
        _host_skydome_manager = myunique_ptr<CudaSkyDomeManager>( new CudaSkyDomeManager( _host_texture_manager->getEnvTexture(), _skydome_type ) );
        _framebuffer_manager = myunique_ptr<CudaFrameBufferManager>( new CudaFrameBufferManager( textureID, w, h ) );

        NOOR::memcopy_symbol( &_mesh_manager, _host_mesh_manager.get() );
        NOOR::memcopy_symbol( &_material_manager, _host_material_manager.get() );
        NOOR::memcopy_symbol( &_texture_manager, _host_texture_manager.get() );
        NOOR::memcopy_symbol( &_light_manager, _host_light_manager.get() );
        NOOR::memcopy_symbol( &_transform_manager, _host_transform_manager.get() );
        NOOR::memcopy_symbol( &_skydome_manager, _host_skydome_manager.get() );
    }

    /*void init_framebuffer( GLuint* textureID, uint w, uint h ) {
        _framebuffer_manager = myunique_ptr<CudaFrameBufferManager>( new CudaFrameBufferManager( textureID, w, h ) );
    }*/

    void update_skydome() {
        if ( _skydome_type == PHYSICAL ) _host_skydome_manager->update();
    }

    void update_camera( const glm::mat4& cameraToWorld, const glm::mat4& rasterToCamera, int w, int h, float lens_radius, float focal_length, CameraType type ) {
        _camera.update( cameraToWorld, rasterToCamera, w, h, lens_radius, focal_length, type );
        _camera._center = w * ( h >> 1 ) + ( w >> 1 );
        NOOR::memcopy_symbol_async( &_constant_camera, &_camera );
    }

    void update_hoseksky() {
        NOOR::memcopy_symbol_async( &_constant_hosek_sky, &_host_hosek_sky );
        update_skydome();
    }
};
#endif /* CUDARENDERMANAGER_CUH */