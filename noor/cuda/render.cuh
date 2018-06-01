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
    myunique_ptr<CudaFrameBufferManager> _framebuffer_manager;

    ~CudaRenderManager() {
        _host_transform_manager.reset();
        _host_light_manager.reset();
        _host_mesh_manager.reset();
        _host_material_manager.reset();
        _host_skydome_manager.reset();
        _host_texture_manager.reset();
        _host_bxdf_manager.reset();
        _framebuffer_manager.reset();
        cudaDeviceReset();
    }
    CudaRenderManager( const std::unique_ptr<CudaPayload>& payload,
                       const CudaHosekSky& hosek,
                       const CudaCamera& camera,
                       const CudaSpec& spec,
                       GLuint* textureID ) :
        _host_hosek( hosek ),
        _host_camera( camera ),
        _host_spec( spec ) {
        cudaSetDevice( _host_spec._gpuID );
        printf( "GPU %d selected\n", _host_spec._gpuID );
        _host_texture_manager = myunique_ptr<CudaTextureManager>( new CudaTextureManager( payload.get() ) );
        _host_mesh_manager = myunique_ptr<CudaMeshManager>( new CudaMeshManager( payload.get() ) );
        _host_material_manager = myunique_ptr<CudaMaterialManager>( new CudaMaterialManager( payload.get() ) );
        _host_light_manager = myunique_ptr<CudaLightManager>( new CudaLightManager( payload.get() ) );
        _host_transform_manager = myunique_ptr<CudaTransformManager>( new CudaTransformManager( payload.get() ) );
        _host_bxdf_manager = myunique_ptr<CudaBxDFManager>( new CudaBxDFManager(1) );
        _host_skydome_manager = myunique_ptr<CudaSkyDomeManager>( new CudaSkyDomeManager( _host_texture_manager->getEnvTexture(), _host_spec._skydome_type ) );
        _framebuffer_manager = myunique_ptr<CudaFrameBufferManager>( new CudaFrameBufferManager( textureID, _host_camera._w, _host_camera._h ) );
        update_spec();
        update_camera();
        update_hoseksky();

        NOOR::memcopy_symbol( &_mesh_manager, _host_mesh_manager.get() );
        NOOR::memcopy_symbol( &_material_manager, _host_material_manager.get() );
        NOOR::memcopy_symbol( &_texture_manager, _host_texture_manager.get() );
        NOOR::memcopy_symbol( &_light_manager, _host_light_manager.get() );
        NOOR::memcopy_symbol( &_transform_manager, _host_transform_manager.get() );
        NOOR::memcopy_symbol( &_skydome_manager, _host_skydome_manager.get() );
        NOOR::memcopy_symbol( &_constant_bxdf_manager, _host_bxdf_manager.get() );
    }

    void update_spec() {
        NOOR::memcopy_symbol_async( &_constant_spec, &_host_spec );
    }

    void update_camera() {
        NOOR::memcopy_symbol_async( &_constant_camera, &_host_camera );
    }

    void update_hoseksky() {
        NOOR::memcopy_symbol_async( &_constant_hosek_sky, &_host_hosek );
        update_skydome();
    }

    void update_skydome() {
        if ( _host_spec._skydome_type == PHYSICAL ) _host_skydome_manager->update();
    }
};
#endif /* CUDARENDERMANAGER_CUH */