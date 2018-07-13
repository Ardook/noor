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
#include "scene.h"
#include "model.h"
#include "bvh.h"
#include "camera.h"
#include "hosek.h"

Scene::~Scene() {
    device_free_memory();
}
Scene::Scene( const Spec& spec ) :
    _host_spec( spec ),
    _frameCount( 1 ) {
    load();
}

glm::uint32 Scene::getWidthPixels()const { return _host_spec._camera_spec._w; }
glm::uint32 Scene::getHeightPixels()const { return _host_spec._camera_spec._h; }
void Scene::reset( int w, int h ) { _camera->reset( w, h ); }
void Scene::mouse( int button, int action ) { _camera->mouse( button, action ); }
void Scene::motion( int x, int y ) { _camera->motion( x, y ); }
const glm::mat4& Scene::getViewMatrix() const { return _camera->getViewMatrix(); }
const glm::mat4& Scene::getProjectionMatrix() const { return _camera->getProjectionMatrix(); }

void Scene::load() {
    _cuda_payload = std::make_unique<CudaPayload>();
    Stat stat;
    _model = std::make_unique<Model>( _host_spec, stat );
    _model->loadCudaPayload( _cuda_payload );
    _scene_bbox = _model->getSceneBBox();
    _scene_radius = _scene_bbox.radius();
    _scene_bias = 0.0001f * _scene_radius;
    _spec = std::make_unique<CudaSpec>();
    _spec->_bvh_height = stat._height;
    _spec->_bounces = _host_spec._camera_spec._bounces;
    _spec->_rr = _host_spec._camera_spec._rr;
    _spec->_white = make_float3( 1.0f, 1.0f, 1.0f );
    _spec->_black = make_float3( 0.0f, 0.0f, 0.0f );
    _spec->_reflection_bias = _scene_bias;
    _spec->_shadow_bias = 0.0001f * _scene_radius;
    _spec->_world_radius = _scene_radius;
    _spec->_wr2 = _scene_radius * _scene_radius;
    _spec->_bvh_root_node = _cuda_payload->_bvh_root_node;
    _spec->_num_gpus = _host_spec._num_gpus;
    _spec->_skydome_type = (SkydomeType)_host_spec._model_spec._skydome_type;
    _camera = std::make_unique<Camera>(
        *this
        , _model->_eye
        , _model->_lookAt
        , _model->_up
        , _model->_fov
        , _model->_orthozoom
        , _host_spec._camera_spec._lens_radius
        , _host_spec._camera_spec._focal_length
        , _host_spec._camera_spec._w
        , _host_spec._camera_spec._h
        );
    _hosek = std::make_unique<HosekSky>( *this );
    if ( _cuda_payload->_num_lights < 2 )
        _spec->enable_sky_light();
    else
        _spec->disable_sky_light();
    std::cout << stat << std::endl;
}

void Scene::updateSky( float theta, float phi ) const {
    _hosek->update( theta, phi );
}

void Scene::enableDebugSky() const {
    _spec->enable_debug_sky();
}

void Scene::disableDebugSky() const {
    _spec->disable_debug_sky();
}

void Scene::setCameraType( CameraType type ) const {
    _camera->setCameraType( type );
}

void Scene::enableSky() const {
    _spec->enable_sky_light();
}

void Scene::disableSky() const {
    _spec->disable_sky_light();
}

void Scene::enableMIS() const {
    _spec->enable_mis();
}

void Scene::disableMIS() const {
    _spec->disable_mis();
}
bool Scene::isSkydomeEnabled() const {
    return _spec->is_sky_light_enabled();
}

void Scene::updateCudaSpec() {
    if ( _spec->_outofsync ) {
        update_cuda_spec();
        _spec->_outofsync = false;
        resetCudaRenderBuffer();
    }
}

void Scene::updateCudaSky() {
    if ( _hosek->_outofsync ) {
        _hosek->updateCudaHosek();
        resetCudaRenderBuffer();
    }
}

void Scene::updateCudaCamera() {
    if ( _camera->_outofsync ) {
        _camera->updateCudaCamera();
        resetCudaRenderBuffer();
    }
}

void Scene::initCudaContext( GLuint* cudaTextureID ) {
    load_cuda_data( _cuda_payload, 
                    _hosek->_cuda_hosek_sky, 
                    _camera->_cuda_camera,
                    *_spec.get(), 
                    cudaTextureID );
}

void Scene::path_tracer() {
    if ( _spec->_debug_sky && _spec->_num_gpus == 1 )
        debug_skydome( _frameCount, _camera->_w, _camera->_h );
    else
        cuda_path_tracer( _frameCount );
}
