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
    _spec( spec ),
    _frameCount( 1 )
{
    load();
}

glm::uint32 Scene::getWidthPixels()const { return _spec._camera_spec._w; }
glm::uint32 Scene::getHeightPixels()const { return _spec._camera_spec._h; }
void Scene::reset( int w, int h ) { _camera->reset( w, h ); }
void Scene::mouse( int button, int action ) { _camera->mouse( button, action ); }
void Scene::motion( int x, int y ) { _camera->motion( x, y ); }
const glm::mat4& Scene::getViewMatrix() const { return _camera->getViewMatrix(); }
const glm::mat4& Scene::getProjectionMatrix() const { return _camera->getProjectionMatrix(); }

void Scene::load() {
    _cuda_payload = std::make_unique<CudaPayload>();
    Stat stat;
    _model = std::make_unique<Model>( _spec, stat );
    _model->loadCudaPayload( _cuda_payload );
    _scene_bbox = _model->getSceneBBox();
    _scene_radius = _scene_bbox.radius();
    _scene_bias = 0.0001f * _scene_radius;
    _cuda_spec = std::make_unique<CudaSpec>();
    _cuda_spec->_bvh_height = stat._height;
    _cuda_spec->_bounces = _spec._camera_spec._bounces;
    _cuda_spec->_rr = _spec._camera_spec._rr;
    _cuda_spec->_white = make_float3( 1.0f, 1.0f, 1.0f );
    _cuda_spec->_black = make_float3( 0.0f, 0.0f, 0.0f );
    _cuda_spec->_reflection_bias = _scene_bias;
    _cuda_spec->_shadow_bias = 0.0001f;
    _cuda_spec->_world_radius = _scene_radius;
    _cuda_spec->_bvh_root_node = _cuda_payload->_bvh_root_node;
    _spec._camera_spec._eye = _model->_eye;
    _spec._camera_spec._up = _model->_up;
    _spec._camera_spec._lookAt = _model->_lookAt;
    _spec._camera_spec._fov = _model->_fov;
    _camera = std::make_unique<Camera>(
        *this
        , _spec._camera_spec._eye
        , _spec._camera_spec._lookAt
        , _spec._camera_spec._up
        , _spec._camera_spec._fov
        , _model->_orthozoom
        , _spec._camera_spec._lens_radius
        , _spec._camera_spec._focal_length
        , _spec._camera_spec._w
        , _spec._camera_spec._h
        );
    _hosek_sky = std::make_unique<HosekSky>( *this );
    if ( _cuda_payload->_num_lights < 2 )
        _cuda_spec->enable_sky_light();
    else
        _cuda_spec->disable_sky_light();
    std::cout << stat << std::endl;
}

void Scene::updateSky( float theta, float phi ) const {
    _hosek_sky->update( theta, phi );
}

void Scene::enableDebugSky() const {
    _cuda_spec->enable_debug_sky();
}

void Scene::disableDebugSky() const {
    _cuda_spec->disable_debug_sky();
}

void Scene::setCameraType( CameraType type ) const {
    _camera->setCameraType( type );
}

void Scene::enableSky() const {
    _cuda_spec->enable_sky_light();
}

void Scene::disableSky() const {
    _cuda_spec->disable_sky_light();
}

void Scene::enableMIS() const {
    _cuda_spec->enable_mis();
}

void Scene::disableMIS() const {
    _cuda_spec->disable_mis();
}
bool Scene::isSkydomeEnabled() const {
    return _cuda_spec->is_sky_light_enabled();
}

void Scene::updateCudaSpec() {
    if ( _cuda_spec->_outofsync ) {
        update_cuda_spec( _cuda_spec );
        resetCudaRenderBuffer();
    }
}

void Scene::updateCudaSky() const {
    _hosek_sky->updateCudaSky();
}

void Scene::updateCudaCamera() {
    _camera->updateCudaCamera();
}

void Scene::initCudaContext() const {
    int gpuID = _spec._gpu;
    SkydomeType skydome_type = (SkydomeType) _spec._model_spec._skydome_type;
    load_cuda_data( _cuda_payload, _hosek_sky->_cuda_hosek_sky, gpuID, skydome_type );
}

void Scene::initFramebuffer( GLuint* textureID ) const {
    init_framebuffer( textureID, getWidthPixels(), getHeightPixels() );
}

void Scene::path_tracer() {
    cuda_path_tracer( _frameCount );
}