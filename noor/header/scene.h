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
#ifndef SCENE_H
#define SCENE_H

class Model;
class Camera;
class HosekSky;
class CudaPayload;
class CudaSpec;
enum CameraType;
class Scene {
public:
    std::unique_ptr<Model> _model;
    std::unique_ptr<Camera> _camera;
    std::unique_ptr<HosekSky> _hosek;
    std::unique_ptr<CudaSpec> _spec;
    std::unique_ptr<CudaPayload> _cuda_payload;
    BBox _scene_bbox;
    float _scene_radius;
    float _scene_bias;
    unsigned int _frameCount;
    void load();
    Spec _host_spec;
    ~Scene();
    Scene( const Spec& spec );
    const glm::mat4& getViewMatrix() const;
    const glm::mat4& getProjectionMatrix() const;
    void resetCudaRenderBuffer() {
        _frameCount = 1;
    }
    bool syncCudaSpecRequired() const { return _frameCount == 1; }
    glm::uint32& getFrameCount() { return _frameCount; }
    glm::uint32 getWidthPixels() const;
    glm::uint32 getHeightPixels() const;
    void reset( int w, int h );
    void mouse( int button, int action, int mods );
    void motion( int x, int y );

    const BBox& getSceneBBox() const { return _scene_bbox; }
    float getSceneRadius() const { return _scene_radius; }
    void setCameraType( CameraType type ) const;
    void updateSky( float theta, float phi ) const;
    void updateCudaSpec();
    void updateCudaSky();
    void updateCudaCamera();
    void enableDebugSky() const;
    void disableDebugSky() const;
    void path_tracer();
    void enableSky() const;
    void disableSky() const;
    void enableMIS() const;
    void disableMIS() const;
    bool isSkydomeEnabled() const;
    void initCudaContext( GLuint* cudaTextureID);
};
#endif /* SCENE_H */