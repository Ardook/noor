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

struct Mouse {
    // GLFW button states
    GLint _button;
    GLint _mods{ 0 };
    GLint _prev_button_state{ GLFW_RELEASE };
    GLint _curr_button_state{ GLFW_RELEASE };
    int _w, _h;
    // pixel coordinate state
    glm::ivec2 _prev_xy{ 0 };
    glm::ivec2 _curr_xy{ 0 };
    glm::ivec2 _delta{ 0 };
    glm::vec2 _dt{ 0 };

    Mouse( int w, int h ) :_w( w ), _h( h ) {}
    void updateState( GLint button, GLint action, GLint mods ) {
        _button = button;
        _prev_button_state = _curr_button_state;
        _curr_button_state = action;
        _mods = mods;
    }
    void updateMotion( int x, int y ) {
        _prev_xy = _curr_xy;
        _curr_xy = glm::ivec2( x, y );
        _delta = _curr_xy - _prev_xy;
        _dt.x = _delta.x / (float)_w;
        _dt.y = _delta.y / (float)_h;
    }

    bool buttonReleased()const {
        return ( _curr_button_state == GLFW_RELEASE &&
                 _prev_button_state == GLFW_PRESS );
    }

    bool buttonPressed( GLuint button )const {
        return ( _curr_button_state == GLFW_PRESS && _button == button );
    }

    bool buttonPressed( GLuint button, GLuint mods )const {
        return  buttonPressed( button ) && _mods == mods;
    }

    bool cameraMode() const {
        return ( _mods == 0 || _mods == GLFW_MOD_SHIFT );
    }

    bool skyMode() const {
        return ( _mods == GLFW_MOD_ALT );
    }
};

class Scene {
public:
    std::unique_ptr<Model> _model;
    std::unique_ptr<Camera> _camera;
    std::unique_ptr<Mouse> _mouse;
    std::unique_ptr<HosekSky> _hosek;
    std::unique_ptr<CudaSpec> _spec;
    std::unique_ptr<CudaPayload> _cuda_payload;
    BBox _scene_bbox;
    Spec _host_spec;
    float _scene_radius;
    float _scene_bias;
    unsigned int _frameCount;
    ~Scene();
    Scene( const Spec& spec );
    void load();

    glm::uint32 getWidthPixels() const;
    glm::uint32 getHeightPixels() const;

    void reset( int w, int h );
    void mouse( int button, int action, int mods );
    void motion( int x, int y );

    bool isSkydomeEnabled() const;
    void resetCudaRenderBuffer() { _frameCount = 1; }
    void setCameraType( CameraType type ) const;
    void updateCudaSpec();
    void updateCudaSky();
    void updateCudaCamera();
    void enableDebugSky() const;
    void disableDebugSky() const;
    void enableSky() const;
    void disableSky() const;
    void enableMIS() const;
    void disableMIS() const;

    void initCudaContext( GLuint* cudaTextureID );
    void path_tracer();
};
#endif /* SCENE_H */