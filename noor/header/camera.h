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

#ifndef NOOR_CAMERA_H
#define NOOR_CAMERA_H
struct Screen {
    glm::vec2 _min{ -1.0f, -1.0f };
    glm::vec2 _max{ 1.0f,  1.0f };
};
class Camera {
    friend class Scene;
    Scene& _scene;
    bool _outofsync{ true };
    // screen boundaries
    Screen _screen;
    // width in pixels
    int _w;
    // height in pixels
    int _h;

    // camera frame
    glm::vec3 _eye;
    glm::vec3 _lookAt;
    glm::vec3 _up;
    glm::vec3 _right;
    glm::vec3 _forward;

    // quaternion rotation
    glm::quat _q;
    // delta rotation
    glm::quat _dq;

    // radius of screen in pixels
    float _r;
    // radius squared
    float _r2;
    // radius square root
    float _rsq2;

    // radius of scene's bounding sphere
    float _scene_radius;

    // rotation speed scale
    float _rotate_speed;
    // zoom speed scale
    float _zoom_speed;
    // slow camera movements by how close to surface
    float _scale;
    // strafe speed scale
    glm::vec2 _strafe_speed;

    // vertical field of view
    float _fov;
    float _lens_radius;
    float _focal_length;
    // aspect ratio
    float _aspect;
    float _orthozoom;

    // GLFW button states
    int _button;
    int _prev_button_state;
    int _curr_button_state;
    // pixel coordinate state
    glm::ivec2 _prev_xy;
    glm::ivec2 _curr_xy;

    // projection transformation
    glm::mat4 _cameraToScreen;
    // screen to raster transformation
    glm::mat4 _screenToRaster;
    // view transformation (inverse of camera to world)
    glm::mat4 _view;
    // camera rotation
    glm::mat4 _rotation;
    // camera transformation
    glm::mat4 _cameraToWorld;
    // raster to camera coordinate transformation
    glm::mat4 _rasterToCamera;

    CameraType _type;
public:
    Camera() = default;
    Camera(
        Scene& scene,
        const glm::vec3& eye,
        const glm::vec3& lookAt,
        const glm::vec3& up,
        float fov,
        float ortho_zoom,
        float lens_radius,
        float focal_length,
        int w,
        int h
    ) :
        _scene( scene ),
        _eye( eye ),
        _lookAt( lookAt ),
        _up( up ),
        _fov( fov ),
        _orthozoom( ortho_zoom ),
        _lens_radius( lens_radius ),
        _focal_length( focal_length ),
        _scene_radius( scene.getSceneRadius() ),
        _scale( glm::length( eye - _lookAt ) ),
        _rotate_speed( 1.5f ),
        _prev_button_state( GLFW_RELEASE ),
        _curr_button_state( GLFW_RELEASE ),
        _view( 1.f ),
        _rotation( 1.f ),
        _cameraToWorld( 1.f ),
        _rasterToCamera( 1.f ),
        _type( PERSP ) {
        reset( w, h );
    }

    void reset( int w, int h ) {
        _w = w;
        _h = h;
        _aspect = _w / (float) _h;
        _r = ( _w > _h ? _h : _w ) * 0.5f;
        _r2 = _r * _r;
        _rsq2 = _r / glm::root_two<float>();

        _strafe_speed.x = 1.0f / _w;
        _strafe_speed.y = 1.0f / _h;
        _zoom_speed = 1.0f / _w;

        _forward = glm::normalize( _eye - _lookAt );
        _right = glm::normalize( glm::cross( _up, _forward ) );
        _rotation[0] = glm::vec4( _right, 0 );
        _rotation[1] = glm::vec4( _up, 0 );
        _rotation[2] = glm::vec4( _forward, 0 );
        updateProjection();
        updateView();
    }

    void setCameraType( CameraType type ) {
        _type = type;
        updateProjection();
        _outofsync = true;
    }

    glm::mat4 orthographic( float n, float f ) {
        return glm::scale( glm::vec3( 1.f, 1.f, 1.f / ( f - n ) ) ) *
            glm::translate( glm::vec3( 0, 0, -n ) );
    }

    glm::mat4 perspective( float fov, float n, float f ) {
        float cot = 1.0f / std::tanf( fov / 2.f );
        // Scale canonical perspective view to specified field of view
        glm::mat4 persp{
            cot, 0, 0, 0
            , 0, cot, 0, 0
            , 0, 0, ( f + n ) / ( n - f ), -1
            , 0, 0, -2 * f*n / ( f - n ), 0 };
        return persp;
    }

    void updateScreenToRaster() {
        _screenToRaster =
            glm::scale( glm::vec3( _w, _h, 1.0f ) ) *
            glm::scale( glm::vec3(
            1.0f / ( _screen._max.x - _screen._min.x ),
            1.0f / ( _screen._max.y - _screen._min.y ),
            1.0f ) ) *
            glm::translate( glm::vec3( -_screen._min.x, -_screen._min.y, 0.0f ) );
        _rasterToCamera = glm::inverse( _cameraToScreen ) *
            glm::inverse( _screenToRaster );
        _outofsync = true;
    }

    void updatePerspProjection() {
        _cameraToScreen = perspective( _fov, 0.01f, 1000.f );
        updateScreenToRaster();
        _outofsync = true;
    }

    void updateOrthoProjection() {
        _cameraToScreen = orthographic( 0.f, 1.f );
        _screen._min *= _orthozoom;
        _screen._max *= _orthozoom;
        updateScreenToRaster();
        _outofsync = true;
    }

    void updateProjection() {
        if ( _aspect > 1.f ) {
            _screen._min.x = -_aspect;
            _screen._max.x = _aspect;
            _screen._min.y = -1.f;
            _screen._max.y = 1.f;
        } else {
            _screen._min.x = -1.f;
            _screen._max.x = 1.f;
            _screen._min.y = -1.f / _aspect;
            _screen._max.y = 1.f / _aspect;
        }
        if ( _type == PERSP ) {
            updatePerspProjection();
        } else if ( _type == ORTHO ) {
            updateOrthoProjection();
        }
        _outofsync = true;
    }

    void updateView() {
        _scale = glm::length( _eye - _lookAt );
        const glm::mat4 R = _rotation * glm::toMat4( _q );
        _cameraToWorld = glm::translate( _lookAt ) * R * glm::translate( glm::vec3( 0, 0, _scale ) );
        _view = glm::translate( glm::vec3( 0, 0, -_scale ) ) * glm::transpose( R ) * glm::translate( -_lookAt );
        _eye = getEye();
        _outofsync = true;
    }

    void updateCudaCamera() {
        if ( _outofsync ) {
            update_cuda_camera( _cameraToWorld, _rasterToCamera, _w, _h, _lens_radius, _focal_length, _type );
            _scene.resetCudaRenderBuffer();
            _outofsync = false;
        }
    }

    glm::vec3 shoemake( float x, float y ) {
        glm::vec3 p;
        const float d = std::sqrt( x*x + y*y );
        if ( d <= _r ) {   /* Inside sphere */
            p = glm::vec3( x, y, sqrt( _r2 - d*d ) );
        } else {        /* On hyperbola */
            p = ( _r / d )*glm::vec3( x, y, 0.0f );
        }
        return glm::normalize( p );
    }

    glm::vec3 holroyd( float x, float y ) {
        glm::vec3 p;
        const float d = std::sqrt( x*x + y*y );
        if ( d <= _rsq2 ) {    /* Inside sphere */
            p = glm::vec3( x, y, sqrt( _r2 - d*d ) );
        } else {            /* On hyperbola */
            p = glm::normalize( glm::vec3( x, y, _r2 / ( 2.0f*d ) ) );
        }
        return glm::normalize( p );
    }

    glm::vec3 onSphere( const glm::ivec2& xy ) {
        const float x = xy.x - _w / 2.0f;
        const float y = -xy.y + _h / 2.0f;
        return holroyd( x, y );
    }

    const glm::vec3 getEye() const {
        return _cameraToWorld[3];
    }

    const glm::vec3 getRight() const {
        return _cameraToWorld[0];
    }

    const glm::vec3 getUp() const {
        return _cameraToWorld[1];
    }

    const glm::vec3 getForward() const {
        return _cameraToWorld[2];
    }

    void mouse( int button, int action ) {
        _button = button;
        _prev_button_state = _curr_button_state;
        _curr_button_state = action;
    }

    void updateLookAt() {
        if ( _type == ENV || _type == ORTHO ) return;
        const glm::vec4 lookAt = F2V4( get_lookAt() );
        _lookAt = ( lookAt.w > 0.0f ) ? lookAt : _lookAt;
    }

    void motion( int x, int y ) {
        _prev_xy = _curr_xy;
        _curr_xy.x = x;
        _curr_xy.y = y;

        if ( _curr_button_state == GLFW_PRESS ) {
            switch ( _button ) {
                case GLFW_MOUSE_BUTTON_RIGHT:
                    zoom();
                    break;
                case GLFW_MOUSE_BUTTON_LEFT:
                    orbit();
                    break;
                case GLFW_MOUSE_BUTTON_MIDDLE:
                    strafe();
                    break;
            }
        } else if ( _curr_button_state == GLFW_RELEASE && _prev_button_state == GLFW_PRESS ) {
            _prev_button_state = GLFW_RELEASE;
            if ( _button != GLFW_MOUSE_BUTTON_LEFT ) {
                updateLookAt();
            }
        }
    }

    void orbit() {
        if ( _curr_xy == _prev_xy ) {
            /* Zero rotation */
            _dq = NOOR::QUAT_IDENTITY;
            return;
        }
        const glm::vec3 _curr = onSphere( _curr_xy );
        const glm::vec3 _prev = onSphere( _prev_xy );
        const glm::vec3 _axis = glm::normalize( glm::cross( _curr, _prev ) );
        const float _angle = glm::angle( _curr, _prev );
        _dq = glm::normalize( glm::angleAxis( _angle*_rotate_speed, _axis ) );
        _q = glm::normalize( _q*_dq );
        updateView();
    }

    void strafe() {
        const glm::vec2 delta( _curr_xy - _prev_xy );
        if ( delta.x != 0.0f ) {
            _lookAt -= getRight() * delta.x * _strafe_speed.x * _scale;
        }
        if ( delta.y != 0.0f ) {
            _lookAt += getUp() * delta.y * _strafe_speed.y * _scale;
        }
        updateView();
    }

    void zoom() {
        if ( _type == ENV || _type == ORTHO ) return;
        const glm::vec2 delta( _curr_xy - _prev_xy );
        if ( delta.x != 0.0f ) {
            _eye -= getForward() * delta.x * _zoom_speed * _scale;
        }
        updateView();
    }

    const glm::mat4& getViewMatrix() const {
        return _view;
    }

    const glm::mat4& getProjectionMatrix() const {
        return _cameraToScreen;
    }

};

#endif /* NOOR_CAMERA_H */
