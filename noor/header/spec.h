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
#ifndef SPEC_H
#define SPEC_H

class BVHSpec {
public:
    glm::uint32  _num_bins;
    glm::uint32  _max_height;
    float   _Ci;
    float   _Ct;
    glm::uint32   _min_leaf_tris;
    glm::uint32   _max_leaf_tris;
    BVHSpec() = default;
    BVHSpec(
        glm::uint32 num_bins,
        glm::uint32 max_height,
        float Ci,
        float Ct,
        glm::uint32 min_leaf_tris,
        glm::uint32 max_leaf_tris
    ) :
        _num_bins( num_bins ),
        _max_height( max_height ),
        _Ci( Ci ),
        _Ct( Ct ),
        _min_leaf_tris( min_leaf_tris ),
        _max_leaf_tris( max_leaf_tris ) {}

};

class CameraSpec {
public:
    // camera settings
    glm::vec3 _eye;
    glm::vec3 _up;
    glm::vec3 _lookAt;
    float	  _fov;
    glm::uint32 _w;
    glm::uint32 _h;
    float _lens_radius;
    float _focal_length;
    glm::uint8 _bounces;
    glm::uint8 _rr;

    CameraSpec() = default;
    CameraSpec(
        const glm::vec3& eye,
        const glm::vec3& up,
        const glm::vec3& lookAt,
        float fov,
        glm::uint32 w,
        glm::uint32 h,
        float lens_radius,
        float focal_length,
        glm::uint8 bounces,
        glm::uint8 rr
    ) :
        _eye( eye ),
        _up( up ),
        _lookAt( lookAt ),
        _fov( fov ),
        _w( w ),
        _h( h ),
        _lens_radius( lens_radius ),
        _focal_length( focal_length ),
        _bounces( bounces ),
        _rr( rr )
    {}
};

class ModelSpec {
public:
    std::string _model_filename;
    std::string _hdr_filename;
    int _skydome_type;
};

class Spec {
public:
    int _gpu;
    Spec(
        const BVHSpec& bvh_spec,
        const CameraSpec& camera_spec,
        const ModelSpec model_spec,
        int gpu = 0
    ) :
        _bvh_spec( bvh_spec ),
        _camera_spec( camera_spec ),
        _model_spec( model_spec ),
        _gpu( gpu ) {}

    BVHSpec _bvh_spec;
    CameraSpec _camera_spec;
    ModelSpec _model_spec;
};
#endif /* SPEC_H */
