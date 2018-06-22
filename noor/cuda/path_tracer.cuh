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
#ifndef PATH_TRACER_CUH
#define PATH_TRACER_CUH
#define GLEW_STATIC 
#include <GL/glew.h>
// CUDA runtime API
#include <cuda.h>
#include <cuda_runtime.h>
// CUDA & OpenGL interoperability
#include <cuda_gl_interop.h>
#include <helper_cuda.h>
// vector types
#include <helper_math.h>
// findCudaGLDevice
#include <helper_gl.h>
#include <math_constants.h>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <glm/ext.hpp>

#include <memory>
#include <vector>

// Kernel configuration 
#define THREAD_W 16
#define THREAD_H 8 
#define THREAD_N THREAD_W * THREAD_H

// BVH masks
#define TRI_LEAF_MASK		0x80000000
#define MESH_LEAF_MASK		0x40000000
#define MESH_INSTANCE_MASK  0x20000000
#define MESH_INNER_MASK		0x10000000
#define MESH_NODE_MASK		0x70000000

#define COUNT_MASK			0x0FFFFFFF
#define TRI_COUNT_MASK		0x7FFFFFFF
#define AXIS_MASK			0x0FFFFFFF
#define TRANS_MASK			0x0FFFFFFF

#define LIGHT_NODE_MASK		0x08000000
#define LIGHT_MASK			0x07FFFFFF

#define F2V4( v ) glm::vec4( v.x, v.y, v.z, v.w )
#define F2V3( v ) glm::vec3( v.x, v.y, v.z )
#define V2F4( v ) make_float4( v.x, v.y, v.z, v.w )
#define V2F3( v ) make_float3( v.x, v.y, v.z )
#define V2F2( v ) make_float2( v.x, v.y )

constexpr float NOOR_EPSILON = std::numeric_limits<float>::epsilon();
constexpr float NOOR_ONE_MINUS_EPSILON = 1.0f - NOOR_EPSILON;
constexpr float NOOR_INF = std::numeric_limits<float>::infinity();
constexpr float NOOR_FLT_MAX = std::numeric_limits<float>::max();
constexpr float NOOR_PI = CUDART_PI_F;
constexpr float NOOR_2PI = 2.0f * CUDART_PI_F;
constexpr float NOOR_4PI = 4.0f * CUDART_PI_F;
constexpr float NOOR_invPI = 1.0f / NOOR_PI;
constexpr float NOOR_inv2PI = 1.0f / NOOR_2PI;
constexpr float NOOR_inv4PI = 1.0f / NOOR_4PI;
constexpr float NOOR_PI_over_2 = NOOR_PI / 2.0f;
constexpr float NOOR_PI_over_4 = NOOR_PI / 4.0f;
constexpr float NOOR_3PI_over_2 = 3.f * NOOR_PI_over_2;

constexpr int2 skydome_res{ 2048, 2048 };
#define SKYDOME_COLOR make_float3(.25f)
enum MaterialType {
    DIFFUSE = 1 << 0,
    ORENNAYAR = 1 << 1,
    TRANSLUCENT = 1 << 2,
    GLOSSY = 1 << 3,
    MIRROR = 1 << 4,
    GLASS = 1 << 5,
    ROUGHGLASS = 1 << 6,
    METAL = 1 << 7,
    SUBSTRATE = 1 << 8,
    EMITTER = 1 << 9,
    CLEARCOAT = 1 << 10,
    SHADOWCATCHER = 1 << 11,
    ALPHA = 1 << 12,
    BUMP = 1 << 13,
    MESHLIGHT = 1 << 14
};
constexpr uint NOOR_NO_BUMP_ALPHA = ~(BUMP | ALPHA);
constexpr uint NOOR_EMITTER = EMITTER | MESHLIGHT;
constexpr uint NOOR_TRANSPARENT = TRANSLUCENT | GLASS | ROUGHGLASS;
constexpr uint NOOR_SPECULAR = MIRROR | GLASS | CLEARCOAT;
constexpr uint NOOR_GLOSSY = NOOR_SPECULAR | ROUGHGLASS | CLEARCOAT | METAL | SUBSTRATE | GLOSSY;
enum AreaMeshLightType { QUAD = 0, SPHERE = 1, DISK = 2 };
enum SkydomeType { HDR = 0, PHYSICAL = 1, CONSTANT = 2 };
enum HandedNess { LEFT_HANDED = 0, RIGHT_HANDED = 1 };
enum CameraType { PERSP = 0, ORTHO = 1, ENV = 2 };

class CudaPayload;
class CudaSpec;
class CudaHosekSky;
class CudaCamera;
extern "C" {
    void load_cuda_data( const std::unique_ptr<CudaPayload>& payload,
                         const CudaHosekSky& hosek,
                         const CudaCamera& camera,
                         const CudaSpec& spec,
                         GLuint* textureID
                         );

    void update_cuda_camera();
    void update_cuda_hosek();
    void update_cuda_spec();
    void cuda_path_tracer( unsigned int& frameCount );
    void device_free_memory();
    void get_lookAt(float4& lookAt);
}
// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define checkNoorErrors(val) checkError ( (val), #val, __FILE__, __LINE__ )
template< typename T >
static void checkError( T result, char const *const func, const char *const file, int const line ) {
    if ( result != cudaSuccess ) {
        fprintf( stderr, "CUDA error at: %s:%d \nErrorCode: %d (%s) \nFunction: \"%s\" \n",
                 file, line, static_cast<unsigned int>( result ), _cudaGetErrorEnum( result ), func );
        // Make sure we call CUDA Device Reset before exiting
        exit( EXIT_FAILURE );
    }
}
using uchar = unsigned char;
// hybrid cuda and host headers
#include "math.cuh"
#include "utils.cuh"
#include "spec.cuh"
#include "shape.cuh"
#include "light.cuh"
#include "hosek.cuh"
#include "material.cuh"
#include "transform.cuh"
#include "camera.cuh"
#include "texture.h"
#include "payload.h"
#endif /* PATH_TRACER_CUH */