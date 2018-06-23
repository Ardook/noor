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
#pragma once

#include "targetver.h"
#include <vector>
#include <array>
#include <unordered_map>
#include <unordered_set>
#include <map>

#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>

#include <memory>
#include <cmath>
#include <chrono>
#include <thread>
#include <experimental/filesystem>

#define GLEW_STATIC 
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#define GLM_FORCE_SSE2
#define GLM_FORCE_CUDA
#define GLM_ENABLE_EXPERIMENTAL
#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/ext.hpp>
#include <glm/gtx/polar_coordinates.hpp>

enum {
    CUDA_VBO_VERTEX = 0
    , CUDA_VBO_UV
    , CUDA_VBO_ELEMENT
    , NUM_VBOS
};
enum {
    CUDA_VAO = 0
    , NUM_VAOS
};

namespace NOOR {
    const glm::quat QUAT_IDENTITY( glm::angleAxis( 0.0f, glm::vec3( 0.0f ) ) );

    __forceinline
        bool contains( const std::string src, const std::string query ) {
        return ( src.find( query ) != std::string::npos );
    }

    __forceinline
        bool isInf( const glm::vec3& v ) {
        return glm::isinf( v.x ) || glm::isinf( v.y ) || glm::isinf( v.z );
    }

    __forceinline
        bool isNan( const glm::vec3& v ) {
        return glm::isnan( v.x ) || glm::isnan( v.y ) || glm::isnan( v.z );
    }

    __forceinline
        glm::uint32 float_as_uint( float f ) {
        return *( reinterpret_cast<glm::uint32*>( &f ) );
    }

    __forceinline
        float uint_as_float( glm::uint32 u ) {
        return *( reinterpret_cast<float*>( &u ) );
    }

    template <typename T>
    __forceinline
        std::string to_string( const T a_value, const int n = 4 ) {
        std::ostringstream out;
        out << std::setprecision( n ) << a_value;
        return out.str();
    }
}
#include "timer.h"
#include "spec.h"
#include "bbox.h"
#include "stat.h"
#include "ArHosekSkyModel.h"