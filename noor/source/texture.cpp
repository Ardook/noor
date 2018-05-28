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
#define TINYEXR_IMPLEMENTATION
#include "tinyexr.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "path_tracer.cuh"


void ImageTexture::load( const std::string& filename ) {
    const char* err;
    if ( NOOR::contains( filename, "exr" ) ) {
        if ( LoadEXR( &_data, &_width, &_height, filename.c_str(), &err ) < 0 ) {
            std::string msg = "File " + std::string( __FILE__ ) + " LINE " + std::to_string( __LINE__ ) + "\n";
            msg += "Error: unable to read/open texture file " + filename;
            std::cerr << msg << std::endl;
            exit( EXIT_FAILURE );
        }
        _num_channels = 4;
    } else {
        if ( ( _data = stbi_loadf( filename.c_str(), &_width, &_height, &_num_channels, 0 ) ) == nullptr ) {
            std::string msg = "File " + std::string( __FILE__ ) + " LINE " + std::to_string( __LINE__ ) + "\n";
            msg += "Error: unable to read/open texture file " + filename;
            std::cerr << msg << std::endl;
            exit( EXIT_FAILURE );
        }
        if ( _num_channels != 4 && _num_channels != 1 ) {
            stbi_image_free( _data );
            if ( ( _data = stbi_loadf( filename.c_str(), &_width, &_height, &_num_channels, 4 ) ) == nullptr ) {
                std::string msg = "File " + std::string( __FILE__ ) + " LINE " + std::to_string( __LINE__ ) + "\n";
                msg += "Error: unable to read/open texture file " + filename;
                std::cerr << msg << std::endl;
                exit( EXIT_FAILURE );
            }
            _num_channels = 4;
        }
        _channel_desc = _num_channels == 4 ? _float4_channelDesc : _float_channelDesc;
        assert( _num_channels == 1 || _num_channels == 4 );
    }
    _size_bytes = _width * _height * _num_channels * sizeof( float );
}