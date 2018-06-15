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
#define STBI_FAILURE_USERMSG
#include "stb_image.h"

#include "path_tracer.cuh"

static inline void print_error( const std::string& filename,
                                const char* file,
                                int line ) {
    std::string msg = "File " + std::string( file ) + " LINE " + std::to_string( line ) + "\n";
    std::cerr << "Noor Error: unable to read/open texture file " + filename;
}

static inline void print_error_stb( const std::string& filename,
                                    const char* file,
                                    int line ) {
    print_error( filename, file, line );
    std::cerr << "\nSTB Error: " + std::string( stbi_failure_reason() ) << std::endl;
}

void ImageTexture::load( const std::string& filename ) {
    const char* err;
    if ( NOOR::contains( filename, "exr" ) ) {
        if ( LoadEXR( &_data, &_width, &_height, filename.c_str(), &err ) < 0 ) {
            print_error( filename, __FILE__, __LINE__ );
            exit( EXIT_FAILURE );
        }
        _num_channels = 4;
    } else {
        if ( stbi_info( filename.c_str(), &_width, &_height, &_num_channels ) != 1 ) {
            print_error_stb( filename, __FILE__, __LINE__ );
            exit( EXIT_FAILURE );
        }
        if ( _num_channels == 3 ) {
            _data = stbi_loadf( filename.c_str(), &_width, &_height, &_num_channels, 4 );
            _num_channels = 4;
        } else {
            _data = stbi_loadf( filename.c_str(), &_width, &_height, &_num_channels, 0 );
        }
        if ( _data == nullptr ) {
            print_error_stb( filename, __FILE__, __LINE__ );
            exit( EXIT_FAILURE );
        }
        _channel_desc = _num_channels == 4 ?
            _float4_channelDesc :
            ( _num_channels == 2 ? _float2_channelDesc : _float_channelDesc );
        assert( _num_channels == 1 || _num_channels == 2 || _num_channels == 4 );
    }
    _size_bytes = _width * _height * _num_channels * sizeof( float );
}