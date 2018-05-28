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
#ifndef CUDAFRAMEBUFFER_CUH
#define CUDAFRAMEBUFFER_CUH
class CudaFrameBufferManager {
public:
    uint _w, _h;
    // output frame buffer of the path tracer
    float4* _buffer;
    // Cuda array mapped to OpenGL texture
    cudaArray* _buffer_array;
    // Cuda to OpenGL mapping resource
    cudaGraphicsResource* _glResource;
    size_t _size_bytes;

    CudaFrameBufferManager() = default;
    CudaFrameBufferManager( GLuint* textureID, uint w, uint h ) :
        _w( w )
        , _h( h )
        , _size_bytes( w * h * sizeof( float4 ) ) {
        glGenTextures( 1, textureID );
        glBindTexture( GL_TEXTURE_2D, *textureID );
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
        glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F, _w, _h, 0, GL_RGBA, GL_FLOAT, nullptr );
        checkNoorErrors( cudaGraphicsGLRegisterImage( &_glResource, *textureID, GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard ) );
        checkNoorErrors( cudaMalloc( (void **) &_buffer, _size_bytes ) );
    }

    void update() {
        checkNoorErrors( cudaGraphicsMapResources( 1, &_glResource, nullptr ) );
        checkNoorErrors( cudaGraphicsSubResourceGetMappedArray( &_buffer_array, _glResource, 0, 0 ) );
        NOOR::memcopy_array( _buffer_array, _buffer, _size_bytes, cudaMemcpyDeviceToDevice );
        checkNoorErrors( cudaGraphicsUnmapResources( 1, &_glResource, nullptr ) );
    }

    void free() {
        cudaFree( _buffer );
    }
};
#endif /* CUDAFRAMEBUFFER_CUH */