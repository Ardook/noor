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
    // output frame buffer of the path tracer
    float4* _buffer;
    bool _managed;
    CudaFrameBufferManager() = default;

    CudaFrameBufferManager( int w, int h, bool managed = false ): _managed(managed)
    {
        if (managed)
            checkNoorErrors( cudaMallocManaged( (void **)&_buffer, w * h * sizeof( float4 ) ) );
            //checkNoorErrors( cudaHostAlloc( (void **)&_buffer, w * h * sizeof( float4 ), cudaHostAllocMapped ) );
            //checkNoorErrors( cudaHostAlloc( (void **)&_buffer, w * h * sizeof( float4 ), cudaHostAllocDefault ) );
            //checkNoorErrors( cudaHostAlloc( (void **)&_buffer, w * h * sizeof( float4 ), cudaHostAllocPortable ) );
        else
            //checkNoorErrors( cudaMallocManaged( (void **)&_buffer, w * h * sizeof( float4 ) ) );
            checkNoorErrors( cudaMalloc( (void **)&_buffer, w * h * sizeof( float4 ) ) );
    }

    __device__
        float4 get( int index, uint frame_number )const {
        return frame_number > 1 ? _buffer[index] : make_float4( 0.0f, 0.0f, 0.0f, 1.0f );
    }
    __device__
        void set( const float4& new_color, int index ) {
        _buffer[index] = new_color;
    }
    __device__
        void set( const float4& new_color, int index, uint frame_number ) {
        const float4 old_color = get( index, frame_number );
        _buffer[index] = lerp( old_color, new_color, 1.0f / static_cast<float>( frame_number ) );
    }
    void free() {
       /* if (_managed)
        checkNoorErrors( cudaFreeHost( _buffer ) );
        else*/
        checkNoorErrors( cudaFree( _buffer ) );
    }
};

__constant__
CudaFrameBufferManager _framebuffer_manager;
#endif /* CUDAFRAMEBUFFER_CUH */