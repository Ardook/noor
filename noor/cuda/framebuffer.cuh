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
struct CudaRenderTask {
    int _w;
    int _h;
    int _gpu_id;
    size_t _size;

    CudaRenderTask() = default;
    CudaRenderTask( int w, int h, int gpu_id ) :
        _w( w ),
        _h( h ),
        _gpu_id( gpu_id ),
        _size( _w*_h * sizeof( float4 ) )
    {}
};
class CudaFrameBufferManager {
public:
    // output frame buffer of the path tracer
    float4* _buffer;
    bool _managed;
    CudaFrameBufferManager() = default;

    CudaFrameBufferManager( const CudaRenderTask& task ):
        _managed(task._gpu_id != 0) 
    {
        //if ( !_managed )
            //checkNoorErrors( cudaMallocManaged( (void **)&_buffer, task._size ) );
            checkNoorErrors( cudaMalloc( (void **)&_buffer, task._size ) );
            //checkNoorErrors( cudaMallocManaged( (void **)&_buffer, w * h * sizeof( float4 ) ) );
            //checkNoorErrors( cudaHostAlloc( (void **)&_buffer, w * h * sizeof( float4 ), cudaHostAllocMapped ) );
            //checkNoorErrors( cudaHostAlloc( (void **)&_buffer, w * h * sizeof( float4 ), cudaHostAllocDefault ) );
            //checkNoorErrors( cudaHostAlloc( (void **)&_buffer, task._size, cudaHostAllocWriteCombined ) );
        //else
            //checkNoorErrors( cudaMallocManaged( (void **)&_buffer, task._size ) );
          //  checkNoorErrors( cudaHostAlloc( (void **)&_buffer, task._size, cudaHostAllocMapped ) );
    }

    __device__
        float4 get( int index, uint frame_number )const {
        return frame_number > 1 ? 
            _buffer[index] : 
            make_float4( 0.0f, 0.0f, 0.0f, 1.0f );
    }
    __device__
        void set( const float4& new_color, int index ) {
        _buffer[index] = new_color;
    }
    __device__
        void set( const float4& new_color, int index, uint frame_number ) {
        const float4 old_color = get( index, frame_number );
        _buffer[index] = lerp( old_color, new_color, 1.0f / frame_number );
    }
    void free() {
        checkNoorErrors( cudaFree( _buffer ) );
    }
};

__constant__
CudaFrameBufferManager _framebuffer_manager;
#endif /* CUDAFRAMEBUFFER_CUH */