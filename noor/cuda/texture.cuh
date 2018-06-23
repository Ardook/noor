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
#ifndef CUDATEXTURE_CUH
#define CUDATEXTURE_CUH
#include "image.cuh"
using NOOR::nearestPow2;
using NOOR::prevPow2;
using NOOR::nextPow2;
class CudaTexture {
protected:
    cudaExtent _extent;
    int _num_channels;

    cudaArray_t _array;
    cudaTextureFilterMode _filter_mode;
    cudaTextureAddressMode _address_mode;
    cudaChannelFormatDesc _channel_desc;
    cudaTextureObject_t _read_tex_obj;
    cudaSurfaceObject_t _write_surface_obj;
public:
    CudaTexture() = default;
    __host__
        CudaTexture( const cudaExtent& extent,
                     int num_channels,
                     cudaTextureFilterMode filter_mode = cudaFilterModeLinear,
                     cudaTextureAddressMode address_mode = cudaAddressModeClamp
        ) :
        _extent( extent ),
        _num_channels( num_channels ),
        _filter_mode( filter_mode ),
        _address_mode( address_mode ) {}

    __device__ __host__
        int width() const {
        return (int) _extent.width;
    }
    __device__ __host__
        int height() const {
        return (int) _extent.height;
    }
    __device__ __host__
        int depth() const {
        return (int) _extent.depth;
    }
    __device__ __host__
        int getNumChannels() const {
        return (int) _num_channels;
    }
    __device__ __host__
        const cudaExtent& getExtent() const {
        return _extent;
    }
    __device__ __host__
        const cudaArray_t& getArray() const {
        return _array;
    }
    __device__ __host__
        const cudaTextureObject_t& getReadTexObj() const {
        return _read_tex_obj;
    }
    __device__ __host__
        const cudaSurfaceObject_t& getWriteSurfaceObj() const {
        return _write_surface_obj;
    }
    __device__ __host__
        const cudaTextureFilterMode& getFilterMode() const {
        return _filter_mode;
    }
    __device__ __host__
        const cudaTextureAddressMode& getAddressMode() const {
        return _address_mode;
    }
    __device__
        float4 evaluateLOD( float u, float v, int level ) const {
        return tex2DLod<float4>( _read_tex_obj, u, v, level );
    }
    __device__
        float4 evaluate( float u, float v ) const {
        return tex2D<float4>( _read_tex_obj, u, v );
    }
    __device__
        float4 evaluate( const float2& uv ) const {
        return tex2D<float4>( _read_tex_obj, uv.x, uv.y );
    }
    template<class T>
    __device__
        T evaluate( const float2& uv, const float2& uvscale ) const {
        const float2 t = uvscale*uv;
        return tex2D<T>( _read_tex_obj, t.x, t.y );
    }
    template<class T>
    __device__
        T evaluate( const CudaIntersection& I, const float2& uvscale ) const {
        const float2 uv = uvscale * I._uv;
        return tex2D<T>( _read_tex_obj, uv.x, uv.y );
    }
    template<class T>
    __device__
        T evaluateGrad( const CudaIntersection& I, const float2& uvscale, float du = 0, float dv = 0 ) const {
        const float2 gradx = uvscale.x*make_float2( I._differential._dudx, I._differential._dvdx );
        const float2 grady = uvscale.y*make_float2( I._differential._dudy, I._differential._dvdy );
        const float2 uv = uvscale * ( I._uv + make_float2( du, dv ) );
        return tex2DGrad<T>( _read_tex_obj, uv.x, uv.y, gradx, grady );
    }
protected:
    __host__
        void load( const ImageTexture& t ) {
        cudaMemcpy3DParms copyParams;
        memset( &copyParams, 0, sizeof( cudaMemcpy3DParms ) );
        copyParams.srcPtr = make_cudaPitchedPtr( (void*)t.getData(), 
                                                 t.getNumChannels() * t.getWidth() * sizeof( float ), 
                                                 t.getWidth(), 
                                                 t.getHeight() );
        copyParams.extent = make_cudaExtent( t.getWidth(), t.getHeight(), 1 );
        copyParams.kind = cudaMemcpyHostToDevice;
        if ( width() == t.getWidth()&& height() == t.getHeight() ) {
            copyParams.dstArray = _array;
            checkNoorErrors( cudaMemcpy3D( &copyParams ) );
        } else {
            cudaArray_t src_array;
            cudaChannelFormatDesc channel_desc = t.getChannelDesc();
            checkNoorErrors( NOOR::malloc_array( &src_array, &channel_desc, t.getWidth(), t.getHeight() ) );
            copyParams.dstArray = src_array;
            checkNoorErrors( cudaMemcpy3D( &copyParams ) );
            resize( src_array, make_cudaExtent( t.getWidth(), t.getHeight(), 0 ), _extent );
        }
    }
    __host__
        void resize( cudaArray_t src_array, const cudaExtent& old_extent, const cudaExtent& new_extent ) {
        cudaSurfaceObject_t src_surfaceobj;
        cudaSurfaceObject_t dst_surfaceobj;
        checkNoorErrors( NOOR::create_surfaceobj( &src_surfaceobj, src_array ) );
        checkNoorErrors( NOOR::create_surfaceobj( &dst_surfaceobj, _array ) );
        dim3 block( 8, 8, 1 );
        dim3 grid( (uint) new_extent.width / block.x, (uint) new_extent.height / block.y, 1 );
        size_t shmsize = 8 * sizeof( ResampleWeight );
        cudaTextureAddressMode address_mode = cudaAddressModeClamp;
        if ( _num_channels == 4 ) {
            resize_kernel<float4> << <grid, block, shmsize >> > ( src_surfaceobj,
                                                                  dst_surfaceobj,
                                                                  old_extent,
                                                                  new_extent,
                                                                  _num_channels,
                                                                  address_mode );
        } else if ( _num_channels == 2 ) {
            resize_kernel<float2> << <grid, block, shmsize >> > ( src_surfaceobj,
                                                                  dst_surfaceobj,
                                                                  old_extent,
                                                                  new_extent,
                                                                  _num_channels,
                                                                  address_mode );
        } else {
            resize_kernel<float> << <grid, block, shmsize >> > ( src_surfaceobj,
                                                                 dst_surfaceobj,
                                                                 old_extent,
                                                                 new_extent,
                                                                 _num_channels,
                                                                 address_mode );
        }
        checkNoorErrors( cudaDeviceSynchronize() );
        checkNoorErrors( cudaGetLastError() );
        checkNoorErrors( cudaDestroySurfaceObject( src_surfaceobj ) );
        checkNoorErrors( cudaDestroySurfaceObject( dst_surfaceobj ) );
        checkNoorErrors( cudaFreeArray( src_array ) );
    }
};

// Cuda mipmapping based on Cuda SDK Samples
template<class T>
__global__
void mipmap( cudaSurfaceObject_t mipOutput, cudaTextureObject_t mipInput, uint imageW, uint imageH ) {
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;

    float px = 1.0 / float( imageW );
    float py = 1.0 / float( imageH );


    if ( ( x < imageW ) && ( y < imageH ) ) {
        // take the average of 4 samples
        // we are using the normalized access to make sure non-power-of-two textures
        // behave well when downsized.
        T color =
            ( tex2D<T>( mipInput, ( x + 0 ) * px, ( y + 0 ) * py ) ) +
            ( tex2D<T>( mipInput, ( x + 1 ) * px, ( y + 0 ) * py ) ) +
            ( tex2D<T>( mipInput, ( x + 1 ) * px, ( y + 1 ) * py ) ) +
            ( tex2D<T>( mipInput, ( x + 0 ) * px, ( y + 1 ) * py ) );
        color /= 4.0f;
        surf2Dwrite( color, mipOutput, x * sizeof( T ), y );
    }
}

class CudaMipMap : public CudaTexture {
    cudaMipmappedArray_t  _mipmapArray;
public:
    CudaMipMap() = default;

    __device__ __host__
        CudaMipMap& operator=( const CudaMipMap& t ) = delete;

    __host__
        CudaMipMap( const ImageTexture& t ) :
        CudaTexture( make_cudaExtent( nearestPow2( t.getWidth() ), nearestPow2( t.getHeight() ), 0 ),
                     t.getNumChannels(),
                     t.getFilterMode(),
                     t.getAddressMode()
        ) {
        // how many mipmaps we need
        int levels = getMipMapLevels( _extent );
        cudaChannelFormatDesc channel_desc = t.getChannelDesc();
        checkNoorErrors( cudaMallocMipmappedArray( &_mipmapArray, &channel_desc, _extent, levels ) );
        checkNoorErrors( cudaGetMipmappedArrayLevel( &_array, _mipmapArray, 0 ) );
        // upload level 0, resize the source if it's not power of 2
        load( t );
        // generate the entire mipmap
        generateMipMaps();

        // generate bindless texture object
        cudaResourceDesc resDescr;
        memset( &resDescr, 0, sizeof( cudaResourceDesc ) );
        resDescr.resType = cudaResourceTypeMipmappedArray;
        resDescr.res.mipmap.mipmap = _mipmapArray;
        cudaTextureDesc texDescr;
        memset( &texDescr, 0, sizeof( cudaTextureDesc ) );
        texDescr.normalizedCoords = 1;
        texDescr.filterMode = cudaFilterModeLinear;
        texDescr.mipmapFilterMode = cudaFilterModeLinear;
        texDescr.addressMode[0] = cudaAddressModeWrap;
        texDescr.addressMode[1] = cudaAddressModeWrap;
        texDescr.addressMode[2] = cudaAddressModeClamp;
        texDescr.maxMipmapLevelClamp = float( levels - 1 );
        texDescr.maxAnisotropy = 64;
        texDescr.readMode = cudaReadModeElementType;
        checkNoorErrors( cudaCreateTextureObject( &_read_tex_obj, &resDescr, &texDescr, NULL ) );
        checkNoorErrors( NOOR::create_surfaceobj( &_write_surface_obj, _array ) );
    }

    __device__ __host__
        const cudaMipmappedArray_t& getMipMapArray()const {
        return _mipmapArray;
    }
    __host__
        uint getMipMapLevels( cudaExtent size ) {
        size_t sz = MAX( MAX( size.width, size.height ), size.depth );

        uint levels = 0;

        while ( sz ) {
            sz /= 2;
            levels++;
        }
        return levels;
    }

    __host__
        void update() {
        generateMipMaps();
    }

    __host__
        void generateMipMaps() {
        size_t width = _extent.width;
        size_t height = _extent.height;
        cudaArray_t levelFirst;
        checkNoorErrors( cudaGetMipmappedArrayLevel( &levelFirst, _mipmapArray, 0 ) );
        uint level = 0;
        while ( width != 1 || height != 1 ) {
            width /= 2;
            width = MAX( (size_t) 1, width );
            height /= 2;
            height = MAX( (size_t) 1, height );

            cudaArray_t levelFrom;
            checkNoorErrors( cudaGetMipmappedArrayLevel( &levelFrom, _mipmapArray, level ) );
            cudaArray_t levelTo;
            checkNoorErrors( cudaGetMipmappedArrayLevel( &levelTo, _mipmapArray, level + 1 ) );
            cudaExtent  levelToSize;
            checkNoorErrors( cudaArrayGetInfo( NULL, &levelToSize, NULL, levelTo ) );
            // generate texture object for reading
            cudaTextureObject_t         texInput;
            cudaResourceDesc            texRes;
            memset( &texRes, 0, sizeof( cudaResourceDesc ) );
            texRes.resType = cudaResourceTypeArray;
            texRes.res.array.array = levelFrom;
            cudaTextureDesc             texDescr;
            memset( &texDescr, 0, sizeof( cudaTextureDesc ) );
            texDescr.normalizedCoords = 1;
            texDescr.filterMode = _filter_mode;
            texDescr.addressMode[0] = _address_mode;
            texDescr.addressMode[1] = _address_mode;
            texDescr.addressMode[2] = cudaAddressModeClamp;
            texDescr.readMode = cudaReadModeElementType;
            checkNoorErrors( cudaCreateTextureObject( &texInput, &texRes, &texDescr, NULL ) );
            // generate surface object for writing
            cudaSurfaceObject_t surfOutput;
            cudaResourceDesc    surfRes;
            memset( &surfRes, 0, sizeof( cudaResourceDesc ) );
            surfRes.resType = cudaResourceTypeArray;
            surfRes.res.array.array = levelTo;
            checkNoorErrors( cudaCreateSurfaceObject( &surfOutput, &surfRes ) );
            // run mipmap kernel
            dim3 blockSize( 16, 16, 1 );
            dim3 gridSize( ( (uint) width + blockSize.x - 1 ) / blockSize.x, ( (uint) height + blockSize.y - 1 ) / blockSize.y, 1 );
            if ( _num_channels == 4 )
                mipmap<float4> << <gridSize, blockSize >> > ( surfOutput, texInput, (uint) width, (uint) height );
            else if ( _num_channels == 2 )
                mipmap<float2> << <gridSize, blockSize >> > ( surfOutput, texInput, (uint) width, (uint) height );
            else
                mipmap<float> << <gridSize, blockSize >> > ( surfOutput, texInput, (uint) width, (uint) height );
            checkNoorErrors( cudaDeviceSynchronize() );
            checkNoorErrors( cudaGetLastError() );
            checkNoorErrors( cudaDestroySurfaceObject( surfOutput ) );
            checkNoorErrors( cudaDestroyTextureObject( texInput ) );
            level++;
        }
    }

    __host__
        void free() {
        if ( _mipmapArray ) {
            checkNoorErrors( cudaFreeMipmappedArray( _mipmapArray ) );
        }
        if ( _read_tex_obj )
            checkNoorErrors( cudaDestroyTextureObject( _read_tex_obj ) );
        checkNoorErrors( cudaDestroySurfaceObject( _write_surface_obj ) );
    }

};

class Cuda2DTexture : public CudaTexture {
public:
    Cuda2DTexture() = default;
    __host__
        Cuda2DTexture( const ImageTexture& t ) :
        CudaTexture( make_cudaExtent( t.getWidth(), t.getHeight(), 0 ), t.getNumChannels() ) {
        cudaChannelFormatDesc channel_desc = t.getChannelDesc();
        checkNoorErrors( NOOR::malloc_array( &_array, &channel_desc, width(), height() ) );
        load( t );
        checkNoorErrors( NOOR::create_2d_texobj( &_read_tex_obj, _array, t.getFilterMode(), t.getAddressMode() ) );
        checkNoorErrors( NOOR::create_surfaceobj( &_write_surface_obj, _array ) );
    }

    __host__
        Cuda2DTexture( const float4& t ) :
        CudaTexture( make_cudaExtent( 1, 1, 0 ), 4 ) {
        checkNoorErrors( NOOR::malloc_array( &_array, &NOOR::_float4_channelDesc, width(), height() ) );
        checkNoorErrors( NOOR::memcopy_array( _array, (void*) &t, sizeof( float4 ) ) );
        checkNoorErrors( NOOR::create_2d_texobj( &_read_tex_obj, _array, _filter_mode, _address_mode ) );
        checkNoorErrors( NOOR::create_surfaceobj( &_write_surface_obj, _array ) );
    }

    __host__
        void update() {}

    __host__
        void free() {
        if ( _array ) checkNoorErrors( cudaFreeArray( _array ) );
        if ( _read_tex_obj ) checkNoorErrors( cudaDestroyTextureObject( _read_tex_obj ) );
        checkNoorErrors( cudaDestroySurfaceObject( _write_surface_obj ) );
    }
};

template<class T>
class CudaTextureManagerTemplate {
    size_t _num_textures;
    T* _device_textures;
public:
    CudaTextureManagerTemplate() = default;
    __host__
        CudaTextureManagerTemplate( const CudaPayload* payload ) :
        _num_textures( payload->_textures.size() ) {
        std::vector<T> host_textures;
        host_textures.reserve( _num_textures );
        for ( auto& t : payload->_textures ) {
            host_textures.push_back( T( t ) );
        }
        checkNoorErrors( NOOR::malloc( &_device_textures, _num_textures * sizeof( T ) ) );
        checkNoorErrors( NOOR::memcopy( _device_textures, &host_textures[0], _num_textures * sizeof( T ) ) );
    }

    __device__
        const T& getTexture( uint tex_idx ) const {
        return _device_textures[tex_idx];
    }

    __host__
        T getEnvTexture() {
        T t;
        checkNoorErrors( NOOR::memcopy( &t, &_device_textures[_num_textures - 1], sizeof( T ), cudaMemcpyDeviceToHost ) );
        return t;
    }

    __host__
        void free() {
        for ( int i = 0; i < _num_textures; ++i ) {
            T t;
            checkNoorErrors( NOOR::memcopy( &t, &_device_textures[i], sizeof( T ), cudaMemcpyDeviceToHost ) );
            t.free();
        }
        checkNoorErrors( cudaFree( _device_textures ) );
    }
};

//using CudaTextureManager = CudaTextureManagerTemplate<Cuda2DTexture>;
using CudaTextureManager = CudaTextureManagerTemplate<CudaMipMap>;
__constant__
CudaTextureManager _texture_manager;
#endif /* CUDATEXTURE_CUH */