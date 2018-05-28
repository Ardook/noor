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

#ifndef TEXTURE_H
#define TEXTURE_H
const cudaChannelFormatDesc _float4_channelDesc{ cudaCreateChannelDesc( 32, 32, 32, 32, cudaChannelFormatKindFloat ) };
const cudaChannelFormatDesc _float2_channelDesc{ cudaCreateChannelDesc( 32, 32, 0, 0, cudaChannelFormatKindFloat ) };
const cudaChannelFormatDesc _float_channelDesc{ cudaCreateChannelDesc( 32, 0, 0, 0, cudaChannelFormatKindFloat ) };
class ImageTexture {
public:
    std::string _filename{ "" };
    int _width{ 0 };
    int _height{ 0 };
    int _num_channels{ 0 };
    size_t _size_bytes{ 0 };
    float* _data{ nullptr };
    cudaTextureFilterMode _filter_mode{ cudaFilterModeLinear };
    cudaTextureAddressMode _address_mode{ cudaAddressModeWrap };
    cudaChannelFormatDesc _channel_desc{ _float4_channelDesc };

    ImageTexture() = default;
    ImageTexture( const ImageTexture& t ) = default;
    ImageTexture( int w,
                  int h,
                  cudaTextureFilterMode filter_mode = cudaFilterModeLinear,
                  cudaTextureAddressMode address_mode = cudaAddressModeWrap ) :
        _width( w )
        , _height( h )
        , _num_channels( 4 )
        , _size_bytes( w*h * sizeof( float4 ) )
        , _filter_mode( filter_mode )
        , _address_mode( address_mode ) {
        _data = new float[_width*_height * 4];
        memset( _data, 0, _size_bytes );
    }
    ImageTexture( const float3& c ) :
        _width( 1 )
        , _height( 1 )
        , _num_channels( 4 )
        , _size_bytes( sizeof( float4 ) )
        , _filter_mode( cudaFilterModePoint )
        , _address_mode( cudaAddressModeClamp ) {
        _data = new float[_num_channels];
        _data[0] = c.x;
        _data[1] = c.y;
        _data[2] = c.z;
        _data[3] = 1.0f;
    }
    ImageTexture( const float4& c ) :
        _width( 1 )
        , _height( 1 )
        , _num_channels( 4 )
        , _size_bytes( sizeof( float4 ) )
        , _filter_mode( cudaFilterModePoint )
        , _address_mode( cudaAddressModeClamp ) {
        _data = new float[_num_channels];
        _data[0] = c.x;
        _data[1] = c.y;
        _data[2] = c.z;
        _data[3] = c.w;
    }
    ImageTexture( const float2& c ) :
        _width( 1 )
        , _height( 1 )
        , _num_channels( 2 )
        , _size_bytes( sizeof( float2 ) )
        , _filter_mode( cudaFilterModePoint )
        , _address_mode( cudaAddressModeClamp )
        , _channel_desc( _float2_channelDesc ) {
        _data = new float[_num_channels];
        _data[0] = c.x;
        _data[1] = c.y;
    }
    ImageTexture& operator=( const ImageTexture& tex ) = delete;
    ImageTexture(
        const std::string& filename
        , cudaTextureFilterMode filter_mode = cudaFilterModeLinear
        , cudaTextureAddressMode address_mode = cudaAddressModeWrap
    ) :
        _filename( filename )
        , _filter_mode( filter_mode )
        , _address_mode( address_mode ) {
        load( filename );
    }

    ImageTexture& operator=( ImageTexture&& tex ) {
        if ( this == &tex )
            return *this;
        if ( _data != nullptr ) {
            delete[] _data; _data = nullptr;
        }
        _filename = tex._filename;
        _width = tex._width;
        _height = tex._height;
        _num_channels = tex._num_channels;
        _size_bytes = tex._size_bytes;
        _data = tex._data;
        _filter_mode = tex._filter_mode;
        _address_mode = tex._address_mode;
        _channel_desc = tex._channel_desc;
        tex._size_bytes = 0;
        tex._data = nullptr;
        return *this;
    }

    ImageTexture( ImageTexture&& tex ) :
        _filename( tex._filename )
        , _width( tex._width )
        , _height( tex._height )
        , _num_channels( tex._num_channels )
        , _size_bytes( tex._size_bytes )
        , _data( tex._data )
        , _filter_mode( tex._filter_mode )
        , _address_mode( tex._address_mode )
        , _channel_desc( tex._channel_desc ) {
        tex._size_bytes = 0;
        tex._data = nullptr;
    }

    ~ImageTexture() {
        if ( _data ) {
            delete[] _data;
            _data = nullptr;
        }
    }

    void load( const std::string& filename );

};
#endif /* TEXTURE_H */
