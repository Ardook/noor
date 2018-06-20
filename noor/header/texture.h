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

class Texture {
protected:
    float* _data{ nullptr };
    int _width;
    int _height;
    int _num_channels;
    size_t _size_bytes;
    cudaTextureFilterMode _filter_mode;
    cudaTextureAddressMode _address_mode;
    cudaChannelFormatDesc _channel_desc;

public:
    Texture() = default;
    Texture(
        int width,
        int height,
        int num_channels,
        cudaTextureFilterMode filter_mode,
        cudaTextureAddressMode address_mode,
        cudaChannelFormatDesc channel_desc
    ) :
        _width( width ),
        _height( height ),
        _num_channels( num_channels ),
        _filter_mode( filter_mode ),
        _address_mode( address_mode ),
        _channel_desc( channel_desc ) 
    {
        _num_channels = _num_channels == 3 ? 4 : _num_channels;
        int n = _width * _height * _num_channels;
        _size_bytes = sizeof( float ) * n;
        _data = new float[n];
        _channel_desc = _num_channels == 4 ?  _float4_channelDesc :
            ( _num_channels == 2 ? _float2_channelDesc : _float_channelDesc );
    }

    ~Texture() {
        if ( _data != nullptr ) {
            delete[] _data;
            _data = nullptr;
        }
    }

    Texture( const Texture& t ) = default;
    //	ImageTexture& operator=(const ImageTexture& tex) = delete;

    Texture( Texture&& tex ) :
        _width( tex._width )
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

    Texture& operator=( Texture&& tex ) {
        if ( this == &tex )
            return *this;
        if ( _data != nullptr ) {
            delete[] _data; _data = nullptr;
        }
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
    const float* getData() const {
        return _data;
    }
    int getWidth()const {
        return _width;
    }
    int getHeight()const {
        return _height;
    }
    int getNumChannels()const {
        return _num_channels;
    }
    cudaChannelFormatDesc getChannelDesc() const {
        return _channel_desc;
    }
    cudaTextureAddressMode getAddressMode() const {
        return _address_mode;
    }
    cudaTextureFilterMode getFilterMode() const {
        return _filter_mode;
    }
};

class ImageTexture : public Texture {
    void load( const std::string& filename );
public:
    ImageTexture() = default;

    template<typename T>
    ImageTexture( const T& c ) :
        Texture( 1, 1,
                 sizeof( T ) / sizeof( float ),
                 cudaFilterModePoint,
                 cudaAddressModeClamp,
                 _float4_channelDesc
        )
    {
        memset( _data, 1, _size_bytes );
        memcpy( _data, &c, sizeof( c ) );
    }

    ImageTexture(
        const std::string& filename,
        cudaTextureFilterMode filter_mode = cudaFilterModeLinear,
        cudaTextureAddressMode address_mode = cudaAddressModeWrap
    ) : Texture() {
        _filter_mode = filter_mode;
        _address_mode = address_mode;
        load( filename );
    }

    ImageTexture(
        int w,
        int h,
        int num_channels = 4,
        cudaTextureFilterMode filter_mode = cudaFilterModeLinear,
        cudaTextureAddressMode address_mode = cudaAddressModeWrap,
        cudaChannelFormatDesc channel_desc = _float4_channelDesc
    ) : Texture( w, h, num_channels, filter_mode, address_mode, channel_desc ) {
        memset( _data, 0, _size_bytes );
    }
};
#endif /* TEXTURE_H */
