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

#ifndef UTILS_CUH
#define UTILS_CUH

namespace NOOR {
    __forceinline__ __device__
        float SchlickWeight( float cosTheta ) {
        float m = clamp( 1.f - cosTheta, 0.f, 1.f );
        return ( m * m ) * ( m * m ) * m;
    }

    __forceinline__ __device__
        float3 sheen( const float3 &wo, const float3 &wi ) {
        float3 wh = wi + wo;
        if ( wh.x == 0 && wh.y == 0 && wh.z == 0 ) return make_float3( 0.f );
        wh = normalize( wh );
        float cosThetaD = dot( wi, wh );
        const float3 R = make_float3( 1.f );
        return R * SchlickWeight( cosThetaD );
    }

    template<typename T>
    __device__ __host__ __forceinline__
        void swap( T& a, T& b ) {
        T c( a ); a = b; b = c;
    }


    __forceinline__ __device__
        bool solveQuadratic( float a, float b, float c, float &x0, float &x1 ) {
        /* Linear case */
        if ( a == 0 ) {
            if ( b != 0 ) {
                x0 = x1 = -c / b;
                return true;
            }
            return false;
        }

        float discrim = b*b - 4.0f*a*c;

        if ( discrim < 0 )
            return false;

        float temp, sqrtDiscrim = sqrtf( discrim );

        /* Numerically stable version of (-b (+/-) sqrtDiscrim) / (2 * a)
        *
        * Based on the observation that one solution is always
        * accurate while the other is not. Finds the solution of
        * greater magnitude which does not suffer from loss of
        * precision and then uses the identity x1 * x2 = c / a
        */
        if ( b < 0 )
            temp = -0.5f * ( b - sqrtDiscrim );
        else
            temp = -0.5f * ( b + sqrtDiscrim );

        x0 = temp / a;
        x1 = c / temp;

        /* Return the results so that x0 < x1 */
        if ( x0 > x1 )
            NOOR::swap( x0, x1 );

        return true;
    }

    template<typename T>
    __device__ __host__ __forceinline__
        T saturate( const T& f ) {
        return clamp( f, 0.f, 1.f );
    }

    __device__ __host__ __forceinline__
        float Lerp( float t, float v1, float v2 ) { return ( 1.f - t ) * v1 + t * v2; }

    __device__ __host__ __forceinline__
        int sign( float val ) {
        return ( float( 0 ) < val ) - ( val < float( 0 ) );
    }

    __forceinline__ __device__ __host__
        /* Finds the closest power of two larger n */
        int nextPow2( int n ) {
        int next = n;
        next--;
        next |= next >> 1;
        next |= next >> 2;
        next |= next >> 4;
        next |= next >> 8;
        next |= next >> 16;
        next++;
        return next;
    }

    __forceinline__ __device__ __host__
        /* Finds the closest power of two smaller than n */
        int prevPow2( int n ) {
        int prev = n;
        prev = (int) log2f( (float) prev );
        prev = (int) powf( 2.0f, (float) prev );
        return prev;
    }

    __forceinline__ __device__ __host__
        /* Finds the closest (larger or smaller) power of two for n */
        int nearestPow2( int n ) {
        int next = nextPow2( n );
        int prev = prevPow2( n );
        return fabsf( float( n - next ) ) < fabsf( float( n - prev ) ) ? next : prev;
    }

    __forceinline__ __device__ __host__
        bool isBlack( const float3& c ) {
        return ( c.x == 0.0f && c.y == 0.0f && c.z == 0.0f );
    }

    __forceinline__ __device__ __host__
        int mod( int a, int b ) {
        const int result = a - ( a / b ) * b;
        return (int) ( ( result < 0 ) ? result + b : result );
    }

    __forceinline__ __device__ __host__
        float pow5f( float v ) {
        const float a = v*v;
        return a*a*v;
    }

    __forceinline__ __device__ __host__
        bool isnan3( const float3& c ) {
        return isnan( c.x ) || isnan( c.y ) || isnan( c.z );
    }

    __forceinline__ __device__ __host__
        bool isinf3( const float3& c ) {
        return isinf( c.x ) || isinf( c.y ) || isinf( c.z );
    }

    __forceinline__ __device__ __host__
        float3 sqrt3f( const float3& n ) {
        return make_float3( sqrtf( n.x ), sqrtf( n.y ), sqrtf( n.z ) );
    }

    __forceinline__ __device__ __host__
        float3 log3f( const float3& n ) {
        return make_float3( logf( n.x ), logf( n.y ), logf( n.z ) );
    }

    __forceinline__ __host__ __device__
        float3 exp3f( const float3& n ) {
        return make_float3( expf( n.x ), expf( n.y ), expf( n.z ) );
    }

    __forceinline__ __device__ __host__
        float3 pow3f( const float3& n, float p ) {
        return make_float3( powf( n.x, p ), powf( n.y, p ), powf( n.z, p ) );
    }
    __forceinline__ __device__ __host__
        float3 max3f( const float3& a, const float3& b ) {
        return make_float3( fmaxf( a.x, b.x ), fmaxf( a.y, b.y ), fmaxf( a.z, b.z ) );
    }
    __forceinline__ __device__ __host__
        float maxcomp( const float3& f ) {
        return fmaxf( fmaxf( f.x, f.y ), f.z );
    }

    __forceinline__ __device__ __host__
        float min3f( const float3& f ) {
        return fminf( fminf( f.x, f.y ), f.z );
    }

    __forceinline__ __device__
        float gammaCorrect( float value ) {
        if ( value <= 0.0031308f ) return 12.92f * value;
        return 1.055f * powf( value, (float) ( 1.f / 2.4f ) ) - 0.055f;
    }

    __forceinline__ __device__
        float3 gammaCorrect( float3 c ) {
        return make_float3( gammaCorrect( c.x ), gammaCorrect( c.y ), gammaCorrect( c.z ) );
    }

    __forceinline__ __device__
        float4 gammaCorrect( float4 c ) {
        return make_float4( gammaCorrect( c.x ), gammaCorrect( c.y ), gammaCorrect( c.z ), 1.0f );
    }

    __forceinline__ __device__
        float inverseGammaCorrect( float value ) {
        if ( value <= 0.04045f ) return value * 1.f / 12.92f;
        return powf( ( value + 0.055f ) * 1.f / 1.055f, ( float )2.4f );
    }

    __forceinline__ __device__
        float3 inverseGammaCorrect( float3 c ) {
        return make_float3( inverseGammaCorrect( c.x ), inverseGammaCorrect( c.y ), inverseGammaCorrect( c.z ) );
    }

    __forceinline__ __device__
        float4 inverseGammaCorrect( float4 c ) {
        return make_float4( inverseGammaCorrect( c.x ), inverseGammaCorrect( c.y ), inverseGammaCorrect( c.z ), 1.f );
    }

    __forceinline__ __device__ __host__
        uint WangHash( uint a ) {
        a = ( a ^ 61 ) ^ ( a >> 16 );
        a *= 9;
        a = a ^ ( a >> 4 );
        a *= 0x27d4eb2d;
        a = a ^ ( a >> 15 );
        return a;
    }

    // Color space conversions
    __forceinline__ __host__ __device__
        float3 Yxy2XYZ( const float3& Yxy ) {
        // avoid division by zero
        if ( Yxy.z < 1e-4 )
            return make_float3( 0.0f, 0.0f, 0.0f );

        return make_float3( Yxy.y * ( Yxy.x / Yxy.z ),
                            Yxy.x,
                            ( 1.0f - Yxy.y - Yxy.z ) * ( Yxy.x / Yxy.z ) );
    }

    __forceinline__ __host__ __device__
        float3 XYZ2rgb( const float3& xyz ) {
        const float R = dot( xyz, make_float3( 3.2410f, -1.5374f, -0.4986f ) );
        const float G = dot( xyz, make_float3( -0.9692f, 1.8760f, 0.0416f ) );
        const float B = dot( xyz, make_float3( 0.0556f, -0.2040f, 1.0570f ) );
        return make_float3( R, G, B );
    }

    __forceinline__ __host__ __device__
        float3 Yxy2rgb( const float3& Yxy ) {
        // avoid division by zero
        if ( Yxy.z < 1e-4 )
            return make_float3( 0.0f, 0.0f, 0.0f );

        // First convert to xyz
        const float3 xyz = make_float3( Yxy.y * ( Yxy.x / Yxy.z ),
                                        Yxy.x,
                                        ( 1.0f - Yxy.y - Yxy.z ) * ( Yxy.x / Yxy.z ) );

        const float R = dot( xyz, make_float3( 3.2410f, -1.5374f, -0.4986f ) );
        const float G = dot( xyz, make_float3( -0.9692f, 1.8760f, 0.0416f ) );
        const float B = dot( xyz, make_float3( 0.0556f, -0.2040f, 1.0570f ) );
        return make_float3( R, G, B );
    }

    __forceinline__ __host__ __device__
        float3 rgb2Yxy( const float3& rgb ) {
        // convert to xyz
        const float X = dot( rgb, make_float3( 0.4124f, 0.3576f, 0.1805f ) );
        const float Y = dot( rgb, make_float3( 0.2126f, 0.7152f, 0.0722f ) );
        const float Z = dot( rgb, make_float3( 0.0193f, 0.1192f, 0.9505f ) );

        // avoid division by zero
        // here we make the simplifying assumption that X, Y, Z are positive
        const float denominator = X + Y + Z;
        if ( denominator < 1e-4 )
            return make_float3( 0.0f, 0.0f, 0.0f );

        // convert xyz to Yxy
        return make_float3( Y,
                            X / ( denominator ),
                            Y / ( denominator ) );
    }

    __forceinline__ __host__ __device__
        float rgb2Y( const float3& rgb ) {
        // convert to xyz
        return dot( rgb, make_float3( 0.2126f, 0.7152f, 0.0722f ) );
    }

    __forceinline__ __host__ __device__
        float rgb2Y( const float4& rgba ) {
        // convert to xyz
        return rgb2Y( make_float3( rgba ) );
    }

    __forceinline__ __host__ __device__
        float3 tonemap( const float3 &hdr_value, float Y_log_av, float Y_max ) {
        const float3 val_Yxy = rgb2Yxy( hdr_value );

        float Y = val_Yxy.x; // Y channel is luminance
        const float a = 0.04f;
        float Y_rel = a * Y / Y_log_av;
        float mapped_Y = Y_rel * ( 1.0f + Y_rel / ( Y_max * Y_max ) ) / ( 1.0f + Y_rel );

        float3 mapped_Yxy = make_float3( mapped_Y, val_Yxy.y, val_Yxy.z );
        float3 mapped_rgb = Yxy2rgb( mapped_Yxy );

        return mapped_rgb;
    }

    __forceinline__ __device__ __host__
        float3 faceforward( const float3& n, const float3& v ) {
        if ( dot( v, n ) < 0 )
            return -1.0f*n;
        else
            return n;
    }

    __forceinline__ __host__ __device__
        float rad2deg( float radians ) {
        return radians * 180.0f / NOOR_PI;
    }

    __forceinline__ __host__ __device__
        float deg2rad( float degrees ) {
        return degrees * NOOR_PI / 180.0f;
    }

    __forceinline__ __host__ __device__
        void coordinateSystem( const float3 &v1, float3& v2, float3& v3 ) {
        if ( fabsf( v1.x ) > fabsf( v1.y ) )
            v2 = make_float3( -v1.z, 0, v1.x ) / sqrtf( v1.x * v1.x + v1.z * v1.z );
        else
            v2 = make_float3( 0, v1.z, -v1.y ) / sqrtf( v1.y * v1.y + v1.z * v1.z );
        v3 = cross( v1, v2 );
    }

    __forceinline__ __host__ __device__
        float3 uniformSampleSphere( const float2 &u, HandedNess handedness = RIGHT_HANDED ) {
        const float z = 1.f - 2.f * u.x;
        const float r = sqrtf( fmaxf( 0.f, 1.f - z * z ) );
        const float phi = 2.f * NOOR_PI * u.y;
        return handedness == RIGHT_HANDED ?
            make_float3( r * cosf( phi ), z, r * sinf( phi ) )
            :
            make_float3( r * cosf( phi ), r * sinf( phi ), z );
    }

    __forceinline__ __host__ __device__
        float uniformSpherePdf() { return NOOR_inv4PI; }

    __forceinline__ __host__ __device__
        float3 uniformSampleHemisphere( const float2 &u, HandedNess handedness = RIGHT_HANDED ) {
        const float z = u.x;
        const float r = sqrtf( fmaxf( 0.f, 1.f - z * z ) );
        const float phi = 2.f * NOOR_PI * u.y;
        return handedness == RIGHT_HANDED ?
            make_float3( r * cosf( phi ), z, r * sinf( phi ) )
            :
            make_float3( r * cosf( phi ), r * sinf( phi ), z );
    }

    __forceinline__ __host__ __device__
        float uniformHemiSpherePdf() { return NOOR_inv2PI; }

    __forceinline__ __host__ __device__
        float2 concentricSampleDisk( const float2& u ) {
        // Map uniform random numbers to $[-1,1]^2$
        const float2 uOffset = 2.f * u - make_float2( 1, 1 );

        // Handle degeneracy at the origin
        if ( uOffset.x == 0 && uOffset.y == 0 ) return make_float2( 0, 0 );

        // Apply concentric mapping to point
        float theta, r;
        if ( fabsf( uOffset.x ) > fabsf( uOffset.y ) ) {
            r = uOffset.x;
            theta = NOOR_PI_over_4 * ( uOffset.y / uOffset.x );
        } else {
            r = uOffset.y;
            theta = NOOR_PI_over_2 - NOOR_PI_over_4 * ( uOffset.x / uOffset.y );
        }
        return r * make_float2( cosf( theta ), sinf( theta ) );
    }

    __forceinline__ __host__ __device__
        float3 cosineSampleHemisphere( const float2 &u, HandedNess handedness = LEFT_HANDED ) {
        const float2 d = concentricSampleDisk( u );
        const float z = sqrtf( fmaxf( 0.0f, 1 - d.x * d.x - d.y * d.y ) );
        return handedness == RIGHT_HANDED ?
            make_float3( d.x, z, d.y ) :
            make_float3( d.x, d.y, z );
    }

    __forceinline__ __host__ __device__
        bool sameHemisphere( const float3 &wi, const float3 &wo ) {
        return dot( wi, wo ) > 0.f;
    }

    __forceinline__ __host__ __device__
        float sphericalPhi( const float3 &v, HandedNess handedness = RIGHT_HANDED ) {
        float p = handedness == RIGHT_HANDED ? atan2f( v.z, v.x ): p = atan2f( v.y, v.x );
        return ( p < 0 ) ? ( p + NOOR_2PI ) : p;
    }

    __forceinline__ __host__ __device__
        float sphericalTheta( const float3 &v, HandedNess handedness = RIGHT_HANDED ) {
        return (handedness == RIGHT_HANDED ? acosf( clamp( v.y, -1.f, 1.f ) ): acosf( clamp( v.z, -1.f, 1.f ) ));
    }

    __forceinline__ __host__ __device__
        float3 uniformSampleCone( const float2 &u, float cosThetaMax ) {
        const float cosTheta = ( 1 - u.x ) + u.x * cosThetaMax;
        const float sinTheta = std::sqrt( (float) 1 - cosTheta * cosTheta );
        const float phi = u.y * NOOR_2PI;
        return make_float3( cosf( phi ) * sinTheta, sinf( phi ) * sinTheta, cosTheta );
    }

    __forceinline__ __host__ __device__
        float3 uniformSampleCone( const float2 &u, float cosThetaMax,
                                  const float3 &x, const float3 &y,
                                  const float3 &z ) {
        const float cosTheta = Lerp( u.x, cosThetaMax, 1.f );
        const float sinTheta = sqrtf( 1.f - cosTheta * cosTheta );
        const float phi = u.y * NOOR_2PI;
        return cosf( phi ) * sinTheta * x + sinf( phi ) * sinTheta * y + cosTheta * z;
    }

    __forceinline__ __host__ __device__
        float uniformConePdf( float cosThetaMax ) {
        return NOOR_inv2PI / ( 1.f - cosThetaMax );
    }

    
    __forceinline__ __host__ __device__
        float3 sphericalDirection( float sinTheta, float cosTheta, float phi,
                                   const float3 &x, const float3 &y, const float3 &z
        ) {
        return sinTheta * cosf( phi ) * x + sinTheta * sinf( phi ) * y + cosTheta * z;
    }

    __forceinline__ __host__ __device__
        float3 sphericalDirection( float sinTheta, float cosTheta, float phi, HandedNess handedness = RIGHT_HANDED ) {
        if ( handedness == RIGHT_HANDED )
            return normalize( make_float3( sinTheta * cosf( phi ), cosTheta, sinTheta * sinf( phi ) ) );
        else
            return normalize( make_float3( sinTheta * cosf( phi ), sinTheta * sinf( phi ), cosTheta ) );
    }
    __forceinline__ __host__ __device__
        float3 sphericalDirection( float theta, float phi, HandedNess handedness = RIGHT_HANDED ) {
        const float sinTheta = sinf( theta );
        const float cosTheta = cosf( theta );
        return sphericalDirection( sinTheta, cosTheta, phi, handedness );
    }
    __forceinline__ __host__ __device__
        float3 sphericalDirection( const float2& u, float& sinTheta ) {
        const float phi = u.x * NOOR_2PI;
        const float theta = u.y * NOOR_PI;
        const float cosTheta = cosf( theta );
        sinTheta = sinf( theta );
        return sphericalDirection( sinTheta, cosTheta, phi );
    }

    __forceinline__ __host__ __device__
        float balanceHeuristic( int nf, float fPdf, int ng, float gPdf ) {
        return ( nf * fPdf ) / ( nf * fPdf + ng * gPdf );
    }
    __forceinline__ __host__ __device__
        float powerHeuristic( int nf, float fPdf, int ng, float gPdf ) {
        float f = nf * fPdf, g = ng * gPdf;
        return ( f * f ) / ( f * f + g * g );
    }
    __forceinline__ __host__ __device__
        float ErfInv( float x ) {
        float w, p;
        x = clamp( x, -.99999f, .99999f );
        w = -logf( ( 1 - x ) * ( 1 + x ) );
        if ( w < 5 ) {
            w = w - 2.5f;
            p = 2.81022636e-08f;
            p = 3.43273939e-07f + p * w;
            p = -3.5233877e-06f + p * w;
            p = -4.39150654e-06f + p * w;
            p = 0.00021858087f + p * w;
            p = -0.00125372503f + p * w;
            p = -0.00417768164f + p * w;
            p = 0.246640727f + p * w;
            p = 1.50140941f + p * w;
        } else {
            w = sqrtf( w ) - 3;
            p = -0.000200214257f;
            p = 0.000100950558f + p * w;
            p = 0.00134934322f + p * w;
            p = -0.00367342844f + p * w;
            p = 0.00573950773f + p * w;
            p = -0.0076224613f + p * w;
            p = 0.00943887047f + p * w;
            p = 1.00167406f + p * w;
            p = 2.83297682f + p * w;
        }
        return p * x;
    }
    __forceinline__ __host__ __device__
        float Erf( float x ) {
        // constants
        float a1 = 0.254829592f;
        float a2 = -0.284496736f;
        float a3 = 1.421413741f;
        float a4 = -1.453152027f;
        float a5 = 1.061405429f;
        float p = 0.3275911f;

        // Save the sign of x
        int sign = 1;
        if ( x < 0 ) sign = -1;
        x = fabsf( x );

        // A&S formula 7.1.26
        float t = 1 / ( 1 + p * x );
        float y =
            1 -
            ( ( ( ( ( a5 * t + a4 ) * t ) + a3 ) * t + a2 ) * t + a1 ) * t * expf( -x * x );

        return sign * y;
    }

    __forceinline__ __host__ __device__
        float length2( const float3& v ) {
        return dot( v, v );
    }
    __forceinline__ __host__ __device__
        float3 normalize( const float3& v ) {
        const float l = length2( v );
        if ( l == 0.0f ) return v;
        return v * rsqrtf( l );
    }
    __forceinline__ __host__ __device__
        float absDot( const float3 &v1, const float3 &v2 ) {
        return fabsf( dot( v1, v2 ) );
    }

    template<class C>
    __forceinline__
        cudaError_t malloc_array( cudaArray_t* cuArray, C* channelDesc, size_t w, size_t h = 0 ) {
        return cudaMallocArray( cuArray, channelDesc, w, h );
    }

    __forceinline__
        cudaError_t memcopy_array( cudaArray_t cuArray, const void *data, size_t size, cudaMemcpyKind kind = cudaMemcpyHostToDevice ) {
        return cudaMemcpyToArray( cuArray, 0, 0, data, size, kind );
    }

    template<class T>
    __forceinline__
        cudaError_t memcopy_symbol( T* d_tex_obj, const T* s_tex_obj, cudaMemcpyKind kind = cudaMemcpyHostToDevice ) {
        return cudaMemcpyToSymbol( *d_tex_obj, (void*) s_tex_obj, sizeof( T ), 0, kind );
    }

    template<class T>
    __forceinline__
        cudaError_t memcopy_symbol_async( T* d_tex_obj, const T* s_tex_obj, cudaMemcpyKind kind = cudaMemcpyHostToDevice ) {
        return cudaMemcpyToSymbolAsync( *d_tex_obj, (void*) s_tex_obj, sizeof( T ), 0, kind );
    }

    template<class T>
    __forceinline__
        cudaError_t malloc( T** devPtr, size_t size ) {
        return cudaMalloc( (void**) devPtr, size );
    }

    template<class T>
    __forceinline__
        cudaError_t malloc_managed( T **devPtr, size_t size ) {
        return cudaMallocManaged( (void**) devPtr, size );
    }

    __forceinline__
        cudaError_t memcopy( void *device, const void *host, size_t size, cudaMemcpyKind kind = cudaMemcpyHostToDevice ) {
        return cudaMemcpy( device, host, size, kind );
    }

    __forceinline__
        cudaError_t memcopy_async( void *device, const void *host, size_t size, cudaMemcpyKind kind = cudaMemcpyHostToDevice ) {
        return cudaMemcpyAsync( device, host, size, kind );
    }

    __forceinline__
        cudaError_t create_1d_texobj(
        cudaTextureObject_t* tex_object
        , void* buffer
        , size_t bytes
        , cudaChannelFormatDesc channel
        ) {
        cudaResourceDesc resDesc;
        memset( &resDesc, 0, sizeof( cudaResourceDesc ) );
        resDesc.resType = cudaResourceTypeLinear;
        resDesc.res.linear.devPtr = buffer;
        resDesc.res.linear.desc = channel;
        resDesc.res.linear.sizeInBytes = bytes;
        cudaTextureDesc texDesc;
        memset( &texDesc, 0, sizeof( cudaTextureDesc ) );
        texDesc.addressMode[0] = cudaAddressModeWrap;
        texDesc.addressMode[1] = cudaAddressModeWrap;
        texDesc.filterMode = cudaFilterModePoint;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = false;
        return cudaCreateTextureObject( tex_object, &resDesc, &texDesc, nullptr );
    }

    __forceinline__
        cudaError_t create_2d_texobj(
        cudaTextureObject_t* tex_object
        , cudaArray* cuda_array
        , cudaTextureFilterMode filter_mode
        , cudaTextureAddressMode address_mode = cudaAddressModeWrap
        , bool normalized = true
        ) {
        cudaResourceDesc resDesc;
        memset( &resDesc, 0, sizeof( cudaResourceDesc ) );
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = cuda_array;
        cudaTextureDesc texDesc;
        memset( &texDesc, 0, sizeof( cudaTextureDesc ) );
        texDesc.addressMode[0] = address_mode;
        texDesc.addressMode[1] = address_mode;
        texDesc.filterMode = filter_mode;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = normalized;
        return cudaCreateTextureObject( tex_object, &resDesc, &texDesc, nullptr );
    }

    __forceinline__
        cudaError_t create_surfaceobj(
        cudaSurfaceObject_t* surface_obj
        , cudaArray* cuda_array
        ) {
        cudaResourceDesc resDesc;
        memset( &resDesc, 0, sizeof( cudaResourceDesc ) );
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = cuda_array;
        return cudaCreateSurfaceObject( surface_obj, &resDesc );
    }
#ifdef __CUDACC__
    template<typename T>
    __forceinline__ __device__
        T print( T v ) {
        return v;
    }
    __forceinline__ __device__
        static void print( const char* s, const float3& v ) {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if ( x == 511 && y == 511 )
            printf( "%s: %f %f %f\n", s, v.x, v.y, v.z );
    }
    __forceinline__ __device__
        static void print( const char* s, const float4& v ) {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if ( x == 511 && y == 511 )
            printf( "%s: %f %f %f %f\n", s, v.x, v.y, v.z, v.w );
    }
    __forceinline__ __device__
        static void print( const char* s, const float2& v ) {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if ( x == 511 && y == 511 )
            printf( "%s: %f %f\n", s, v.x, v.y );
    }
    cudaChannelFormatDesc _float4_channelDesc = cudaCreateChannelDesc( 32, 32, 32, 32, cudaChannelFormatKindFloat );
    cudaChannelFormatDesc _float2_channelDesc = cudaCreateChannelDesc( 32, 32, 0, 0, cudaChannelFormatKindFloat );
    cudaChannelFormatDesc _float_channelDesc = cudaCreateChannelDesc( 32, 0, 0, 0, cudaChannelFormatKindFloat );
    cudaChannelFormatDesc _uint4_channelDesc = cudaCreateChannelDesc( 32, 32, 32, 32, cudaChannelFormatKindUnsigned );
    cudaChannelFormatDesc _uint2_channelDesc = cudaCreateChannelDesc( 32, 32, 0, 0, cudaChannelFormatKindUnsigned );
    cudaChannelFormatDesc _uint_channelDesc = cudaCreateChannelDesc( 32, 0, 0, 0, cudaChannelFormatKindUnsigned );
#endif /* __CUDACC__ */
} // namespace NOOR
#endif /* UTILS_CUH */
