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
/*
pbrt source code is Copyright(c) 1998-2016
Matt Pharr, Greg Humphreys, and Wenzel Jakob.

This file is part of pbrt.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

- Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

- Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/
#ifndef CUDADISTRIBUTION_CUH
#define CUDADISTRIBUTION_CUH
/* based on PBRTTrowbridge Distribution */

class CudaTrowbridgeReitz {
    // TrowbridgeReitzDistribution Private Data
    float _alphax, _alphay;
public:
        CudaTrowbridgeReitz() = default;
    __device__
        CudaTrowbridgeReitz(
        float alphax
        , float alphay
        ) :
        _alphax( RoughnessToAlpha( alphax ) )
        , _alphay( RoughnessToAlpha( alphay ) ) {}

    __device__
        CudaTrowbridgeReitz( const float2 alpha ) :
        _alphax( RoughnessToAlpha( alpha.x ) )
        , _alphay( RoughnessToAlpha( alpha.y ) ) {}

    __device__ 
        float RoughnessToAlpha( float roughness ) {
        return fmaxf( roughness*roughness, 1e-3f );
    }
    __device__
        float D( const float3 &wh ) const {
        const float tan2Theta = Tan2Theta( wh );
        if ( isinf( tan2Theta ) ) return 0.f;
        const float cos4Theta = Cos2Theta( wh ) * Cos2Theta( wh );
        const float e =
            ( Cos2Phi( wh ) / ( _alphax * _alphax ) + Sin2Phi( wh ) / ( _alphay * _alphay ) ) *
            tan2Theta;
        return 1.0f / ( NOOR_PI * _alphax * _alphay * cos4Theta * ( 1.0f + e ) * ( 1.0f + e ) );
    }
    __device__
        float G( const float3 &wo, const float3 &wi ) const {
        return 1.f / ( 1.f + Lambda( wo ) + Lambda( wi ) );
    }
    __device__
        float Pdf( const float3 &wo, const float3 &wh ) const {
        return D( wh ) * G1( wo ) * NOOR::absDot( wo, wh ) / AbsCosTheta( wo );
    }

    __device__
        float3 Sample_wh( const float3 &wo, const float2 &u ) const {
        const float flip = wo.z < 0.0f ? -1.f : 1.f;
        return flip*TrowbridgeReitzSample( flip*wo, _alphax, _alphay, u.x, u.y );
    }
private:
    __device__
        float G1( const float3 &w ) const {
        return 1.f / ( 1.f + Lambda( w ) );
    }
    __device__
        float Lambda( const float3 &w ) const {
        const float absTanTheta = fabsf( TanTheta( w ) );
        if ( isinf( absTanTheta ) ) return 0.f;
        const float alpha =
            sqrtf( Cos2Phi( w ) * _alphax * _alphax + Sin2Phi( w ) * _alphay * _alphay );
        float alpha2Tan2Theta = ( alpha * absTanTheta ) * ( alpha * absTanTheta );
        return ( -1.0f + sqrtf( 1.f + alpha2Tan2Theta ) ) / 2.0f;
    }
    __device__
        void TrowbridgeReitzSample11( float cosTheta, float U1, float U2, float &slope_x, float &slope_y ) const {
        if ( cosTheta > .9999f ) {
            const float r = sqrtf( U1 / ( 1.0f - U1 ) );
            const float phi = 6.28318530718f * U2;
            slope_x = r * cosf( phi );
            slope_y = r * sinf( phi );
            return;
        }

        const float sinTheta = sqrtf( fmaxf( 0.0f, 1.0f - cosTheta * cosTheta ) );
        const float tanTheta = sinTheta / cosTheta;
        const float a = 1.0f / tanTheta;
        const float G1 = 2.0f / ( 1.0f + sqrtf( 1.f + 1.f / ( a * a ) ) );

        // sample slope_x
        const float A = 2.0f * U1 / G1 - 1.0f;
        float tmp = 1.f / ( A * A - 1.f );
        if ( tmp > 1e10f ) tmp = 1e10f;
        const float B = tanTheta;
        const float D = sqrtf( fmaxf( B * B * tmp * tmp - ( A * A - B * B ) * tmp, 0.0f ) );
        const float slope_x_1 = B * tmp - D;
        const float slope_x_2 = B * tmp + D;
        slope_x = ( A < 0 || slope_x_2 > 1.f / tanTheta ) ? slope_x_1 : slope_x_2;

        // sample slope_y
        float S;
        if ( U2 > 0.5f ) {
            S = 1.f;
            U2 = 2.f * ( U2 - .5f );
        } else {
            S = -1.f;
            U2 = 2.f * ( .5f - U2 );
        }
        float z =
            ( U2 * ( U2 * ( U2 * 0.27385f - 0.73369f ) + 0.46341f ) ) /
            ( U2 * ( U2 * ( U2 * 0.093073f + 0.309420f ) - 1.000000f ) + 0.597999f );
        slope_y = S * z * sqrtf( 1.f + slope_x * slope_x );
    }
    __device__
        float3 TrowbridgeReitzSample( const float3 &wi, float alpha_x, float alpha_y, float U1, float U2 ) const {
        // 1. stretch wi
        const float3 wiStretched =
            normalize( make_float3( alpha_x * wi.x, alpha_y * wi.y, wi.z ) );

        // 2. simulate P22_{wi}(x_slope, y_slope, 1, 1)
        float slope_x, slope_y;
        TrowbridgeReitzSample11( CosTheta( wiStretched ), U1, U2, slope_x, slope_y );

        // 3. rotate
        const float tmp = CosPhi( wiStretched ) * slope_x - SinPhi( wiStretched ) * slope_y;
        slope_y = SinPhi( wiStretched ) * slope_x + CosPhi( wiStretched ) * slope_y;
        slope_x = tmp;

        // 4. unstretch
        slope_x *= alpha_x;
        slope_y *= alpha_y;

        // 5. compute normal
        return normalize( make_float3( -slope_x, -slope_y, 1.f ) );
    }
};

class CudaBeckmannDistribution {
    float _alphax, _alphay;
public:
    // BeckmannDistribution Public Methods
    __forceinline__ __device__ __host__
        static float RoughnessToAlpha( float roughness ) {
        roughness = fmaxf( roughness, ( float )1e-4f );
        const float x = logf( roughness );
        return 1.62142f + 0.819955f * x + 0.1734f * x * x +
            0.0171201f * x * x * x + 0.000640711f * x * x * x * x;
    }
    __device__
        CudaBeckmannDistribution(
        float alphax
        , float alphay
        ) :
        _alphax( alphax )
        , _alphay( alphay ) {}
    __device__
        float D( const float3 &wh ) const {
        const float tan2Theta = Tan2Theta( wh );
        if ( isinf( tan2Theta ) ) return 0.0f;
        const float cos4Theta = Cos2Theta( wh ) * Cos2Theta( wh );
        return expf( -tan2Theta * ( Cos2Phi( wh ) / ( _alphax * _alphax ) +
                     Sin2Phi( wh ) / ( _alphay * _alphay ) ) ) /
                     ( NOOR_PI * _alphax * _alphay * cos4Theta );
    }
    __device__
        float G( const float3 &wo, const float3 &wi ) const {
        return 1.f / ( 1.f + Lambda( wo ) + Lambda( wi ) );
    }
    __device__
        float Pdf( const float3 &wo, const float3 &wh ) const {
        return D( wh ) * G1( wo ) * fabs( dot( wo, wh ) ) / AbsCosTheta( wo );
    }

    __device__
        float3 Sample_wh( const float3 &wo, const float2 &u ) const {
            // Sample visible area of normals for Beckmann distribution
        float3 wh;
        bool flip = wo.z < 0.0f;
        wh = BeckmannSample( flip ? -1.0f*wo : wo, _alphax, _alphay, u.x, u.y );
        if ( flip ) wh = -1.0f*wh;
        return wh;
    }
private:
    __device__
        float G1( const float3 &w ) const {
        return 1.f / ( 1.f + Lambda( w ) );
    }
    __device__
        // BeckmannDistribution Private Methods
        float Lambda( const float3 &w ) const {
        const float absTanTheta = abs( TanTheta( w ) );
        if ( isinf( absTanTheta ) ) return 0.0f;
        // Compute _alpha_ for direction _w_
        const float alpha = sqrtf( Cos2Phi( w ) * _alphax * _alphax + Sin2Phi( w ) * _alphay * _alphay );
        const float a = 1.f / ( alpha * absTanTheta );
        if ( a >= 1.6f ) return 0.0f;
        return ( 1.0f - 1.259f * a + 0.396f * a * a ) / ( 3.535f * a + 2.181f * a * a );
    }
    __device__
        void BeckmannSample11( float cosThetaI, float U1, float U2, float &slope_x, float &slope_y ) const {
        /* Special case (normal incidence) */
        if ( cosThetaI > .9999f ) {
            const float r = sqrtf( -logf( 1.0f - U1 ) );
            const float sinPhi = sinf( 2.0f * NOOR_PI * U2 );
            const float cosPhi = cosf( 2.0f * NOOR_PI * U2 );
            slope_x = r * cosPhi;
            slope_y = r * sinPhi;
            return;
        }

        /* The original inversion routine from the paper contained
        discontinuities, which causes issues for QMC integration
        and techniques like Kelemen-style MLT. The following code
        performs a numerical inversion with better behavior */
        const float sinThetaI = sqrtf( fmaxf( ( float ) 0.0f, ( float ) 1.0f - cosThetaI * cosThetaI ) );
        const float tanThetaI = sinThetaI / cosThetaI;
        const float cotThetaI = 1.0f / tanThetaI;

        /* Search interval -- everything is parameterized
        in the Erf() domain */
        float a = -1.0f;
        float c = NOOR::Erf( cotThetaI );
        const float sample_x = fmaxf( U1, ( float )1e-6f );

        /* Start with a good initial guess */
        // float b = (1-sample_x) * a + sample_x * c;

        /* We can do better (inverse of an approximation computed in
        * Mathematica) */
        const float thetaI = acosf( cosThetaI );
        const float fit = 1.0f + thetaI * ( -0.876f + thetaI * ( 0.4265f - 0.0594f * thetaI ) );
        float b = c - ( 1.0f + c ) * powf( 1.0f - sample_x, fit );

        /* Normalization factor for the CDF */
        float normalization =
            1.0f /
            ( 1.0f + c + sqrtf( NOOR_invPI ) * tanThetaI * expf( -cotThetaI * cotThetaI ) );

        int it = 0;
        while ( ++it < 10 ) {
            /* Bisection criterion -- the oddly-looking
            Boolean expression are intentional to check
            for NaNs at little additional cost */
            if ( !( b >= a && b <= c ) ) b = 0.5f * ( a + c );

            /* Evaluate the CDF and its derivative
            (i.e. the density function) */
            const float invErf = NOOR::ErfInv( b );
            const float value =
                normalization *
                ( 1.0f + b + sqrtf( NOOR_invPI ) * tanThetaI * expf( -invErf * invErf ) ) -
                sample_x;
            const float derivative = normalization * ( 1.0f - invErf * tanThetaI );

            if ( fabsf( value ) < 1e-5f ) break;

            /* Update bisection intervals */
            if ( value > 0.0f )
                c = b;
            else
                a = b;

            b -= value / derivative;
        }

        /* Now convert back into a slope value */
        slope_x = NOOR::ErfInv( b );

        /* Simulate Y component */
        slope_y = NOOR::ErfInv( 2.0f * fmaxf( U2, ( float )1e-6f ) - 1.0f );

    }
    __device__
        float3 BeckmannSample( const float3 &wi, float alpha_x, float alpha_y, float U1, float U2 ) const {
        // 1. stretch wi
        const float3 wiStretched =
            normalize( make_float3( alpha_x * wi.x, alpha_y * wi.y, wi.z ) );

        float slope_x, slope_y;
        BeckmannSample11( CosTheta( wiStretched ), U1, U2, slope_x, slope_y );

        // 3. rotate
        const float tmp = CosPhi( wiStretched ) * slope_x - SinPhi( wiStretched ) * slope_y;
        slope_y = SinPhi( wiStretched ) * slope_x + CosPhi( wiStretched ) * slope_y;
        slope_x = tmp;

        // 4. unstretch
        slope_x = alpha_x * slope_x;
        slope_y = alpha_y * slope_y;

        // 5. compute normal
        return normalize( make_float3( -slope_x, -slope_y, 1.f ) );
    }

};


#endif /* CUDADISTRIBUTION_CUH */
