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
#ifndef CUDAFRESNEL_CUH
#define CUDAFRESNEL_CUH
/* based on PBRT Fresnel */

__forceinline__ __device__
float fresnelDielectric( float cosThetaI, float etaI, float etaT ) {
    cosThetaI = clamp( cosThetaI, -1.0f, 1.0f );
    const bool entering = cosThetaI > 0.f;
    if ( !entering ) {
        NOOR::swap( etaI, etaT );
        cosThetaI = fabsf( cosThetaI );
    }

    // Compute _cosThetaT_ using Snell's law
    const float sinThetaI = sqrtf( fmaxf( 0.0f, 1.0f - cosThetaI * cosThetaI ) );
    const float sinThetaT = etaI / etaT * sinThetaI;

    // Handle total internal reflection
    if ( sinThetaT >= 1.0f ) return 1.0f;
    const float cosThetaT = sqrtf( fmaxf( 0.0f, 1.0f - sinThetaT * sinThetaT ) );
    const float Rparl = ( ( etaT * cosThetaI ) - ( etaI * cosThetaT ) ) / 
        ( ( etaT * cosThetaI ) + ( etaI * cosThetaT ) );
    const float Rperp = ( ( etaI * cosThetaI ) - ( etaT * cosThetaT ) ) / 
        ( ( etaI * cosThetaI ) + ( etaT * cosThetaT ) );
    const float result = ( Rparl * Rparl + Rperp * Rperp ) / 2.0f;
    return result;
}
__forceinline__ __device__
float3 fresnelConductor( float cosThetaI, 
                         const float3& etaI, 
                         const float3& etaT, 
                         const float3& k ) {
    cosThetaI = clamp( cosThetaI, -1.0f, 1.0f );
    const float3 eta = etaT / etaI;
    const float3 etak = k / etaI;

    const float cosThetaI2 = cosThetaI * cosThetaI;
    const float sinThetaI2 = 1.0f - cosThetaI2;
    const float3 eta2 = eta * eta;
    const float3 etak2 = etak * etak;

    const float3 t0 = eta2 - etak2 - sinThetaI2;
    const float3 a2plusb2 = NOOR::sqrt3f( t0 * t0 + 4.0f * eta2 * etak2 );
    const float3 t1 = a2plusb2 + cosThetaI2;
    const float3 a = NOOR::sqrt3f( 0.5f * ( a2plusb2 + t0 ) );
    const float3 t2 = 2.0f * cosThetaI * a;
    const float3 Rs = ( t1 - t2 ) / ( t1 + t2 );

    const float3 t3 = cosThetaI2 * a2plusb2 + sinThetaI2 * sinThetaI2;
    const float3 t4 = t2 * sinThetaI2;
    const float3 Rp = Rs * ( t3 - t4 ) / ( t3 + t4 );

    return 0.5f * ( Rp + Rs );
}
enum FresnelType {
    DIELECTRIC = 0,
    CONDUCTOR,
    NOOP
};
class CudaFresnel{
public:
    float3 _etaI;
    float3 _etaT;
    float3 _k;
    FresnelType _type{ DIELECTRIC };
    __device__
        CudaFresnel( const float3 &etaI, const float3 &etaT, const float3 &k )
        : _etaI( etaI ), _etaT( etaT ), _k( k ), _type(CONDUCTOR) {}
    __device__
        CudaFresnel( const float3& etaI, const float3& etaT ) : 
        _etaI( etaI ), _etaT( etaT ), _type(DIELECTRIC)  {}
    __device__
        CudaFresnel() : _type(NOOP)  {}
    __device__
        float3 evaluate( float cosThetaI ) const {
        switch ( _type ) {
            case CONDUCTOR:
                return fresnelConductor( cosThetaI, _etaI, _etaT, _k );
            case DIELECTRIC: 
                return make_float3( fresnelDielectric( cosThetaI, _etaI.x, _etaT.x ) );
            default:
                return make_float3( 1.f );
        }
    }
};

#endif /* CUDAFRESNEL_CUH */
