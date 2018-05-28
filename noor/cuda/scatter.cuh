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
#ifndef SCATTER_CUH
#define SCATTER_CUH

template<typename BSDF>
__forceinline__ __device__
float3 scatter(
    const BSDF& bsdf
    , CudaIntersection& I
    , const CudaRNG& rng
    , float3& wi
) {
    BxDFType bsdfFlags = BSDF_ALL;
    BxDFType sampledType;
    const float2 u = make_float2( rng(), rng() );
    float pdf = 0.0f;
    const float3 f = bsdf.Sample_f( I, I._wo, wi, u, pdf, bsdfFlags, sampledType );
    if ( pdf == 0.0f )
        return _constant_spec._black;
    else
        return f * NOOR::absDot( wi, I._shading._n ) / pdf;
}
#endif /* SCATTER_CUH */