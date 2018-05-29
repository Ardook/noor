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
#ifndef CUDAMATERIAL_CUH
#define CUDAMATERIAL_CUH
class CudaMaterial {
public:
    float2 _diffuse_uvscale{ make_float2( 1 ) };
    float2 _specular_uvscale{ make_float2( 1 ) };
    float2 _transparency_uvscale{ make_float2( 1 ) };
    float2 _emittance_uvscale{ make_float2( 1 ) };
    float2 _bump_uvscale{ make_float2( 1 ) };
    float2 _roughness_uvscale{ make_float2( 1 ) };
    float2 _metalness_uvscale{ make_float2( 1 ) };

    uint _diffuse_tex_idx{ 0 };
    uint _specular_tex_idx{ 0 };
    uint _transmission_tex_idx{ 0 };
    uint _emittance_tex_idx{ 0 };
    uint _bump_tex_idx{ 0 };
    uint _transparency_tex_idx{ 0 };
    uint _ior_tex_idx{ 0 };
    uint _k_tex_idx{ 0 };
    uint _roughness_tex_idx{ 0 };
    uint _metalness_tex_idx{ 0 };

    float _bumpfactor{ 1.f };
    float _coat_weight{ 1.f };
    float _coat_roughness{ .001f };
    float _coat_thickness{ 1.f };
    float _coat_sigma{ 0.f };
    float _coat_ior{ 1.5f };

    uint _type;

    CudaMaterial() = default;
};

#endif