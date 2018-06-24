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
#version 430 core

subroutine vec4 tonemapType( vec3 color );
subroutine uniform tonemapType tonemap;

in vec2 UV;
layout( location = 0 ) out vec4 FragColor;

uniform sampler2D sampler;
uniform float exposure = 1.0f;
const vec3 gamma = vec3( 1.0f / 2.2f );

subroutine( tonemapType )
vec4 gammaCorrect( vec3 color ) {
    return vec4( pow( color.rgb, gamma ), 1.0f );
}

subroutine( tonemapType )
vec4 tonemapReinhard( vec3 color ) {
    return gammaCorrect( color / ( color + vec3( 1.0f ) ) );
}

subroutine( tonemapType )
vec4 tonemapFilmic( vec3 color ) {
    vec3 x = max( vec3( 0.0f ), color - 0.004f );
    return vec4( ( x * ( 6.2f * x + 0.5f ) ) / ( x * ( 6.2f * x + 1.7f ) + 0.06f ), 1.0f );
}

float A = 0.15f;
float B = 0.50f;
float C = 0.10f;
float D = 0.20f;
float E = 0.02f;
float F = 0.30f;
float W = 11.2f;

vec3 Uncharted2Tonemap( vec3 x ) {
    return ( ( x * ( A * x + C * B ) + D * E ) / ( x * ( A * x + B ) + D * F ) ) - E / F;
}

//Based on Filmic Tonemapping Operators http://filmicgames.com/archives/75
subroutine( tonemapType )
vec4 tonemapUncharted2( vec3 color ) {
    float ExposureBias = 4.0f;
    vec3 curr = Uncharted2Tonemap( ExposureBias * color );

    vec3 whiteScale = 1.0f / Uncharted2Tonemap( vec3( W ) );
    return gammaCorrect( curr * whiteScale );
}

void main() {
    // output gamma corrected color
    vec3 color = exposure *  texture( sampler, UV ).rgb;
    FragColor = tonemap( color );
}