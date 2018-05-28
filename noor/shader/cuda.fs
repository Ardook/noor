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
#version 330 core

// Interpolated values from the vertex shaders
in vec2 UV;

// Values that stay constant for the whole mesh.
uniform sampler2D sampler;

const vec3 gamma = vec3( 0.45f );

vec3 gammaCorrect( vec3 color ) {
	return pow( color.rgb, gamma );
}

vec4 gammaCorrect( vec4 color ) {
	return vec4( pow( color.rgb, gamma ), 1.0 );
}

vec3 tonemapReinhard( vec3 color ) {
	return gammaCorrect( color / ( color + vec3( 1.0 ) ) );
}

vec3 tonemapFilmic( vec3 color ) {
	vec3 x = max( vec3( 0.0 ), color - 0.004 );
	return ( x * ( 6.2 * x + 0.5 ) ) / ( x * ( 6.2 * x + 1.7 ) + 0.06 );
}

float A = 0.15;
float B = 0.50;
float C = 0.10;
float D = 0.20;
float E = 0.02;
float F = 0.30;
float W = 11.2;

vec3 Uncharted2Tonemap( vec3 x ) {
	return ( ( x * ( A * x + C * B ) + D * E ) / ( x * ( A * x + B ) + D * F ) ) - E / F;
}

//Based on Filmic Tonemapping Operators http://filmicgames.com/archives/75
vec3 tonemapUncharted2( vec3 color ) {
	float ExposureBias = 2.0;
	vec3 curr = Uncharted2Tonemap( ExposureBias * color );

	vec3 whiteScale = 1.0 / Uncharted2Tonemap( vec3( W ) );
	return gammaCorrect( curr * whiteScale );
}


void main() {
	// output gamma corrected color
	vec3 color = vec3( texture( sampler, UV ) );
	// TODO: divide by color.w for filtering
	// gl_FragColor = vec4(pow(color.rgb/color.w, gamma),1.0);
	//gl_FragColor = vec4( pow( color, gamma ), 1.0 );
	//gl_FragColor = vec4(tonemapFilmic(vec3(color)), 1.f);
	gl_FragColor = vec4( tonemapReinhard( color ), 1.f );
	//gl_FragColor = vec4( gammaCorrect( color ), 1.f );
	//gl_FragColor = vec4( tonemapUncharted2(vec3(color)), 1.f);
}