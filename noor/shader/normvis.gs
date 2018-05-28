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

layout( triangles ) in;
layout( line_strip, max_vertices = 6 ) out;

uniform float normal_length;

in VS_OUT{
	vec4 normal;
} gs_in[];

out GS_OUT{
	vec4 color;
} gs_out;


vec4 normal_color = vec4( 1, 0, 0, 1 );

void main( void ) {
	float scale = normal_length * 0.01f;
	for ( int i = 0; i < 3; ++i ) {
		// normal
		gl_Position = gl_in[i].gl_Position + scale*gs_in[i].normal;
		gs_out.color = normal_color;
		EmitVertex();
		gl_Position = ( gl_in[i].gl_Position + gs_in[i].normal * normal_length );
		gs_out.color = normal_color;
		EmitVertex();
		EndPrimitive();
	}
}
