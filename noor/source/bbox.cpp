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
#include "pch.h"

void BBox::glBuffers(
	std::vector<glm::vec3>& vertices
	, std::vector<glm::vec3>& colors
	, std::vector<glm::uint32>& indices
	, const glm::vec3& color
	, bool isMeshBBox
) const {
	assert( this->isValid() );
	const glm::vec3 d = _max - _min;
	float extend = isMeshBBox ? glm::length( d ) / 400.0f : 0.0f;
	glm::vec3 delta( extend );
	const glm::vec3 dx( d.x + extend, 0, 0 );
	const glm::vec3 dy( 0, d.y + extend, 0 );
	const glm::vec3 dz( 0, 0, d.z + extend );
	//           6-------7
	//          /|      /|
	//         / |     / |
	//        2-------4  |
	//        |  3----|--5
	//        | /     | /
	//        0-------1
	vertices.push_back( _min - delta );	//0
	vertices.push_back( _min + dx );	//1
	vertices.push_back( _min + dy );	//2
	vertices.push_back( _min + dz );	//3
	vertices.push_back( _min + dx + dy );	//4
	vertices.push_back( _min + dx + dz );	//5
	vertices.push_back( _min + dy + dz );	//6
	vertices.push_back( _min + dx + dy + dz );	//7

	// colors
	colors.push_back( color );	//0
	colors.push_back( color );	//1
	colors.push_back( color );	//2
	colors.push_back( color );	//3
	colors.push_back( color );	//4
	colors.push_back( color );	//5
	colors.push_back( color );	//6
	colors.push_back( color );	//7

	const glm::uint32 n = static_cast<glm::uint32>( vertices.size() ) - 8u;

	indices.push_back( n + 0 );
	indices.push_back( n + 1 );

	indices.push_back( n + 0 );
	indices.push_back( n + 2 );

	indices.push_back( n + 0 );
	indices.push_back( n + 3 );

	indices.push_back( n + 1 );
	indices.push_back( n + 4 );

	indices.push_back( n + 1 );
	indices.push_back( n + 5 );

	indices.push_back( n + 3 );
	indices.push_back( n + 6 );

	indices.push_back( n + 3 );
	indices.push_back( n + 5 );

	indices.push_back( n + 4 );
	indices.push_back( n + 2 );

	indices.push_back( n + 4 );
	indices.push_back( n + 7 );

	indices.push_back( n + 5 );
	indices.push_back( n + 7 );

	indices.push_back( n + 2 );
	indices.push_back( n + 6 );

	indices.push_back( n + 7 );
	indices.push_back( n + 6 );
}
