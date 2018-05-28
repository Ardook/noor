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
#include "splitter.h"
#include "model.h"
#include "bvh.h"
inline const BBox& Splitter::getBBox( const Bin& bin, glm::uint32 i ) const {
    if ( bin._type == TRIBIN )
        return _bvh._model.getTriBBox( i );
    else
        return _bvh._model.getInstanceMeshBBox( i );
}

inline glm::uint32 Splitter::binSearch( const glm::vec3& centroid, glm::uint8 axis ) {
    int first = 1, last = _num_bins - 1;

    float left = _split._boundries[0].getMax()[axis];
    float right = _split._boundries[last].getMin()[axis];

    const float center = centroid[axis];
    if ( center <= left ) return 0;
    if ( center > right ) return last;
    int h = 0, m = 0;
    while ( last > 0 ) {
        h = last >> 1;
        m = first + h;
        left = _split._boundries[m].getMin()[axis];
        right = _split._boundries[m].getMax()[axis];
        if ( left < center && center <= right )
            break;
        else if ( center <= left )
            last = h;
        else {
            first = m + 1;
            last -= h + 1;
        }
    }
    return m;
}

void Splitter::performBinning( const Bin& bin, glm::uint8 axis ) {
    const TriangleIndex begin = bin._start;
    const TriangleIndex end = bin._end;
    const TriangleIndex bin_end;

    glm::uint32 index;
    auto lambda = [&]( glm::uint32 i ) {
        const BBox& bbox = getBBox( bin, i );
        const glm::vec3 centroid = bbox.centroid();
        index = binSearch( centroid, axis );
        _split._left[axis][index].merge( bbox );
        _split._left_centbox[axis][index].include( centroid );
        ++_split._enter[axis][index];
    };
    std::for_each( begin, end, lambda );
}

SAH Splitter::findBestSplit( const Bin& bin, glm::uint8 axis ) {
    performBinning( bin, axis );

    const glm::uint32 n = _bvh._spec._num_bins - 1;

    _split._right[axis][n] = _split._left[axis][n];
    _split._right_centbox[axis][n] = _split._left_centbox[axis][n];
    _split._exit[axis][n] = _split._enter[axis][n];

    for ( glm::uint32 i = n - 1; i > 0; --i ) {
        _split._right[axis][i] = merge( _split._left[axis][i], _split._right[axis][i + 1] );
        _split._right_centbox[axis][i] = merge( _split._left_centbox[axis][i], _split._right_centbox[axis][i + 1] );
        _split._exit[axis][i] = _split._enter[axis][i] + _split._exit[axis][i + 1];
    }

    for ( glm::uint32 i = 1; i < n; ++i ) {
        _split._left[axis][i].merge( _split._left[axis][i - 1] );
        _split._left_centbox[axis][i].merge( _split._left_centbox[axis][i - 1] );
        _split._enter[axis][i] += _split._enter[axis][i - 1];
    }

    float curr_sah_cost = 0.0f;

    SAH min_sah;
    for ( glm::uint32 i = 0; i < n; ++i ) {
        const glm::uint32 left_count = _split._enter[axis][i];
        const glm::uint32 right_count = _split._exit[axis][i + 1];
        if ( left_count == 0 || right_count == 0 ) {
            continue;
        }
        const float left_cost = _split._enter[axis][i] * _split._left[axis][i].surfaceArea();
        const float right_cost = _split._exit[axis][i + 1] * _split._right[axis][i + 1].surfaceArea();

        curr_sah_cost = _bvh._spec._Ct + _bvh._spec._Ci * ( left_cost + right_cost ) / bin._bbox.surfaceArea();
        if ( _split._enter[axis][i] != 0 && _split._exit[axis][i + 1] != 0 && curr_sah_cost < min_sah._sah_cost ) {
            min_sah._sah_cost = curr_sah_cost;
            min_sah._index = i;
            min_sah._loc = _split._boundries[i].getMax()[axis];
            min_sah._axis = axis;
            min_sah._left_count = _split._enter[axis][i];
            min_sah._right_count = _split._exit[axis][i + 1];
        }
    }
    return min_sah;
}

SAH Splitter::findBestSplit( const Bin& bin ) {
    _split.reset( bin );

    SAH curr_sah;
    SAH min_sah;
    for ( int axis = 0; axis < 3; ++axis ) {
        curr_sah = findBestSplit( bin, axis );
        if ( curr_sah.isValid() && curr_sah < min_sah ) {
            min_sah = curr_sah;
        }
    }
    return min_sah;
}

void Splitter::split( Bin& bin, Bin& left, Bin& right, const SAH& sah ) {
    auto lambda = [&]( glm::uint32 i ) {
        const BBox& bb = ( bin._type == TRIBIN ) ? _bvh._model.getTriBBox( i ) : _bvh._model.getInstanceMeshBBox( i );
        return bb.centroid()[sah._axis] <= _split._boundries[sah._index].getMax()[sah._axis];
    };
    const TriangleIndex left_end = std::partition( bin._start, bin._end, lambda );
    left = Bin(
        bin._start
        , left_end
        , _split._left[sah._axis][sah._index]
        , _split._left_centbox[sah._axis][sah._index]
        , bin._type
    );
    right = Bin(
        left_end
        , bin._end
        , _split._right[sah._axis][sah._index + 1]
        , _split._right_centbox[sah._axis][sah._index + 1]
        , bin._type
    );
    assert( left.count() == _split._enter[sah._axis][sah._index] );
    assert( right.count() == _split._exit[sah._axis][sah._index + 1] );
}


