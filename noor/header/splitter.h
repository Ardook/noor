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

#ifndef Splitter_h
#define Splitter_h

using TriangleIndices = std::vector<glm::uint32>;
using TriangleIndex = std::vector<glm::uint32>::iterator;

using MeshIndices = std::vector<glm::uint32>;
using MeshIndex = std::vector<glm::uint32>::iterator;
struct SAH {
    float _sah_cost;
    float _loc;
    glm::uint8 _axis;
    glm::uint32 _index;
    glm::uint32 _left_count;
    glm::uint32 _right_count;
    float _overlap_area;


    SAH() :
        _sah_cost( std::numeric_limits<float>::infinity() )
        , _loc( 0.0f )
        , _axis( 0 )
        , _index( 0 )
        , _left_count( 0 )
        , _right_count( 0 )
        , _overlap_area( 0.0f ) {}

    SAH( const SAH& s ) = default;
    SAH& operator=( const SAH& s ) = default;
    bool isValid()const {
        const bool inf = ( _sah_cost != std::numeric_limits<float>::infinity() );
        const bool cnt = ( _left_count != 0 && _right_count != 0 );
        return ( inf && cnt );
    }
    bool operator<( const SAH& s )const {
        return _sah_cost < s._sah_cost;
    }
};

enum Bintype { TRIBIN = 0, MESHBIN };
struct Bin {
    TriangleIndex _start;
    TriangleIndex _end;

    BBox _bbox;
    BBox _centbox;

    glm::uint32 _count;
    Bintype _type = TRIBIN;
    Bin() = default;

    Bin( Bin&& bin ) = default;
    Bin& operator=( Bin&& bin ) = default;

    Bin(
        TriangleIndex begin
        , TriangleIndex end
        , const BBox& bbox
        , const BBox& centbox
        , Bintype type = TRIBIN
    ) :
        _start( begin )
        , _end( end )
        , _bbox( bbox )
        , _centbox( centbox )
        , _type( type ) {
        _count = this->count();
    }

    glm::uint32 count() const {
        return static_cast<glm::uint32>( std::distance( _start, _end ) );
    }

};

struct Split {
    glm::uint32 _num_bins;
    float _inv_num_bins;

    glm::vec3 _dt;
    glm::vec3 _lb;

    std::array<std::vector<BBox>, 3> _left;
    std::array<std::vector<BBox>, 3> _right;

    std::array<std::vector<BBox>, 3> _left_centbox;
    std::array<std::vector<BBox>, 3> _right_centbox;

    std::array<std::vector<glm::uint32>, 3> _enter;
    std::array<std::vector<glm::uint32>, 3> _exit;

    std::vector<BBox> _boundries;

    Split( glm::uint32 num_bins ) :
        _num_bins( num_bins )
        , _dt( glm::vec3( 0.0f ) )
        , _lb( glm::vec3( 0.0f ) ) {
        _inv_num_bins = 1.0f / _num_bins;
        _boundries.resize( _num_bins );
        for ( char i = 0; i < 3; ++i ) {
            _left[i].resize( _num_bins );
            _right[i].resize( _num_bins );
            _left_centbox[i].resize( _num_bins );
            _right_centbox[i].resize( _num_bins );
            _enter[i].resize( _num_bins );
            _exit[i].resize( _num_bins );
        }
    }

    void reset( const Bin& bin ) {
        _dt = ( bin._centbox.getMax() - bin._centbox.getMin() ) * _inv_num_bins;
        _lb = bin._centbox.getMin();

        _boundries[0].setMin( _lb );
        _boundries[0].setMax( _lb + _dt );
        for ( glm::uint32 i = 1; i < _num_bins; ++i ) {
            _boundries[i].setMin( _boundries[i - 1].getMin() + _dt );
            _boundries[i].setMax( _boundries[i - 1].getMax() + _dt );
        }
        _boundries[_num_bins - 1].setMax( bin._centbox.getMax() );
        static BBox empty;
        for ( char i = 0; i < 3; ++i ) {
            std::fill_n( _enter[i].begin(), _num_bins, 0 );
            std::fill_n( _exit[i].begin(), _num_bins, 0 );
            std::fill_n( _left[i].begin(), _num_bins, empty );
            std::fill_n( _right[i].begin(), _num_bins, empty );
            std::fill_n( _left_centbox[i].begin(), _num_bins, empty );
            std::fill_n( _right_centbox[i].begin(), _num_bins, empty );
        }
    }
};

class BVH;
class Splitter {
    BVH& _bvh;
    Split _split;
    glm::uint32 _num_bins;
public:
    Splitter(
        BVH& bvh
        , const BVHSpec& spec
    ) :
        _bvh( bvh )
        , _split( Split( spec._num_bins ) )
        , _num_bins( spec._num_bins ) {}
    SAH findBestSplit( const Bin& bin );
    void split( Bin& bin, Bin& left, Bin& right, const SAH& sah );
private:
    glm::uint32 binSearch( const glm::vec3& centroid, glm::uint8 axis );
    void performBinning( const Bin& bin, glm::uint8 axis );
    SAH findBestSplit( const Bin& bin, glm::uint8 axis );
    const BBox& getBBox( const Bin& bin, glm::uint32 i ) const;
};

#endif /* splitter_h */
