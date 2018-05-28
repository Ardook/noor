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
#ifndef BBOX_H
#define BBOX_H

class BBox {
    glm::vec3 _min{ std::numeric_limits<float>::infinity() };
    glm::vec3 _max{ -std::numeric_limits<float>::infinity() };
public:
    BBox() = default;

    BBox(
        const glm::vec3& min
        , const glm::vec3& max
    ) {
        _min = min;
        _max = max;
    }

    BBox(
        const glm::vec3& p0
        , const glm::vec3& p1
        , const glm::vec3& p2
    ) {
        _min = glm::min( p0, p1, p2 );
        _max = glm::max( p0, p1, p2 );
    }

    BBox(
        const glm::vec4& p0
        , const glm::vec4& p1
        , const glm::vec4& p2
    ) {
        _min = glm::min( p0, p1, p2 );
        _max = glm::max( p0, p1, p2 );
    }

    const glm::vec3& getMin() const { return _min; }
    const glm::vec3& getMax() const { return _max; }

    template<typename T>
    void setMin( const T& m ) { this->_min = m; }

    template<typename T>
    void setMax( const T& m ) { this->_max = m; }

    template<typename T>
    void include( const T& p ) {
        _min = glm::min( _min, glm::vec3( p ) );
        _max = glm::max( _max, glm::vec3( p ) );
    }

    bool isDegenerate() const {
        return glm::length( diagonal() ) <= std::numeric_limits<float>::epsilon();
    }

    bool contains( const glm::vec3& p ) const {
        const bool a = glm::all( glm::greaterThanEqual( p, _min ) );
        const bool b = glm::all( glm::lessThanEqual( p, _max ) );
        return a && b;
    }

    bool contains( const BBox& bbox ) const {
        const bool a = glm::all( glm::greaterThanEqual( bbox._min, _min ) );
        const bool b = glm::all( glm::lessThanEqual( bbox._max, _max ) );
        return a && b;
    }

    void merge( const BBox& bbox ) {
        _min = glm::min( _min, bbox._min );
        _max = glm::max( _max, bbox._max );
    }

    glm::uint8 maxExtent() const {
        const glm::vec3 d = diagonal();
        if ( d.x > d.y && d.x > d.z )
            return 0;
        else if ( d.y > d.z )
            return 1;
        else
            return 2;
    }

    glm::vec3 diagonal() const {
        return glm::vec3( _max - _min );
    }

    glm::vec3 centroid() const {
        return ( _min + _max )*0.5f;
    }

    float surfaceArea() const {
        glm::vec3 d = diagonal();
        if ( !this->isValid() ) {
            return  std::numeric_limits<float>::infinity();
        }
        return 2.0f*( d.x*d.y + d.y*d.z + d.x*d.z );
    }

    bool isValid()const {
        const bool isinf = !NOOR::isInf( _min ) && !NOOR::isInf( _max );
        const bool isnan = !NOOR::isNan( _min ) && !NOOR::isNan( _max );
        return isinf && isnan;
    }

    void boundingSphere( glm::vec3& center, float& radius ) const {
        center = ( _min - _max )*0.5f;
        radius = glm::length( center - _max );
    }

    float radius() const {
        return 0.5f*glm::length( _max - _min );
    }
    friend BBox operator*( const glm::mat4& m, const BBox& bb );
};

inline BBox operator*( const glm::mat4& m, const BBox& bb ) {
    const glm::vec3 t( m[3] );
    const glm::mat3 r( m );

    BBox result;
    const glm::vec3 p0 = m * glm::vec4( bb._min.x, bb._min.y, bb._min.z, 1.0f );
    const glm::vec3 p1 = m * glm::vec4( bb._min.x, bb._min.y, bb._max.z, 1.0f );
    const glm::vec3 p2 = m * glm::vec4( bb._min.x, bb._max.y, bb._min.z, 1.0f );
    const glm::vec3 p3 = m * glm::vec4( bb._min.x, bb._max.y, bb._max.z, 1.0f );

    const glm::vec3 p4 = m * glm::vec4( bb._max.x, bb._min.y, bb._min.z, 1.0f );
    const glm::vec3 p5 = m * glm::vec4( bb._max.x, bb._min.y, bb._max.z, 1.0f );
    const glm::vec3 p6 = m * glm::vec4( bb._max.x, bb._max.y, bb._min.z, 1.0f );
    const glm::vec3 p7 = m * glm::vec4( bb._max.x, bb._max.y, bb._max.z, 1.0f );

    result._min = glm::min( glm::min( glm::min( p0, p1, p2, p3 ), p4, p5, p6 ), p7 );
    result._max = glm::max( glm::max( glm::max( p0, p1, p2, p3 ), p4, p5, p6 ), p7 );
    return result;
}

inline bool overlap( const BBox& a, const BBox& b ) {
    if ( glm::all( glm::lessThan( a.getMax(), b.getMin() ) ) ) return false;
    if ( glm::all( glm::lessThan( b.getMax(), a.getMin() ) ) ) return false;
    return true;
}

inline BBox intersect( const BBox& a, const BBox& b ) {
    BBox result;
    result.setMin( glm::max( a.getMin(), b.getMin() ) );
    result.setMax( glm::min( a.getMax(), b.getMax() ) );
    return result;
}

inline float overlap_area( const BBox& a, const BBox& b ) {
    if ( !overlap( a, b ) ) return 0.0f;
    return intersect( a, b ).surfaceArea();
}

inline BBox merge( const BBox& lbbox, const BBox& rbbox ) {
    BBox result = lbbox;
    result.merge( rbbox );
    return result;
}


#endif /* BBOX_H */
