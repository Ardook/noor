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

#ifndef TRIANGLE_H
#define TRIANGLE_H

struct WoopTriangle {
    glm::vec4 _woop[3];

    WoopTriangle(
        const glm::vec3& v0
        , const glm::vec3& v1
        , const glm::vec3& v2
        , const glm::vec3& fn ) {
        glm::mat4 m;
        m[0] = glm::vec4( v0 - v2, 0.0f );
        m[1] = glm::vec4( v1 - v2, 0.0f );
        m[2] = glm::vec4( fn, 0.0f );
        m[3] = glm::vec4( v2, 1.0f );

        m = glm::inverse( m );

        _woop[0] = glm::row( m, 2 ); _woop[0].w *= -1;
        _woop[1] = glm::row( m, 0 );
        _woop[2] = glm::row( m, 1 );
        if ( _woop[0].x == 0.0f )
            _woop[0].x = 0.0f;  // avoid degenerate coordinates
    }

    bool degenerate() const {
        return _woop[0].x == 0.0f;
    }
};

struct WaldTriangle {
    glm::uint8 _axis;
    float _nu;
    float _nv;
    float _nd;

    float _bnu;
    float _bnv;
    float _au;
    float _av;

    float _cnu;
    float _cnv;
    glm::uint32 _mat = DIFFUSE;
    glm::uint32 _deg = 0;

    WaldTriangle() = default;
    WaldTriangle(
        const glm::vec3& A
        , const glm::vec3& B
        , const glm::vec3& C
    ) {
        _axis = 0;
        const glm::vec3 b = C - A;
        const glm::vec3 c = B - A;
        const glm::vec3 N = glm::normalize( glm::cross( c, b ) );
        static const glm::uint32 waldModulo[4] = { 1u, 2u, 0u, 1u };
        for ( glm::uint32 i = 0; i < 3; ++i )
            _axis = fabsf( N[i] ) > fabsf( N[_axis] ) ? i : _axis;

        const glm::uint32 u = waldModulo[_axis + 0];
        const glm::uint32 v = waldModulo[_axis + 1];

        const float denom = b[u] * c[v] - b[v] * c[u];
        const float Nk = N[_axis];
        if ( denom == 0.0f ) {
            _deg = 1;
            return;
        }
        _nu = N[u] / Nk;
        _nv = N[v] / Nk;
        _nd = glm::dot( N, A ) / Nk;

        _bnu = b[u] / denom;
        _bnv = -b[v] / denom;
        _au = A[u];
        _av = A[v];
        _cnu = c[v] / denom;
        _cnv = -c[u] / denom;
        _deg = 0;
        _mat = 0;
    }
    glm::uvec4 get_ax_nu_nv_nd() {
        return glm::uvec4( _axis, NOOR::float_as_uint( _nu ), NOOR::float_as_uint( _nv ), NOOR::float_as_uint( _nd ) );
    }
    glm::uvec4 get_bnu_bnv_au_av() {
        return glm::uvec4( NOOR::float_as_uint( _bnu ), NOOR::float_as_uint( _bnv ), NOOR::float_as_uint( _au ), NOOR::float_as_uint( _av ) );
    }
    glm::uvec4 get_cnu_cnv_deg_mat() {
        return glm::uvec4( NOOR::float_as_uint( _cnu ), NOOR::float_as_uint( _cnv ), _deg, _mat );
    }
    bool degenerate() {
        return _deg == 1;
    }
};
#endif /* TRIANGLE_H */
