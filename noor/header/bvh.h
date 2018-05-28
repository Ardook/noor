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
#ifndef BVH_h
#define BVH_h
#include "splitter.h"
class Model;
class BVH {
    friend class Splitter;
    Model&   _model;
    std::vector<glm::uvec4> _bvh_min_right_start_nodes;
    std::vector<glm::uvec4> _bvh_max_axis_count_nodes;
    std::vector<glm::uint32> _bvh_mesh_start;

    glm::uint32 _bvh_root_node = 0;
    TriangleIndices& _triangle_indices;
    MeshIndices& _mesh_indices;
    Stat& _stat;
    const BVHSpec& _spec;
    Bin _root_bin;
    std::unique_ptr<Splitter> _splitter;
    glm::uint32 _num_boxes;
public:
    BVH( Model& model, const BVHSpec& spec, Stat& stat );
    ~BVH() = default;

    void loadCudaPayload( std::unique_ptr<CudaPayload>& payload );
private:
    glm::uint32 makeInner( const Bin& bin, glm::uint8 axis, glm::uint32 level );
    glm::uint32 makeLeaf( const Bin& bin, glm::uint32 level );
    glm::uint32 recursiveBuild( Bin& bin, glm::uint32 level );
    void build();
};

#endif /* BVH_h */
