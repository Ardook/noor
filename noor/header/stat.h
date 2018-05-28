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
#ifndef STAT_H
#define STAT_H
class Stat {
public:
    glm::uint32 _max_leaf_tris = 0;
    glm::uint32 _num_innernodes = 0;
    glm::uint32 _num_leafnodes = 0;
    glm::uint32 _height = 0;

    glm::uint32 _num_meshes = 0;
    glm::uint32 _num_triangles = 0;
    glm::uint32 _num_vertices = 0;
    glm::uint32 _num_normals = 0;
    glm::uint32 _num_tex_coordinates = 0;
    glm::uint32 _num_material_groups = 0;
    glm::uint32 _num_materials = 0;
    glm::uint32 _num_area_lights = 0;
    glm::uint32 _num_point_lights = 0;
    glm::uint32 _num_spot_lights = 0;
    glm::uint32 _num_distant_lights = 0;
    glm::uint32 _num_infinite_lights = 0;

    float _process_model_duration = 0;
    float _process_bvh_duration = 0;
    float _model_payload_duration = 0;
};

inline std::ostream& operator<<( std::ostream& os, const Stat& stat ) {
    std::string duration_stat( "duration stat:\n" );
    duration_stat += "\t-process model duration:  " + NOOR::to_string( stat._process_model_duration ) + " seconds\n";
    duration_stat += "\t-process BVH duration:    " + NOOR::to_string( stat._process_bvh_duration ) + " seconds\n";
    duration_stat += "\t-model payload duration:  " + NOOR::to_string( stat._model_payload_duration ) + " seconds\n";
    std::string model_stat( "\nmodel stat:\n" );
    model_stat += "\t-num meshes:	" + NOOR::to_string( stat._num_meshes ) + "\n";
    model_stat += "\t-num triangles:  " + NOOR::to_string( stat._num_triangles ) + "\n";
    model_stat += "\t-num vertices:   " + NOOR::to_string( stat._num_vertices ) + "\n";
    model_stat += "\t-num normals:    " + NOOR::to_string( stat._num_normals ) + "\n";
    model_stat += "\t-num tex coords: " + NOOR::to_string( stat._num_tex_coordinates ) + "\n";
    model_stat += "\t-num materials:  " + NOOR::to_string( stat._num_materials ) + "\n";
    model_stat += "\t-num groups:	 " + NOOR::to_string( stat._num_material_groups ) + "\n";
    model_stat += "\t-num area lights:  " + NOOR::to_string( stat._num_area_lights ) + "\n";
    model_stat += "\t-num point lights: " + NOOR::to_string( stat._num_point_lights ) + "\n";
    model_stat += "\t-num spot lights:  " + NOOR::to_string( stat._num_spot_lights ) + "\n";
    model_stat += "\t-num dist lights:  " + NOOR::to_string( stat._num_distant_lights ) + "\n";
    model_stat += "\t-num infinite lights:  " + NOOR::to_string( stat._num_infinite_lights ) + "\n";
    std::string bvh_stat( "\nbvh stat:\n" );
    bvh_stat += "\t-max_leaf:      " + NOOR::to_string( stat._max_leaf_tris ) + "\n";
    bvh_stat += "\t-inner nodes:   " + NOOR::to_string( stat._num_innernodes ) + "\n";
    bvh_stat += "\t-leaf nodes:    " + NOOR::to_string( stat._num_leafnodes ) + "\n";
    bvh_stat += "\t-total nodes:   " + NOOR::to_string( stat._num_innernodes + stat._num_leafnodes ) + "\n";
    bvh_stat += "\t-bvh height:    " + NOOR::to_string( stat._height ) + "\n";
    os << duration_stat + model_stat + bvh_stat;
    return os;
}
#endif /* STAT_H */