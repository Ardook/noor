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
#ifndef OPTIONS_H
#define OPTIONS_H

const std::string texture_dir( "../../docs/scenes/textures/" );
const std::vector<std::string> hdr_files = {
    /*0*/std::string( texture_dir + "hdri/env.hdr" ),
    /*1*/std::string( texture_dir + "hdri/skylight-morn.exr" ),
    /*2*/std::string( texture_dir + "hdri/pisa_latlong.exr" ),
    /*3*/std::string( texture_dir + "hdri/sky.exr" ),
    /*4*/std::string( texture_dir + "hdri/hdri_sky_01_full.hdr" ),
    /*5*/std::string( texture_dir + "hdri/hdri_sky_02_full.hdr" ),
    /*6*/std::string( texture_dir + "hdri/hdri_sky_03_full.hdr" ),
    /*7*/std::string( texture_dir + "hdri/georgentor_4k.hdr" ),
    /*8*/std::string( texture_dir + "hdri/vignaioli_night_4k.hdr" ),
    /*9*/std::string( texture_dir + "hdri/hotel_room_4k.hdr" ),
    /*10*/std::string( texture_dir + "hdri/hall_con.hdr" ),
    /*11*/std::string( texture_dir + "hdri/envmap.exr" ),
    /*12*/std::string( texture_dir + "hdri/harties_4k.hdr" ),
    /*13*/std::string( texture_dir + "hdri/flower_road_4k.hdr" ),
    /*14*/std::string( texture_dir + "hdri/noon_grass_4k.hdr" ),
    /*15*/std::string( texture_dir + "hdri/umhlanga_sunrise_4k.hdr" ),
    /*16*/std::string( texture_dir + "hdri/moonlit_golf_4k.hdr" ),
    /*17*/std::string( texture_dir + "hdri/kiara_6_afternoon_4k.hdr" ),
    /*18*/std::string( texture_dir + "hdri/flower_road_4k.hdr" ),
    /*19*/std::string( texture_dir + "hdri/symmetrical_garden_4k.hdr" ),
    /*20*/std::string( texture_dir + "hdri/aft_lounge_4k.hdr" ),
    /*21*/std::string( texture_dir + "hdri/royal_esplanade_4k.hdr" ),
    /*22*/std::string( texture_dir + "hdri/satara_night_8k.hdr" ),
    /*23*/std::string( texture_dir + "hdri/umhlanga_sunrise_8k.hdr" ),
    /*24*/std::string( texture_dir + "hdri/cayley_interior_8k.hdr" ),
    /*25*/std::string( texture_dir + "hdri/canada_montreal_loft_max_sunny.exr" ),
    /*26*/std::string( texture_dir + "hdri/canada_montreal_nad_cafeteria_bright.exr" ),
    /*27*/std::string( texture_dir + "hdri/canada_montreal_nad_photorealism.exr" ),
    /*28*/std::string( texture_dir + "hdri/canada_montreal_pierre_bathroom.exr" ),
    /*29*/std::string( texture_dir + "hdri/canada_montreal_pierre_kitchen.exr" ),
    /*30*/std::string( texture_dir + "hdri/canada_montreal_thea.exr" ),
};


const std::string models_dir( "../../docs/scenes/models/" );
const std::vector<std::string> model_files = {
    /*0*/   std::string( models_dir + "obj/cornell-box/CornellBox-Original.obj" )
    /*1*/   ,std::string( models_dir + "obj/cornell-box/CornellBox-Sphere.obj" )
    /*2*/   ,std::string( models_dir + "obj/cornell-box/CornellBox-Glossy.obj" )
    /*3*/   ,std::string( models_dir + "obj/cornell-box/CornellBox-Mirror.obj" )
    /*4*/   ,std::string( models_dir + "obj/mitsuba/mitsuba.obj" )
    /*5*/   ,std::string( models_dir + "obj/head/head.obj" )
    /*6*/   ,std::string( models_dir + "obj/f16/f16.obj" )
    /*7*/   ,std::string( models_dir + "obj/dabrovic-sponza/sponza.obj" )
    /*8*/   ,std::string( models_dir + "obj/lost-empire/lost_empire.obj" )
    /*9*/   ,std::string( models_dir + "obj/rungholt/rungholt3.obj" )
    /*10*/  ,std::string( models_dir + "obj/sgi/sgi.obj" )
    /*11*/  ,std::string( models_dir + "obj/bunny/bunny.obj" )
    /*12*/  ,std::string( models_dir + "obj/chess/chessbox.obj" )
    /*13*/  ,std::string( models_dir + "obj/treespaceship/treespaceship.obj" )
    /*14*/  ,std::string( models_dir + "obj/trees/tree.obj" )
    /*15*/  ,std::string( models_dir + "obj/rose/rose.obj" )
    /*16*/  ,std::string( models_dir + "obj/corvette/corvette.obj" )
    /*17*/  ,std::string( models_dir + "obj/lonelytree/lonelytree.obj" )
    /*18*/  ,std::string( models_dir + "obj/A380/A380.obj" )
    /*19*/  ,std::string( models_dir + "obj/transparency/treecube.obj" )
    /*20*/  ,std::string( models_dir + "obj/transparency/leaf.obj" )
    /*21*/  ,std::string( models_dir + "obj/sibenik/sibenik.obj" )
    /*22*/  ,std::string( models_dir + "obj/powerplant/powerplant.obj" )
    /*23*/  ,std::string( models_dir + "obj/dragon.obj" )
    /*24*/  ,std::string( models_dir + "obj/buddha.obj" )
    /*25*/  ,std::string( models_dir + "obj/hairball.obj" )
    /*26*/  ,std::string( models_dir + "obj/dragon_girl/dragon_girl.obj" )
    /*27*/  ,std::string( models_dir + "obj/teapot/teapot.obj" )
    /*28*/  ,std::string( models_dir + "obj/dragon/dragon.obj" )
    /*29*/  ,std::string( models_dir + "obj/testing/testing.obj" )
    /*30*/  ,std::string( models_dir + "obj/testing/baloondog.obj" )
    /*31*/  ,std::string( models_dir + "obj/testing/dragon.obj" )
    /*32*/  ,std::string( models_dir + "fbx/cornellbox_original.fbx" )
    /*33*/  ,std::string( models_dir + "fbx/cornellbox_spheres.fbx" )
    /*34*/  ,std::string( models_dir + "fbx/dragon.fbx" )
    /*35*/  ,std::string( models_dir + "fbx/museum.fbx" )
    /*36*/  ,std::string( models_dir + "fbx/train.fbx" )
    /*37*/  ,std::string( models_dir + "fbx/treasure.fbx" )
    /*38*/  ,std::string( models_dir + "fbx/cabin.fbx" )
    /*39*/  ,std::string( models_dir + "fbx/carnival.fbx" )
    /*40*/  ,std::string( models_dir + "fbx/lighthouse.fbx" )
    /*41*/  ,std::string( models_dir + "fbx/frozentest.fbx" )
    /*42*/  ,std::string( models_dir + "fbx/instancedtest.fbx" )
    /*43*/  ,std::string( models_dir + "fbx/lighttest.fbx" )
    /*44*/  ,std::string( models_dir + "fbx/indoor.fbx" )
    /*45*/  ,std::string( models_dir + "fbx/outdoor.fbx" )
    /*46*/  ,std::string( models_dir + "fbx/combo_lores.fbx" )
    /*47*/  ,std::string( models_dir + "fbx/combo_hires.fbx" )
    /*48*/  ,std::string( models_dir + "fbx/combo.fbx" )
    /*49*/  ,std::string( models_dir + "fbx/test.fbx" )
    /*50*/  ,std::string( models_dir + "fbx/temp.fbx" )
};

std::unordered_map<std::string, float> options = {
{ "-model",(float) 0},
{ "-hdr",(float) 0 },
{ "-skydome",(float) 0 },
{ "-width",(float) 1024 },
{ "-height",(float) 1024 },
{ "-lens_radius", 0.0f },
{ "-focal_length", 5.0f },
{ "-samples",(float) 1 },
{ "-bounces",(float) 2 },
{ "-rr",(float) 1 },
{ "-num_gpus",(float) 2 },
{ "-bvh_num_bins",(float) 16 },
{ "-bvh_max_height",(float) 32 },
{ "-bvh_min_leaf_tris",(float) 2 },
{ "-bvh_max_leaf_tris",(float) 2 },
{ "-bvh_ci", 1.0f },
{ "-bvh_ct", 0.125f }
};
const std::string help = "\
Example Options:\n \
 -model 0 (index of the model file)\n \
 -hdr 0  (index of the hdr file)\n \
 -skydome 0  (0-hdr, 1-physical )\n \
 -width 1024 (width in pixels)\n \
 -height 1024 (height in pixels)\n \
 -lens_radius 0 (lens radius)\n \
 -focal_length 5 (focal distance)\n \
 -bounces 3 (max path tracing bounces)\n \
 -rr 2 (Russian roulette depth)\n \
 -bvh_max_height 32 (max height of tree (make leaf)\n \
 -bvh_num_bins 16 (bvh build number of bins)\n \
 -bvh_min_leaf_tris 2 (min num tris to create leaf node)\n \
 -bvh_max_leaf_tris 2 (max num tris early leaf node creation)\n \
 -bvh_ci 1.0 (traversal cost object splitter)\n \
 -bvh_ct 0.125 (traversal cost object splitter)\n ";

const std::string keyboard_shortcuts = "\
Keyboard shortcuts:\n \
 esc\t(exit)\n \
 1	(draw fps)\n \
 2	(enable/disable env light)\n \
 3	(enable/disable env debug view)\n \
 4	(enable/disable env debug view importance sample)\n \
 5	(switch to persp cam)\n \
 6	(switch to ortho cam)\n \
 7	(switch to env cam)\n \
 F9	    (tonemap: Gamma correct)\n \
 F10	(tonemap: Reinhard)\n \
 F11	(tonemap: Filmic)\n \
 F12	(tonemap: Uncharted)\n \
\nMouse Control:\n \
 -Left mouse button orbit/rotate. \n \
 -Right mouse button zoom in/out. \n \
 -Middle mouse button strafe. \n";

BVHSpec bvh_spec{
    static_cast<glm::uint32>( options["-bvh_num_bins"] ) /* num_bins */
    ,static_cast<glm::uint32>( options["-bvh_max_height"] ) /* max_height */
    , options["-bvh_ci"] /* Ci */
    , options["-bvh_ct"] /* Ct object*/
    ,static_cast<glm::uint32>( options["-bvh_min_leaf_tris"] ) /* min_leaf_tris */
    ,static_cast<glm::uint32>( options["-bvh_max_leaf_tris"] ) /* max_leaf_tris */
};

CameraSpec camera_spec{
    glm::vec3( 0.0f, 0.0f, 3.0f ), // eye
    glm::vec3( 0.0f, 1.0f, 0.0f ), // up
    glm::vec3( 0.0f, 0.0f, 0.0f ), // lookAt
    glm::pi<float>() / 4.f,
    static_cast<glm::uint32>( options["-width"] ), // width in pixels
    static_cast<glm::uint32>( options["-height"] ), // height in pixels
    options["-lens_radius"], // lens_radius
    options["-focal_lenth"], // focal_length
    static_cast<glm::uint8>( options["-bounces"] ),
    static_cast<glm::uint8>( options["-rr"] ), // Russian Roulette
};

ModelSpec model_spec;
Spec noor_spec( bvh_spec, camera_spec, model_spec );

inline void printHelp() {
    std::cout << keyboard_shortcuts << std::endl;
    std::cout << help << std::endl;
    std::cout << "model files:  \n";
    int count = 0;
    for ( const std::string& model_file : model_files ) {
        std::cout << "\t" << count++ << "- " << model_file << std::endl;
    }
    count = 0;
    std::cout << "\nhdr files:  \n";
    for ( const std::string& hdr_file : hdr_files ) {
        std::cout << "\t" << count++ << "- " << hdr_file << std::endl;
    }
}
inline bool is_numeric( const char *s ) {
    bool single_dot = false;
    return std::all_of( s, s + std::strlen( s ),
                        [&single_dot]( char c ) {
        if ( c == '.' ) {
            if ( single_dot )
                return false;
            else
                single_dot = true;
        }
        return isdigit( c ) || c == '.';
    } );
}
inline bool parseOptions( int argc, char * argv[] ) {
    int num_models = static_cast<int>( model_files.size() );
    int num_hdrs = static_cast<int>( hdr_files.size() );
    for ( int i = 1; i < argc; i += 2 ) {
        if ( strcmp( argv[i], "help" ) == 0 || strcmp( argv[i], "-help" ) == 0 ) {
            printHelp();
            return false;
        }
        if ( options.find( argv[i] ) != options.end() ) {
            if ( argv[i + 1] == nullptr || !is_numeric( argv[i + 1] ) ) {
                std::cerr << "Invalid argument for " << argv[i] << std::endl;
                return false;
            }
            float argument = std::stof( argv[i + 1] );
            if ( strcmp( argv[i], "-model" ) == 0 && argument >= num_models ) {
                std::cerr << argv[i + 1] << " exceeds number of available models.  ";
                std::cerr << "there are only " << num_models << " models." << std::endl;
                return false;
            }
            if ( strcmp( argv[i], "-hdr" ) == 0 && argument >= num_hdrs ) {
                std::cerr << argv[i + 1] << " exceeds number of available hdrs.   ";
                std::cerr << "there are only " << num_hdrs << " hdrs." << std::endl;
                return false;
            }
            options[argv[i]] = argument;
        } else {
            std::cerr << "Flag " << argv[i] << " not recognized!" << std::endl;
            std::cerr << "Use -help or help for list of valid options." << std::endl;
            return false;
        }
    }

    noor_spec._model_spec._model_filename = model_files[static_cast<glm::uint32>( options["-model"] )];
    noor_spec._model_spec._hdr_filename = hdr_files[static_cast<glm::uint32>( options["-hdr"] )];
    noor_spec._model_spec._skydome_type = static_cast<int>( options["-skydome"] );
    noor_spec._camera_spec._w = static_cast<glm::uint32>( options["-width"] );
    noor_spec._camera_spec._h = static_cast<glm::uint32>( options["-height"] );
    noor_spec._camera_spec._bounces = static_cast<glm::uint32>( options["-bounces"] );
    noor_spec._camera_spec._rr = static_cast<glm::uint32>( options["-rr"] );
    noor_spec._camera_spec._lens_radius = options["-lens_radius"];
    noor_spec._camera_spec._focal_length = options["-focal_length"];
    noor_spec._bvh_spec._num_bins = static_cast<glm::uint32>( options["-bvh_num_bins"] );
    noor_spec._bvh_spec._max_height = static_cast<glm::uint32>( options["-bvh_max_height"] );
    noor_spec._bvh_spec._min_leaf_tris = static_cast<glm::uint32>( options["-bvh_min_leaf_tris"] );
    noor_spec._bvh_spec._max_leaf_tris = static_cast<glm::uint32>( options["-bvh_max_leaf_tris"] );
    noor_spec._bvh_spec._Ci = options["-bvh_ci"];
    noor_spec._bvh_spec._Ct = options["-bvh_ct"];
    noor_spec._num_gpus = options["-num_gpus"];

    return true;
}
#endif /* OPTIONS_H */