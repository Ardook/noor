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
#include "scene.h"
#include "glShaders.h"
#include "path_tracer.cuh"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBI_MSC_SECURE_CRT
#include "stb_image_write.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"

#define Millis_to_Secs(x) 0.001f*x

namespace {
    enum {
        DISPLAY_FPS = GLFW_KEY_1,
        CUDA_SKY_LIGHT = GLFW_KEY_2,
        CUDA_DEBUG_SKY = GLFW_KEY_3,
        CUDA_MIS = GLFW_KEY_4,
        CUDA_PERSP = GLFW_KEY_5,
        CUDA_ORTHO = GLFW_KEY_6,
        CUDA_ENV = GLFW_KEY_7,
        SUN_CW = GLFW_KEY_LEFT,
        SUN_CCW = GLFW_KEY_RIGHT,
        SUN_UP = GLFW_KEY_UP,
        SUN_DOWN = GLFW_KEY_DOWN,
        SCREEN_SHOT_FLIP = GLFW_KEY_F1,
        SCREEN_SHOT_NO_FLIP = GLFW_KEY_F2,
        QUIT = GLFW_KEY_ESCAPE
    };

    const GLfloat cuda_vertex[] = {
        -1.0f, -1.0f, 0.0f
        , 1.0f, -1.0f, 0.0f
        , 1.0f,  1.0f, 0.0f
        ,-1.0f,  1.0f, 0.0f
    };

    const GLfloat cuda_uv[] = {
        0.0f, 0.0f
        , 1.0f, 0.0f
        , 1.0f, 1.0f
        , 0.0f, 1.0f
    };

    const GLuint cuda_ea[] = {
        0,1,2
        ,0,2,3
    };
    const char* cuda_vs( "shader/cuda.vs" );
    const char* cuda_fs( "shader/cuda.fs" );

    GLuint cudaTextureID;
    GLuint cudaSamplerID;

    GLuint render_cuda_program;

    GLuint mvp_uniform;

    GLuint vaos[NUM_VAOS];
    GLuint vbos[NUM_VBOS];

    GLuint num_geometry_elements;
    GLuint num_bvh_elements;

    glm::mat4 projection;
    glm::mat4 modelview;
    glm::mat4 mvp;

    const float scene_near = 0.001f;
    const float scene_far = 1000.0f;

    float sun_lat = 45.0f;
    float sun_lon = 45.0f;
    const float sun_lat_delta = 1.5;
    const float sun_lon_delta = 2.0f;

    Timer fpsTimer;
    unsigned frame_count = 0;

    std::unique_ptr<Scene> scene;

    GLFWwindow* window;

    std::unordered_map<int, bool> keymap = {
        { DISPLAY_FPS, true }			// 1
        ,{ CUDA_SKY_LIGHT, false }		// 2
        ,{ CUDA_DEBUG_SKY, false }		// 3
        ,{ CUDA_MIS, false }			// 4
        ,{ CUDA_PERSP, false }			// 5
        ,{ CUDA_ORTHO, false }			// 6
        ,{ CUDA_ENV, false }			// 7
    };
}

void cleanup() {
    glfwTerminate();
}

void loadScene( const Spec& spec ) {
    scene = std::make_unique<Scene>( spec );
    keymap[CUDA_SKY_LIGHT] = scene->isSkydomeEnabled();
}

void displayFps( unsigned int frame_count ) {
    static double fps = -1.0;
    static unsigned last_frame_count = 0;
    static double last_update_time = glfwGetTime();
    static double current_time = 0.0;
    static char fps_text[256];

    current_time = glfwGetTime();
    if ( current_time - last_update_time > 0.05f ) {
        fps = ( frame_count - last_frame_count ) / ( current_time - last_update_time );
        last_frame_count = frame_count;
        last_update_time = current_time;
    }
    if ( keymap[DISPLAY_FPS] && frame_count > 0 && fps >= 0.0 ) {
        int l = sprintf_s( fps_text, "fps: %7.2f", fps );
    }
    if ( !keymap[DISPLAY_FPS] ) {
        fps_text[0] = '\0';
    }
    glfwSetWindowTitle( window, fps_text );
}

void screenshot( bool flip = true ) {
    const std::string root_dir( "C:\\Users\\ardavan\\Desktop\\noor\\docs\\screenshots\\" );
    const std::string dir_50( root_dir + "50percent\\" );
    const std::string dir_100( root_dir + "100percent\\" );

    std::stringstream filename;
    time_t rawtime = std::time( nullptr );
    tm now;
    localtime_s( &now, &rawtime );
    filename << std::put_time( &now, "screenshot-%d-%m-%Y-%H-%M-%S.jpg" );

    const std::string file_50 = dir_50 + filename.str();
    const std::string file_100 = dir_100 + filename.str();

    const int w = scene->getWidthPixels();
    const int h = scene->getHeightPixels();
    const int channels = 4;
    GLubyte* data = new GLubyte[channels * w * h];
    glReadPixels(
        0, //GLint   x,
        0, //GLint   y,
        w, //GLsizei width,
        h, //GLsizei height,
        GL_RGBA, //GLenum  format,
        GL_UNSIGNED_BYTE, //GL_GLenum  type,
        data
    );
    stbi_flip_vertically_on_write( flip );
    stbi_write_jpg( file_100.c_str(), w, h, channels, data, 100 );
    GLubyte* resized_data = new GLubyte[channels * w*h / 4];
    stbir_resize_uint8( data, w, h, 0, resized_data, w / 2, h / 2, 0, channels );
    stbi_write_jpg( file_50.c_str(), w / 2, h / 2, channels, resized_data, 100 );
    delete[] data;
    delete[] resized_data;
}

void displayFPS( float frame_duration_ms ) {
    static char fps_text[16];
    if ( keymap[DISPLAY_FPS] )
        sprintf_s( fps_text, "%4.2f fps   ", 1.0f / frame_duration_ms );
    else
        fps_text[0] = '\0';
    glfwSetWindowTitle( window, fps_text );
}

void reshape( GLFWwindow* window, int w, int h ) {
    glViewport( 0, 0, w, h );
}

void keyboard( GLFWwindow* window, int key, int scancode, int action, int mode ) {
    if ( action == GLFW_PRESS ) {
        keymap[key] ^= true;
        switch ( key ) {
            case QUIT:
                glfwSetWindowShouldClose( window, GLFW_TRUE );
                return;
            case SCREEN_SHOT_FLIP:
                screenshot( true );
                break;
            case SCREEN_SHOT_NO_FLIP:
                screenshot( false );
                break;
            case CUDA_SKY_LIGHT:
                if ( keymap[CUDA_SKY_LIGHT] )
                    scene->enableSky();
                else
                    scene->disableSky();
                break;
            case CUDA_DEBUG_SKY:
                if ( keymap[CUDA_DEBUG_SKY] )
                    scene->enableDebugSky();
                else
                    scene->disableDebugSky();
                break;
            case CUDA_PERSP:
                scene->setCameraType( PERSP );
                break;
            case CUDA_ORTHO:
                scene->setCameraType( ORTHO );
                break;
            case CUDA_ENV:
                scene->setCameraType( ENV );
                break;
            case CUDA_MIS:
                if ( keymap[CUDA_MIS] ) scene->enableMIS();
                else  scene->disableMIS();
                break;
            default:
                break;
        }
        return;
    }
    if ( action == GLFW_REPEAT && keymap[CUDA_SKY_LIGHT] ) {
        switch ( key ) {
            case SUN_CW:
                sun_lon += sun_lon_delta;
                sun_lon = sun_lon >= 360.0f ? 0.0f : sun_lon;
                scene->updateSky( glm::radians( sun_lat ), glm::radians( sun_lon ) );
                break;
            case SUN_CCW:
                sun_lon -= sun_lon_delta;
                sun_lon = sun_lon <= 0.0f ? 360.0f : sun_lon;
                scene->updateSky( glm::radians( sun_lat ), glm::radians( sun_lon ) );
                break;
            case SUN_UP:
                sun_lat -= sun_lat_delta;
                sun_lat = glm::clamp( sun_lat, 0.0f, 90.0f );
                scene->updateSky( glm::radians( sun_lat ), glm::radians( sun_lon ) );
                break;
            case SUN_DOWN:
                sun_lat += sun_lat_delta;
                sun_lat = glm::clamp( sun_lat, 0.0f, 90.0f );
                scene->updateSky( glm::radians( sun_lat ), glm::radians( sun_lon ) );
                break;
        }
    }
}

void mouse( GLFWwindow* window, int button, int action, int mods ) {
    scene->mouse( button, action );
}

void motion( GLFWwindow* window, double x, double y ) {
    scene->motion( static_cast<int>( x ), static_cast<int>( y ) );

}

void renderCuda() {
    scene->updateCudaSpec();
    scene->updateCudaCamera();
    scene->updateCudaSky();
    scene->path_tracer();
    glDrawElements(
        GL_TRIANGLES,
        6,
        GL_UNSIGNED_INT,
        ( void* ) nullptr
    );
}

void mainloop() {
    glPolygonMode( GL_FRONT_AND_BACK, GL_FILL );
    glClearColor( 0.0f, 0.0f, 0.0f, 0.0f );
    while ( !glfwWindowShouldClose( window ) ) {
        glfwPollEvents();
        glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
        renderCuda();
        displayFps( ++frame_count );
        glfwSwapBuffers( window );
    }
    cleanup();
}

void initGLFW() {
    // Init GLFW
    if ( !glfwInit() ) {
        std::cout << "glfw initialization failed.\n";
        return;
        // Initialization failed
    }
    // Set all the required options for GLFW
    glfwWindowHint( GLFW_CONTEXT_VERSION_MAJOR, 4 );
    glfwWindowHint( GLFW_CONTEXT_VERSION_MINOR, 3 );
    glfwWindowHint( GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE );
    glfwWindowHint( GLFW_RESIZABLE, GL_FALSE );

    // Create a GLFWwindow object that we can use for GLFW's functions
    window = glfwCreateWindow( scene->getWidthPixels(), scene->getHeightPixels(),
                               "NOOR", nullptr, nullptr );
    glfwMakeContextCurrent( window );

    // Set the required callback functions
    glfwSetKeyCallback( window, keyboard );
    glfwSetMouseButtonCallback( window, mouse );
    glfwSetCursorPosCallback( window, motion );
    glfwSetFramebufferSizeCallback( window, reshape );
    glfwSwapInterval( 0 );
    // Set this to true so GLEW knows to use a modern approach to retrieving 
    // function pointers and extensions
    glewExperimental = GL_TRUE;
    // Initialize GLEW to setup the OpenGL Function pointers
    glewInit();
    // Define the viewport dimensions
    reshape( window, scene->getWidthPixels(), scene->getHeightPixels() );
    glEnable( GL_DEPTH_TEST );

}

void initGLBuffers() {
    glGenVertexArrays( NUM_VAOS, &vaos[0] );
    glGenBuffers( NUM_VBOS, &vbos[0] );
    glBindVertexArray( vaos[CUDA_VAO] );
    glBindBuffer( GL_ARRAY_BUFFER, vbos[CUDA_VBO_VERTEX] );
    glBufferData( GL_ARRAY_BUFFER, sizeof( cuda_vertex ), cuda_vertex, GL_STATIC_DRAW );
    glVertexAttribPointer(
        0,                  // attribute
        3,                  // size : x,y,z
        GL_FLOAT,           // type
        GL_FALSE,           // normalized
        0,                  // stride
        ( void* ) nullptr   // data 
    );
    glEnableVertexAttribArray( 0 );

    glBindBuffer( GL_ARRAY_BUFFER, vbos[CUDA_VBO_UV] );
    glBufferData( GL_ARRAY_BUFFER, sizeof( cuda_uv ), cuda_uv, GL_STATIC_DRAW );
    glVertexAttribPointer(
        1,                  // attribute
        2,                  // size : u,v
        GL_FLOAT,           // type
        GL_FALSE,           // normalized
        0,                  // stride
        ( void* ) nullptr   // data 
    );
    glEnableVertexAttribArray( 1 );
    glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, vbos[CUDA_VBO_ELEMENT] );
    glBufferData( GL_ELEMENT_ARRAY_BUFFER, sizeof( cuda_ea ), cuda_ea, GL_STATIC_DRAW );
    glBindVertexArray( 0 );

    glUseProgram( render_cuda_program );
    cudaSamplerID = glGetUniformLocation( render_cuda_program, "sampler" );
    mvp_uniform = glGetUniformLocation( render_cuda_program, "mvp" );
    mvp = glm::ortho( -1.0, 1.0, -1.0, 1.0, -1.0, 1.0 );
    glUniformMatrix4fv( mvp_uniform, 1, GL_FALSE, &mvp[0][0] );
    glBindTexture( GL_TEXTURE_2D, cudaTextureID );
    glEnable( GL_TEXTURE_2D );
    glUniform1i( cudaSamplerID, 0 );
    glBindVertexArray( vaos[CUDA_VAO] );
}

void initGLShaders() {
    GLuint vs, fs;
    // vertex and fragment shader to render cuda result 
    loadShader( vs, GL_VERTEX_SHADER, cuda_vs );
    loadShader( fs, GL_FRAGMENT_SHADER, cuda_fs );

    if ( !linkShaders( render_cuda_program, vs, fs, 0 ) ) {
        std::string msg = "File " + std::string( __FILE__ ) + " LINE " + std::to_string( __LINE__ ) + "\n";
        msg += "Error: error linking shaders " + std::string( cuda_vs ) + " and " + std::string( cuda_fs );
        std::cerr << msg << std::endl;
        exit( EXIT_FAILURE );
    }
}

void initCudaConext() {
    scene->initCudaContext( &cudaTextureID );
}

void initGLContext() {
    initGLFW();
    initGLShaders();
    initGLBuffers();
}

void startRender() {
    mainloop();
}



