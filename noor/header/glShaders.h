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
#ifndef GLSHADERS_H
#define GLSHADERS_H

GLuint loadShader(
    GLuint& shader_id
    , GLenum shader_type
    , const char* shader_file
) {

    shader_id = glCreateShader( shader_type );

    std::ifstream file( shader_file );
    if ( !file ) {
        fprintf( stderr, "Error opening file: %s\n", shader_file );
        return 0;
    }

    std::stringstream s;
    s << file.rdbuf();
    file.close();
    std::string ss = s.str();
    const char* shader_code = ss.c_str();

    GLint result = GL_FALSE;
    printf( "Compiling shader : %s\n", shader_file );
    glShaderSource( shader_id, 1, &shader_code, nullptr );
    glCompileShader( shader_id );

    glGetShaderiv( shader_id, GL_COMPILE_STATUS, &result );
    if ( result == GL_FALSE ) {
        fprintf( stderr, "shader compilation failed!\n" );

        GLint logLen;
        glGetShaderiv( shader_id, GL_INFO_LOG_LENGTH, &logLen );

        if ( logLen > 0 ) {
            char * log = (char *) malloc( logLen );

            GLsizei written;
            glGetShaderInfoLog( shader_id, logLen, &written, log );

            fprintf( stderr, "Shader log: \n%s", log );

            free( log );
        }
        return 0;
    }
    return shader_id;
}

GLuint linkShaders( GLuint& program, GLuint vs, GLuint fs, GLuint gs ) {
    program = glCreateProgram();
    if ( program == 0 ) {
        fprintf( stderr, "Error creating program object.\n" );
        return 0;
    }

    glAttachShader( program, vs );
    glAttachShader( program, fs );
    if ( gs )
        glAttachShader( program, gs );


    glLinkProgram( program );

    GLint status;
    glGetProgramiv( program, GL_LINK_STATUS, &status );
    if ( GL_FALSE == status ) {

        fprintf( stderr, "Failed to link shader program!\n" );

        GLint logLen;
        glGetProgramiv( program, GL_INFO_LOG_LENGTH, &logLen );

        if ( logLen > 0 ) {
            char * log = (char *) malloc( logLen );

            GLsizei written;
            glGetProgramInfoLog( program, logLen, &written, log );
            fprintf( stderr, "Program log: \n%s", log );

            free( log );
        }
        return 0;
    } else {
        return program;
    }
}
#endif /* GLSHADERS_H */