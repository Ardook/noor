/*
* Copyright (c) 2017 NVIDIA CORPORATION. All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions
* are met:
*  * Redistributions of source code must retain the above copyright
*    notice, this list of conditions and the following disclaimer.
*  * Redistributions in binary form must reproduce the above copyright
*    notice, this list of conditions and the following disclaimer in the
*    documentation and/or other materials provided with the distribution.
*  * Neither the name of NVIDIA CORPORATION nor the names of its
*    contributors may be used to endorse or promote products derived
*    from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
* PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
* CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
* EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
* PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
* PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
* OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/* Based on "GPU Random Numbers via the Tiny Encryption Algorithm"
* by Fahad Zafar, Marc Olano, and Aaron Curtis
* Implementation from Nvidia Optix Examples/Samples */

#ifndef CUDARNG_CUH
#define CUDARNG_CUH 


class CudaRNG {
    mutable uint _seed;
public:
    __device__
        CudaRNG( uint id, uint frame_number ) :
        _seed( tea<16>( id, frame_number ) ) {}

    __device__
        float operator()() const {
        return rnd( _seed );
    }
    template<uint N>
    __device__
        uint tea( uint val0, uint val1 ) {
        uint v0 = val0;
        uint v1 = val1;
        uint s0 = 0;

        for ( uint n = 0; n < N; n++ ) {
            s0 += 0x9e3779b9;
            v0 += ( ( v1 << 4 ) + 0xa341316c ) ^ ( v1 + s0 ) ^ ( ( v1 >> 5 ) + 0xc8013ea4 );
            v1 += ( ( v0 << 4 ) + 0xad90777d ) ^ ( v0 + s0 ) ^ ( ( v0 >> 5 ) + 0x7e95761e );
        }
        return v0;
    }
    // Generate random uint in [0, 2^24)
    __device__
        uint lcg( uint &prev ) const {
        const uint LCG_A = 1664525u;
        const uint LCG_C = 1013904223u;
        prev = ( LCG_A * prev + LCG_C );
        return prev & 0x00FFFFFF;
    }

    // Generate random float in [0, 1)
    __device__
        float rnd( uint &prev ) const {
        return ( (float) lcg( prev ) / (float) 0x01000000 );
    }
};
#endif // CUDARNG_CUH
