/*
* Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
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
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
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

#ifndef CUDASKY_CUH
#define CUDASKY_CUH 
// Based on Nvidia Optix advanced samples 
class CudaSky {
public:
	float3 _c0;
	float3 _c1;
	float3 _c2;
	float3 _c3;
	float3 _c4;
	float3 _inv_divisor_Yxy;
	float3 _sun_dir;
	float3 _sun_color;
	float3 _up;
	float _overcast;
	float _sun_radius;
	float _sun_scale;

	CudaSky() = default;
	__device__
		float3 querySkyModel( const float3& direction, bool CEL = true ) const {
		if ( direction.y <= 0.0f ) return _constant_spec._black;
		float3 sunlit_sky_color = make_float3( 0.0f );
		// Preetham skylight model
		if ( _overcast < 1.0f ) {
			float3 ray_direction = direction;
			if ( CEL && dot( ray_direction, _sun_dir ) > 94.0f / sqrtf( 94.0f*94.0f + 0.45f*0.45f ) ) {
				sunlit_sky_color = _sun_color;
			} else {
				float inv_dir_dot_up = 1.f / dot( ray_direction, _up );
				if ( inv_dir_dot_up < 0.f ) {
					ray_direction = reflect( ray_direction, _up );
					inv_dir_dot_up = -inv_dir_dot_up;
				}

				const float gamma = dot( _sun_dir, ray_direction );
				const float acos_gamma = acosf( gamma );
				float3 color_Yxy =
					( make_float3( 1.0f ) + _c0*NOOR::exp3f( _c1 * inv_dir_dot_up ) )
					*
					( make_float3( 1.0f ) + _c2*NOOR::exp3f( _c3 * acos_gamma ) + _c4*gamma*gamma );
				color_Yxy *= _inv_divisor_Yxy;
				color_Yxy.y = 0.33f + 1.2f * ( color_Yxy.y - 0.33f );
				color_Yxy.z = 0.33f + 1.2f * ( color_Yxy.z - 0.33f );
				const float3 color_XYZ = NOOR::Yxy2XYZ( color_Yxy );
				sunlit_sky_color = NOOR::XYZ2rgb( color_XYZ );
				sunlit_sky_color /= 1000.0f;
			}
		}

		// CIE standard overcast sky model
		const float Y = 15.0f;
		const float3 overcast_sky_color = make_float3( ( 1.0f + 2.0f * fabsf( direction.y ) ) / 3.0f * Y );

		// return linear combo of the two
		return lerp( sunlit_sky_color, overcast_sky_color, _overcast ) / 35.0f;
	}
};
//__constant__ CudaSky _constant_sky;
#endif // CUDASKY_CUH
