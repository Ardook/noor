##### Noor
Noor is a CUDA/GPU Monte-Carlo path tracer that I've been working on for the past couple of years.  Although Noor is a powerful renderer it's still an experimental one that I wrote as a personal research tool (It is a work in progress).  I keep rewriting, redesigning, breaking, and fixing it on daily basis, so please keep that in mind if you decide to download the [source code](https://github.com/Ardook/Noor).    

The PBRT book has been instrumental in the development of Noor.  The book has been my constant companion for the past two years.  Noor started as a completely independent system, as I made progress in its development I gravitated more towards the design of the PBRT system.  In fact, some components such as the BSDF/BXDF, 1D/2D distributions, and cameras are the CUDA ports of the PBRT ones and as such the credit should be given to them. 
##### Development environment:
* Windows 10 64 bit 
* Visual Studio 2015 with Visual Assist 
* Nvidia CUDA ToolKit 9.1 and Nsight for Visual Studio 
* C++ (11 & 14)
* Autodesk Maya
* Gimp 

##### Acceleration Data Structures:
* BVH
* SBVH (removed)

##### Instancing:
* Multi-level BVH tree construction and traversal
* Support for large scenes with complex instancing of geometry

##### Primitives:
* Triangles
* Implicit surfaces like quads, disks, spheres, and cylinders (mesh lights)

##### Supported model/scene file formats:
* Alias Wavefront OBJ
* Autodesk FBX
   * Custom materials, lights, and cameras using MEL scripts

##### Supported image formats:
* JPG, jpg, TGA, BMP, PSD, GIF, HDR, PIC [(Sean Barrett)](https://github.com/nothings/stb). 
* EXR [(Syoyo Fujita)](https://github.com/syoyo/tinyexr).

##### Supported lights:
* Environment lighting (HDR and physical sky)
* Distant lights
* Area lights
* Point lights
* Spot lights
* Mesh area lights (quad, sphere, and disk)
* Infinite/environment light with importance sampling
* Physical sun/sky (Hosek) with importance sampling  
   * Cuda port of the Hosek & Wilkie implementation [(Hosek & Wilkie Sun & Sky Model)](http://cgg.mff.cuni.cz/projects/SkylightModelling/)

##### Texture:
* HDR and LDR
* Mipmapping
* Point, bi-linear, tri-linear, EWA (anisotropic) sampling
* Realtime Cuda image resizer and mipmap generator

##### Camera:
* Perspective with depth of field
* Orthographic
* Environment (360 degrees)

##### Work in Progress:
* Bidirectional path tracing
* Volume path tracing
* Subsurface scattering
* Multi-threaded BVH build
* Maya plugin

##### Resources and Credits: 
* PBRT (super bible of ray tracing)
* Mitsuba
* Nvidia Optix: excellent resource on the GPU side of things
* Solid Angle Arnold Renderer: ground truth
* Octane Renderer: ground truth
* Shader ball/knob License: CC BY 3.0 © Yasutoshi Mori
   * Removed the equations
   * Removed the backdrop
   * Removed all the textures
   * Only the knob is used
* HDRI 
   * Outdoor scenes [hdriheaven.com](https://hdriheaven.com)
   * Indoor scenes courtesy of Bernhard Vogl (used in Mitsuba also)
* Art assets
   * Train, cabin (only the cabin), and museum are from [3drender.com lighting challenge](http://www.3drender.com/challenges/)
   * The trees in the cabin scene are from [ORCA: Open Research Content Archive](https://developer.nvidia.com/orca/speedtree)
* PBR textures
   * [textures.com](https://textures.com)
   * [source allegorithmic](https://source.allegorithmic.com/)
* All the rest of the assets are created by me using Autodesk Maya

##### ScreenShots:
[![metals](screenshots/50percent/metals.jpg){:height="128px" width="512px"}](screenshots/100percent/metal.jpg)
[![smooth plastic](screenshots/50percent/screenshot-23-05-2018-10-05-35.jpg)](screenshots/100percent/screenshot-23-05-2018-10-05-35.jpg)
[![rough plastic](screenshots/50percent/screenshot-23-05-2018-10-08-01.jpg)](screenshots/100percent/screenshot-23-05-2018-10-08-01.jpg)
[![smooth gold](screenshots/50percent/screenshot-22-05-2018-13-28-56.jpg)](screenshots/100percent/screenshot-22-05-2018-13-28-56.jpg)
[![rough gold](screenshots/50percent/screenshot-22-05-2018-13-27-03.jpg)](screenshots/100percent/screenshot-22-05-2018-13-27-03.jpg)
[![rough gold smooth coat](screenshots/50percent/screenshot-22-05-2018-13-13-45.jpg)](screenshots/100percent/screenshot-22-05-2018-13-13-45.jpg)
[![aniso rough gold](screenshots/50percent/screenshot-23-05-2018-10-15-17.jpg)](screenshots/100percent/screenshot-23-05-2018-10-15-17.jpg)
[![smooth glass](screenshots/50percent/screenshot-21-05-2018-13-47-19.jpg)](screenshots/100percent/screenshot-21-05-2018-13-47-19.jpg)
[![rough glass](screenshots/50percent/screenshot-23-05-2018-18-10-35.jpg)](screenshots/100percent/screenshot-23-05-2018-18-10-35.jpg)
[![flannel knob](screenshots/50percent/screenshot-21-05-2018-12-46-57.jpg)](screenshots/100percent/screenshot-21-05-2018-12-46-57.jpg)
[![aluminium foil](screenshots/50percent/screenshot-21-05-2018-13-09-10.jpg)](screenshots/100percent/screenshot-21-05-2018-13-09-10.jpg)
[![red leather](screenshots/50percent/screenshot-21-05-2018-14-21-41.jpg)](screenshots/100percent/screenshot-21-05-2018-14-21-41.jpg)
[![substrate](screenshots/50percent/screenshot-22-05-2018-15-44-15.jpg)](screenshots/100percent/screenshot-22-05-2018-15-44-15.jpg)
[![smooth metal](screenshots/50percent/screenshot-23-05-2018-10-28-41.jpg)](screenshots/100percent/screenshot-23-05-2018-10-28-41.jpg)
[![rough metal](screenshots/50percent/screenshot-23-05-2018-10-32-29.jpg)](screenshots/100percent/screenshot-23-05-2018-10-32-29.jpg)
[![aniso rough metal](screenshots/50percent/screenshot-23-05-2018-10-27-41.jpg)](screenshots/100percent/screenshot-23-05-2018-10-27-41.jpg)
[![translucent](screenshots/50percent/screenshot-23-05-2018-11-56-01.jpg)](screenshots/100percent/screenshot-23-05-2018-11-56-01.jpg)
[![2side lights](screenshots/50percent/screenshot-23-05-2018-12-13-36.jpg)](screenshots/100percent/screenshot-23-05-2018-12-13-36.jpg)
[![1side lights](screenshots/50percent/screenshot-23-05-2018-12-14-17.jpg)](screenshots/100percent/screenshot-23-05-2018-12-14-17.jpg)
[![ortho museum](screenshots/50percent/screenshot-23-05-2018-12-46-45.jpg)](screenshots/100percent/screenshot-23-05-2018-12-46-45.jpg)
[![persp museum](screenshots/50percent/screenshot-23-05-2018-12-49-44.jpg)](screenshots/100percent/screenshot-23-05-2018-12-49-44.jpg)
[![360 museum](screenshots/50percent/screenshot-23-05-2018-12-52-59.jpg)](screenshots/100percent/screenshot-23-05-2018-12-52-59.jpg)
[![cabin](screenshots/50percent/screenshot-23-05-2018-13-26-10.jpg)](screenshots/100percent/screenshot-23-05-2018-13-26-10.jpg)
[![train](screenshots/50percent/screenshot-23-05-2018-14-12-20.jpg)](screenshots/100percent/screenshot-23-05-2018-14-12-20.jpg)
[![dof train](screenshots/50percent/screenshot-23-05-2018-14-19-09.jpg)](screenshots/100percent/screenshot-23-05-2018-14-19-09.jpg)
[![red leather](screenshots/50percent/screenshot-23-05-2018-14-34-53.jpg)](screenshots/100percent/screenshot-23-05-2018-14-34-53.jpg)
[![brick](screenshots/50percent/screenshot-23-05-2018-14-35-44.jpg)](screenshots/100percent/screenshot-23-05-2018-14-35-44.jpg)
[![brown leather](screenshots/50percent/screenshot-23-05-2018-14-41-06.jpg)](screenshots/100percent/screenshot-23-05-2018-14-41-06.jpg)
[![red marble](screenshots/50percent/screenshot-23-05-2018-15-31-13.jpg)](screenshots/100percent/screenshot-23-05-2018-15-31-13.jpg)
[![wood](screenshots/50percent/screenshot-23-05-2018-15-34-58.jpg)](screenshots/100percent/screenshot-23-05-2018-15-34-58.jpg)
[![semi dark wood](screenshots/50percent/screenshot-23-05-2018-15-38-04.jpg)](screenshots/100percent/screenshot-23-05-2018-15-38-04.jpg)
[![dark wood](screenshots/50percent/screenshot-23-05-2018-15-39-32.jpg)](screenshots/100percent/screenshot-23-05-2018-15-39-32.jpg)
[![checked tile](screenshots/50percent/screenshot-23-05-2018-15-46-56.jpg)](screenshots/100percent/screenshot-23-05-2018-15-46-56.jpg)
[![spandex](screenshots/50percent/screenshot-23-05-2018-15-51-09.jpg)](screenshots/100percent/screenshot-23-05-2018-15-51-09.jpg)
[![polyester](screenshots/50percent/screenshot-23-05-2018-15-56-55.jpg)](screenshots/100percent/screenshot-23-05-2018-15-56-55.jpg)
[![chainmail tile](screenshots/50percent/screenshot-23-05-2018-15-58-59.jpg)](screenshots/100percent/screenshot-23-05-2018-15-58-59.jpg)
[![no MIS](screenshots/50percent/screenshot-26-05-2018-15-47-11.jpg)](screenshots/100percent/screenshot-26-05-2018-15-47-11.jpg)
[![MIS](screenshots/50percent/screenshot-26-05-2018-15-47-17.jpg)](screenshots/100percent/screenshot-26-05-2018-15-47-17.jpg)
[![MIS](screenshots/50percent/screenshot-26-05-2018-18-14-15.jpg)](screenshots/100percent/screenshot-26-05-2018-18-14-15.jpg)
[![MIS](screenshots/50percent/screenshot-26-05-2018-18-14-22.jpg)](screenshots/100percent/screenshot-26-05-2018-18-14-22.jpg)
[![MIS](screenshots/50percent/screenshot-26-05-2018-18-15-29.jpg)](screenshots/100percent/screenshot-26-05-2018-18-15-29.jpg)
[![MIS](screenshots/50percent/screenshot-26-05-2018-18-15-35.jpg)](screenshots/100percent/screenshot-26-05-2018-18-15-35.jpg)
[![MIS](screenshots/50percent/screenshot-26-05-2018-18-18-04.jpg)](screenshots/100percent/screenshot-26-05-2018-18-18-04.jpg)
[![MIS](screenshots/50percent/screenshot-26-05-2018-18-18-09.jpg)](screenshots/100percent/screenshot-26-05-2018-18-18-09.jpg)
[![MIS](screenshots/50percent/screenshot-26-05-2018-18-25-14.jpg)](screenshots/100percent/screenshot-26-05-2018-18-25-14.jpg)
[![MIS](screenshots/50percent/screenshot-26-05-2018-18-25-20.jpg)](screenshots/100percent/screenshot-26-05-2018-18-25-20.jpg)
[![MIS](screenshots/50percent/screenshot-26-05-2018-18-25-28.jpg)](screenshots/100percent/screenshot-26-05-2018-18-25-28.jpg)
[![MIS](screenshots/50percent/screenshot-26-05-2018-18-25-30.jpg)](screenshots/100percent/screenshot-26-05-2018-18-25-30.jpg)
[![MIS](screenshots/50percent/screenshot-26-05-2018-18-25-38.jpg)](screenshots/100percent/screenshot-26-05-2018-18-25-38.jpg)
[![MIS](screenshots/50percent/screenshot-26-05-2018-18-25-41.jpg)](screenshots/100percent/screenshot-26-05-2018-18-25-41.jpg)
[![MIS](screenshots/50percent/screenshot-26-05-2018-18-25-51.jpg)](screenshots/100percent/screenshot-26-05-2018-18-25-51.jpg)
[![MIS](screenshots/50percent/screenshot-26-05-2018-18-25-54.jpg)](screenshots/100percent/screenshot-26-05-2018-18-25-54.jpg)