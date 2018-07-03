
#### Noor
Noor is an experimental yet powerful CUDA pathtracer (with support for multiple GPUs).  It started (Noor 1.0) as part of a final class project (real-time Whitted raytracer) in 2009 and was recently resurrected as a pathtracer to assist me in learning and reinforcing the ideas and the topics covered in the PBRT book. 
 
#### Acceleration Data Structures:
* BVH
* SBVH (removed)

#### Instancing:
* Multi-level BVH tree construction and traversal
* Support for large scenes with complex instancing of geometry

#### Primitives:
* Triangles
* Implicit surfaces like quads, disks, spheres, and cylinders (mesh lights)

#### Supported model/scene file formats:
* Alias Wavefront OBJ
* Autodesk FBX
   * Custom materials, lights, and cameras using MEL scripts

#### Supported image formats:
* JPG, jpg, TGA, BMP, PSD, GIF, HDR, PIC [(Sean Barrett)](https://github.com/nothings/stb). 
* EXR [(Syoyo Fujita)](https://github.com/syoyo/tinyexr).

#### Supported lights:
* Environment lighting (HDR and physical sky)
* Distant lights
* Area lights
* Point lights
* Spot lights
* Mesh area lights (quad, sphere, and disk)
* Infinite/environment light with importance sampling
* Physical sun/sky (Hosek) with importance sampling  
   * Cuda port of the Hosek & Wilkie implementation [(Hosek & Wilkie Sun & Sky Model)](http://cgg.mff.cuni.cz/projects/SkylightModelling/)

#### Texture:
* HDR and LDR
* Mipmapping
* Point, bi-linear, tri-linear, EWA (anisotropic) sampling
* Realtime Cuda image resizer and mipmap generator

#### Camera:
* Perspective with depth of field
* Orthographic
* Environment (360 degrees)

#### Work in Progress:
* Bidirectional path tracing
* Volume path tracing
* Subsurface scattering
* Multi-threaded BVH build
* Maya plugin

#### Resources and Credits: 
* PBRT (super bible of ray tracing)
* Mitsuba
* Nvidia Optix: excellent resource on the GPU side of things
* Solid Angle Arnold Renderer: ground truth
* Octane Renderer: ground truth
* Pixar Luxo Jr. and Luxo Ball License: 
   * Lamp CC BY 3.0 © [pistiwique](https://www.blendswap.com/blends/view/75677)
   * Ball CC BY 0.0 © [ethomson92](https://www.blendswap.com/blends/view/91066) 
   * Exported from Blender and Imported to Maya	
   * Added Noor Custom PBR Material
* Utah Teapot: [University of Utah Rendering Competition](https://graphics.cs.utah.edu/trc/)
* SGI Logo: Whoever owns SGI today, [HP](https://www.hpe.com/us/en/solutions/hpc-high-performance-computing.html)
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

#### ScreenShots:
|[![metals](screenshots/50percent/metals.jpg "metals")](screenshots/100percent/metal.jpg)|
|Metals|
|[![infinite mirrors](screenshots/50percent/screenshot-13-06-2018-15-34-01.jpg "infinite mirrors")](screenshots/100percent/screenshot-13-06-2018-15-34-01.jpg)|
|Infinite Mirror (64 max bounces)|
|[![trio](screenshots/50percent/screenshot-07-06-2018-11-47-47.jpg "smooth plastic")](screenshots/100percent/screenshot-07-06-2018-11-47-47.jpg)|
|Luxo Jr. & Ball, SGI logo, and Utah Teapot|
|[![trio](screenshots/50percent/screenshot-07-06-2018-14-19-36.jpg "smooth plastic")](screenshots/100percent/screenshot-07-06-2018-14-19-36.jpg)|
|Luxo Jr. & Ball, SGI logo, and Utah Teapot|
|[![smooth plastic](screenshots/50percent/screenshot-23-05-2018-10-05-35.jpg "smooth plastic")](screenshots/100percent/screenshot-23-05-2018-10-05-35.jpg)|
|Smooth Plastic Material|
|[![rough plastic](screenshots/50percent/screenshot-23-05-2018-10-08-01.jpg "rough plastic")](screenshots/100percent/screenshot-23-05-2018-10-08-01.jpg)|
|Rough Plastic Material|
|[![smooth gold](screenshots/50percent/screenshot-22-05-2018-13-28-56.jpg "smooth gold")](screenshots/100percent/screenshot-22-05-2018-13-28-56.jpg)|
|Smooth Metal Material (Gold)|
|[![rough gold](screenshots/50percent/screenshot-22-05-2018-13-27-03.jpg "rough gold")](screenshots/100percent/screenshot-22-05-2018-13-27-03.jpg)|
|Rough Metal Material (Gold)|
|[![rough gold clear coat](screenshots/50percent/screenshot-22-05-2018-13-13-45.jpg "rough gold clear coat")](screenshots/100percent/screenshot-22-05-2018-13-13-45.jpg)|
|Clear Coat Material (Rough Gold Substrate + Specular Coating)|
|[![aniso rough gold](screenshots/50percent/screenshot-23-05-2018-10-15-17.jpg "anisotropic rough gold")](screenshots/100percent/screenshot-23-05-2018-10-15-17.jpg)|
|Anisotropic Roughness Texture|
|[![smooth glass](screenshots/50percent/screenshot-21-05-2018-13-47-19.jpg "smooth glass")](screenshots/100percent/screenshot-21-05-2018-13-47-19.jpg)|
|Smooth Glass|
|[![rough glass](screenshots/50percent/screenshot-29-05-2018-18-30-13.jpg "rough glass")](screenshots/100percent/screenshot-29-05-2018-18-30-13.jpg)|
|Rough Glass|
|[![flannel knob](screenshots/50percent/screenshot-21-05-2018-12-46-57.jpg "flannel")](screenshots/100percent/screenshot-21-05-2018-12-46-57.jpg)|
|Flannel|
|[![aluminium foil](screenshots/50percent/screenshot-21-05-2018-13-09-10.jpg "aluminium foil")](screenshots/100percent/screenshot-21-05-2018-13-09-10.jpg)|
|Aluminium Foil|
|[![red leather](screenshots/50percent/screenshot-21-05-2018-14-21-41.jpg)](screenshots/100percent/screenshot-21-05-2018-14-21-41.jpg)|
|Leather|
|[![substrate](screenshots/50percent/screenshot-22-05-2018-15-44-15.jpg)](screenshots/100percent/screenshot-22-05-2018-15-44-15.jpg)|
|Fresnel Blend|
|[![red leather](screenshots/50percent/screenshot-23-05-2018-14-34-53.jpg)](screenshots/100percent/screenshot-23-05-2018-14-34-53.jpg)|
|Leather|
|[![brick](screenshots/50percent/screenshot-23-05-2018-14-35-44.jpg)](screenshots/100percent/screenshot-23-05-2018-14-35-44.jpg)|
|Brick|
|[![red marble](screenshots/50percent/screenshot-23-05-2018-15-31-13.jpg)](screenshots/100percent/screenshot-23-05-2018-15-31-13.jpg)|
|Marble|
|[![wood](screenshots/50percent/screenshot-23-05-2018-15-34-58.jpg)](screenshots/100percent/screenshot-23-05-2018-15-34-58.jpg)|
|Wood|
|[![semi dark wood](screenshots/50percent/screenshot-23-05-2018-15-38-04.jpg)](screenshots/100percent/screenshot-23-05-2018-15-38-04.jpg)|
|Semi Dark Wood|
|[![dark wood](screenshots/50percent/screenshot-23-05-2018-15-39-32.jpg)](screenshots/100percent/screenshot-23-05-2018-15-39-32.jpg)|
|Dark Wood|
|[![checked tile](screenshots/50percent/screenshot-23-05-2018-15-46-56.jpg)](screenshots/100percent/screenshot-23-05-2018-15-46-56.jpg)|
|Tile|
|[![spandex](screenshots/50percent/screenshot-23-05-2018-15-51-09.jpg)](screenshots/100percent/screenshot-23-05-2018-15-51-09.jpg)|
|Spandex|
|[![chainmail tile](screenshots/50percent/screenshot-23-05-2018-15-58-59.jpg)](screenshots/100percent/screenshot-23-05-2018-15-58-59.jpg)|
|Chainmail Tile|
|[![diamond plate](screenshots/50percent/screenshot-23-05-2018-15-56-55.jpg)](screenshots/100percent/screenshot-23-05-2018-15-56-55.jpg)|
|Diamond Plate|
|[![rust](screenshots/50percent/screenshot-29-05-2018-13-18-04.jpg)](screenshots/100percent/screenshot-29-05-2018-13-18-04.jpg)|
|Rust|
|[![scratch](screenshots/50percent/screenshot-29-05-2018-14-21-34.jpg)](screenshots/100percent/screenshot-29-05-2018-14-21-34.jpg)|
|Scratched|
|[![diamond smooth leather](screenshots/50percent/screenshot-29-05-2018-15-12-52.jpg)](screenshots/100percent/screenshot-29-05-2018-15-12-52.jpg)|
|Leather|
|[![metal mesh](screenshots/50percent/screenshot-29-05-2018-16-00-16.jpg)](screenshots/100percent/screenshot-29-05-2018-16-00-16.jpg)|
|Metal Mesh|
|[![smooth metal](screenshots/50percent/screenshot-23-05-2018-10-28-41.jpg)](screenshots/100percent/screenshot-23-05-2018-10-28-41.jpg)|
|Smooth Metal|
|[![rough metal](screenshots/50percent/screenshot-23-05-2018-10-32-29.jpg)](screenshots/100percent/screenshot-23-05-2018-10-32-29.jpg)|
|Rough Metal|
|[![aniso rough metal](screenshots/50percent/screenshot-23-05-2018-10-27-41.jpg)](screenshots/100percent/screenshot-23-05-2018-10-27-41.jpg)|
|Anisotropic Roughness Metal|
|[![translucent](screenshots/50percent/screenshot-23-05-2018-11-56-01.jpg)](screenshots/100percent/screenshot-23-05-2018-11-56-01.jpg)|
|left: No Trasparency Middle: Transparency Right: Translucency (spot light behind the leaf)|
[![2side lights](screenshots/50percent/screenshot-23-05-2018-12-13-36.jpg)](screenshots/100percent/screenshot-23-05-2018-12-13-36.jpg)
|Quad (double sided), Disk (double sided), Sphere Area Lights, Point Light, and Spot Lights|
[![1side lights](screenshots/50percent/screenshot-23-05-2018-12-14-17.jpg)](screenshots/100percent/screenshot-23-05-2018-12-14-17.jpg)
|Quad (single sided), Disk (single sided), Sphere Area Lights, Point Light, and Spot Lights|
[![ortho museum](screenshots/50percent/screenshot-23-05-2018-12-46-45.jpg "orthographic camera")](screenshots/100percent/screenshot-23-05-2018-12-46-45.jpg)
|Museum Orthographic Camera|
[![persp museum](screenshots/50percent/screenshot-23-05-2018-12-49-44.jpg "perspective camera")](screenshots/100percent/screenshot-23-05-2018-12-49-44.jpg)
|Museum Perspective Camera|
[![360 museum](screenshots/50percent/screenshot-23-05-2018-12-52-59.jpg "360/environment camera")](screenshots/100percent/screenshot-23-05-2018-12-52-59.jpg)
|Museum 360/Environment Camera|
[![cabin](screenshots/50percent/screenshot-23-05-2018-13-26-10.jpg "instanced trees, shadow catcher, composit")](screenshots/100percent/screenshot-23-05-2018-13-26-10.jpg)
|Cabin Instanced Trees, Shadow Catcher Plane, and Environemnt Lighting|
[![train](screenshots/50percent/screenshot-23-05-2018-14-12-20.jpg "pinhole camera")](screenshots/100percent/screenshot-23-05-2018-14-12-20.jpg)
|Train Scene Pinhole Camera|
[![dof train](screenshots/50percent/screenshot-23-05-2018-14-19-09.jpg "DOF thin lens")](screenshots/100percent/screenshot-23-05-2018-14-19-09.jpg)
|Train Scene Thin Lens DOF|
[![no MIS](screenshots/50percent/screenshot-26-05-2018-15-47-11.jpg "veach no MIS")](screenshots/100percent/screenshot-26-05-2018-15-47-11.jpg)
|Veach no MIS|
[![MIS](screenshots/50percent/screenshot-26-05-2018-15-47-17.jpg "veach MIS")](screenshots/100percent/screenshot-26-05-2018-15-47-17.jpg)
|Veach with MIS|
