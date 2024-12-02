# Coding Plan

## Steps

1. Debug Glasgow stereo disparity algorithm
2. OpenCL->Vulkan for direct image display from GPU
3. Implement forward rendering from maps, using default values
    - depth/surface normal, illumination, shading, reflectance, relative velocity maps.
4. Parsimony regularizer
5. Direct (aka primal) TV regularizer
6. Priors - also using Parsimony style Particle space
7. Reduced pixel count - Multi-scale Sobel edges  with Pyramids for fine edges
    1. track only edge pixels
    2. fit parsimony sets of pixels
8. Residual tracking in stereo flow - find 2nd & 3rd components
9. Joint optimization
    - round robin  5 adaptive steps
    - comput gradients & 2nd order gradient
10. Acquire & update priors
11. Recognition 
     - objects, actions, affordances, adjectives, adverbs, prepositions
     - class vs instance
     - heirarchies of parts
     - scales, aspects of change - the real thing vs image vs model
12. Scene Graph
    - Episodic memory
    - Procedural memory
    - Imagination
13. Abstract causal inference 
    - recursive abstraction
    - causal model search
    - causal prediction from model
    - experiment design - to distinguish between alternative models
14. Theory of mind
    - Frontal pole - to do list
    - Limbic system - drives, urgency, physiology
    - Action selection - cycle of maps - multi-objective optimization
    - Modelling other agents

## Stages

1. get it working with rigid lambertian SLAM
2. Add specularity
3. Add reative velocity map - use 2nd & 3rd components
4. Add particle model 
    - constraint
    - Fit -> model materials & structure
5. Particle refinement - multi scale
6. Relation to Radiance fields & Particle based Radiance Field
7. later (3D Convex Splatting - variable sharpness, roundness, relation to discrete element models. )
8. output models -> for Unreal, Blender, Unity ?/


## Business

1. Cloud service - scene capture - Google Play , pay for compute.
2. Edge Capture (computing edge) - run on local hardware - Android & Windows
3. Plugin - vulkan renderer of SPMP models.
4. Robot motion planning - needs to learn the robot (self-identification)
    - Vehicles
    - Limbed manipulation
5. Professional knowledge capture - medicine etc...

## Applications

1. Anatomy/surg/training - with Haptic Omni
2. Relighting
    - cast AR shadows & light on real objects in video
    - cast real shadows & light on AR objects in video

## SPH surface for a particle cloud
- (compare with "3D Convex splatting paper" which controlls roundedness and fogginess of polyhedra/polygons)
1. If we sample at intervals along a ray, given an octree
2. Find density + gradient of density along ray -> locate surface
3. Interpolate normal & reflectance
4. Compute (sample) onward rays (NB recursion limit)

Q. How to do parallel Octtree hit detections on GPU ?
- NB Growing ray width  with depth
- Could sort particles by ray & merge/refine particle model.
- Surface will be associated with 1st particle in te ray.
- Need the adjacent 9 rays (3x3 beam of rays)
- Consider 3 sets of rays - specular, diffuse, transmitted (refracted)

## Macro-pixels by sorting :
1. Cluster by 6D (rgb + 3D location)

## Other
1. Anisotropic blurring / relaxation with g1mem
2. Total-variation without primal-dual
3. Frame-to-frame optial flow: expect repeat of previous warp (tracking pixels aka follow the particles) i.e. lagrangian prediction.
Update and merge through the image pyramid.
4. Tracking and mapping from disparity ?
- Camera tracking from flow vector direction
- Mapping from flow vector relative magnitudes
5. Relative velocity map ?
Where regions don't fit depth + camera motion
6. Reflectance map
- specular component (metallic/glassy, roughness, tint)
- -> Predicted image
7. Curvature map <-> surface normal map <-> Depthmap
- -> shading map -> predicted image
8. Illumination map
- -> shading map -> predicted image
- Light source model - where are they ?
- Map of shperical illumination - may loose lobes when in shadows

## Priors
1. Parsimony - sample from current scene, apply to sample, then to the rest of the image. See SPH based parsimony.
2. Anisotropic smoothing - primal TV + g1mem edges
3. Absolute prior
- previous distribution of points
- can be top-dpwn from recognition
- 2nd order space of spaces
- Does the is scene resemble one or more canonical priors about scene spaces for this variable ?


Q. How to make rapid large changes ?
E.g. Parsimony of shape + Expectation of specular reflectance.

- When, where, why should you believe you are looking at a reflective (mirror) surface ?
- If you expect the surfaces are highly mirrored, how do you work out their shape ?
- Depends on the movement over multiple frames.
- If you have dense optical flow over several time steps,
- May consider different spatial frequencies in their own pyramids - or at least "high" vs "low"
