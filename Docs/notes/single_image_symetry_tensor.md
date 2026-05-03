# Single image smmetry tennsor

## To find repeating patterns of shapes in the image, translate patches to find optima in 2D e.g.

- sample 5 steps (u,v) at each pyramid level -> initial symetry tensor,
- then iterate LK steps to locate optima
- link optima -> chains of repeating patterns
- find 3D transform to explain, ideally the same between optima in a chain.


Consider perception of 3D struture from a edge image,  and the chararcteristic structure of 3D perspective.
