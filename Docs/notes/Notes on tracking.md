# Notes on tracking etc ...

1. Dense photometric tracking has an expected catchment radius roughly half the blurring radius, for thin edges, but arbitrarily wide for homogeneous patches.

2. Sparse tracking has potentially a much wider catchment iff the feature identification is correct.





## Glasgow stereo algorithm

This uses _image_correlation_ of a patch UxV of at least 4x4 or more pixels, weighted by the discrete normal Gaussian for the patch size.


Tian, X., Cockshott, P., and Oehler, S. (2014) Acceleration of stereo-
matching on multi-core CPU and GPU. In: 16th IEEE International
Conference on High Performance Computing and Communications (HPCC
2014), 20-22 Aug 2014, Paris, France.


- Image correlation    cor_{l,r} = {cov_{l,r}(x,y) } / { sqr(var_l(x,y))  *  sqr(var_r(x,y)) }


- where covariance   cov_{l,r}(x,y) = sum_{u,v} {  f_l(x+u, y+v)  *  f_r(x+u, y+v)  *  w(u.v)  }

- where  w(u.v)  is the discrete normal Gaussian for the patch size.

- variance var_i(x,y) = sum_(u,v) { f_i(x+u, y+v)**2 * w(u,v) }



- The algorithm then samples +-1 step in u & v  to give total 5 samples around the current estimate.

- Fron this it computes the 2D (u,v) shift for the 2nd order polynomial optimum fit, but clips it to maximum 1 pixel step.

- It iterates within the layer, and then steps down the image pyramid, refining the 2D image warp map.


- It also performs anisotropic diffusion on the disparity map, based on the confidence map ( = the current correlation map).



NB this makes it agnostic to the epipolar lines or other image flow, when computing the binocular disparity map.

The algorithm does require the camera intrinsic matricies and the pose transform to convert the dispartity into an absolute depth map.



## Newton–Raphson accelerated dense photometric tracking

For image a & b, with image gradients a' & b' :

- Estimate of step required, assuming constant gradient   =  sum{ (a-b) / a'  + (b-a) / b'  }

BUT this gives _very_large_ steps where the image gradient is near zero.

Therefore, must clip pixels with < threshold image gradient.



## ESM Gauss-Newton image tracking in LSD-SLAM

ESM = Efficient Second Order Minimization (Ezio & Malis)

NB LSD-Slam is semi-dense and operates on +1% of pixels with the highest image gradient.

1. Semi-Dense Visual Odometry for a Monocular Camera∗
Jakob Engel, Jürgen Sturm, Daniel Cremers
TU München, Germany

2. LSD-SLAM: Large-Scale
Direct Monocular SLAM
Jakob Engel and Thomas Schöps and Daniel Cremers


Increment in SE3:

-  delta Xi_n = -(J_k^T * W * J_c)^{-1}  * J^T * W *rho(Xi_n)


Where

- Xi_n is the current estimate of SE3

- rho is the photometric error

- J_k and J_c are the image jacobians for the keyframe and current image respectively.

- The error function,  E(Xi) = sum{ rho**2(Xi) }

- J^T * J is the Gauss-Newtom approximation of the Hessian of E

- ESM uses ( J_k^T * J_c )  with the Moore-Penrose pseudo-inverse.

- ^{-1} is the inverse of the 6x6 matrix. This approximates the Hessian. (Can use opencv Matx66f.inv() and specify the method.)

- W = W(Xi_n) , is a diagonal weighting matrix, which down-weights large rho pixels. i.e. excludes outliers & mismatched pixels.


### Loop closure in LSD-SLAM

This uses 7DoF Sim3 = SE3 + scaling , to adjust for scale drift when finding the match between keyframes.


## LSD-SLAM mapping

- exhaustive search for the pixel's intensity along the epipolar line (similar to DTAM costvol ray for 1 img pair)

- Followed by sub-pixel matching of disparity, using summ squared deviation error, for 5 equidistant points along the epipolar line.

- NB this is purely 1D search.

- search interval is limited to previous depth estimate +- 2*variance

### depth map variance

- error variance is esimated as a fn of keyframe, curret frame, projection matrix and depth.

    sigma**2_d  =  J_d * SIGMA  * J_d^T

    where SIGMA = covariance of the input error. depends on  photometric disparity error,  geometric disparity error, pixel to inverse depth ratio

    which boild down to: magnitude of the img gradient along the epipolar line,  and the expected parallax movement given ST3 and k camera intrinsic matrix.

    For LSD Slam, sigma**2_d determines (i) which pixels are worth the cost of updating, (ii) the weight of thene measurement in updating the depth map.

### depth map regularization

- "For each frame – after all observations have been incorporated –

    we perform one regularization iteration by assigning each inverse depth value the average of the surrounding inverse depths,

    weighted by their respective inverse variance."

- Edge preservaion : if adjacent pixels are > 2*sigma_d   apart in depth, they do not diffuse to each other.


### outlier removal

- For each successful tracking of a pixel the probability that it is valid is increased

- For each failed tracking of a pixel, the probability tha it is valid is decreased.

- If it falls below some threshold, that depth hypothesis is removed from the map.

- If during depth regularization, all contributing pixels are below some threshold, the hypothesis is removed.

### hole filling

- If an empty pixel has valid neighbours, such that its probability of being a valid trackable point rises above some threshold,

    then it is added, with depth diffused from its valid neighbours.

- This fills holes due to new points being revealed after occlusion.

- It also broadens the depth map around thin edges, which improves tracking.

### depth map propgation

- LSD-slam proagates the depth map and the variance map forwards to each new frame, once the camera tracking step has been completed.



## DTAM tracking component

- SO3 for the first iteration of the first image pyramid layer, SE3 for the rest.

- Project keyframe to current frame to find photometrc error.

- SE3 pose Psi = arg_min_Psi Rho**2(Psi)

- They write "Taylor series expansion about zero, of f(psi)", which requires the first differetial of Rho wt SE3, i.e the Jacobian. This implies that they use Newton–Raphson and chain partial Jacobians wrt pixel coord, to link Rho to SE3.

NB we can compute what steps in SO3 and ST3 at min depth, produce 1 pixel of image flow at each image pyramid layer.

### Robustified tracking

- disregard pixels whose Rho > threshold

- ramp down threshold in each iteration as they converge.

### New keyframe

- when num pixels in predicted image < threshold



## Options for Dynamic SLAM tracking

1. have a sparse tracking recovery mode for very large SE3

2. Bin-sort keyframe pixels by image_gradient**2, and use top n%.

    NB this would still focus on the most trackable points, regardless of the degraa of blurring.

    - need to do it per DoF, otherwise may not be using the right pixels.

    - need to ensure coverage in the image, but bank sky etc may have none.

3. for debugging, need to display larger rho_maps

4. Need to check stepsize, fitting gradient, and rho wrt changing image pyramid layers.


## Options for Dynamic SLAM mapping

1. confidence map : img_grad / Rho , use for anisotropic diffusion,

2. limit diffusion if difference in depth > 2 sigma_d

3. If we model 2D image disparity using image co-variance, then we can explain motion wrt parallax + deformation + rendering ?

