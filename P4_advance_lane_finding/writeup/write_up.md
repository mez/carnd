
# Advanced Lane Finding Project

The latest lane finding techniques are most likely using a combination of traditional and deep learning computer vision tactics and probably have multi redundant systems. In this project, we will explore and understand traditional computer vision methods.

### The goals / steps of this project are the following:

1. Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
2. Apply a distortion correction to raw images.
3. Use color transforms, gradients, etc., to create a thresholded binary image.
4. Apply a perspective transform to rectify binary image ("birds-eye view").
5. Detect lane pixels and fit to find the lane boundary.
6. Determine the curvature of the lane and vehicle position with respect to center.
7. Warp the detected lane boundaries back onto the original image.
8. Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

## Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The `calibrate_camera_and_pickle_mtx_dist()` method in the `calc_calibration.py` takes care of this step.

First, I create `objpoints`, which are the (x, y, z) coordinates of the chessboard corners in the real world space. Then I create `imgpoints`, which are the (x, y) coordinates of the corners in the image plane.

I used `cv2.findChessboardCorners()` method to populate the `imgpoints` list with the corners of each successful chessboard detection.

I then used the output `objpoints` and `imgpoints` to compute the camera matrix and distortion coefficients using the `cv2.calibrateCamera()` method. I used the camera matrix and distortion coefficients to the test image using the `cv2.undistort()` method and obtained this result:

[image1]: undistored_calibration_5.png "Undistorted"
![alt text][image1]


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

[image2]: undistored_comp_test4.jpg "Road Transformed"
![alt text][image2]


### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I tried many combinations including directional, Sobel-y, etc; I settled on the S channel with thresholds (20, 100) from the HLS color space and Sobel-x gradient with thresholds (170, 255) and combined them for a binary image. The code can be found in the `main.py` file and the method is called `binary_image_via_threshold()`.

An example is below:

[image3]: combined.png "Combined Binary Image Example"
![alt text][image3]

### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is in `WarpTransformer` class in the `warp_transformer.py`. I hardcoded the source and destination points. Then I used those points to create an instance of the `WarpTransformer` class. Example is below...

```
src = np.array([[262, 677], [580, 460], [703, 460], [1040, 677]]).astype(np.float32)
dst = np.array([[262, 720], [262, 0], [1040, 0], [1040, 720]]).astype(np.float32)

# Create transformer object
transformer = WarpTransformer(src, dst)

#to get bird-eye view
binary_warped = transformer.to_birdview(combined)

#to get original view back
unwarped = transformer.to_normal(color_warped)
```

Example image is below:

[image4]: bird-eye-view.png "Warp Example"
![alt text][image4]

### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

In the `main.py` file there is a `sliding_window() and non_sliding()` method that implements this feature; I simply followed the techniques from the lectures:

1. Read an image
2. Undistort the image
3. Create a combined binary image using S channel and Sobel-x gradient thresholds
4. Warp to a bird-eye view
5. Calculate a histogram of the lower half of the image
6. Use the two maximum peaks from the histogram as starting points for finding the lanes
7. Use the sliding window technique to find the lane pixels
9. Use `np.polyfit()` to fit a 2nd order polynomial to the pixels

I only use the sliding window technique the first frame using the `sliding_window()` method; then I search for lane pixels around the fitted line from the previous frame using the `non_sliding()` method.

### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The code for this is located in the `main.py` file in the `pipeline()` method located in lines 184-194 and 217.

First, adjust the points of the found lane lines to account for meters per pixel in both X and Y. Next I take those adjusted points and fit a polynomial with `np.polyfit()`.
Snippet of the code is below:

```
# Fit new polynomials to x,y in world space
left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, deg=2)
right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, deg=2)

# Calculate radii of curvature in meters
y_eval = np.max(ploty)  # Where radius of curvature is measured
left_curverad = ((1 +(2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
right_curverad=((1 +(2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) /np.absolute(2*right_fit_cr[0])

```

Position of the vehicle with respect to the center is in lines 196-198 in the same `pipeline()` method:

```
midpoint = np.int(start_img.shape[1]/2)
middle_of_lane = (right_fitx[-1] - left_fitx[-1]) / 2.0 + left_fitx[-1]
offset = (midpoint - middle_of_lane) * xm_per_pix
```

### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

[image6]: final_pipeline.png "Output"
![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

[Final video link](./final_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

This pipeline does not work well on the challenge videos.

In my pipeline I probably could of first ran the warp to remove the lines that are not the lane lines so we don't waste compute on this with the thresholding. To make it more robust, I would of added redunant methods of detecting the lines and explore more color spaces to handle different lighting conditions etc. Also, I am not sure how this pipeline would handle night time or different weather patterns.

Overall, after I understood the general idea, most of the time was spent tweaking many nobs (boring). This quickly became time consuming! I wish we could of had additional data like GPS and other sensors, so that we could use deep learning with something like particle filters to try and detect lane lines/localization!
