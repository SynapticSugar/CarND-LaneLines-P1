# **Finding Lane Lines on the Road** 

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./flow.png

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

The basic lane finding pipeline consisted of 5 steps. First the image was converted to grayscale, smoothed and an area mask was applied. Second, lines were detected by applying Hough line detection to the Canny edges. Third, the lines were classified into left and right lines and pruned for outliers. Fourth, the left and right candidate line sets were averaged to find the best lane line for each side. The resulting lane lines were then passed through a low pass filter. Finally, the final lane lines were overlaid back on to the original image. A global dashboard object was used to hold threshold and masking parameters, the currnent lane lines, as well as a left/right bias adjustment for where the camera is placed with respect to the center of the car.

![alt text][image1]

1. Image processing

	There are 3 steps in the image processing section of the pipeline.

	1. Grayscale

		 After opening the image, the first thing I did was apply the grayscale function. This converts the RGB image into intensity values between 0 and 255. This makes the pipeline simplier in that it only needs to work on one channel instead of three and the computation time is reduced. The intensity contains the most relevant information for this project, although, we do give up some information that could be used to find yellow lines in this step.

	2. Gaussian Smoothing
		
		Applying a 5x5 Gaussian kernel to the image will remove noise and puts a lower bound on the size of features that contribute to the line detection algorithm. For busy sceens, this removes clutter from the Hough lines transform making it easier to pick out the lane lines.
	
	3. Area Mask

		Applying the area mask helps remove lines that would otherwise be detected in the areas of the image we know lane lines would not exist.  For example, lane lines would not exist above the horizon or they would be in the air and we are not driving a hover craft, so it makes sense to block those out.  A level camera would generally have the horizon at about 50% of the height of the image, assuming the ground is generally flat and level. The same can be said for the road way itself.  Since the road is generally a set of parallel lines that disapear into the horizon or "vanishing point", we generally have a mask that looks like a pyramid where the left and right sides are angled inward. As we get close to the vanishing point, the lane lines start to blur together and even warp or bend.  The area mask cuts the top off the pyramid in this area as it may lead to inaccuracies in line detection. In the challenging video case, it also became necessarry to mask off the hood of the car at the bottom of the image, as it would introduce false line detections.

2. Hough Lines Detection

	Once the area mask is applied, the remaining area has a Canny edge detection filter applied, which allows lines to be detected with the Hough line transform.  This basically maps each point into a curve in the Hough space (angle, radius).  The intersection of curves is where the most number of points, or votes, exists.  The lines with the most votes are used to generate the list of lines.  The min and max of these points in image space are used to form the verticies of the lines.  The Hough thresholds were tuned using a snapshot of the video under the worst case conditions approximately 5 seconds into the "challenge.mp4" video called "atest.jpg".  Only one static set of Canny thresholds were used for all image and videos.
	https://en.wikipedia.org/wiki/Canny_edge_detector

3. Classification and Outlier Removal
	
	Lines were classified into left and right by calculating thier slopes. Not all lines were true lane lines however, some followed cracks and shadows on the road, some followed cars, and some followed road repair lines.  Outliner removal was absolutely needed in these cases.  It was assumed that the car would not be changing lanes or altering course dramatically on the road.  This allowed the removal of lines which did not fit into a certain angle range for the left and right lanes.  As well, any lines that crossed the boundary of the middle of the image were assumed to not be lanes and were removed from the list of candidates.
	Outlier removal was the most important step and had the most impact in making the lane finding project work well for all the videos.

4. Fit and Filter Lane Lines

	In order to draw a single line on the left and right lanes, I added a *draw_filtered_lines()* function that performs the classification and outlier removal step above, average fits left and right lane lines to the Hough lines, and low pass filters the result.

	1. Average Fit Lanes

		It was found that a averaging the verticies of all the left and right lane line candidates produced a very representative set of lane lines. Averaging is very sensitive to outliers, and it was very important to remove them before we got to this step. The lane lines were artifically extended to the top of the area mask and down the bottom of the image.

	2. Low Pass Filter

		It was found that a low pass filter improved the jitter in the lanes lines present from frame to frame, and made them easier to observe.  When a lane was not found, it would keep the last lane detected for display.  The problem with filtering is that it adds a time delay to the lane detection response. The filter rate had to be tuned to avoid delays while reducing jitter.

5. Overlay

	The last step in the pipeline was to generate a new image or video file that contained a set of lane lines overlaid on the original.  The 'weighted_img' function was not altered.  The *line_img* itself was created in the *draw_filtered_lines()* function.  It was useful to add debug drawing to the *line_img* during development.


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be the outcome if the car changes lanes.  The outlier rejection makes the assumption that the car is not going to change lanes. This would result in a lack of lane detection during the move until the car is completely in the new lane.

Another shortcoming could be if the car is on a road that is curved.  It would violate the assumption that the lane lines are more or less straight to the vanishing point.  It may still manage to track the road close to the vehicle, but the lane lines would be incorreclty mapped straight instead of curved.

Another problem would occur if a car merged into the lane in front of you.  We make the assumption that the whole lane is in free-space.  A car could cause false lane line detection.

Another problem with the pipeline is its inability to interpret light, shadow, and varying shades of the road.

### 3. Suggest possible improvements to your pipeline

Possible improvements to the pipeline are listed below.

1. Improve yellow lane detection.

	During or before the conversion to grayscale, one could increase the intensity of pixels which match the yellow lane colour, thus boosting its chances of being detected in the grayscale image.

2. Use RANSAC to fit the best lane line
	
	Outlier rejection is crucial for fitting lines.  Although we rejected many false line detections with the current outlier removal, this step sacrificed the ability of being able to change lanes. A better way would be to relax the outlier removal step and let RANSAC be the primary outlier rejector.  It interratively forms a best match based on a local consensus of inliers vs outliers.
	https://en.wikipedia.org/wiki/Random_sample_consensus

3. Dynamic thresholding

	Light and shadow can to some degree be mittigated by performing a threshold based in the local neighbourhood of intensity values. The drawback to this would be added complexity and computation time. 
	OpenCV  adaptiveThreshold(): http://docs.opencv.org/3.2.0/d7/d1b/group__imgproc__misc.html#ga72b913f352e4a1b1b397736707afcde3

4. Multiple area masks

	Define multiple sets of area masks that divide the highway freespace into a far, middle, and close set.  Each could have a different set of lane lines and angle.  Ideally, the lane lines should be joined at the verticies.  This will help with lanes following around corners, or have a better chance that at least one area will see the road when a car is in front of us, or when we are passing over light and dark patches on the road.

5. Find the homography

	This really only applies to image sequences, but if you can find the homography of the scene from image to image, you could find the camera ego motion. This is the change in position and rotation of the camera or car, with respect to the world, from frame to frame. This would give us an additional prediction step in the lane finding pipeline.  We could also map our lanes onto the surface of the road in different perspectives (ie. top down), which could be used to measure ground surface distances to various objects.
	https://en.wikipedia.org/wiki/Homography

