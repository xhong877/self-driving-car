# **Finding Lane Lines on the Road**


---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps.
1. Turn it into gray for later process
2. Use Gaussian_blur to make the pic vague, so we can avoid noises that may be mistakenly consider as edge
3. Use Canny edge detector to find out edges
4. Find Lines in the region of interest through edges using hough transformation.
5. Color the lines we find in step 4 and add it to the original image. This is the tricky part, especially the draw_lines function. We need to modified it to draw a single line. The basic idea here is we need to calculate the slope and bias of each line, and find two points to draw a line.(one on the upper limit of the region of interest, one on the down limit of the region of interest)


### 2. Identify potential shortcomings with your current pipeline

1. Throught the pipeline, there are still some irrelevent points left, thus leads to lines that didn't fit the lanes.

2.  It couldn't see far enough, specifically, the failure to recognize curve.


### 3. Suggest possible improvements to your pipeline

1. Use the color filter to screen out irrelevent points.
2. Find a line detector other than houghline to identify curves.
